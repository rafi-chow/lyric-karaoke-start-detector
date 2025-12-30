"""Train a simple lyric-presence classifier on Harmonix melspecs.

Key invariant:
  Everything is aligned to the canonical 0.5s frame grid.

This script:
  - builds a canonical frame grid per track
  - aggregates Harmonix mel frames into that grid (aggregate_mel_to_frames)
  - builds labels aligned to the same grid
  - trains a (StandardScaler + LogisticRegression) pipeline

Example (Windows):
  python scripts/train_harmonix.py \
    --harmonix-root C:\\path\\to\\harmonixset-main \
    --melspec-dir   C:\\path\\to\\melspecs \
    --out-model     models\\harmonix_lr.pkl
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lyric_karaoke.datasets.frame_grid import build_frame_grid
from lyric_karaoke.datasets.harmonix import (
    load_harmonix_intervals,
    intervals_to_frame_labels_for_grid,
    load_harmonix_mel_and_times,
)
from lyric_karaoke.features.aggregate import aggregate_mel_to_frames


def track_id_from_segment_file(p: Path) -> str:
    return p.stem


def load_metadata_track_ids(metadata_csv: Path) -> set[str]:
    ids: set[str] = set()
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["File"])
    return ids


def find_available_track_ids(segments_dir: Path, metadata_csv: Path) -> list[str]:
    seg_ids = {
        track_id_from_segment_file(p)
        for p in segments_dir.glob("*.txt")
        if not p.name.startswith("._")
    }
    meta_ids = load_metadata_track_ids(metadata_csv)
    return sorted(seg_ids & meta_ids)


def mel_path_for_track(track_id: str, melspec_dir: Path) -> Path:
    p = melspec_dir / f"{track_id}-mel.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing mel file for {track_id}: {p}")
    return p


def predict_with_threshold(model, X: np.ndarray, thr: float) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    return (probs >= thr).astype(int)


def build_song_example(
    track_id: str,
    *,
    segments_dir: Path,
    metadata_csv: Path,
    melspec_dir: Path,
    frame_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one song as (X, y) aligned to the canonical frame grid."""

    intervals, song_duration = load_harmonix_intervals(
        track_id=track_id,
        segments_dir=segments_dir,
        metadata_csv=metadata_csv,
    )

    frame_grid = build_frame_grid(song_duration, frame_duration=frame_duration)
    y = intervals_to_frame_labels_for_grid(intervals, frame_grid)

    mel_path = mel_path_for_track(track_id, melspec_dir)
    mel, mel_times = load_harmonix_mel_and_times(mel_path, song_duration)
    X = aggregate_mel_to_frames(mel, mel_times, frame_grid)

    if len(X) != len(y):
        # Should not happen; both are defined on frame_grid.
        n = min(len(X), len(y))
        X = X[:n]
        y = y[:n]

    return X, y


def concat_songs(
    track_ids: list[str],
    *,
    segments_dir: Path,
    metadata_csv: Path,
    melspec_dir: Path,
    frame_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for tid in track_ids:
        try:
            X, y = build_song_example(
                tid,
                segments_dir=segments_dir,
                metadata_csv=metadata_csv,
                melspec_dir=melspec_dir,
                frame_duration=frame_duration,
            )
            Xs.append(X)
            ys.append(y)
            print(f"[OK] {tid}: frames={len(y)} positives={int(y.sum())}")
        except Exception as e:
            print(f"[SKIP] {tid}: {e}")

    if not Xs:
        raise RuntimeError("No training data built.")

    return np.vstack(Xs), np.concatenate(ys)


def split_track_ids(
    track_ids: list[str],
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    assert abs((train + val + test) - 1.0) < 1e-6
    ids = track_ids[:]
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)

    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--harmonix-root", type=Path, default=None,
                   help="Path to harmonixset-main (contains dataset/segments + dataset/metadata.csv)")
    p.add_argument("--segments-dir", type=Path, default=None,
                   help="Override: path to dataset/segments")
    p.add_argument("--metadata-csv", type=Path, default=None,
                   help="Override: path to dataset/metadata.csv")
    p.add_argument("--melspec-dir", type=Path, required=True,
                   help="Directory containing Harmonix mel .npy files (e.g. <track_id>-mel.npy)")

    p.add_argument("--frame-duration", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tracks", type=int, default=None,
                   help="Optional: cap number of tracks for faster experiments")

    p.add_argument("--out-model", type=Path, default=Path("models/harmonix_lr.pkl"))
    p.add_argument("--out-config", type=Path, default=Path("models/harmonix_lr_config.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve annotation paths
    if args.harmonix_root is None and (args.segments_dir is None or args.metadata_csv is None):
        raise SystemExit(
            "Provide either --harmonix-root, or both --segments-dir and --metadata-csv"
        )

    segments_dir = args.segments_dir
    metadata_csv = args.metadata_csv

    if args.harmonix_root is not None:
        segments_dir = segments_dir or (args.harmonix_root / "dataset" / "segments")
        metadata_csv = metadata_csv or (args.harmonix_root / "dataset" / "metadata.csv")

    assert segments_dir is not None and metadata_csv is not None

    if not segments_dir.exists():
        raise SystemExit(f"segments_dir not found: {segments_dir}")
    if not metadata_csv.exists():
        raise SystemExit(f"metadata_csv not found: {metadata_csv}")
    if not args.melspec_dir.exists():
        raise SystemExit(f"melspec_dir not found: {args.melspec_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    track_ids = find_available_track_ids(segments_dir, metadata_csv)
    if args.max_tracks is not None:
        track_ids = track_ids[: args.max_tracks]
    print(f"Found {len(track_ids)} usable tracks")

    train_ids, val_ids, test_ids = split_track_ids(track_ids, seed=args.seed)
    print(f"Split → train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    print("\n--- Building TRAIN set ---")
    X_train, y_train = concat_songs(
        train_ids,
        segments_dir=segments_dir,
        metadata_csv=metadata_csv,
        melspec_dir=args.melspec_dir,
        frame_duration=args.frame_duration,
    )

    print("\n--- Building VAL set ---")
    X_val, y_val = (
        concat_songs(
            val_ids,
            segments_dir=segments_dir,
            metadata_csv=metadata_csv,
            melspec_dir=args.melspec_dir,
            frame_duration=args.frame_duration,
        ) if val_ids else (None, None)
    )

    print("\n--- Building TEST set ---")
    X_test, y_test = (
        concat_songs(
            test_ids,
            segments_dir=segments_dir,
            metadata_csv=metadata_csv,
            melspec_dir=args.melspec_dir,
            frame_duration=args.frame_duration,
        ) if test_ids else (None, None)
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])
    model.fit(X_train, y_train)

    print("\n=== TRAIN REPORT ===")
    print(classification_report(y_train, model.predict(X_train), digits=3))

    if X_val is not None:
        print("\n=== VAL REPORT ===")
        print(classification_report(y_val, model.predict(X_val), digits=3))

    if X_test is not None:
        print("\n=== TEST REPORT ===")
        print(classification_report(y_test, model.predict(X_test), digits=3))

    if X_val is not None:
        for thr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            yv = predict_with_threshold(model, X_val, thr)
            print(f"\n=== VAL thr={thr} ===")
            print(classification_report(y_val, yv, digits=3))

    # Save outputs
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)

    config = {
        "seed": args.seed,
        "frame_duration": args.frame_duration,
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "mel_bins": 80,
        "notes": "Trained on Harmonix mel-spectrograms; features aggregated to canonical frame grid.",
    }
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(json.dumps(config, indent=2))

    print(f"\nSaved model → {args.out_model}")
    print(f"Saved config → {args.out_config}")


if __name__ == "__main__":
    main()
