from pathlib import Path
import random
import json
import csv

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.harmonix import (
    load_harmonix_intervals,
    intervals_to_frame_labels,
    load_harmonix_mel,
)

# ---------------- CONFIG ----------------
SEED = 42
FRAME_DURATION = 0.5

# Harmonix annotation paths
HARMONIX_ROOT = Path(r"C:\Users\psult\Downloads\harmonixset-main\harmonixset-main")
SEGMENTS_DIR = HARMONIX_ROOT / "dataset" / "segments"
METADATA_CSV = HARMONIX_ROOT / "dataset" / "metadata.csv"

# Harmonix mel-spectrograms
MELSPEC_DIR = Path(r"C:\Users\psult\Downloads\Harmonix_melspecs\melspecs")

# Output
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "harmonix_lr.pkl"
CONFIG_PATH = MODELS_DIR / "harmonix_lr_config.json"


# ---------------- UTILITIES ----------------
def track_id_from_segment_file(p: Path) -> str:
    return p.stem


def load_metadata_track_ids(metadata_csv: Path) -> set[str]:
    ids = set()
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["File"])
    return ids


def find_available_track_ids() -> list[str]:
    seg_ids = {
        track_id_from_segment_file(p)
        for p in SEGMENTS_DIR.glob("*.txt")
        if not p.name.startswith("._")
    }
    meta_ids = load_metadata_track_ids(METADATA_CSV)

    ids = sorted(seg_ids & meta_ids)
    return ids


def mel_path_for_track(track_id: str) -> Path:
    p = MELSPEC_DIR / f"{track_id}-mel.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing mel file for {track_id}")
    return p

def predict_with_threshold(model, X, thr):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= thr).astype(int)


# ---------------- DATA BUILDING ----------------
def build_song_example(track_id: str):
    """
    Returns:
        X: (num_frames, 80)
        y: (num_frames,)
    """
    intervals, song_duration = load_harmonix_intervals(
        track_id=track_id,
        segments_dir=SEGMENTS_DIR,
        metadata_csv=METADATA_CSV,
    )

    _, y = intervals_to_frame_labels(
        intervals=intervals,
        song_duration=song_duration,
        frame_duration=FRAME_DURATION,
    )
    y = np.array(y, dtype=np.int32)

    mel_path = mel_path_for_track(track_id)
    mel = load_harmonix_mel(mel_path)  # (T, 80)

    T = mel.shape[0]

    # seconds per mel frame
    sec_per_mel_frame = song_duration / T

    # mel frames per 0.5s chunk
    frames_per_chunk = max(1, int(round(FRAME_DURATION / sec_per_mel_frame)))

    X_chunks = []

    for start in range(0, T, frames_per_chunk):
        end = start + frames_per_chunk
        if end > T:
            break  # drop incomplete tail
        chunk = mel[start:end].mean(axis=0)
        X_chunks.append(chunk)

    X = np.vstack(X_chunks)  # (num_chunks, 80)

    # Final alignment (rounding only)
    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    return X, y


def concat_songs(track_ids: list[str]):
    Xs, ys = [], []

    for tid in track_ids:
        try:
            X, y = build_song_example(tid)
            Xs.append(X)
            ys.append(y)
            print(f"[OK] {tid}: frames={len(y)} positives={y.sum()}")
        except Exception as e:
            print(f"[SKIP] {tid}: {e}")

    if not Xs:
        raise RuntimeError("No training data built.")

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    return X_all, y_all


def split_track_ids(track_ids, train=0.8, val=0.1, test=0.1):
    ids = track_ids[:]
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# ---------------- MAIN ----------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    track_ids = find_available_track_ids()
    print(f"Found {len(track_ids)} usable tracks")

    train_ids, val_ids, test_ids = split_track_ids(track_ids)
    print(f"Split → train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    print("\n--- Building TRAIN set ---")
    X_train, y_train = concat_songs(train_ids)

    print("\n--- Building VAL set ---")
    X_val, y_val = concat_songs(val_ids) if val_ids else (None, None)

    print("\n--- Building TEST set ---")
    X_test, y_test = concat_songs(test_ids) if test_ids else (None, None)

    # Train model
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
    
    for thr in [0.3, 0.4, 0.5, 0.6]:
        yv = predict_with_threshold(model, X_val, thr)
        print(f"\n=== VAL thr={thr} ===")
        print(classification_report(y_val, yv, digits=3))


    # Save outputs
    joblib.dump(model, MODEL_PATH)

    config = {
        "seed": SEED,
        "frame_duration": FRAME_DURATION,
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "mel_bins": 80,
        "notes": "Trained on Harmonix mel-spectrograms; labels derived from segments.",
    }

    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved config → {CONFIG_PATH}")


if __name__ == "__main__":
    main()
