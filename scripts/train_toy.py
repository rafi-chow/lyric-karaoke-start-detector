"""
Reproducible toy training (no external datasets).

Goal:
- Train a tiny logistic regression model on synthetic frame-aligned features
- Produce models/toy_lr.pkl (NOT meant to be committed)
- Used by CI to validate training → inference plumbing
"""

from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lyric_karaoke.datasets.frame_grid import build_frame_grid


OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)


def generate_toy_song(duration_sec: float = 30.0, frame_duration: float = 0.5):
    """
    Generate synthetic frame-aligned features.

    - First 10s: "no lyrics" (label 0)
    - After 10s: "lyrics" (label 1) with a stronger feature signal
    """
    frame_grid = build_frame_grid(duration_sec, frame_duration=frame_duration)
    n_frames = len(frame_grid)

    X = np.random.normal(0, 0.6, size=(n_frames, 80)).astype(np.float32)
    y = np.zeros(n_frames, dtype=int)

    for i, f in enumerate(frame_grid):
        if f["t_start"] >= 10.0:
            y[i] = 1
            X[i] += 2.0  # separable signal during lyric region

    return X, y


def main():
    Xs = []
    ys = []

    # Make a few synthetic tracks to avoid trivially overfitting a single sequence
    for _ in range(6):
        X, y = generate_toy_song()
        Xs.append(X)
        ys.append(y)

    X_train = np.vstack(Xs)
    y_train = np.concatenate(ys)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    out_path = OUT_DIR / "toy_lr.pkl"
    joblib.dump(model, out_path)

    print(f"Saved toy model → {out_path}")


if __name__ == "__main__":
    main()
