"""
CI smoke test:
- Load toy-trained model
- Run a minimal predict pass on frame-aligned synthetic input
- Assert shapes + numeric sanity
"""

from pathlib import Path
import numpy as np
import joblib

from lyric_karaoke.datasets.frame_grid import build_frame_grid


def main():
    model_path = Path("models/toy_lr.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Toy model not found. Did scripts/train_toy.py run?")

    model = joblib.load(model_path)

    frame_grid = build_frame_grid(5.0, frame_duration=0.5)
    n_frames = len(frame_grid)

    X = np.random.normal(0, 1, size=(n_frames, 80)).astype(np.float32)
    probs = model.predict_proba(X)[:, 1]

    # One prob per canonical frame
    assert probs.shape == (n_frames,)
    # No NaNs or infs
    assert np.isfinite(probs).all()

    print("Smoke predict OK")


if __name__ == "__main__":
    main()
