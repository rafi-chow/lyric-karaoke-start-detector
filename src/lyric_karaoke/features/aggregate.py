import numpy as np
from typing import List, Dict


def aggregate_mel_to_frames(
    mel: np.ndarray,
    mel_times: np.ndarray,
    frame_grid: List[Dict],
) -> np.ndarray:
    num_frames = len(frame_grid)
    n_mels = mel.shape[1]

    X = np.zeros((num_frames, n_mels), dtype=np.float32)

    for i, frame in enumerate(frame_grid):
        t0 = frame["t_start"]
        t1 = frame["t_end"]

        # find mel frames that overlap this canonical frame
        mask = (mel_times >= t0) & (mel_times < t1)

        if np.any(mask):
            X[i] = mel[mask].mean(axis=0)
        # else: leave zeros (silence)

    return X
