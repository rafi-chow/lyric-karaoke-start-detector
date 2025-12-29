import numpy as np

def median_smooth(y_pred, window_size=5):
    """
    Median filter over binary predictions.
    window_size should be odd (3, 5, 7).
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    pad = window_size // 2
    y = np.pad(y_pred, (pad, pad), mode="edge")

    smoothed = []
    for i in range(len(y_pred)):
        smoothed.append(int(np.median(y[i:i + window_size])))

    return np.array(smoothed, dtype=int)

def enforce_min_consecutive(y_pred, min_consecutive=3):
    y = y_pred.copy()
    n = len(y)
    out = np.zeros_like(y)

    run_start = None
    run_len = 0

    for i in range(n):
        if y[i] == 1:
            if run_start is None:
                run_start = i
            run_len += 1

            if run_len >= min_consecutive:
                out[run_start:i+1] = 1
        else:
            run_start = None
            run_len = 0

    return out


