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

def predictions_to_segments(times, y_pred, frame_duration=0.5):
    """
    Converts frame-level vocal predictions into contiguous
    time segments.
    """
    segments = []
    in_segment = False
    seg_start = None

    for t, label in zip(times, y_pred):
        if label == 1 and not in_segment:
            in_segment = True
            seg_start = t
        elif label == 0 and in_segment:
            in_segment = False
            seg_end = t + frame_duration
            segments.append({
                "start": float(seg_start),
                "end": float(seg_end)
            })


    if in_segment:
        seg_end = times[-1] + frame_duration
        segments.append({
            "start": float(seg_start),
            "end": float(seg_end)
        })

    return segments

def clean_segments(segments, min_duration=1.5, merge_gap=1.0):
    if not segments:
        return []

    cleaned = []
    current = segments[0].copy()

    for seg in segments[1:]:
        # If close enough, merge
        if seg["start"] - current["end"] <= merge_gap:
            current["end"] = seg["end"]
        else:
            duration = current["end"] - current["start"]
            if duration >= min_duration:
                cleaned.append(current)
            current = seg.copy()

    # Handle last segment
    duration = current["end"] - current["start"]
    if duration >= min_duration:
        cleaned.append(current)

    return cleaned

