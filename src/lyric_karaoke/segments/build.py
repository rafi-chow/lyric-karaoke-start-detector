
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
