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
