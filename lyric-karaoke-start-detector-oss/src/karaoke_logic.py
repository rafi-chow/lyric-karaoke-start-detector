INTRO_IGNORE = 12.0         # keeps you out of true intro noise
MIN_SEG_DURATION = 1.0      # segment must be at least this long to be a candidate

WINDOW = 45.0               # lookahead window for "does lyrics continue?"
MIN_SEGMENTS_IN_WINDOW = 2  # must see at least this many segments soon after
MIN_TOTAL_LYRIC_IN_WINDOW = 6.0  # total lyric seconds inside the window


def choose_karaoke_start(segments, song_duration):
    """
    Pick karaoke start as the first segment that looks like the start of
    a sustained lyrics region.

    We prefer the earliest candidate segment that has enough "lyrics density"
    soon after it (multiple segments + enough total lyric time in a lookahead window).
    """
    if not segments:
        return None

    # sort by start time
    segs = sorted(segments, key=lambda s: float(s["start"]))

    # filter to plausible candidates (after intro + long enough)
    candidates = []
    for seg in segs:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start

        if start < INTRO_IGNORE:
            continue
        if dur < MIN_SEG_DURATION:
            continue
        candidates.append(seg)

    # helper: total lyric seconds and segment count in [t0, t0+WINDOW]
    def window_stats(t0):
        t1 = t0 + WINDOW
        total = 0.0
        count = 0
        for s in segs:
            s0 = float(s["start"])
            s1 = float(s["end"])

            if s0 < t0:
                continue
            if s0 > t1:
                break

            # add overlap with window
            overlap_start = max(s0, t0)
            overlap_end = min(s1, t1)
            if overlap_end > overlap_start:
                total += (overlap_end - overlap_start)
                count += 1
        return count, total

    # 1) preferred: earliest candidate with enough density
    for seg in candidates:
        t0 = float(seg["start"])
        count, total = window_stats(t0)
        if count >= MIN_SEGMENTS_IN_WINDOW and total >= MIN_TOTAL_LYRIC_IN_WINDOW:
            return t0

    # 2) fallback: earliest plausible candidate
    if candidates:
        return float(candidates[0]["start"])

    # 3) final fallback: first segment start
    return float(segs[0]["start"])
