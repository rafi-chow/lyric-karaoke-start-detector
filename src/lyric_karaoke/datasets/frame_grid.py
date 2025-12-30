import math
from typing import List, Dict


def build_frame_grid(duration_sec: float, frame_duration: float = 0.5) -> List[Dict]:
    """
    Build the canonical frame grid for a track.

    Returns a list of dicts with:
      - frame_idx
      - t_start
      - t_end (clipped to duration)
    """
    frames: List[Dict] = []

    # total number of frames (include final partial frame)
    num_frames = math.ceil(duration_sec / frame_duration)

    for frame_idx in range(num_frames):
        t_start = frame_idx * frame_duration
        t_end = min(t_start + frame_duration, duration_sec)

        frames.append({
            "frame_idx": frame_idx,
            "t_start": t_start,
            "t_end": t_end,
        })

    return frames
