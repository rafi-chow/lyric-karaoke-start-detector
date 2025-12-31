import numpy as np

from lyric_karaoke.segments.build import predictions_to_segments
from lyric_karaoke.segments.clean import clean_segments
from lyric_karaoke.segments.smooth import median_smooth, enforce_min_consecutive
def test_flicker_noise_does_not_create_segments():
    """
    Isolated positive frames (flicker noise) should be removed
    by smoothing + min-consecutive enforcement.
    """

    frame_duration = 0.5
    times = np.arange(0, 10) * frame_duration  # 10 frames, 5 seconds total

    # Flicker pattern: isolated 1s
    y = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=int)

    # Apply the same smoothing used in inference
    y_smooth = median_smooth(y, window_size=5)
    y_clean = enforce_min_consecutive(y_smooth, min_consecutive=3)

    raw_segments = predictions_to_segments(times, y_clean, frame_duration)
    segments = clean_segments(raw_segments, min_duration=1.5, merge_gap=1.0)

    assert segments == []
