import numpy as np
from lyric_karaoke.features.aggregate import aggregate_mel_to_frames


def test_aggregate_mel_to_frames_basic_alignment():
    """
    Mel frames fall into canonical frames using:
      t_start <= mel_time < t_end

    Frame grid:
      [0.0, 0.5)
      [0.5, 1.0)
      [1.0, 1.5)

    Mel frames at:
      0.1, 0.2  -> frame 0
      0.6       -> frame 1
      (none)    -> frame 2
    """

    mel = np.array([
        [1.0, 1.0],   # t = 0.1
        [3.0, 3.0],   # t = 0.2
        [10.0, 10.0], # t = 0.6
    ], dtype=np.float32)

    mel_times = np.array([0.1, 0.2, 0.6], dtype=np.float32)

    frame_grid = [
        {"frame_idx": 0, "t_start": 0.0, "t_end": 0.5},
        {"frame_idx": 1, "t_start": 0.5, "t_end": 1.0},
        {"frame_idx": 2, "t_start": 1.0, "t_end": 1.5},
    ]

    X = aggregate_mel_to_frames(mel, mel_times, frame_grid)

    # frame 0: mean of first two rows
    assert np.allclose(X[0], [2.0, 2.0])

    # frame 1: single mel frame
    assert np.allclose(X[1], [10.0, 10.0])

    # frame 2: silence -> zeros
    assert np.allclose(X[2], [0.0, 0.0])
