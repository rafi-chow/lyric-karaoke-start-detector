import numpy as np
from lyric_karaoke.features.aggregate import aggregate_mel_to_frames


def test_aggregate_mel_to_frames_basic():
    # 3 mel frames, 2 mel bins
    mel = np.array([
        [1.0, 1.0],  # t=0.0
        [3.0, 3.0],  # t=0.25
        [5.0, 5.0],  # t=0.75
    ])

    mel_times = np.array([0.0, 0.25, 0.75])

    frame_grid = [
        {"frame_idx": 0, "t_start": 0.0, "t_end": 0.5},
        {"frame_idx": 1, "t_start": 0.5, "t_end": 1.0},
    ]

    X = aggregate_mel_to_frames(mel, mel_times, frame_grid)

    # frame 0 averages first two mel frames
    assert np.allclose(X[0], [2.0, 2.0])

    # frame 1 averages last mel frame
    assert np.allclose(X[1], [5.0, 5.0])
