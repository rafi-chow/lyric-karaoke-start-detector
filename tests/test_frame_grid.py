from lyric_karaoke.datasets.frame_grid import build_frame_grid


def test_frame_grid_includes_final_partial_frame():
    frames = build_frame_grid(duration_sec=1.1, frame_duration=0.5)

    # Expect 3 frames:
    # 0.0–0.5
    # 0.5–1.0
    # 1.0–1.1 (shortened)
    assert len(frames) == 3

    assert frames[0]["frame_idx"] == 0
    assert frames[0]["t_start"] == 0.0
    assert frames[0]["t_end"] == 0.5

    assert frames[2]["frame_idx"] == 2
    assert frames[2]["t_start"] == 1.0
    assert frames[2]["t_end"] == 1.1
