from lyric_karaoke.karaoke_logic.choose_start import choose_karaoke_start
def test_single_long_early_segment_is_valid_start():
    """
    If there is one long lyric segment early in the song,
    karaoke_start should be the start of that segment.
    """

    segments = [
        {"start": 20.0, "end": 60.0},   # early, long
        {"start": 90.0, "end": 120.0},  # later segment
    ]

    song_duration = 180.0

    karaoke_start = choose_karaoke_start(segments, song_duration)

    assert karaoke_start == 20.0

