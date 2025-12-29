# tests/test_choose_start.py
import importlib

import pytest


def seg(start, end):
    """Helper to build segments in the same dict format your pipeline uses."""
    return {"start": float(start), "end": float(end)}


def load_choose_module():
    """
    Import the module that contains choose_karaoke_start.
    Update the import path here if you moved files.
    """
    # ✅ preferred final path (recommended structure)
    return importlib.import_module("lyric_karaoke.karaoke_logic.choose_start")


@pytest.fixture
def chooser(monkeypatch):
    """
    Returns choose_karaoke_start with constants patched to deterministic test values.
    This makes tests stable even if you tweak defaults later.
    """
    mod = load_choose_module()

    # Patch constants to make tests deterministic and fast
    monkeypatch.setattr(mod, "INTRO_IGNORE", 12.0, raising=False)
    monkeypatch.setattr(mod, "MIN_SEG_DURATION", 1.0, raising=False)
    monkeypatch.setattr(mod, "WINDOW", 45.0, raising=False)
    monkeypatch.setattr(mod, "MIN_SEGMENTS_IN_WINDOW", 2, raising=False)
    monkeypatch.setattr(mod, "MIN_TOTAL_LYRIC_IN_WINDOW", 6.0, raising=False)

    return mod.choose_karaoke_start


def test_returns_none_when_no_segments(chooser):
    assert chooser([], song_duration=180.0) is None


def test_skips_intro_segments_before_intro_ignore(chooser):
    # segment starts at 5s (intro noise), real verse at 22s
    segments = [seg(5, 20), seg(22, 35)]
    assert chooser(segments, song_duration=180.0) == 22.0


def test_single_long_segment_qualifies_and_prevents_late_start(chooser):
    """
    Regression: early verse gets merged into one long segment.
    Old logic required >=2 segments in window, causing a late pick.
    New logic should accept the long segment directly.
    """
    segments = [
        seg(20, 60),     # long early verse (should be chosen)
        seg(104.5, 110), # later vocals that previously got chosen
        seg(112.5, 115),
    ]
    assert chooser(segments, song_duration=180.0) == 20.0


def test_fragmented_lyrics_can_qualify_via_window_density_rule(chooser):
    """
    If the first segment is short (< MIN_TOTAL_LYRIC_IN_WINDOW), it can still qualify
    if the following WINDOW contains enough lyric time across multiple segments.
    """
    segments = [
        seg(20, 22),  # short
        seg(23, 26),  # +3 sec
        seg(27, 30),  # +3 sec => total in window from 20 is 8 sec across 3 segs
    ]
    assert chooser(segments, song_duration=180.0) == 20.0


def test_overlap_aware_window_stats_counts_segments_starting_before_t0(monkeypatch):
    """
    Candidate at 35 is short, but it should qualify because a segment starting
    before 35 overlaps into the window.

    We force 30 to NOT be a candidate by setting INTRO_IGNORE > 30.
    """
    import importlib
    mod = importlib.import_module("lyric_karaoke.karaoke_logic.choose_start")

    # Make 30 NOT a candidate, but keep it available for overlap accounting
    monkeypatch.setattr(mod, "INTRO_IGNORE", 31.0, raising=False)

    # Keep normal thresholds
    monkeypatch.setattr(mod, "MIN_TOTAL_LYRIC_IN_WINDOW", 6.0, raising=False)
    monkeypatch.setattr(mod, "MIN_SEGMENTS_IN_WINDOW", 2, raising=False)

    segments = [
        seg(30, 40),  # overlaps into [35, 80) by 5 seconds (35-40)
        seg(35, 36),  # candidate
        seg(40, 48),  # adds 8 seconds
        seg(100, 110),
    ]

    # At t0=35: overlap totals = 5 + 1 + 8 = 14 sec, count=3 => qualifies
    assert mod.choose_karaoke_start(segments, song_duration=180.0) == 35.0


