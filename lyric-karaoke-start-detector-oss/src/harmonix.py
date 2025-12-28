from pathlib import Path
import csv
import numpy as np


LYRIC_LABELS = {"verse", "chorus", "bridge", "prechorus"}


def load_harmonix_intervals(
    track_id: str,
    segments_dir: Path,
    metadata_csv: Path,
):
    """
    Load Harmonix segment boundaries and convert them into
    (start, end, label) intervals using song duration.

    Returns:
        intervals: list of (start, end, label)
        song_duration: float (seconds)
    """

    segment_file = segments_dir / f"{track_id}.txt"
    if not segment_file.exists():
        raise FileNotFoundError(f"Missing segment file: {segment_file}")


    lines = segment_file.read_text().strip().splitlines()
    boundaries = []
    for line in lines:
        time_str, label = line.split()
        boundaries.append((float(time_str), label))


    intervals = []
    for i in range(len(boundaries) - 1):
        start, label = boundaries[i]
        end = boundaries[i + 1][0]
        intervals.append((start, end, label))

    song_duration = None
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["File"] == track_id:
                song_duration = float(row["Duration"])
                break

    if song_duration is None:
        raise ValueError(f"Track {track_id} not found in metadata")


    last_start, last_label = boundaries[-1]
    intervals.append((last_start, song_duration, last_label))

    return intervals, song_duration


def intervals_to_frame_labels(
    intervals,
    song_duration: float,
    frame_duration: float = 0.5,
):
    """
    Convert Harmonix intervals into frame-level binary labels.

    Rule:
    - Lyrics start ONLY at the first verse
    - verse / chorus / bridge / prechorus AFTER first verse = 1
    - everything else = 0

    Returns:
        times: list of frame start times
        y: list of binary labels (0/1)
    """

    first_verse_start = None
    for start, end, label in intervals:
        if label == "verse":
            first_verse_start = start
            break

    if first_verse_start is None:
        raise ValueError("No verse found in intervals")


    times = []
    t = 0.0
    while t < song_duration:
        times.append(t)
        t += frame_duration


    y = []
    for t in times:
        label_for_frame = 0

        if t >= first_verse_start:
            for start, end, seg_label in intervals:
                if start <= t < end:
                    if seg_label in LYRIC_LABELS:
                        label_for_frame = 1
                    break

        y.append(label_for_frame)

    return times, y

def load_harmonix_mel(mel_path):
    """
    Load a Harmonix mel-spectrogram and return frame-wise features.

    Input shape:
        (n_mels, n_frames)  e.g. (80, 3066)

    Output shape:
        (n_frames, n_mels)  e.g. (3066, 80)
    """
    mel = np.load(mel_path)

    if mel.ndim != 2:
        raise ValueError(f"Unexpected mel shape: {mel.shape}")

    # (80, T) -> (T, 80)
    X = mel.T.astype(np.float32)
    return X
