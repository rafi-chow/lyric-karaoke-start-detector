import librosa
import numpy as np
from typing import List, Dict

from lyric_karaoke.features.aggregate import aggregate_mel_to_frames


def extract_features(audio_path, frame_duration=0.5):
    """
    DEPRECATED:
    This function defines its own time grid and can cause training–serving skew.
    Use extract_mel_features_for_frames with a canonical frame grid instead.
    """
    sr = 22050
    hop_length = 1024
    n_fft = 2048
    n_mels = 80

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Linear/power mel spectrogram (positive values)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,  # power mel (matches Harmonix-looking scale)
    )  # (80, T)

    T = S.shape[1]
    if T == 0:
        return np.zeros((0, n_mels), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Match training's "duration / frames" approach
    song_duration = librosa.get_duration(y=y, sr=sr)
    sec_per_mel_frame = song_duration / T
    frames_per_chunk = max(1, int(round(frame_duration / sec_per_mel_frame)))

    X_chunks = []
    for start in range(0, T, frames_per_chunk):
        end = start + frames_per_chunk
        if end > T:
            break  # drop incomplete tail
        feat = S[:, start:end].mean(axis=1)  # (80,)
        X_chunks.append(feat)

    X = np.vstack(X_chunks) if X_chunks else np.zeros((0, n_mels), dtype=np.float32)
    times = np.arange(len(X), dtype=np.float32) * frame_duration
    return X.astype(np.float32), times

def extract_mel_features_for_frames(
    audio_path: str,
    frame_grid: List[Dict],
    sr: int = 22050,
    n_mels: int = 80,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Inference adapter:
    - loads audio
    - computes mel spectrogram
    - aggregates mel frames into canonical frame grid
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        power=2.0,
    )

    # Convert to log-mel (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Transpose to shape (n_mel_frames, n_mels)
    mel_db = mel_db.T

    # Build mel frame timestamps
    mel_times = librosa.frames_to_time(
        np.arange(mel_db.shape[0]),
        sr=sr,
        hop_length=hop_length,
    )

    # Aggregate to canonical frames
    X = aggregate_mel_to_frames(mel_db, mel_times, frame_grid)

    return X