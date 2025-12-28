import librosa
import numpy as np

def extract_features(audio_path, frame_duration=0.5):
    """
    Match Harmonix-trained features:
    - Use linear/power mel (NO dB conversion)
    - Compute mel on full song once
    - Chunk mel frames into 0.5s windows and average
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
