# Datasets

This repository intentionally **does not** redistribute:
- copyrighted audio
- HarmonixSet features (`.npy`) or segment files

You must obtain datasets yourself and place them in your local filesystem.

## HarmonixSet
If you have access to HarmonixSet, place it like:

```
<DATA_ROOT>/harmonixset/
  dataset/
    metadata.csv
    segments/
    mels/
```

### Notes
- Ignore macOS sidecar files (e.g. `._0001...npy`).
- Harmonix mel files have shape `(80, T)` (n_mels, frames).

## Local audio files
For local tests, use your own legally-owned audio.

We recommend WAV for reliability. MP3/M4A require ffmpeg and may introduce decode differences.
