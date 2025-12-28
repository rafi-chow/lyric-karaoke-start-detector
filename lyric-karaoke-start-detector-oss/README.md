# Lyric Karaoke Start Detector

Detects the **best karaoke start time** in a song (ideally the first verse) by predicting vocal/lyric presence over time, grouping predictions into segments, and selecting a robust start point.

This repo is being open-sourced as a **data-pipeline-first** project (data contracts, reproducibility, and quality gates), with an ML baseline (Logistic Regression) as one component.

## What it does
Given an audio file (WAV recommended):
1. Extracts mel features per fixed frame (default: 0.5s)
2. Predicts lyric/vocal presence per frame
3. Converts frame predictions into time segments: `[{start, end}, ...]`
4. Chooses a `karaoke_start` timestamp

## Quickstart (local)
> Note: By default, this repo does **not** ship trained model weights or copyrighted audio.

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Run the API
```bash
python -m app.app
```
Then open the site in your browser (see console output).

## Datasets
- HarmonixSet is supported for training/evaluation **but is not redistributed here**.
- See `DATASETS.md` for setup instructions and compliance notes.

## Project status
Early-stage. The priority is correctness + reproducibility:
- Canonical feature grid (0.5s frames)
- Versioned configs and artifacts
- Data validation gates

## Contributing
See `CONTRIBUTING.md`.

## License
Apache 2.0 — see `LICENSE`.
