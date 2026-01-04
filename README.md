# Lyric Karaoke Start Detector

Detects the best **karaoke start** time in a song (ideally the first sustained lyric section) and returns:

- `segments`: contiguous lyric regions `[{start, end}, ...]`
- `karaoke_start`: recommended jump time

This project uses a simple per-frame lyric-presence model plus deterministic post-processing (smoothing → segmentation → start selection).

---

## Features

- Canonical 0.5s frame grid (training/inference parity)
- Per-frame lyric presence prediction (fast, simple baseline)
- Robust deterministic post-processing:
  - smoothing to reduce flicker
  - segment extraction + cleanup
  - karaoke start chooser tuned to avoid “too early” and “too late”
- CLI (`karaoke-predict`)
- Flask demo app (upload + jump-to-start)
- CI: reproducible toy training + smoke inference + unit tests

---

## Requirements

- Python 3.10+

---

## Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

---

## Run tests

```bash
pytest -q
```

---

## Quickstart (CLI)

> The CLI requires a local `.pkl` model (this repo does not ship trained weights).

Show help:

```bash
karaoke-predict --help
```

Run prediction:

```bash
karaoke-predict --model models/harmonix_lr.pkl --audio path/to/song.wav
```

Windows / PowerShell (paths with spaces must be quoted):

```powershell
karaoke-predict --model models\harmonix_lr.pkl --audio "C:\Users\you\Downloads\My Song (Official).mp3"
```

Example output:

```json
{
  "segments": [{"start": 23.5, "end": 45.0}],
  "karaoke_start": 23.5
}
```

---

## Quickstart (Web demo)

> The web demo requires a local `.pkl` model (this repo does not ship trained weights).

Set `KARAOKE_MODEL_PATH` to a local model file.

Windows (PowerShell):

```powershell
$env:KARAOKE_MODEL_PATH="models\harmonix_lr.pkl"
```

macOS / Linux:

```bash
export KARAOKE_MODEL_PATH="models/harmonix_lr.pkl"
```

Run the app:

```bash
python app/app.py
```

Open `http://127.0.0.1:5000` and upload an audio file (WAV recommended; MP3 supported).

---

## Reproducible pipeline (no external datasets)

This repository does not ship copyrighted audio or trained model weights.
To validate the training → inference pipeline end-to-end using synthetic data:

```bash
python scripts/train_toy.py
python scripts/smoke_toy_predict.py
```

Notes:
- This generates `models/toy_lr.pkl` locally.
- Do not commit generated model files.

---

## Evaluation (hand-labeled)
```bash
python scripts/eval_handset.py --csv data/eval_handset.csv --print-outliers
```

---

## Documentation

- `docs/design.md` — architecture, invariants (canonical time grid), post-processing rationale
- `DATASETS.md` — dataset setup notes (if training on Harmonix locally)

---




## Contributing

Issues and PRs are welcome. If you’re changing core behavior, please add or update unit tests under `tests/`.

---

## License

See `LICENSE`.
