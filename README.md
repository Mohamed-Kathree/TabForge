# TabForge

MP3 → guitar tab generator for short riff clips. Not a full-song transcriber.

Full pipeline: clip → (optional Demucs guitar separation) → tuning
detection → basic-pitch transcription → range/confidence filtering
(+ optional 16th-note quantization) → Viterbi fingering optimizer →
ASCII tab / MIDI / Guitar Pro export. CLI and a Streamlit UI both wrap
the same `tabforge.pipeline.run()`.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

(Python 3.11 is used deliberately — basic-pitch/demucs depend on
TensorFlow/PyTorch, which lag behind the newest CPython releases.)

## Usage

CLI:

```
.venv\Scripts\python -m tabforge.pipeline path\to\clip.mp3 --start 10 --end 20
```

Useful flags: `--demucs` (separate the guitar stem first, slow, cached),
`--quantize` (snap to a 16th-note grid), `--verify-dir DIR` (write the
original clip + re-rendered MIDI for A/B listening), `--gp5 out.gp5`
(export Guitar Pro).

Streamlit UI:

```
.venv\Scripts\streamlit run app.py
```

Upload a file, pick a clip range, tune the basic-pitch/confidence
thresholds, toggle Demucs/quantization, transcribe, fix any wrong frets
in the editable grid, and download MIDI or a .gp5.

## Tests

```
.venv\Scripts\pytest
```
