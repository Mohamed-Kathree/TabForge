# TabForge

MP3 → guitar tab generator for short riff clips. Not a full-song transcriber.

## Status: Milestone 1 — ugly but complete

MP3 + time range → basic-pitch → greedy fingering → ASCII tab on stdout.
No source separation, no tuning detection, no quantization, no UI yet.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

(Python 3.11 is used deliberately — basic-pitch/demucs depend on
TensorFlow/PyTorch, which lag behind the newest CPython releases.)

## Usage

```
.venv\Scripts\python -m tabforge.pipeline path\to\clip.mp3 --start 10 --end 20
```

## Tests

```
.venv\Scripts\pytest
```
