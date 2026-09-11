"""Stage 1: load, clip, resample. That's it."""

import numpy as np
import librosa

SAMPLE_RATE = 22050


def load_clip(path: str, start: float, end: float) -> tuple[np.ndarray, int]:
    """Load `path` as mono audio at SAMPLE_RATE, sliced to [start, end] seconds."""
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=start, duration=end - start)
    return y, sr
