"""Stage 2: Demucs source separation, with an aggressive disk cache.

Demucs is the slow stage, and the fingering optimizer gets re-run
hundreds of times during development -- nobody should wait for
separation twice. The cache is keyed on (file content hash, start, end,
model name) and stores the extracted guitar stem already resampled to
io_audio's contract (mono, 22050 Hz), so a cache hit skips both Demucs
and the resample.

Bypassable: pipeline.py only calls this when asked to; io_audio.load_clip
is the non-Demucs path and the pipeline runs fine without this module.
"""

import hashlib
import os

import numpy as np
import librosa
import soundfile as sf

from .io_audio import SAMPLE_RATE

MODEL_NAME = "htdemucs_6s"
CACHE_DIR = ".cache"

_model = None  # lazy-loaded: weights download ~300MB on first use


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model

        _model = get_model(MODEL_NAME)
        _model.eval()
    return _model


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(path: str, start: float, end: float) -> str:
    key = f"{_file_hash(path)}__{start:.3f}__{end:.3f}__{MODEL_NAME}.wav"
    return os.path.join(CACHE_DIR, key)


def separate_clip(path: str, start: float, end: float) -> tuple[np.ndarray, int]:
    """Run Demucs on `path[start:end]`, return the guitar stem as
    (mono, 22050 Hz) audio -- the same contract as io_audio.load_clip.
    """
    cache_path = _cache_path(path, start, end)
    if os.path.exists(cache_path):
        y, sr = librosa.load(cache_path, sr=SAMPLE_RATE, mono=True)
        return y, sr

    import torch
    from demucs.apply import apply_model

    model = _get_model()
    wav, _ = librosa.load(path, sr=model.samplerate, mono=False, offset=start, duration=end - start)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])  # Demucs expects stereo input

    wav_t = torch.tensor(wav, dtype=torch.float32)
    with torch.no_grad():
        sources = apply_model(model, wav_t[None], device="cpu", progress=False)[0]

    guitar_idx = model.sources.index("guitar")
    guitar = sources[guitar_idx].mean(dim=0).numpy()  # downmix stereo -> mono
    guitar_resampled = librosa.resample(guitar, orig_sr=model.samplerate, target_sr=SAMPLE_RATE)

    os.makedirs(CACHE_DIR, exist_ok=True)
    sf.write(cache_path, guitar_resampled, SAMPLE_RATE)

    return guitar_resampled.astype(np.float32), SAMPLE_RATE
