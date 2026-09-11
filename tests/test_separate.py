import os

import numpy as np
import soundfile as sf

from tabforge.separate import _cache_path, separate_clip
from tabforge.io_audio import SAMPLE_RATE


def test_cache_hit_avoids_running_demucs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    cache_path = _cache_path(str(audio_path), 0.0, 1.0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sf.write(cache_path, np.full(SAMPLE_RATE, 0.25, dtype=np.float32), SAMPLE_RATE)

    def _boom():
        raise AssertionError("Demucs should not run on a cache hit")

    monkeypatch.setattr("tabforge.separate._get_model", _boom)

    y, sr = separate_clip(str(audio_path), 0.0, 1.0)
    assert sr == SAMPLE_RATE
    assert np.allclose(y, 0.25, atol=1e-3)
