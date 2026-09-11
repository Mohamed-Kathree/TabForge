import numpy as np
import soundfile as sf

from tabforge.io_audio import load_clip, SAMPLE_RATE


def test_load_clip_slices_and_resamples(tmp_path):
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr)

    clip, out_sr = load_clip(str(path), start=1.0, end=2.0)

    assert out_sr == SAMPLE_RATE
    assert abs(len(clip) / out_sr - 1.0) < 0.01


def test_end_must_be_after_start(tmp_path):
    sr = 22050
    y = np.zeros(sr)
    path = tmp_path / "silence.wav"
    sf.write(path, y, sr)

    try:
        load_clip(str(path), start=2.0, end=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
