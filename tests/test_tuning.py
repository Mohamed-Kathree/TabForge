import numpy as np

from tabforge.tuning import detect_tuning
from tabforge.models import OPEN_STRING_PITCHES

SR = 22050


def _tone_clip(midi_notes, duration=0.6, sr=SR):
    segments = []
    for midi in midi_notes:
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        segments.append(0.5 * np.sin(2 * np.pi * freq * t))
    return np.concatenate(segments).astype(np.float32)


def test_standard_tuning_reports_standard():
    # Open low E, A, D played repeatedly -- clearly centered at MIDI 40.
    y = _tone_clip([40, 45, 50, 40, 45])
    report = detect_tuning(y, SR)
    assert report.named_tuning == "standard"
    assert report.open_string_pitches == list(OPEN_STRING_PITCHES)


def test_eb_standard_shifts_open_strings_down_a_half_step():
    # Everything played a semitone below standard: 39 (Eb), 44 (Ab), 49 (Db)
    y = _tone_clip([39, 44, 49, 39, 44])
    report = detect_tuning(y, SR)
    assert report.named_tuning == "eb_standard"
    assert report.open_string_pitches == [p - 1 for p in OPEN_STRING_PITCHES]


def test_drop_d_only_shifts_low_string():
    # Low string down a whole step (38 = D2) but the A string stays put (45).
    y = _tone_clip([38, 45, 38, 45, 38])
    report = detect_tuning(y, SR)
    assert report.named_tuning == "drop_d"
    expected = list(OPEN_STRING_PITCHES)
    expected[0] -= 2
    assert report.open_string_pitches == expected


def test_silence_reports_unknown_without_crashing():
    y = np.zeros(SR, dtype=np.float32)
    report = detect_tuning(y, SR)
    assert report.named_tuning == "unknown"
    assert report.open_string_pitches == list(OPEN_STRING_PITCHES)
