import os

import numpy as np

from tabforge.models import Note, TabEvent, OPEN_STRING_PITCHES
from tabforge.tuning import TuningReport
from tabforge.pipeline import Result, write_verification, write_gp5


def _make_result(**overrides):
    defaults = dict(
        clip=np.zeros(22050, dtype=np.float32),
        sr=22050,
        tuning=TuningReport(0.0, "standard", list(OPEN_STRING_PITCHES), "Tuning looks like standard E."),
        bpm=120.0,
        notes=[Note(start=0.0, duration=0.5, pitch=40, confidence=1.0)],
        events=[TabEvent(start=0.0, duration=0.5, string=0, fret=0)],
        ascii_tab="e|-|\nB|-|\nG|-|\nD|-|\nA|-|\nE|0|",
    )
    defaults.update(overrides)
    return Result(**defaults)


def test_write_verification_writes_expected_files(tmp_path):
    result = _make_result()

    out_dir = str(tmp_path / "verify")
    paths = write_verification(result, out_dir)

    assert set(paths.keys()) == {"clip_wav", "midi", "midi_wav"}
    for p in paths.values():
        assert os.path.isfile(p)


def test_write_gp5_writes_a_valid_file(tmp_path):
    result = _make_result()
    path = str(tmp_path / "tab.gp5")

    write_gp5(result, path)

    assert os.path.isfile(path)
    import guitarpro

    song = guitarpro.parse(path)
    assert song.tracks[0].measures
