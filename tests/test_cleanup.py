import numpy as np

from tabforge.models import Note
from tabforge.cleanup import filter_range, filter_confidence, quantize, MIN_MIDI, MAX_MIDI


def test_filter_range_drops_out_of_range_pitches():
    notes = [
        Note(start=0.0, duration=0.1, pitch=20, confidence=1.0),  # too low
        Note(start=0.1, duration=0.1, pitch=60, confidence=1.0),  # in range
        Note(start=0.2, duration=0.1, pitch=110, confidence=1.0),  # too high
    ]
    result = filter_range(notes)
    assert [n.pitch for n in result] == [60]


def test_filter_range_is_inclusive_at_boundaries():
    notes = [
        Note(start=0.0, duration=0.1, pitch=MIN_MIDI, confidence=1.0),
        Note(start=0.1, duration=0.1, pitch=MAX_MIDI, confidence=1.0),
    ]
    result = filter_range(notes)
    assert len(result) == 2


def test_filter_confidence_drops_low_confidence_notes():
    notes = [
        Note(start=0.0, duration=0.1, pitch=60, confidence=0.9),
        Note(start=0.1, duration=0.1, pitch=61, confidence=0.1),
    ]
    result = filter_confidence(notes, threshold=0.5)
    assert [n.pitch for n in result] == [60]


def test_quantize_snaps_onset_to_nearest_grid_slot():
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    notes = [Note(start=0.24, duration=0.2, pitch=60, confidence=1.0)]
    result = quantize(notes, grid)
    assert len(result) == 1
    assert result[0].start == 0.25


def test_quantize_preserves_note_end_time():
    # A note starting at 0.24 with duration 0.2 ends at 0.44; snapping the
    # onset to 0.25 should shrink duration to keep that same end time.
    grid = np.array([0.0, 0.25, 0.5])
    notes = [Note(start=0.24, duration=0.2, pitch=60, confidence=1.0)]
    result = quantize(notes, grid)
    assert abs(result[0].duration - 0.19) < 1e-9


def test_quantize_merges_same_slot_notes_within_a_semitone():
    grid = np.array([0.0, 0.25, 0.5])
    notes = [
        Note(start=0.01, duration=0.2, pitch=60, confidence=0.6),
        Note(start=0.02, duration=0.3, pitch=61, confidence=0.9),  # same slot, 1 semitone away
    ]
    result = quantize(notes, grid)
    assert len(result) == 1
    # the higher-confidence note's pitch wins; duration is recomputed per
    # note to preserve each one's original end time before the max is
    # taken, so it's not simply the longer of the two raw durations
    assert result[0].pitch == 61
    assert abs(result[0].duration - 0.32) < 1e-9


def test_quantize_does_not_merge_notes_more_than_a_semitone_apart():
    grid = np.array([0.0, 0.25, 0.5])
    notes = [
        Note(start=0.01, duration=0.1, pitch=60, confidence=1.0),
        Note(start=0.02, duration=0.1, pitch=64, confidence=1.0),  # same slot, far in pitch
    ]
    result = quantize(notes, grid)
    assert len(result) == 2


def test_quantize_empty_grid_is_a_no_op():
    notes = [Note(start=0.13, duration=0.1, pitch=60, confidence=1.0)]
    result = quantize(notes, np.array([]))
    assert result == notes
