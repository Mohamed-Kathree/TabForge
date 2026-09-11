from tabforge.models import Note
from tabforge.fingering import fingering_greedy, OPEN_STRING_PITCHES


def test_chromatic_run_stays_in_one_position():
    # E2, F2, F#2, G2 on the low E string: frets 0, 1, 2, 3
    notes = [
        Note(start=0.0, duration=0.25, pitch=40, confidence=1.0),
        Note(start=0.25, duration=0.25, pitch=41, confidence=1.0),
        Note(start=0.5, duration=0.25, pitch=42, confidence=1.0),
        Note(start=0.75, duration=0.25, pitch=43, confidence=1.0),
    ]
    events = fingering_greedy(notes)
    assert [e.string for e in events] == [0, 0, 0, 0]
    assert [e.fret for e in events] == [0, 1, 2, 3]


def test_open_string_riff_uses_open_strings():
    # E2 and A2 are both open strings (fret 0) on strings 0 and 1
    notes = [
        Note(start=0.0, duration=0.25, pitch=OPEN_STRING_PITCHES[0], confidence=1.0),
        Note(start=0.25, duration=0.25, pitch=OPEN_STRING_PITCHES[1], confidence=1.0),
    ]
    events = fingering_greedy(notes)
    assert all(e.fret == 0 for e in events)


def test_chord_assigns_distinct_strings():
    # Two notes at the same onset must not land on the same string.
    notes = [
        Note(start=0.0, duration=0.5, pitch=40, confidence=1.0),  # low E open
        Note(start=0.0, duration=0.5, pitch=45, confidence=1.0),  # A open
    ]
    events = fingering_greedy(notes)
    assert len(events) == 2
    assert events[0].string != events[1].string


def test_out_of_range_pitch_produces_no_event():
    notes = [Note(start=0.0, duration=0.25, pitch=20, confidence=1.0)]
    events = fingering_greedy(notes)
    assert events == []
