from tabforge.models import Note
from tabforge.fingering import fingering_dp, chord_span_ok, OPEN_STRING_PITCHES


def test_chromatic_run_stays_in_one_position():
    # E2, F2, F#2, G2: below the second string's open pitch (45), so each
    # of these has exactly one candidate -- string 0, frets 0-3.
    notes = [
        Note(start=0.0, duration=0.25, pitch=40, confidence=1.0),
        Note(start=0.25, duration=0.25, pitch=41, confidence=1.0),
        Note(start=0.5, duration=0.25, pitch=42, confidence=1.0),
        Note(start=0.75, duration=0.25, pitch=43, confidence=1.0),
    ]
    events = fingering_dp(notes)
    assert [e.string for e in events] == [0, 0, 0, 0]
    assert [e.fret for e in events] == [0, 1, 2, 3]


def test_open_string_riff_uses_open_strings():
    # E2 (open) then A2: A2 is playable as (string0, fret5) or (string1,
    # fret0) -- the DP should prefer the open string over the higher fret.
    notes = [
        Note(start=0.0, duration=0.25, pitch=OPEN_STRING_PITCHES[0], confidence=1.0),
        Note(start=0.25, duration=0.25, pitch=OPEN_STRING_PITCHES[1], confidence=1.0),
    ]
    events = fingering_dp(notes)
    assert all(e.fret == 0 for e in events)
    assert [e.string for e in events] == [0, 1]


def test_chord_assigns_distinct_strings():
    notes = [
        Note(start=0.0, duration=0.5, pitch=40, confidence=1.0),  # low E open
        Note(start=0.0, duration=0.5, pitch=45, confidence=1.0),  # A open
    ]
    events = fingering_dp(notes)
    assert len(events) == 2
    assert events[0].string != events[1].string


def test_chord_never_exceeds_span_or_shares_strings():
    # Pitches chosen to have few/no open-string candidates, forcing the DP
    # to actually negotiate a chord shape rather than defaulting to opens.
    notes = [
        Note(start=0.0, duration=0.5, pitch=44, confidence=1.0),
        Note(start=0.0, duration=0.5, pitch=53, confidence=1.0),
        Note(start=0.0, duration=0.5, pitch=61, confidence=1.0),
    ]
    events = fingering_dp(notes)
    strings = [e.string for e in events]
    assert len(strings) == len(set(strings))
    fretted = [e.fret for e in events if e.fret != 0]
    if len(fretted) >= 2:
        assert max(fretted) - min(fretted) <= 4


def test_out_of_range_pitch_produces_no_event():
    notes = [Note(start=0.0, duration=0.25, pitch=20, confidence=1.0)]
    events = fingering_dp(notes)
    assert events == []


def test_chord_span_ok_rejects_five_fret_span():
    assert chord_span_ok(((0, 3), (1, 8))) is False


def test_chord_span_ok_accepts_four_fret_span():
    assert chord_span_ok(((0, 3), (1, 7))) is True


def test_chord_span_ok_ignores_open_strings():
    # A wide-looking chord is fine if the wide note is an open string.
    assert chord_span_ok(((0, 0), (1, 10))) is True
