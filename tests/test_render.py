from tabforge.models import TabEvent, OPEN_STRING_PITCHES
from tabforge.render import render_ascii, render_midi, render_gp5


def test_empty_events():
    assert render_ascii([]) == "(no notes)"


def test_six_lines_high_e_on_top():
    events = [
        TabEvent(start=0.0, duration=0.25, string=0, fret=3),
        TabEvent(start=0.25, duration=0.25, string=5, fret=0),
    ]
    tab = render_ascii(events)
    lines = tab.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("e|")  # high e on top
    assert lines[-1].startswith("E|")  # low E on bottom


def test_chord_shares_one_column():
    events = [
        TabEvent(start=0.0, duration=0.5, string=0, fret=0),
        TabEvent(start=0.0, duration=0.5, string=1, fret=0),
    ]
    tab = render_ascii(events)
    lines = tab.splitlines()
    # both notes share an onset -> one column, so each line has exactly
    # one cell between the leading string name and the closing bar
    assert len(lines[0].split("|")) == 3  # "E", body, "" (trailing from closing |)


def test_wraps_into_blocks():
    events = [TabEvent(start=float(i), duration=0.1, string=0, fret=i) for i in range(20)]
    tab = render_ascii(events, positions_per_line=16)
    blocks = tab.split("\n\n")
    assert len(blocks) == 2


def test_render_midi_roundtrips_pitch_and_timing():
    events = [
        TabEvent(start=0.0, duration=0.5, string=0, fret=3),
        TabEvent(start=0.5, duration=0.25, string=5, fret=2),
    ]
    pm = render_midi(events)
    assert len(pm.instruments) == 1
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    assert len(notes) == 2
    assert notes[0].pitch == OPEN_STRING_PITCHES[0] + 3
    assert notes[0].start == 0.0
    assert notes[1].pitch == OPEN_STRING_PITCHES[5] + 2
    assert notes[1].start == 0.5


def test_render_midi_extends_very_short_notes_to_be_audible():
    events = [TabEvent(start=0.0, duration=0.001, string=0, fret=0)]
    pm = render_midi(events)
    note = pm.instruments[0].notes[0]
    assert note.end - note.start >= 0.05


def test_render_gp5_string_tuning_matches_open_string_pitches():
    song = render_gp5([], open_string_pitches=OPEN_STRING_PITCHES)
    track = song.tracks[0]
    # GuitarString number 1 = high e ... 6 = low E (reverse of our indexing)
    values = [s.value for s in sorted(track.strings, key=lambda s: s.number)]
    assert values == list(reversed(OPEN_STRING_PITCHES))


def test_render_gp5_places_note_at_correct_string_and_fret():
    # 120 bpm -> 16th note = 0.125s, so a note at start=0.0 lands in slot 0
    events = [TabEvent(start=0.0, duration=0.125, string=0, fret=3)]
    song = render_gp5(events, bpm=120.0)
    beat = song.tracks[0].measures[0].voices[0].beats[0]
    assert beat.status.name == "normal"
    assert len(beat.notes) == 1
    assert beat.notes[0].string == 6  # low E is GP string 6
    assert beat.notes[0].value == 3


def test_render_gp5_empty_slots_are_rests():
    events = [TabEvent(start=0.0, duration=0.125, string=0, fret=0)]
    song = render_gp5(events, bpm=120.0)
    beats = song.tracks[0].measures[0].voices[0].beats
    assert beats[0].status.name == "normal"
    assert all(b.status.name == "rest" for b in beats[1:])


def test_render_gp5_chord_shares_one_beat():
    events = [
        TabEvent(start=0.0, duration=0.125, string=0, fret=0),
        TabEvent(start=0.0, duration=0.125, string=1, fret=2),
    ]
    song = render_gp5(events, bpm=120.0)
    beat = song.tracks[0].measures[0].voices[0].beats[0]
    assert len(beat.notes) == 2


def test_render_gp5_spans_multiple_measures():
    # 20 sixteenth-note slots at 120 bpm -> needs 2 measures of 16 each
    events = [TabEvent(start=i * 0.125, duration=0.1, string=0, fret=0) for i in range(20)]
    song = render_gp5(events, bpm=120.0)
    assert len(song.tracks[0].measures) == 2
