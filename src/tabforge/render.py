"""Stage 7: render TabEvents to ASCII tab, MIDI, and Guitar Pro.

MIDI is the verification tool (milestone 2), not a deliverable feature --
it exists so "is this transcription actually right" can be answered by ear
against the original clip. Guitar Pro (.gp5) is the deliverable meant for
TuxGuitar, for slowed-down looping.
"""

import pretty_midi
import guitarpro

from .models import TabEvent, OPEN_STRING_PITCHES, STRING_NAMES

POSITIONS_PER_LINE = 16
BEATS_PER_MEASURE = 16  # 4/4 time at 16th-note resolution
DEFAULT_BPM = 120.0

# General MIDI program 30 = Overdriven Guitar. Doesn't matter for
# correctness, just makes the A/B playback sound guitar-ish.
GM_OVERDRIVEN_GUITAR = 29


def render_ascii(events: list[TabEvent], positions_per_line: int = POSITIONS_PER_LINE) -> str:
    """Render events as a 6-line ASCII tab, high e on top.

    Without beat-tracked quantization (milestone 5), each distinct onset
    becomes one column -- this is a direct, literal rendering of what was
    played, not a rhythm notation.
    """
    if not events:
        return "(no notes)"

    columns: list[dict[int, int]] = []
    current_start = None
    for ev in sorted(events, key=lambda e: e.start):
        if current_start is None or ev.start - current_start > 1e-6:
            columns.append({})
            current_start = ev.start
        columns[-1][ev.string] = ev.fret

    blocks = []
    for block_start in range(0, len(columns), positions_per_line):
        block = columns[block_start : block_start + positions_per_line]
        cells = [[str(col.get(string, "-")) for col in block] for string in range(5, -1, -1)]
        width = max((len(c) for row in cells for c in row), default=1)

        lines = []
        for string_idx, row in zip(range(5, -1, -1), cells):
            body = "-".join(c.rjust(width, "-") for c in row)
            lines.append(f"{STRING_NAMES[string_idx]}|{body}|")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def render_midi(
    events: list[TabEvent],
    program: int = GM_OVERDRIVEN_GUITAR,
    open_string_pitches: list[int] = OPEN_STRING_PITCHES,
) -> pretty_midi.PrettyMIDI:
    """Render TabEvents back out as MIDI, for A/B playback against the clip.

    Pass the same `open_string_pitches` used by fingering.fingering_dp
    (e.g. a detected alternate tuning) so playback reflects the pitch
    that was actually detected, not the standard-tuning pitch.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    for ev in events:
        pitch = open_string_pitches[ev.string] + ev.fret
        instrument.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=ev.start, end=ev.start + max(ev.duration, 0.05))
        )
    pm.instruments.append(instrument)
    return pm


def render_gp5(
    events: list[TabEvent],
    open_string_pitches: list[int] = OPEN_STRING_PITCHES,
    bpm: float = DEFAULT_BPM,
) -> guitarpro.Song:
    """Render TabEvents as a Guitar Pro Song, with correct string tunings.

    Events are laid out on a 16th-note grid at `bpm` (4/4 time), one Beat
    per grid slot -- notes sharing a slot become one chorded Beat, empty
    slots become rests. `bpm` should come from cleanup.detect_beat_grid
    when available; without a detected tempo, 120 is just a layout
    convenience (Guitar Pro's timeline needs *some* tempo), not a claim
    about the actual tempo.
    """
    bpm = bpm if bpm and bpm > 0 else DEFAULT_BPM
    sixteenth_seconds = 60.0 / bpm / 4

    slots: dict[int, list[TabEvent]] = {}
    for ev in events:
        slot = round(ev.start / sixteenth_seconds)
        slots.setdefault(slot, []).append(ev)

    num_slots = (max(slots) + 1) if slots else 0
    num_measures = max(1, -(-num_slots // BEATS_PER_MEASURE))  # ceil div

    song = guitarpro.Song()
    song.tempo = round(bpm)
    track = song.tracks[0]
    # GuitarString numbering is 1 = highest string, matching TabEvent's
    # string 5 = high e -- the reverse of our own low-to-high indexing.
    track.strings = [
        guitarpro.GuitarString(number=i + 1, value=open_string_pitches[5 - i]) for i in range(6)
    ]

    measures = []
    for measure_idx in range(num_measures):
        header = guitarpro.MeasureHeader(number=measure_idx + 1)
        measure = guitarpro.Measure(track, header)
        voice = measure.voices[0]

        beats = []
        for slot_in_measure in range(BEATS_PER_MEASURE):
            slot = measure_idx * BEATS_PER_MEASURE + slot_in_measure
            chord = slots.get(slot, [])
            duration = guitarpro.Duration(value=16)
            if chord:
                beat = guitarpro.Beat(voice, duration=duration, status=guitarpro.BeatStatus.normal)
                beat.notes = [
                    guitarpro.Note(beat, value=ev.fret, string=6 - ev.string, type=guitarpro.NoteType.normal)
                    for ev in chord
                ]
            else:
                beat = guitarpro.Beat(voice, duration=duration, status=guitarpro.BeatStatus.rest)
            beats.append(beat)

        voice.beats = beats
        measures.append(measure)

    track.measures = measures
    return song
