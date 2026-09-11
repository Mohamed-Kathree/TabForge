"""Stage 6: map pitches to (string, fret) positions.

Milestone-1 placeholder: a greedy nearest-fret chooser. This is explicitly
called out in the spec as acceptable only until the Viterbi/DP version
lands in milestone 3 -- do not tune this further, replace it instead.
"""

from .models import Note, TabEvent

MAX_FRET = 22

# Open string MIDI pitches, index 0 = low E ... 5 = high e (standard tuning).
OPEN_STRING_PITCHES = [40, 45, 50, 55, 59, 64]


def _candidates(pitch: int) -> list[tuple[int, int]]:
    """All (string, fret) pairs that can produce `pitch` within fret range."""
    out = []
    for string, open_pitch in enumerate(OPEN_STRING_PITCHES):
        fret = pitch - open_pitch
        if 0 <= fret <= MAX_FRET:
            out.append((string, fret))
    return out


def fingering_greedy(notes: list[Note]) -> list[TabEvent]:
    """Pick a playable position per note, greedily minimizing hand movement.

    Notes sharing an exact onset are treated as a chord: each gets a
    distinct string, preferring the candidate closest to the previous
    hand position.
    """
    events: list[TabEvent] = []
    prev_fret = 0
    prev_string = 0

    notes_sorted = sorted(notes, key=lambda n: n.start)
    i = 0
    while i < len(notes_sorted):
        start = notes_sorted[i].start
        chord = [notes_sorted[i]]
        j = i + 1
        while j < len(notes_sorted) and abs(notes_sorted[j].start - start) < 1e-6:
            chord.append(notes_sorted[j])
            j += 1

        used_strings: set[int] = set()
        for note in chord:
            candidates = [c for c in _candidates(note.pitch) if c[0] not in used_strings]
            if not candidates:
                # No free string can play this pitch; skip rather than
                # double up on a string (that's not physically playable).
                continue

            def cost(c: tuple[int, int]) -> float:
                string, fret = c
                return abs(fret - prev_fret) + 0.1 * abs(string - prev_string)

            string, fret = min(candidates, key=cost)
            used_strings.add(string)
            prev_fret, prev_string = fret, string
            events.append(TabEvent(start=note.start, duration=note.duration, string=string, fret=fret))

        i = j

    events.sort(key=lambda e: e.start)
    return events
