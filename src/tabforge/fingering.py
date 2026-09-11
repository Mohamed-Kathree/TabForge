"""Stage 6: map pitches to (string, fret) positions via Viterbi/DP.

Replaces the milestone-1 greedy placeholder. Notes sharing an onset (a
chord) are solved jointly: the DP's states at each step are whole valid
chord shapes, not per-note choices, so two notes never land on one string
and a chord's fret span is enforced across the group, not per note.

Chord "hand position" for transition-cost purposes is the average
(fret, string) of its non-open notes -- open strings need no hand
position (spec), so they're excluded, and a chord that's entirely open
strings collapses to position (0, 0), same as a single open note.
"""

import itertools

from .models import Note, TabEvent, OPEN_STRING_PITCHES

MAX_FRET = 22
MAX_CHORD_SPAN = 4
HIGH_FRET_THRESHOLD = 17

# Tunable cost weights (spec: "all weights are named module-level
# constants, tunable").
W_MOVE = 1.0
W_STRING = 0.3
W_HIGH = 0.05
W_OPEN = -0.5
W_HIGH_FRET_PENALTY = 5.0

Position = tuple[int, int]  # (string, fret)
Assignment = tuple[Position, ...]  # one position per note in a group


def _candidates(pitch: int, open_string_pitches: list[int]) -> list[Position]:
    """All (string, fret) pairs that can produce `pitch` within fret range."""
    out = []
    for string, open_pitch in enumerate(open_string_pitches):
        fret = pitch - open_pitch
        if 0 <= fret <= MAX_FRET:
            out.append((string, fret))
    return out


def _group_notes(notes: list[Note]) -> list[list[Note]]:
    notes_sorted = sorted(notes, key=lambda n: n.start)
    groups: list[list[Note]] = []
    for note in notes_sorted:
        if groups and abs(note.start - groups[-1][0].start) < 1e-6:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def chord_span_ok(assignment: Assignment) -> bool:
    """Hard-reject chords with an unplayable fret span.

    Open strings (fret 0) don't count against the span -- they need no
    hand position, so a chord mixing an open string with a fretted note
    four frets wide is still a one-hand-shape chord.
    """
    fretted = [fret for _, fret in assignment if fret != 0]
    if len(fretted) < 2:
        return True
    return max(fretted) - min(fretted) <= MAX_CHORD_SPAN


def _distinct_strings(assignment: Assignment) -> bool:
    strings = [s for s, _ in assignment]
    return len(set(strings)) == len(strings)


def _joint_states(notes: list[Note], open_string_pitches: list[int]) -> tuple[list[Note], list[Assignment]]:
    """Valid joint (string, fret) assignments for a group of simultaneous
    notes, and the subset of notes they apply to (a note with literally no
    playable candidate, or one that can't be reconciled with the rest of
    the chord, is dropped rather than blocking the whole group).
    """
    per_note_candidates = [_candidates(n.pitch, open_string_pitches) for n in notes]
    if any(not c for c in per_note_candidates):
        playable = [n for n, c in zip(notes, per_note_candidates) if c]
        if not playable:
            return [], []
        return _joint_states(playable, open_string_pitches)

    states = [
        a
        for a in itertools.product(*per_note_candidates)
        if _distinct_strings(a) and chord_span_ok(a)
    ]

    if not states and len(notes) > 1:
        # No combination plays the full chord (string clash or span > 4 in
        # every case). Power chords are the only chords in scope, so this
        # exists for robustness against bad transcriptions, not as a
        # feature: drop the least confident note and retry.
        weakest = min(range(len(notes)), key=lambda i: notes[i].confidence)
        return _joint_states(notes[:weakest] + notes[weakest + 1 :], open_string_pitches)

    return notes, states


def _emission_cost(assignment: Assignment, any_low_candidate: list[bool]) -> float:
    cost = 0.0
    for (string, fret), has_low_alt in zip(assignment, any_low_candidate):
        cost += W_HIGH * fret
        if fret == 0:
            cost += W_OPEN
        if fret > HIGH_FRET_THRESHOLD and has_low_alt:
            cost += W_HIGH_FRET_PENALTY
    return cost


def _group_position(assignment: Assignment) -> tuple[float, float]:
    fretted = [(s, f) for s, f in assignment if f != 0]
    if not fretted:
        return (0.0, 0.0)
    avg_fret = sum(f for _, f in fretted) / len(fretted)
    avg_string = sum(s for s, _ in fretted) / len(fretted)
    return (avg_fret, avg_string)


def _transition_cost(prev_pos: tuple[float, float], pos: tuple[float, float]) -> float:
    prev_fret, prev_string = prev_pos
    fret, string = pos
    if prev_fret == 0 or fret == 0:
        return 0.0  # open strings need no hand position
    return W_MOVE * abs(fret - prev_fret) + W_STRING * abs(string - prev_string)


def fingering_dp(notes: list[Note], open_string_pitches: list[int] = OPEN_STRING_PITCHES) -> list[TabEvent]:
    """Minimum-cost (string, fret) assignment across the whole note sequence.

    `open_string_pitches` defaults to standard tuning; pass
    tuning.TuningReport.open_string_pitches to map pitches to frets
    correctly on a detected alternate tuning.
    """
    groups = []
    for group in _group_notes(notes):
        notes_used, states = _joint_states(group, open_string_pitches)
        if states:
            groups.append((notes_used, states, [_group_position(a) for a in states]))

    if not groups:
        return []

    # dp[i][k] = (min cost to reach state k of group i, backpointer into dp[i-1])
    dp: list[list[tuple[float, int]]] = []
    for i, (notes_used, states, positions) in enumerate(groups):
        any_low = [
            any(f <= HIGH_FRET_THRESHOLD for _, f in _candidates(n.pitch, open_string_pitches))
            for n in notes_used
        ]
        row: list[tuple[float, int]] = []
        for assignment, pos in zip(states, positions):
            emission = _emission_cost(assignment, any_low)
            if i == 0:
                row.append((emission, -1))
                continue
            prev_positions = groups[i - 1][2]
            best_cost, best_prev = min(
                (dp[i - 1][p_idx][0] + _transition_cost(prev_pos, pos) + emission, p_idx)
                for p_idx, prev_pos in enumerate(prev_positions)
            )
            row.append((best_cost, best_prev))
        dp.append(row)

    idx = min(range(len(dp[-1])), key=lambda k: dp[-1][k][0])
    chosen: list[tuple[int, int]] = []
    for i in range(len(groups) - 1, -1, -1):
        chosen.append((i, idx))
        idx = dp[i][idx][1]
    chosen.reverse()

    events: list[TabEvent] = []
    for i, s_idx in chosen:
        notes_used, states, _positions = groups[i]
        for note, (string, fret) in zip(notes_used, states[s_idx]):
            events.append(TabEvent(start=note.start, duration=note.duration, string=string, fret=fret))

    events.sort(key=lambda e: e.start)
    return events
