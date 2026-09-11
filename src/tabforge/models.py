"""Data contracts shared across pipeline stages.

Stages 3-5 (transcribe, tuning, cleanup) speak list[Note].
Stage 6 (fingering) emits list[TabEvent]. Stage 7 (render) consumes it.
Nothing else crosses stage boundaries.
"""

from dataclasses import dataclass


@dataclass
class Note:
    start: float
    duration: float
    pitch: int
    confidence: float


@dataclass
class TabEvent:
    start: float
    duration: float
    string: int
    fret: int


# Standard tuning, index 0 = low E ... 5 = high e. Shared by fingering
# (pitch -> position) and render (position -> pitch, for MIDI/GP export).
OPEN_STRING_PITCHES = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]
