"""Milestone 1: MP3 + time range -> basic-pitch -> greedy fingering -> ASCII.

No Demucs, no tuning detection, no quantization, no UI yet -- those land in
later milestones. This is the "ugly but complete" end-to-end path.
"""

import argparse

from .io_audio import load_clip
from .transcribe import (
    transcribe,
    ONSET_THRESHOLD_DEFAULT,
    FRAME_THRESHOLD_DEFAULT,
    MINIMUM_NOTE_LENGTH_DEFAULT,
)
from .fingering import fingering_greedy
from .render import render_ascii


def run(
    path: str,
    start: float,
    end: float,
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT,
    frame_threshold: float = FRAME_THRESHOLD_DEFAULT,
    minimum_note_length: float = MINIMUM_NOTE_LENGTH_DEFAULT,
) -> str:
    y, sr = load_clip(path, start, end)
    notes = transcribe(
        y,
        sr,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    events = fingering_greedy(notes)
    return render_ascii(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="MP3 clip -> ASCII guitar tab")
    parser.add_argument("path", help="path to an audio file")
    parser.add_argument("--start", type=float, required=True, help="clip start, seconds")
    parser.add_argument("--end", type=float, required=True, help="clip end, seconds")
    parser.add_argument("--onset-threshold", type=float, default=ONSET_THRESHOLD_DEFAULT)
    parser.add_argument("--frame-threshold", type=float, default=FRAME_THRESHOLD_DEFAULT)
    parser.add_argument("--minimum-note-length", type=float, default=MINIMUM_NOTE_LENGTH_DEFAULT)
    args = parser.parse_args()

    tab = run(
        args.path,
        args.start,
        args.end,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length=args.minimum_note_length,
    )
    print(tab)


if __name__ == "__main__":
    main()
