"""Milestones 1-6: MP3 + time range -> (Demucs) -> tuning detection ->
basic-pitch -> cleanup (range/confidence filter, optional quantize) ->
DP fingering -> ASCII tab, plus MIDI verification and Guitar Pro export.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import guitarpro

from .io_audio import load_clip, SAMPLE_RATE
from .transcribe import (
    transcribe,
    ONSET_THRESHOLD_DEFAULT,
    FRAME_THRESHOLD_DEFAULT,
    MINIMUM_NOTE_LENGTH_DEFAULT,
)
from .tuning import detect_tuning, TuningReport
from .cleanup import cleanup, detect_beat_grid, CONFIDENCE_THRESHOLD_DEFAULT
from .fingering import fingering_dp
from .render import render_ascii, render_midi, render_gp5
from .models import Note, TabEvent


@dataclass
class Result:
    clip: np.ndarray
    sr: int
    tuning: TuningReport
    bpm: float
    notes: list[Note]
    events: list[TabEvent]
    ascii_tab: str


def run(
    path: str,
    start: float,
    end: float,
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT,
    frame_threshold: float = FRAME_THRESHOLD_DEFAULT,
    minimum_note_length: float = MINIMUM_NOTE_LENGTH_DEFAULT,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    use_demucs: bool = False,
    quantize: bool = False,
) -> Result:
    if use_demucs:
        from .separate import separate_clip

        y, sr = separate_clip(path, start, end)
    else:
        y, sr = load_clip(path, start, end)

    tuning = detect_tuning(y, sr)
    notes = transcribe(
        y,
        sr,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    notes = cleanup(
        notes,
        y=y,
        sr=sr,
        confidence_threshold=confidence_threshold,
        quantize_enabled=quantize,
    )
    events = fingering_dp(notes, open_string_pitches=tuning.open_string_pitches)
    ascii_tab = render_ascii(events)

    bpm, _grid = detect_beat_grid(y, sr)
    return Result(clip=y, sr=sr, tuning=tuning, bpm=bpm, notes=notes, events=events, ascii_tab=ascii_tab)


def write_verification(result: Result, out_dir: str) -> dict[str, str]:
    """Write the original clip and the re-rendered MIDI (as .mid and .wav)
    to `out_dir`, so the transcription can be checked by ear, A/B, against
    the source.

    Returns the written paths keyed by "clip_wav", "midi", "midi_wav".
    """
    os.makedirs(out_dir, exist_ok=True)

    clip_wav = os.path.join(out_dir, "clip.wav")
    sf.write(clip_wav, result.clip, result.sr)

    pm = render_midi(result.events, open_string_pitches=result.tuning.open_string_pitches)
    midi_path = os.path.join(out_dir, "transcription.mid")
    pm.write(midi_path)

    midi_wav = os.path.join(out_dir, "transcription.wav")
    synthesized = pm.synthesize(fs=SAMPLE_RATE)
    sf.write(midi_wav, synthesized, SAMPLE_RATE)

    return {"clip_wav": clip_wav, "midi": midi_path, "midi_wav": midi_wav}


def write_gp5(result: Result, path: str) -> None:
    """Export the tab as a .gp5 file, for TuxGuitar / slowed-down looping."""
    song = render_gp5(result.events, open_string_pitches=result.tuning.open_string_pitches, bpm=result.bpm)
    guitarpro.write(song, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MP3 clip -> ASCII guitar tab")
    parser.add_argument("path", help="path to an audio file")
    parser.add_argument("--start", type=float, required=True, help="clip start, seconds")
    parser.add_argument("--end", type=float, required=True, help="clip end, seconds")
    parser.add_argument("--onset-threshold", type=float, default=ONSET_THRESHOLD_DEFAULT)
    parser.add_argument("--frame-threshold", type=float, default=FRAME_THRESHOLD_DEFAULT)
    parser.add_argument("--minimum-note-length", type=float, default=MINIMUM_NOTE_LENGTH_DEFAULT)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD_DEFAULT)
    parser.add_argument(
        "--demucs",
        action="store_true",
        help="separate out the guitar stem before transcribing (slow, cached)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="snap onsets to a 16th-note grid from beat tracking (destroys swing/rubato)",
    )
    parser.add_argument(
        "--verify-dir",
        default=None,
        help="if set, write clip.wav + transcription.mid/.wav here for A/B playback",
    )
    parser.add_argument("--gp5", default=None, help="if set, export a Guitar Pro (.gp5) file to this path")
    args = parser.parse_args()

    result = run(
        args.path,
        args.start,
        args.end,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length=args.minimum_note_length,
        confidence_threshold=args.confidence_threshold,
        use_demucs=args.demucs,
        quantize=args.quantize,
    )
    print(result.tuning.message)
    print()
    print(result.ascii_tab)

    if args.verify_dir:
        paths = write_verification(result, args.verify_dir)
        print()
        print(f"original clip:  {paths['clip_wav']}")
        print(f"MIDI:           {paths['midi']}")
        print(f"MIDI rendered:  {paths['midi_wav']}")

    if args.gp5:
        write_gp5(result, args.gp5)
        print(f"Guitar Pro:     {args.gp5}")


if __name__ == "__main__":
    main()
