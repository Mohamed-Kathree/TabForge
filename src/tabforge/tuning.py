"""Stage 3: report (don't correct) the clip's tuning.

Two different questions, both read off a CQT peak-pitch histogram:

1. Fine intonation drift -- is the whole clip uniformly a few cents flat
   or sharp of concert pitch? A genuinely mistuned instrument shows up as
   a consistent non-zero mode in "deviation from the nearest semitone".

2. Named alternate tuning (Eb standard, D standard, Drop D) -- this is
   NOT visible in that same deviation histogram: a guitar tuned down an
   exact half step is, by construction, perfectly in tune to its own
   (different) notes, so "deviation from nearest semitone" reads as ~0
   either way -- shifting a 12-TET grid by a whole number of semitones
   gives back the same grid. The only way to see it is to look at which
   ABSOLUTE pitch the low end of the clip clusters around, and compare
   that against known reference points.

The result feeds fingering.py's open-string table: on a down-tuned
guitar, getting fret numbers right requires knowing which pitch the open
string actually sounds at. That's a mapping detail, not "correcting" the
transcription -- the detected pitches themselves are never altered.
"""

from dataclasses import dataclass

import numpy as np
import librosa

from .models import OPEN_STRING_PITCHES

CQT_FMIN_HZ = librosa.note_to_hz("C2")
N_BINS = 84
BINS_PER_OCTAVE = 12

FINE_DRIFT_REPORT_THRESHOLD_CENTS = 15

# Offset of the lowest string from standard low E (MIDI 40) -> label.
NAMED_LOW_STRING_OFFSET = {
    0: "standard",
    -1: "eb_standard",
    -2: "d_standard_or_drop_d",
}


@dataclass
class TuningReport:
    fine_offset_cents: float
    named_tuning: str
    open_string_pitches: list[int]
    message: str


def _dominant_pitches(y: np.ndarray, sr: int) -> np.ndarray:
    """One dominant pitch (continuous MIDI number) per non-silent CQT frame."""
    C = np.abs(librosa.cqt(y, sr=sr, fmin=CQT_FMIN_HZ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
    freqs = librosa.cqt_frequencies(n_bins=N_BINS, fmin=CQT_FMIN_HZ, bins_per_octave=BINS_PER_OCTAVE)
    frame_energy = C.sum(axis=0)
    if frame_energy.size == 0 or frame_energy.max() == 0:
        return np.array([])

    threshold = frame_energy.max() * 0.1
    voiced = frame_energy > threshold
    peak_bins = np.argmax(C, axis=0)
    peak_freqs = freqs[peak_bins[voiced]]
    return 12 * np.log2(peak_freqs / 440.0) + 69


def _fine_offset_cents(pitches: np.ndarray) -> float:
    residuals = (pitches - np.round(pitches)) * 100
    hist, edges = np.histogram(residuals, bins=40, range=(-50, 50))
    peak = np.argmax(hist)
    return float((edges[peak] + edges[peak + 1]) / 2)


def detect_tuning(y: np.ndarray, sr: int) -> TuningReport:
    pitches = _dominant_pitches(y, sr)
    if pitches.size == 0:
        return TuningReport(0.0, "unknown", list(OPEN_STRING_PITCHES), "Clip too quiet to estimate tuning.")

    fine_offset_cents = _fine_offset_cents(pitches)

    # The bottom 15% of detected pitches stand in for "the low string" --
    # riffs tend to hit it often as a pedal tone, so it dominates the low
    # end of the pitch distribution.
    low_cut = np.percentile(pitches, 15)
    low_pitches = pitches[pitches <= low_cut]
    low_mode_midi = round(float(np.median(low_pitches))) if low_pitches.size else round(float(pitches.min()))

    offset_semitones = max(-2, min(0, low_mode_midi - OPEN_STRING_PITCHES[0]))
    named_tuning = NAMED_LOW_STRING_OFFSET[offset_semitones]
    open_strings = list(OPEN_STRING_PITCHES)

    if named_tuning == "eb_standard":
        open_strings = [p - 1 for p in open_strings]
        message = "This section sounds like Eb standard -- tune down a half step."
    elif named_tuning == "d_standard_or_drop_d":
        # Distinguish "all six strings down a step" from "just the low
        # string": Drop D leaves the A string (and everything above it)
        # at standard pitch, D standard shifts it too.
        mid_cut = np.percentile(pitches, 35)
        mid_pitches = pitches[(pitches > low_cut) & (pitches <= mid_cut)]
        a_string_shifted = mid_pitches.size > 0 and round(float(np.median(mid_pitches))) <= OPEN_STRING_PITCHES[1] - 1
        if a_string_shifted:
            open_strings = [p - 2 for p in open_strings]
            named_tuning = "d_standard"
            message = "This section sounds like D standard -- tune everything down a whole step."
        else:
            open_strings[0] -= 2
            named_tuning = "drop_d"
            message = "This section sounds like Drop D."
    else:
        message = "Tuning looks like standard E."

    if abs(fine_offset_cents) >= FINE_DRIFT_REPORT_THRESHOLD_CENTS:
        message += f" Also drifting ~{fine_offset_cents:.0f} cents off pitch overall."

    return TuningReport(fine_offset_cents, named_tuning, open_strings, message)
