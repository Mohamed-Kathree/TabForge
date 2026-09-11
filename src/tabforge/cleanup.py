"""Stage 5: filter, beat-track, quantize.

In order: drop notes outside the guitar range, drop low-confidence notes,
then (optionally) snap onsets to a 16th-note grid derived from librosa's
beat tracker and merge notes that land on the same slot.

Quantization is toggleable -- it destroys swung or rubato playing, so
callers that care about feel should skip `quantize()` entirely.
"""

import numpy as np
import librosa

from .models import Note

MIN_MIDI = 40
MAX_MIDI = 88
CONFIDENCE_THRESHOLD_DEFAULT = 0.3

# Notes on the same quantized onset within this many semitones are
# treated as one detection (basic-pitch sometimes reports the same
# physical note twice at adjacent pitches -- vibrato, noise).
MERGE_SEMITONE_THRESHOLD = 1


def filter_range(notes: list[Note], min_midi: int = MIN_MIDI, max_midi: int = MAX_MIDI) -> list[Note]:
    """Drop notes outside the guitar range -- octave errors from
    distortion harmonics are the single most common failure mode."""
    return [n for n in notes if min_midi <= n.pitch <= max_midi]


def filter_confidence(notes: list[Note], threshold: float = CONFIDENCE_THRESHOLD_DEFAULT) -> list[Note]:
    return [n for n in notes if n.confidence >= threshold]


def detect_beat_grid(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Beat-track the clip and return (bpm, 16th-note grid times in seconds).

    The grid is extended backward/forward from the detected beats (using
    the average beat interval) so notes near the edges of a short clip
    still have a nearby slot to snap to.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        return bpm, np.array([])

    grid = []
    for start, end in zip(beat_times[:-1], beat_times[1:]):
        step = (end - start) / 4
        grid.extend(start + step * k for k in range(4))
    grid.append(beat_times[-1])

    step = float(np.mean(np.diff(beat_times))) / 4
    clip_duration = y.shape[-1] / sr
    t = grid[0] - step
    while t > 0:
        grid.insert(0, t)
        t -= step
    t = grid[-1] + step
    while t < clip_duration:
        grid.append(t)
        t += step

    return bpm, np.array(sorted(grid))


def _collapse(cluster: list[Note]) -> Note:
    best = max(cluster, key=lambda n: n.confidence)
    duration = max(n.duration for n in cluster)
    return Note(start=best.start, duration=duration, pitch=best.pitch, confidence=best.confidence)


def _merge_same_slot(notes: list[Note], semitone_threshold: int = MERGE_SEMITONE_THRESHOLD) -> list[Note]:
    groups: dict[float, list[Note]] = {}
    for n in notes:
        groups.setdefault(n.start, []).append(n)

    merged: list[Note] = []
    for group in groups.values():
        group = sorted(group, key=lambda n: n.pitch)
        cluster = [group[0]]
        for note in group[1:]:
            if note.pitch - cluster[-1].pitch <= semitone_threshold:
                cluster.append(note)
            else:
                merged.append(_collapse(cluster))
                cluster = [note]
        merged.append(_collapse(cluster))
    return merged


def quantize(notes: list[Note], grid: np.ndarray) -> list[Note]:
    """Snap each note's onset to the nearest grid slot, then merge
    same-slot notes within a semitone of each other."""
    if grid.size == 0 or not notes:
        return notes

    snapped = []
    for n in notes:
        idx = int(np.argmin(np.abs(grid - n.start)))
        new_start = float(grid[idx])
        original_end = n.start + n.duration
        new_duration = max(original_end - new_start, 0.01)
        snapped.append(Note(start=new_start, duration=new_duration, pitch=n.pitch, confidence=n.confidence))

    return sorted(_merge_same_slot(snapped), key=lambda n: n.start)


def cleanup(
    notes: list[Note],
    y: np.ndarray | None = None,
    sr: int | None = None,
    min_midi: int = MIN_MIDI,
    max_midi: int = MAX_MIDI,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    quantize_enabled: bool = False,
) -> list[Note]:
    notes = filter_range(notes, min_midi, max_midi)
    notes = filter_confidence(notes, confidence_threshold)
    if quantize_enabled:
        if y is None or sr is None:
            raise ValueError("quantize_enabled=True requires audio (y, sr) for beat tracking")
        _bpm, grid = detect_beat_grid(y, sr)
        notes = quantize(notes, grid)
    return notes
