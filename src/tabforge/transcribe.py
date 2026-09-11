"""Stage 4: basic-pitch wrapper -> list[Note].

basic-pitch's `predict` takes a file path, not an in-memory array, so we
round-trip the clip through a temp wav. This keeps the stage boundary clean:
audio samples in, list[Note] out.
"""

import tempfile
import os

import numpy as np
import soundfile as sf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from .models import Note

# Guitar-ish range gate happens in cleanup.py, not here. This stage just
# converts whatever basic-pitch finds into our Note contract.

ONSET_THRESHOLD_DEFAULT = 0.6
FRAME_THRESHOLD_DEFAULT = 0.4
MINIMUM_NOTE_LENGTH_DEFAULT = 100  # ms


def transcribe(
    y: np.ndarray,
    sr: int,
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT,
    frame_threshold: float = FRAME_THRESHOLD_DEFAULT,
    minimum_note_length: float = MINIMUM_NOTE_LENGTH_DEFAULT,
) -> list[Note]:
    """Run basic-pitch on a clip already loaded via io_audio.load_clip."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_path, y, sr)
        _, _, note_events = predict(
            tmp_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
        )
    finally:
        os.remove(tmp_path)

    notes = []
    for start_s, end_s, pitch, amplitude, _pitch_bends in note_events:
        notes.append(
            Note(
                start=float(start_s),
                duration=float(end_s - start_s),
                pitch=int(pitch),
                confidence=float(amplitude),
            )
        )
    notes.sort(key=lambda n: n.start)
    return notes
