"""Milestone 6: Streamlit UI wrapping the pipeline for interactive use.

Not a pipeline stage itself -- just wires stages 1-7 (via tabforge.pipeline)
to widgets, plus the editable tab grid the spec calls for: the user fixes
the last 10% by ear, so a wrong fret needs to be a click away, not a
re-run.
"""

import io
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import librosa
import soundfile as sf
import guitarpro

from tabforge import pipeline
from tabforge.io_audio import SAMPLE_RATE
from tabforge.transcribe import (
    ONSET_THRESHOLD_DEFAULT,
    FRAME_THRESHOLD_DEFAULT,
    MINIMUM_NOTE_LENGTH_DEFAULT,
)
from tabforge.cleanup import CONFIDENCE_THRESHOLD_DEFAULT
from tabforge.render import render_midi, render_gp5
from tabforge.grid import events_to_grid, grid_to_events

st.set_page_config(page_title="TabForge", layout="wide")
st.title("TabForge")
st.caption("MP3 -> guitar tab, for a short riff clip. Not a full-song transcriber.")


def audio_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac", "m4a", "ogg"])

if uploaded is None:
    st.stop()

suffix = "." + uploaded.name.rsplit(".", 1)[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getvalue())
    audio_path = tmp.name

y_full, sr_full = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration = len(y_full) / sr_full

st.subheader("1. Select a clip")
st.caption("10-20s recommended -- this tool transcribes a riff, not a whole song.")

fig, ax = plt.subplots(figsize=(10, 1.8))
times = np.linspace(0, duration, len(y_full))
ax.plot(times, y_full, linewidth=0.4, color="#4a7")
ax.set_xlim(0, duration)
ax.set_yticks([])
ax.set_xlabel("seconds")
st.pyplot(fig)
plt.close(fig)

default_end = min(15.0, duration)
start, end = st.slider(
    "Clip range (seconds)",
    min_value=0.0,
    max_value=float(duration),
    value=(0.0, float(default_end)),
    step=0.1,
)

st.subheader("2. Settings")
col1, col2, col3 = st.columns(3)
onset_threshold = col1.slider("Onset threshold", 0.0, 1.0, ONSET_THRESHOLD_DEFAULT, 0.05)
frame_threshold = col2.slider("Frame threshold", 0.0, 1.0, FRAME_THRESHOLD_DEFAULT, 0.05)
minimum_note_length = col3.slider(
    "Minimum note length (ms)", 20, 500, int(MINIMUM_NOTE_LENGTH_DEFAULT), 10
)

col4, col5, col6 = st.columns(3)
confidence_threshold = col4.slider("Confidence threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD_DEFAULT, 0.05)
use_demucs = col5.checkbox("Separate guitar first (Demucs)", help="Slow; helps on busy mixes. Downloads ~300MB the first time.")
quantize = col6.checkbox("Quantize to 16th-note grid", help="Snaps onsets to a beat grid. Destroys swing/rubato.")

run_clicked = st.button("Transcribe", type="primary")

if run_clicked:
    with st.spinner("Running the pipeline..."):
        st.session_state["result"] = pipeline.run(
            audio_path,
            start,
            end,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            confidence_threshold=confidence_threshold,
            use_demucs=use_demucs,
            quantize=quantize,
        )
        st.session_state.pop("grid_state", None)

result = st.session_state.get("result")
if result is None:
    st.stop()

st.info(result.tuning.message)

st.subheader("3. Tab")
grid_df, col_starts, col_durations = events_to_grid(result.events)

if "grid_state" not in st.session_state:
    st.session_state["grid_state"] = grid_df

st.caption("ASCII preview")
st.code(result.ascii_tab or "(no notes)", language=None)

st.caption("Editable grid -- fix a wrong fret by hand, columns are onsets left to right")
edited_df = st.data_editor(st.session_state["grid_state"], key="tab_editor")
st.session_state["grid_state"] = edited_df
edited_events = grid_to_events(edited_df, col_starts, col_durations)

st.subheader("4. Verify by ear")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Original clip")
    st.audio(audio_bytes(result.clip, result.sr))
with col_b:
    st.caption("Rendered from the (possibly edited) tab")
    pm = render_midi(edited_events, open_string_pitches=result.tuning.open_string_pitches)
    synthesized = pm.synthesize(fs=result.sr)
    st.audio(audio_bytes(synthesized.astype(np.float32), result.sr))

st.subheader("5. Download")
col_d1, col_d2 = st.columns(2)

midi_buf = io.BytesIO()
pm.write(midi_buf)
col_d1.download_button("Download MIDI", midi_buf.getvalue(), file_name="tabforge.mid", mime="audio/midi")

gp_song = render_gp5(edited_events, open_string_pitches=result.tuning.open_string_pitches, bpm=result.bpm)
gp_buf = io.BytesIO()
guitarpro.write(gp_song, gp_buf, version=(5, 1, 0))
col_d2.download_button(
    "Download Guitar Pro (.gp5)",
    gp_buf.getvalue(),
    file_name="tabforge.gp5",
    mime="application/octet-stream",
)
