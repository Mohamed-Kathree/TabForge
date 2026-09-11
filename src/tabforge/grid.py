"""Convert between TabEvents and an editable grid (pandas DataFrame).

Not a pipeline stage -- this exists for app.py's editable tab grid (the
spec: "the user will fix the last 10% by ear regardless, and being able
to correct a wrong fret is worth more than squeezing out another 2% of
accuracy"). Kept separate from render.py and free of any Streamlit
import so it's unit-testable without a running app.
"""

import pandas as pd

from .models import TabEvent, STRING_NAMES


def events_to_grid(events: list[TabEvent]) -> tuple[pd.DataFrame, list[float], list[float]]:
    """TabEvents -> (editable DataFrame, onset per column, duration per column).

    Rows are string names, high e first (top), matching the ASCII tab.
    Columns are onsets in order; a note's fret goes in its (string, onset)
    cell, empty cells are pandas NA.
    """
    columns: list[dict[int, int]] = []
    col_starts: list[float] = []
    col_durations: list[float] = []
    for ev in sorted(events, key=lambda e: e.start):
        if not col_starts or ev.start - col_starts[-1] > 1e-6:
            columns.append({})
            col_starts.append(ev.start)
            col_durations.append(ev.duration)
        columns[-1][ev.string] = ev.fret
        col_durations[-1] = max(col_durations[-1], ev.duration)

    data = {
        str(i): [columns[i].get(string_idx) for string_idx in range(5, -1, -1)] for i in range(len(columns))
    }
    df = pd.DataFrame(data, index=[STRING_NAMES[s] for s in range(5, -1, -1)], dtype="Int64")
    return df, col_starts, col_durations


def grid_to_events(df: pd.DataFrame, col_starts: list[float], col_durations: list[float]) -> list[TabEvent]:
    """Inverse of events_to_grid -- read an edited grid back into TabEvents."""
    name_to_string = {STRING_NAMES[s]: s for s in range(6)}
    events = []
    for row_name in df.index:
        string_idx = name_to_string[row_name]
        for col_pos, col_label in enumerate(df.columns):
            val = df.loc[row_name, col_label]
            if pd.isna(val):
                continue
            events.append(
                TabEvent(
                    start=col_starts[col_pos],
                    duration=col_durations[col_pos],
                    string=string_idx,
                    fret=int(val),
                )
            )
    return events
