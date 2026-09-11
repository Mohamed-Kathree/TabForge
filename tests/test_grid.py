from tabforge.models import TabEvent
from tabforge.grid import events_to_grid, grid_to_events


def test_events_to_grid_places_frets_in_correct_row_and_column():
    events = [
        TabEvent(start=0.0, duration=0.25, string=0, fret=3),  # low E
        TabEvent(start=0.25, duration=0.25, string=5, fret=0),  # high e
    ]
    df, col_starts, col_durations = events_to_grid(events)
    assert col_starts == [0.0, 0.25]
    assert col_durations == [0.25, 0.25]
    assert list(df.index) == ["e", "B", "G", "D", "A", "E"]  # high e first
    assert df.loc["E", "0"] == 3
    assert df.loc["e", "1"] == 0
    assert df.isna().sum().sum() == 10  # 12 cells total, 2 filled


def test_chord_shares_one_column():
    events = [
        TabEvent(start=0.0, duration=0.5, string=0, fret=0),
        TabEvent(start=0.0, duration=0.5, string=1, fret=2),
    ]
    df, col_starts, _ = events_to_grid(events)
    assert len(col_starts) == 1
    assert df.loc["E", "0"] == 0
    assert df.loc["A", "0"] == 2


def test_grid_to_events_round_trips():
    events = [
        TabEvent(start=0.0, duration=0.25, string=0, fret=3),
        TabEvent(start=0.25, duration=0.5, string=5, fret=0),
    ]
    df, col_starts, col_durations = events_to_grid(events)
    result = grid_to_events(df, col_starts, col_durations)
    assert sorted(result, key=lambda e: e.start) == sorted(events, key=lambda e: e.start)


def test_grid_to_events_reflects_manual_edit():
    events = [TabEvent(start=0.0, duration=0.25, string=0, fret=3)]
    df, col_starts, col_durations = events_to_grid(events)
    df.loc["E", "0"] = 5  # user corrects a wrong fret by hand
    result = grid_to_events(df, col_starts, col_durations)
    assert result == [TabEvent(start=0.0, duration=0.25, string=0, fret=5)]


def test_empty_events_produce_empty_grid():
    df, col_starts, col_durations = events_to_grid([])
    assert col_starts == []
    assert col_durations == []
    assert df.shape == (6, 0)
    assert grid_to_events(df, col_starts, col_durations) == []
