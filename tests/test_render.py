from tabforge.models import TabEvent
from tabforge.render import render_ascii


def test_empty_events():
    assert render_ascii([]) == "(no notes)"


def test_six_lines_high_e_on_top():
    events = [
        TabEvent(start=0.0, duration=0.25, string=0, fret=3),
        TabEvent(start=0.25, duration=0.25, string=5, fret=0),
    ]
    tab = render_ascii(events)
    lines = tab.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("e|")  # high e on top
    assert lines[-1].startswith("E|")  # low E on bottom


def test_chord_shares_one_column():
    events = [
        TabEvent(start=0.0, duration=0.5, string=0, fret=0),
        TabEvent(start=0.0, duration=0.5, string=1, fret=0),
    ]
    tab = render_ascii(events)
    lines = tab.splitlines()
    # both notes share an onset -> one column, so each line has exactly
    # one cell between the leading string name and the closing bar
    assert len(lines[0].split("|")) == 3  # "E", body, "" (trailing from closing |)


def test_wraps_into_blocks():
    events = [TabEvent(start=float(i), duration=0.1, string=0, fret=i) for i in range(20)]
    tab = render_ascii(events, positions_per_line=16)
    blocks = tab.split("\n\n")
    assert len(blocks) == 2
