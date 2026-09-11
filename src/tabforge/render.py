"""Stage 7: render TabEvents. Milestone 1 covers ASCII only."""

from .models import TabEvent

# index 0 = low E ... 5 = high e, matching TabEvent.string
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]

POSITIONS_PER_LINE = 16


def render_ascii(events: list[TabEvent], positions_per_line: int = POSITIONS_PER_LINE) -> str:
    """Render events as a 6-line ASCII tab, high e on top.

    Without beat-tracked quantization (milestone 5), each distinct onset
    becomes one column -- this is a direct, literal rendering of what was
    played, not a rhythm notation.
    """
    if not events:
        return "(no notes)"

    columns: list[dict[int, int]] = []
    current_start = None
    for ev in sorted(events, key=lambda e: e.start):
        if current_start is None or ev.start - current_start > 1e-6:
            columns.append({})
            current_start = ev.start
        columns[-1][ev.string] = ev.fret

    blocks = []
    for block_start in range(0, len(columns), positions_per_line):
        block = columns[block_start : block_start + positions_per_line]
        cells = [[str(col.get(string, "-")) for col in block] for string in range(5, -1, -1)]
        width = max((len(c) for row in cells for c in row), default=1)

        lines = []
        for string_idx, row in zip(range(5, -1, -1), cells):
            body = "-".join(c.rjust(width, "-") for c in row)
            lines.append(f"{STRING_NAMES[string_idx]}|{body}|")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
