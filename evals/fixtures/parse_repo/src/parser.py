"""A tiny field splitter."""

SEPARATOR = " "  # defect


def split_fields(line: str) -> list[str]:
    return line.split(SEPARATOR)
