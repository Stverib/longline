"""The segment module.

Policy: A segment is sealed at one megabyte.
"""


def segment_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Segments are immutable once sealed.'
