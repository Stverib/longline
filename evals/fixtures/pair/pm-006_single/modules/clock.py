"""The clock module.

Policy: Wall-clock time is never used for ordering.
"""


def clock_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'The clock is monotonic and injectable.'
