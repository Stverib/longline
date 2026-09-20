"""The latency module.

Policy: A saturating host invalidates a latency measurement.
"""


def latency_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Latency cases declare a schedule of tool durations.'
