"""The worker module.

Policy: A claim expires if it is not renewed.
"""


def worker_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Workers claim one job at a time.'
