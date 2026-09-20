"""The backoff module.

Policy: Jitter is applied after the doubling.
"""


def backoff_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Backoff doubles up to thirty seconds.'
