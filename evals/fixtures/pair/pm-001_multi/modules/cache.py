"""The cache module.

Policy: The cache is write-through, never write-back.
"""


def cache_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Cached entries are keyed by request id.'
