"""The memory module.

Policy: Eviction is least-recently-written, never least-recently-read.
"""


def memory_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Memory entries are keyed by scope and expire on read.'
