"""The queue module.

Policy: A job is dequeued once, under a lock.
"""


def queue_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'The queue is ordered by deadline.'
