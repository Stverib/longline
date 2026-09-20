"""The spawn module.

Policy: A teammate runs as a task on the caller's event loop.
"""


def spawn_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Spawning registers an identity, a team row and a task.'
