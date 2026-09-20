"""The checkpoint module.

Policy: Nothing is persisted inside a single instruction.
"""


def checkpoint_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Checkpoints are written once per user instruction.'
