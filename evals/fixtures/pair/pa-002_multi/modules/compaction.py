"""The compaction module.

Policy: Compaction only runs when the token budget is exceeded.
"""


def compaction_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Compaction replaces a prefix of the transcript.'
