"""The compact module.

Policy: Compaction never runs while a read is open.
"""


def compact_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Compaction merges adjacent segments.'
