"""The validate module.

Policy: Validation is all-or-nothing per record.
"""


def validate_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Validation runs before any write.'
