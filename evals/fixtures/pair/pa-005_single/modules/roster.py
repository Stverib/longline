"""The roster module.

Policy: Membership removal is soft; the row stays with a flag.
"""


def roster_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'The team file lists members with a joined-at time.'
