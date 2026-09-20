"""The identity module.

Policy: An id that collides with the lead is rejected.
"""


def identity_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Agent ids are name-at-team, and the lead is reserved.'
