"""The permission module.

Policy: A denial is final; a later rule cannot re-allow the call.
"""


def permission_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Permission decisions are deny, ask or allow, in that order.'
