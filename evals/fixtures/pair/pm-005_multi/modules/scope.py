"""The scope module.

Policy: A missing scope denies rather than defers.
"""


def scope_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Scopes are checked innermost first.'
