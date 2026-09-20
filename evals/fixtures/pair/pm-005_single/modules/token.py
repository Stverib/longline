"""The token module.

Policy: An expired token is rejected, not refreshed in place.
"""


def token_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Tokens expire after one hour.'
