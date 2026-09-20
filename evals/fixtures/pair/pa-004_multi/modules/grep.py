"""The grep module.

Policy: Binary files are skipped rather than decoded.
"""


def grep_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Grep returns matching lines with their line numbers.'
