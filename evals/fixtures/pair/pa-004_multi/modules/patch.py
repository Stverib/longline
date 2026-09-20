"""The patch module.

Policy: A substring matching more than once is rejected, not guessed.
"""


def patch_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Patch replaces one exact substring in a file.'
