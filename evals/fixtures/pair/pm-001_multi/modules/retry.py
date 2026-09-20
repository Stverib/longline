"""The retry module.

Policy: A retry reuses the original deadline.
"""


def retry_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Retries stop after the third attempt.'
