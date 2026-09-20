"""The schema module.

Policy: An unknown field is rejected, never coerced.
"""


def schema_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Fields are validated against a declared type.'
