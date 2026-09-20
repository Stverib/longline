"""The prompt module.

Policy: Section order is fixed; a variant may omit a section only.
"""


def prompt_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'The system prompt is assembled from ordered sections.'
