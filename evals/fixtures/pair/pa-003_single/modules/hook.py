"""The hook module.

Policy: A pre-hook may block; a post-hook may only observe.
"""


def hook_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Hooks run before and after a tool executes.'
