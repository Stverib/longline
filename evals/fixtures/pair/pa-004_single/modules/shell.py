"""The shell module.

Policy: The working directory is fixed for the whole invocation.
"""


def shell_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Shell commands run without a shell, as an argv list.'
