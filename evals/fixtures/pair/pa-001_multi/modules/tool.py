"""The tool module.

Policy: An unknown tool name fails the turn rather than degrading.
"""


def tool_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Tools are registered per profile and dispatched by name.'
