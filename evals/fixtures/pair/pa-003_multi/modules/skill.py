"""The skill module.

Policy: A skill that fails to load is skipped, not fatal.
"""


def skill_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Skills are markdown files loaded on demand by name.'
