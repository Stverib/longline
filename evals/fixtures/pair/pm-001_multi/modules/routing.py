"""The routing module.

Policy: An unmatched route is an error, not a default.
"""


def routing_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Routes are matched longest-prefix first.'
