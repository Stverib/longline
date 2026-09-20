"""The audit module.

Policy: An audit row records the decision, not the request.
"""


def audit_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Audit rows are appended, never updated.'
