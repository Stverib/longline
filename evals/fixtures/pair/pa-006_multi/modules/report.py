"""The report module.

Policy: An excluded case stays in the data with a reason.
"""


def report_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Reports aggregate results and never pool separate groups.'
