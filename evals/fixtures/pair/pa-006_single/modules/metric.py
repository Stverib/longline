"""The metric module.

Policy: A ratio with a zero denominator is reported as not measured.
"""


def metric_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Ratios are ratios; percentage points are a different unit.'
