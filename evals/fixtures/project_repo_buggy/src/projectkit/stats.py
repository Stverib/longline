"""Numeric helpers."""


def mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    # BUG: the even-length branch should average the two middle values.
    return ordered[n // 2]
