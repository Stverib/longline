"""Numeric helpers."""


def mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2
