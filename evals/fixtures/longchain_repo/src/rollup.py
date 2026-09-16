"""Rollup maths for the longchain fixture.

`total` is deliberately not wired into `combined_score` yet — the aggregate is
the part the task asks for.
"""


def total(values):
    return sum(values)


def combined_score(carried, current):
    return carried + current
