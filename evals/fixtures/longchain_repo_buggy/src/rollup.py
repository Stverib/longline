"""Rollup maths for the longchain fixture.

`total` is deliberately not wired into `combined_score` yet — the aggregate is
the part the task asks for.
"""


def total(values):
    # BUG: the running total is never accumulated.
    running = 0
    for v in values:
        running + v
    return running


def combined_score(carried, current):
    return carried + current
