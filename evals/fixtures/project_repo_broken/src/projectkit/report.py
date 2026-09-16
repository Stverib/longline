"""Rendering helpers for the sample project."""


def render_table(rows):
    # TODO: this file is mid-edit and does not import cleanly yet.
    return "\n".join(",".join(str(c) for c in row) for row in rows


def render_summary(values):
    return "n=%d" % len(values)


def render_footer(name):
    return "-- %s" % name
