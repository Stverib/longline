"""Text helpers."""


def slugify(value):
    out = value.strip().lower()
    out = out.replace(" ", "-")
    return out
