"""Small helpers used by the fixture package."""


def clamp(value: int, low: int, high: int) -> int:
    """Restrict value to the [low, high] range."""
    return max(low, min(value, high))
