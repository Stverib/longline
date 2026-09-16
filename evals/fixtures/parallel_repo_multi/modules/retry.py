"""Retry policy.

Exponential backoff with a hard ceiling. The ceiling matters more than the
growth rate: an unbounded retry loop turns a slow dependency into an outage.
"""

from __future__ import annotations

BASE_DELAY_MS = 100
MAX_ATTEMPTS = 4


def backoff_ms(attempt: int) -> int:
    """Delay before `attempt` (0-based). Doubles, capped by MAX_ATTEMPTS."""
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    return BASE_DELAY_MS * (2**attempt)


def should_retry(attempt: int, *, status: int | None) -> bool:
    """Whether one more attempt is worth making."""
    if attempt + 1 >= MAX_ATTEMPTS:
        return False
    if status is None:
        return True
    return status in (429, 500, 502, 503, 504)
