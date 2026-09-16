"""Response cache.

Keyed by request path. The cache is process-local and intentionally not
shared across workers: a shared cache would need an eviction protocol this
service does not have.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TTL_S = 30
MAX_ENTRIES = 256


class ResponseCache:
    """A bounded, TTL'd cache of response bodies."""

    def __init__(self, ttl_s: int = DEFAULT_TTL_S, max_entries: int = MAX_ENTRIES) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, now: float) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if now - stored_at > self.ttl_s:
            del self._entries[key]
            return None
        return value

    def put(self, key: str, value: Any, now: float) -> None:
        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k][0])
            del self._entries[oldest]
        self._entries[key] = (now, value)
