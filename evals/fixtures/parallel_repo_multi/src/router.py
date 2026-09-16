"""Request router.

Owns the dispatch table and the 404 path. Deliberately does not import any
module from `modules/`: routing must not depend on a cache or a retry policy
being importable, or a broken module would take the router down with it.
"""

from __future__ import annotations

from typing import Any

ROUTES: dict[str, str] = {
    "/health": "health",
    "/v1/echo": "echo",
}

# Timeout budget for the whole request, in milliseconds. A downstream module
# may shorten its own deadline but never extend this one.
REQUEST_BUDGET_MS = 5000


def resolve(path: str) -> str:
    """Map a request path to a handler name, or raise KeyError."""
    try:
        return ROUTES[path]
    except KeyError:
        raise KeyError(f"no route for {path!r}") from None


def dispatch(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    handler = resolve(path)
    return {"handler": handler, "payload": payload or {}}
