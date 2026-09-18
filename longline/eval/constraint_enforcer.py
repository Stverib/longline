"""Program-level constraint enforcement for eval cases.

An instruction-following case can fail when the agent's own judgement wins
over an explicit user constraint. A prompt reminder loses that fight part of
the time; a tool that is not in the registry never loses at all. This module
removes declared-forbidden tools so the harness, not the model's goodwill,
carries the constraint.

Tool-level only: file-level scoping ("only touch this file") is not enforced
here and is deliberately left to later -- YAGNI until tool-level shows it is
not enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from longline.tools.base import ToolRegistry


def strip_forbidden(registry: ToolRegistry, forbidden: Iterable[str]) -> list[str]:
    """Remove each forbidden tool from the registry; return the removed names.

    Unknown names are ignored, not errors: a case may forbid a tool whose
    family the profile never registered, and a prohibition on an absent tool
    is already being enforced by the registry itself. The return value is the
    removed names so the caller can audit what was actually enforced.
    """
    removed: list[str] = []
    for name in forbidden:
        if registry.get(name) is not None:
            registry.remove(name)
            removed.append(name)
    return removed
