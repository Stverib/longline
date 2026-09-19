"""Content digests, in one place.

Extracted so the runtime (`longline/session/tool_journal.py`) can hash a file
without importing the evaluation harness. `longline/eval/faults.py` re-exports
this name rather than keeping a second copy -- two copies of a digest function
are two sets of semantics, and every comparison across the boundary would then
be comparing different things.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# What an absent file digests to. A sentinel rather than an exception: "the
# artifact is not there" is a fact about a run that a report must be able to
# state, and the alternative is a try/except at every call site.
MISSING = "missing"


def sha256_bytes(data: bytes) -> str:
    """Digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Digest of a file's bytes, or `MISSING` when it is not there."""
    if not path.is_file():
        return MISSING
    return sha256_bytes(path.read_bytes())


def input_fingerprint(tool: str, tool_input: Mapping[str, Any]) -> str:
    """Stable fingerprint of a tool request.

    Keyed on the tool name as well as the arguments: `Bash({"command": "x"})`
    and `Write({"content": "x"})` are not the same request, and a fingerprint
    that ignored the name would report one as a replay of the other.
    `sort_keys=True` makes it independent of dict insertion order, which the two
    legs of a kill-and-resume do not share.

    Lives here rather than in the evaluation harness because the RUNTIME now
    records the same fingerprint (`longline/session/tool_journal.py`) and the
    runtime may not import from `longline/eval/`. Two implementations would be
    two answers to "was this the same request", compared across a process
    boundary where the disagreement would be invisible.
    """
    payload = json.dumps(
        {"tool": tool, "input": dict(tool_input)},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256_bytes(payload)[:16]


__all__ = ["MISSING", "input_fingerprint", "sha256_bytes", "sha256_file"]
