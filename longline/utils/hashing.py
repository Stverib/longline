"""Content digests, in one place.

Extracted so the runtime (`longline/session/tool_journal.py`) can hash a file
without importing the evaluation harness. `longline/eval/faults.py` re-exports
this name rather than keeping a second copy -- two copies of a digest function
are two sets of semantics, and every comparison across the boundary would then
be comparing different things.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


__all__ = ["MISSING", "sha256_bytes", "sha256_file"]
