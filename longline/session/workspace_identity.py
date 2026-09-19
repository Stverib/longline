"""Workspace identity: what a checkpoint was taken against, and whether it still holds.

=== Why not a hash of the whole repository ===

Hashing every file is expensive on a real checkout and produces a boolean that
is wrong in both directions: an unrelated file changing blocks a resume that was
perfectly safe, and a file changing under `git checkout` is indistinguishable
from one edited by hand. Git already answers "what changed" cheaply, and the
operation journal already knows which files this session depends on
(`Tool.workload`). The check is the intersection.

=== The git primitives are here; the verdict is below ===

`current_git_head` and `changed_paths` are thin wrappers that answer "cannot
tell" as `None` rather than raising, because `git` may be absent or the directory
may not be a repository -- and "the check could not run" is a different fact from
"the workspace is clean", which the caller must be able to report separately.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from longline.utils.hashing import sha256_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from longline.session.tool_journal import OperationRecord


class DriftVerdict(Enum):
    """What the workspace did while this session was not running."""

    CLEAN = "clean"          # nothing this session depends on moved
    UNRELATED = "unrelated"  # files changed, none of them ours: resume and warn
    RELEVANT = "relevant"    # a dependency moved, or HEAD did: refuse to resume


@dataclass
class DriftReport:
    verdict: DriftVerdict
    relevant: list[str] = field(default_factory=list)
    unrelated: list[str] = field(default_factory=list)
    git_head_changed: bool = False
    git_available: bool = False

    @property
    def rejected(self) -> bool:
        return self.verdict is DriftVerdict.RELEVANT


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """Run git, returning None when it is absent or the directory is not a repo."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return proc if proc.returncode == 0 else None


def current_git_head(root: Path) -> str | None:
    """The commit the workspace is on, or None when there is no repository."""
    proc = _git(Path(root), "rev-parse", "HEAD")
    return proc.stdout.strip() if proc else None


def changed_paths(root: Path) -> set[str] | None:
    """Absolute paths git reports as changed, or None when git cannot say.

    `--porcelain` includes untracked files, which matters: a file another writer
    just created is a change to the workspace even though no tracked content
    moved.
    """
    root = Path(root)
    proc = _git(root, "status", "--porcelain")
    if proc is None:
        return None
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip().strip('"')
        if " -> " in rel:  # a rename reports both sides
            rel = rel.split(" -> ", 1)[1]
        out.add(str((root / rel).resolve()))
    return out


__all__ = [
    "DriftReport",
    "DriftVerdict",
    "changed_paths",
    "classify_drift",
    "current_git_head",
]


def classify_drift(
    *,
    root: Path,
    header: dict[str, object] | None,
    records: Sequence[OperationRecord],
) -> DriftReport:
    """Compare the workspace now against the identity the session recorded.

    `header` is the session header written by `ToolJournal.write_session_header`,
    and `records` are that journal's records. A session with no header makes no
    claim about the workspace and is therefore never rejected -- the alternative
    would be refusing to resume every session recorded before this existed.

    Anything other than RELEVANT continues. A recovery mechanism that refuses
    whenever the tree is dirty is a mechanism nobody can use.
    """
    from longline.session.tool_journal import workspace_from_records

    root = Path(root)
    if header is None:
        return DriftReport(verdict=DriftVerdict.CLEAN, git_available=False)

    read_set, write_set = workspace_from_records(records)
    # Resolved on this side too: `changed_paths` reports resolved paths, and a
    # set difference between two spellings of the same file is a silent miss.
    recorded = {
        str(Path(path).resolve()): digest
        for path, digest in {**read_set, **write_set}.items()
    }

    pinned_head = header.get("git_head")
    head_changed = bool(pinned_head) and current_git_head(root) != pinned_head

    # A file that is gone digests to "missing", which equals no recorded digest --
    # so a deleted dependency reads as relevant, which it is.
    relevant = sorted(
        path for path, digest in recorded.items() if sha256_file(Path(path)) != digest
    )

    changed = changed_paths(root)
    unrelated = sorted(changed - set(recorded)) if changed is not None else []

    if relevant or head_changed:
        return DriftReport(
            verdict=DriftVerdict.RELEVANT,
            relevant=relevant,
            unrelated=unrelated,
            git_head_changed=head_changed,
            git_available=changed is not None,
        )
    if unrelated:
        return DriftReport(
            verdict=DriftVerdict.UNRELATED,
            unrelated=unrelated,
            git_available=True,
        )
    return DriftReport(verdict=DriftVerdict.CLEAN, git_available=changed is not None)
