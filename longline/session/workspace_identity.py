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
from pathlib import Path


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


__all__ = ["changed_paths", "current_git_head"]
