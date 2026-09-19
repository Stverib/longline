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
    # Files whose interrupted write the TOOL verified it had applied, and which
    # are therefore this session's own revision rather than drift. Named apart
    # from `relevant`/`unrelated` because they were settled by evidence, and a
    # reader should be able to see which files that applied to.
    verified_applied: list[str] = field(default_factory=list)
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

    `--untracked-files=all` matters for the same reason, one level down. Without
    it git collapses a wholly-untracked directory to the DIRECTORY (`?? src/`),
    and the caller compares this set against recorded FILE paths -- so a file the
    session itself just created inside a new directory matched nothing and was
    reported as somebody else's unrelated change. Listing every file makes the
    two sides comparable.
    """
    root = Path(root)
    proc = _git(root, "status", "--porcelain", "--untracked-files=all")
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


def _verified_applied_writes(
    records: Sequence[OperationRecord],
) -> dict[str, str]:
    """Files whose interrupted write the TOOL verified landed, and their revision.

    A write that was interrupted has no `post_state`, so `workspace_from_records`
    leaves it out. That is right as far as it goes -- substituting the operation's
    `pre_state` would compare the world against a revision the session never
    claimed. But leaving it out entirely has a cost the old suite could not see:
    the file is still in the READ set (the session read it before editing it), so
    its digest no longer matching reads as `RELEVANT` and the resume is REFUSED.
    A crash inside an edit made the session permanently unresumable, refused for
    having done the very thing it was asked to do.

    **Reconciliation is the evidence, and the only evidence that will do.** When
    the process came back, `reconcile_pending` asked the tool whether its effect
    was present. A tool that can read its own effect answers, and `RECONCILED`
    means it looked and found it. The revision on disk is then attributable to
    this session, and comparing it against itself is not the vacuous move the
    read side would be -- the verification is what makes it meaningful.

    `ABORTED` and `INDETERMINATE` settle nothing, and that is the whole point.
    `Edit.reconcile` answers `NOT_APPLIED` by finding the OLD text still intact,
    which is exactly what an injected drift on an edit that never ran looks like;
    settling there would mask the drift this check exists to catch. `Bash` cannot
    read its own effect at all and answers `INDETERMINATE`, so it never settles
    either -- its file stays classified as somebody else's change, which is the
    documented limitation rather than a new one.

    An earlier version of this settled any interrupted write whose file had
    changed, on the reasoning that the session's own write is the likely
    explanation. It passed every recovery test and broke
    `test_the_dependent_drift_arm_is_refused`: the arm's Edit is interrupted
    BEFORE it runs, the parent then appends to the same file, and "the digest
    moved" cannot tell those apart. Verification can.
    """
    from longline.session.tool_journal import PREPARED, RECONCILED

    starts: dict[str, OperationRecord] = {}
    latest: dict[str, str] = {}
    for record in records:
        if record.status == PREPARED:
            # First start wins, as in `workspace_from_records`: a retried call is
            # a new operation with its own id.
            starts.setdefault(record.operation_id, record)
        latest[record.operation_id] = record.status

    settled: dict[str, str] = {}
    for operation_id, start in starts.items():
        if latest.get(operation_id) != RECONCILED:
            continue
        for path, mode in start.access.items():
            if mode == "write":
                settled[path] = sha256_file(Path(path))
    return settled


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
    # Last, so a write the tool verified it applied wins over the digest the READ
    # set carries for the same path: the read is the older claim.
    verified = _verified_applied_writes(records)
    # Resolved on this side too: `changed_paths` reports resolved paths, and a
    # set difference between two spellings of the same file is a silent miss.
    recorded = {
        str(Path(path).resolve()): digest
        for path, digest in {**read_set, **write_set, **verified}.items()
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
            verified_applied=sorted(verified),
            git_head_changed=head_changed,
            git_available=changed is not None,
        )
    if unrelated:
        return DriftReport(
            verdict=DriftVerdict.UNRELATED,
            unrelated=unrelated,
            verified_applied=sorted(verified),
            git_available=True,
        )
    return DriftReport(
        verdict=DriftVerdict.CLEAN,
        verified_applied=sorted(verified),
        git_available=changed is not None,
    )
