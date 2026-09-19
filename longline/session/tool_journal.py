"""A durable journal of tool operations, written outside the transcript.

=== The window this closes ===

A kill can land between a tool changing the world and its result reaching the
transcript. The transcript then says the call was never issued, and a resumed
agent re-issues it. `evals/README.md` §5.8 records the consequence: the
`after_tool` arm duplicated the side effect in 10 of 10 injections.

The transcript is not the only place a fact can be written down. This journal is
append-only, lives beside the session file rather than in the workspace, and is
fsynced per record, so it survives the event that erases the tool result.

=== What this is NOT ===

It is not exactly-once. An operation the runtime cannot read back -- any `Bash`
command, any third-party API -- is reconciled to UNKNOWN and reported to the
model rather than replayed, and that is deliberately weaker than exactly-once.
A report built on this module may say "the runtime did not blindly replay an
unverified side effect"; it may never say "the runtime ran it exactly once".

=== Why the record carries the tool input ===

Reconciliation asks the TOOL whether its effect is present, and a tool can only
answer from its own arguments plus the world. `Write` needs the content it would
have written; `Edit` needs both strings. The input is already in the transcript's
`tool_use` block in the common case, but "in the common case" is not a thing a
recovery path may rely on -- the uncommon case is the whole point.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.tools.base import ReconcileOutcome
from longline.utils.hashing import input_fingerprint, sha256_file

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# One file per session, beside the transcript. Not inside the workspace: the
# workspace is what is under test, and a journal written into it would change
# the very digests it records.
OP_JOURNAL_NAME = "tool_ops.jsonl"
HEADER_NAME = "session_header.json"

# The status machine. PREPARED and COMMITTED are written by the executor; the
# other three are assigned by a resume, to a PREPARED that never committed.
PREPARED = "PREPARED"
COMMITTED = "COMMITTED"
RECONCILED = "RECONCILED"        # the effect was verified present
ABORTED = "ABORTED"              # the effect was verified absent; safe to retry
INDETERMINATE = "INDETERMINATE"  # the tool cannot read its own effect

# The statuses that mean "this operation is finished, one way or another".
TERMINAL_STATUSES: tuple[str, ...] = (COMMITTED, RECONCILED, ABORTED, INDETERMINATE)

# The statuses whose post_state is a real reading of the world.
DIGESTED_STATUSES: tuple[str, ...] = (COMMITTED,)

# How a reconcile verdict maps onto a terminal status.
OUTCOME_STATUS: dict[ReconcileOutcome, str] = {
    ReconcileOutcome.APPLIED: RECONCILED,
    ReconcileOutcome.NOT_APPLIED: ABORTED,
    ReconcileOutcome.UNKNOWN: INDETERMINATE,
}

# The texts a reconciled operation contributes to the transcript. Prefixed so the
# model -- and a scripted model -- can tell the three apart mechanically, which
# is the whole reason there are three and not one.
RECONCILE_APPLIED_PREFIX = "[tool journal] already applied before the interruption"
RECONCILE_ABORTED_PREFIX = "[tool journal] did not take effect; safe to retry"
RECONCILE_UNKNOWN_PREFIX = "[tool journal] outcome unknown; not re-run"

_VERDICT_TEXT: dict[str, str] = {
    RECONCILED: RECONCILE_APPLIED_PREFIX,
    ABORTED: RECONCILE_ABORTED_PREFIX,
    INDETERMINATE: RECONCILE_UNKNOWN_PREFIX,
}


def journal_path(claude_dir: Path, session_id: str) -> Path:
    """Where one session's operation journal lives."""
    from longline.session.storage import get_sessions_dir

    return get_sessions_dir(claude_dir) / f"{session_id}.{OP_JOURNAL_NAME}"


def header_path(claude_dir: Path, session_id: str) -> Path:
    """Where one session's workspace identity lives."""
    from longline.session.storage import get_sessions_dir

    return get_sessions_dir(claude_dir) / f"{session_id}.{HEADER_NAME}"


@dataclass(frozen=True)
class OperationRecord:
    """One line of the journal: a start, an end, or a recovery verdict.

    The verdict records carry only `operation_id`, `status` and `outcome`: they
    are the last word on an operation, and repeating the start's fields into them
    would create a second copy of a fact that a later edit could disagree with.
    `pending()` and `workspace_from_records()` both read the START for identity
    and the LATEST for status, which is the only arrangement that keeps the two
    from drifting.
    """

    operation_id: str
    session_id: str
    turn_id: int
    tool_call_id: str
    tool_name: str
    input_fingerprint: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    status: str = PREPARED
    pre_state: dict[str, str] = field(default_factory=dict)
    post_state: dict[str, str] = field(default_factory=dict)
    access: dict[str, str] = field(default_factory=dict)
    outcome: str = ""
    timestamp: float = 0.0

    @property
    def settled(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> OperationRecord:
        return cls(
            operation_id=str(row["operation_id"]),
            session_id=str(row.get("session_id", "")),
            turn_id=int(row.get("turn_id", 0)),
            tool_call_id=str(row.get("tool_call_id", "")),
            tool_name=str(row.get("tool_name", "")),
            input_fingerprint=str(row.get("input_fingerprint", "")),
            tool_input=dict(row.get("tool_input") or {}),
            status=str(row.get("status", PREPARED)),
            pre_state={str(k): str(v) for k, v in dict(row.get("pre_state") or {}).items()},
            post_state={str(k): str(v) for k, v in dict(row.get("post_state") or {}).items()},
            access={str(k): str(v) for k, v in dict(row.get("access") or {}).items()},
            outcome=str(row.get("outcome", "")),
            timestamp=float(row.get("timestamp", 0.0)),
        )


class ToolJournal:
    """Append-only JSONL of tool operations for one session."""

    def __init__(self, claude_dir: Path, session_id: str) -> None:
        self._claude_dir = Path(claude_dir)
        self._session_id = session_id

    @property
    def path(self) -> Path:
        return journal_path(self._claude_dir, self._session_id)

    # -- writing --

    def _append(self, record: OperationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_row(), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def prepare(
        self,
        *,
        turn_id: int,
        tool_call_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        workload: Mapping[str, str],
    ) -> str:
        """Record the intent to run, and digest the declared paths first.

        This record exists so that a process which dies during `execute` leaves
        behind the fact that it was ABOUT to do something -- the one fact the
        transcript cannot supply, because the transcript is not written here.
        """
        operation_id = uuid.uuid4().hex[:16]
        inputs = dict(tool_input)
        self._append(
            OperationRecord(
                operation_id=operation_id,
                session_id=self._session_id,
                turn_id=turn_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                input_fingerprint=input_fingerprint(tool_name, inputs),
                tool_input=inputs,
                status=PREPARED,
                pre_state={path: sha256_file(Path(path)) for path in workload},
                access=dict(workload),
                timestamp=time.time(),
            )
        )
        return operation_id

    def commit(
        self, operation_id: str, *, outcome: str, post_state: Mapping[str, str]
    ) -> None:
        """Record that the operation returned, and what the world looks like now."""
        self._append(
            OperationRecord(
                operation_id=operation_id,
                session_id=self._session_id,
                turn_id=0,
                tool_call_id="",
                tool_name="",
                input_fingerprint="",
                status=COMMITTED,
                post_state=dict(post_state),
                outcome=outcome,
                timestamp=time.time(),
            )
        )

    def resolve(self, operation_id: str, *, status: str, outcome: ReconcileOutcome) -> None:
        """Close an orphaned PREPARED with a recovery verdict.

        Terminal by contract: the verdict is the last word on the operation, so a
        status that is not final is refused rather than written.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"resolve needs a terminal status, got {status!r}")
        self._append(
            OperationRecord(
                operation_id=operation_id,
                session_id=self._session_id,
                turn_id=0,
                tool_call_id="",
                tool_name="",
                input_fingerprint="",
                status=status,
                outcome=outcome.value,
                timestamp=time.time(),
            )
        )

    def write_session_header(self, *, workspace_root: str, git_head: str | None) -> None:
        path = header_path(self._claude_dir, self._session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"workspace_root": workspace_root, "git_head": git_head}, sort_keys=True
            ),
            encoding="utf-8",
        )

    # -- reading --

    def session_header(self) -> dict[str, Any] | None:
        """The workspace identity, or None when there is none to check."""
        path = header_path(self._claude_dir, self._session_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def records(self) -> list[OperationRecord]:
        """Every record, dropping an unparseable tail.

        Dropping rather than raising is the same choice `side_effect_journal`
        makes and for the same reason: a torn last line is a plausible result of
        a kill mid-append, and refusing to read the journal because of it would
        turn the fault under study into a harness crash.
        """
        path = self.path
        if not path.is_file():
            return []
        out: list[OperationRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(OperationRecord.from_row(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return out

    def _latest(self) -> dict[str, OperationRecord]:
        """The last record per operation, which is the one carrying its status."""
        latest: dict[str, OperationRecord] = {}
        for record in self.records():
            latest[record.operation_id] = record
        return latest

    def pending(self) -> list[OperationRecord]:
        """Operations that started and never committed, in the order they started.

        Returns the START record, not the latest one: the verdict records carry
        no `tool_call_id`, and that id is what pairs the recovered result with
        the `tool_use` already sitting on the transcript.

        Keyed on the OPERATION id rather than the tool call id, because a retried
        tool call is a new operation with its own outcome.
        """
        by_id = self._latest()
        seen: set[str] = set()
        out: list[OperationRecord] = []
        for record in self.records():
            if record.status != PREPARED or record.operation_id in seen:
                continue
            seen.add(record.operation_id)
            if by_id[record.operation_id].status == PREPARED:
                out.append(record)
        return out


@dataclass(frozen=True)
class ReconciledOperation:
    """One orphaned operation, closed, with the result the transcript should carry."""

    operation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    tool_input: dict[str, Any] = field(default_factory=dict)

    @property
    def result_text(self) -> str:
        summary = json.dumps(self.tool_input, sort_keys=True, ensure_ascii=False, default=str)
        return f"{_VERDICT_TEXT[self.status]}: {self.tool_name} {summary[:200]}".strip()

    @property
    def is_error(self) -> bool:
        """Only a verified-applied operation is not an error.

        The other two are errors because the model must not treat the step as
        cleanly finished -- one it should retry, and one it must not.
        """
        return self.status != RECONCILED


def reconcile_pending(journal: ToolJournal, registry: Any) -> list[ReconciledOperation]:
    """Ask each tool about its orphaned operation, and close it.

    A tool that is no longer registered answers UNKNOWN rather than "failed": the
    absence of a tool is not evidence about what it did before it went away. A
    tool whose `reconcile` raises answers UNKNOWN for the same reason -- a broken
    reconciler must not be able to block a resume, and it certainly must not be
    able to authorise a retry it did not earn.
    """
    out: list[ReconciledOperation] = []
    for record in journal.pending():
        tool = registry.get(record.tool_name) if registry is not None else None
        if tool is None:
            outcome = ReconcileOutcome.UNKNOWN
        else:
            try:
                outcome = tool.reconcile(record.tool_input)
            except Exception:
                outcome = ReconcileOutcome.UNKNOWN
        status = OUTCOME_STATUS[outcome]
        journal.resolve(record.operation_id, status=status, outcome=outcome)
        out.append(
            ReconciledOperation(
                operation_id=record.operation_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                status=status,
                tool_input=dict(record.tool_input),
            )
        )
    return out


def workspace_from_records(
    records: Sequence[OperationRecord],
) -> tuple[dict[str, str], dict[str, str]]:
    """`(read_set, write_set)` of `{path: digest}`, from committed operations only.

    A path's digest is the one the LAST committed operation on it recorded -- the
    revision the session left the file at, which is the only revision a later
    reading of the same file can be compared against.

    An uncommitted operation contributes nothing. Its `post_state` does not
    exist, and substituting its `pre_state` would make the identity check compare
    the world against a revision the session never claimed -- so every
    interrupted write would read as drift.

    A RECONCILED operation contributes nothing either, and that is a real limit
    rather than an oversight: the digest would have to be taken at resume time,
    from the very world the check is about to compare against, so it would match
    by construction and say nothing.
    """
    read_set: dict[str, str] = {}
    write_set: dict[str, str] = {}
    starts: dict[str, OperationRecord] = {}
    commits: dict[str, OperationRecord] = {}
    for record in records:
        if record.status == PREPARED:
            # First start wins: a retried call is a new operation with its own id.
            starts.setdefault(record.operation_id, record)
        elif record.status in DIGESTED_STATUSES:
            commits[record.operation_id] = record

    # `commits` iterates in the order the operations committed, so a path touched
    # twice ends up with the digest from the LATER commit -- the revision the
    # session actually left it at.
    for operation_id, commit in commits.items():
        start = starts.get(operation_id)
        if start is None:
            continue
        for path, mode in start.access.items():
            digest = commit.post_state.get(path)
            if digest is None:
                continue
            (read_set if mode == "read" else write_set)[path] = digest
    return read_set, write_set


__all__ = [
    "ABORTED",
    "COMMITTED",
    "DIGESTED_STATUSES",
    "HEADER_NAME",
    "INDETERMINATE",
    "OP_JOURNAL_NAME",
    "OUTCOME_STATUS",
    "PREPARED",
    "RECONCILED",
    "RECONCILE_ABORTED_PREFIX",
    "RECONCILE_APPLIED_PREFIX",
    "RECONCILE_UNKNOWN_PREFIX",
    "TERMINAL_STATUSES",
    "OperationRecord",
    "ReconciledOperation",
    "ToolJournal",
    "header_path",
    "journal_path",
    "reconcile_pending",
    "workspace_from_records",
]
