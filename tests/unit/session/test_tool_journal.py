"""The durable tool journal.

Written OUTSIDE the transcript and fsynced per record, so it survives exactly
the event the transcript does not: a process that stops existing between a
tool's effect and its result being recorded.

The status machine is `PREPARED -> COMMITTED`, plus three terminal states a
resume can assign to an orphaned `PREPARED`. Nothing here decides WHEN to
reconcile or what a reconcile means for the model -- this module is the log.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from longline.session.tool_journal import (
    ABORTED,
    COMMITTED,
    EXECUTING,
    INDETERMINATE,
    PREPARED,
    RECONCILED,
    ToolJournal,
    journal_path,
    workspace_from_records,
)
from longline.tools.base import ReconcileOutcome
from longline.utils.hashing import sha256_bytes

if TYPE_CHECKING:
    from pathlib import Path


def _journal(tmp_path: Path) -> ToolJournal:
    return ToolJournal(tmp_path, "s1")


def test_prepare_writes_a_record_and_fsyncs_it(tmp_path: Path) -> None:
    """fsync, not flush: the reader is a different process on the other side of a kill.

    A record sitting in a buffered stream is destroyed by SIGKILL and by
    TerminateProcess alike, and the whole point of this file is that it is not.
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1,
        tool_call_id="tu-1",
        tool_name="Write",
        tool_input={"file_path": "a.txt", "content": "hello"},
        workload={"a.txt": "write"},
    )
    assert op
    lines = journal_path(tmp_path, "s1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] == PREPARED
    assert row["tool_call_id"] == "tu-1"
    assert row["tool_name"] == "Write"
    assert row["operation_id"] == op
    assert row["turn_id"] == 1


def test_prepare_digests_the_world_before_the_tool_can_change_it(tmp_path: Path) -> None:
    """`pre_state` is captured before execution, which is the only time it exists."""
    target = tmp_path / "a.txt"
    target.write_text("before", encoding="utf-8")
    journal = _journal(tmp_path)
    journal.prepare(
        turn_id=1,
        tool_call_id="tu-1",
        tool_name="Write",
        tool_input={"file_path": str(target), "content": "after"},
        workload={str(target): "write"},
    )
    assert journal.records()[0].pre_state == {str(target): sha256_bytes(b"before")}


def test_a_path_that_does_not_exist_yet_digests_to_missing(tmp_path: Path) -> None:
    """A `Write` creating a file has no pre-image, and that is a fact, not an error."""
    target = tmp_path / "new.txt"
    journal = _journal(tmp_path)
    journal.prepare(
        turn_id=1,
        tool_call_id="tu-1",
        tool_name="Write",
        tool_input={"file_path": str(target), "content": "x"},
        workload={str(target): "write"},
    )
    assert journal.records()[0].pre_state[str(target)] == "missing"


def test_commit_appends_rather_than_rewriting(tmp_path: Path) -> None:
    """Two records per operation, never one mutated in place.

    An append-only log is what makes a half-written tail a recoverable fact
    instead of a corrupted file: the reader drops the torn line and still has
    every earlier record.
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1,
        tool_call_id="tu-1",
        tool_name="Read",
        tool_input={"file_path": "a"},
        workload={},
    )
    journal.commit(op, outcome="ok", post_state={})
    lines = journal_path(tmp_path, "s1").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["status"] for line in lines] == [PREPARED, COMMITTED]


def test_pending_returns_only_operations_that_never_committed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    done = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Read", tool_input={}, workload={}
    )
    journal.commit(done, outcome="ok", post_state={})
    orphan = journal.prepare(
        turn_id=1,
        tool_call_id="tu-2",
        tool_name="Bash",
        tool_input={"command": "echo x"},
        workload={},
    )

    assert [pending.record.operation_id for pending in journal.pending()] == [orphan]
    assert journal.pending()[0].started is False


def test_a_resolved_operation_is_no_longer_pending(tmp_path: Path) -> None:
    """Reconciling twice must not append a second verdict.

    A second resume after a first one is a normal thing to do, and re-deciding an
    already-decided operation would let the later, less informed reading of the
    world override the earlier one.
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Write", tool_input={}, workload={}
    )
    journal.resolve(op, status=RECONCILED, outcome=ReconcileOutcome.APPLIED)
    assert journal.pending() == []
    assert journal.records()[-1].status == RECONCILED


def test_the_pending_record_keeps_the_call_id_the_start_carried(tmp_path: Path) -> None:
    """The verdict record carries no call id; the START is what the transcript needs.

    `pending()` must therefore return the start record, not the latest one --
    returning the verdict would lose the `tool_call_id` that pairs the recovered
    result with the `tool_use` already on the transcript.
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=4, tool_call_id="tu-7", tool_name="Edit", tool_input={"a": 1}, workload={}
    )
    journal.commit(op, outcome="ok", post_state={})
    orphan = journal.prepare(
        turn_id=4, tool_call_id="tu-8", tool_name="Edit", tool_input={"b": 2}, workload={}
    )

    record = journal.pending()[0].record
    assert record.operation_id == orphan
    assert record.tool_call_id == "tu-8"
    assert record.turn_id == 4
    assert record.tool_input == {"b": 2}


def test_the_marker_does_not_end_the_operation(tmp_path: Path) -> None:
    """`EXECUTING` is a third state, not a fourth terminal one.

    An operation that reported its point of no return and then stopped is still
    unfinished -- it is exactly the case reconciliation exists for. Folding it
    into the terminal set would quietly resolve every interrupted `Bash` call as
    "nothing to see here".
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1,
        tool_call_id="tu-1",
        tool_name="Bash",
        tool_input={"command": "echo x"},
        workload={},
    )
    journal.mark_executing(op)

    pending = journal.pending()
    assert [p.record.operation_id for p in pending] == [op]
    assert pending[0].started is True
    # The START is still what comes back, because the marker record carries no
    # call id and the transcript needs one.
    assert pending[0].record.tool_call_id == "tu-1"
    assert journal.records()[-1].status == EXECUTING


def test_started_separates_entered_from_never_entered(tmp_path: Path) -> None:
    """The two facts a resume has to tell apart, and could not before.

    Same journal shape, same missing COMMITTED, opposite meanings: one call
    reached the line past which its effect was possible and one never did. The
    marker is the whole difference, and it is the difference between an
    interruption that can be retried and one that cannot.
    """
    journal = _journal(tmp_path)
    never_entered = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Bash",
        tool_input={"command": "echo a"}, workload={},
    )
    entered = journal.prepare(
        turn_id=1, tool_call_id="tu-2", tool_name="Bash",
        tool_input={"command": "echo b"}, workload={},
    )
    journal.mark_executing(entered)

    by_id = {p.record.operation_id: p.started for p in journal.pending()}
    assert by_id == {never_entered: False, entered: True}


def test_a_marker_does_not_survive_a_commit(tmp_path: Path) -> None:
    """A call that ran to completion has an answer, so nothing is left to ask."""
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Bash",
        tool_input={"command": "echo x"}, workload={},
    )
    journal.mark_executing(op)
    journal.commit(op, outcome="ok", post_state={})
    assert journal.pending() == []


def test_the_marker_record_contributes_no_workspace_digests(tmp_path: Path) -> None:
    """It carries an id and a status and nothing else, by construction.

    Repeating the start's `access` into it would be a second copy of a fact a
    later edit could disagree with -- and the start is what owns that fact.
    """
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Edit",
        tool_input={}, workload={"src/a.py": "write"},
    )
    journal.mark_executing(op)
    journal.commit(op, outcome="ok", post_state={"src/a.py": "h"})
    assert workspace_from_records(journal.records())[1] == {"src/a.py": "h"}


def test_resolve_refuses_a_non_terminal_status(tmp_path: Path) -> None:
    """The verdict is the last word on an operation, so it must be a final one."""
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Write", tool_input={}, workload={}
    )
    import pytest

    with pytest.raises(ValueError, match="terminal status"):
        journal.resolve(op, status=INDETERMINATE + "-ish", outcome=ReconcileOutcome.UNKNOWN)


def test_a_torn_tail_is_dropped_and_the_earlier_records_survive(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Read", tool_input={}, workload={}
    )
    path = journal_path(tmp_path, "s1")
    path.write_bytes(path.read_bytes() + b'{"status": "COMMI')
    assert [record.operation_id for record in journal.records()] == [op]


def test_an_unknown_session_has_an_empty_journal_rather_than_raising(tmp_path: Path) -> None:
    """A session with no journal is a session from before this existed."""
    assert _journal(tmp_path).records() == []
    assert _journal(tmp_path).pending() == []


def test_the_session_header_records_root_and_git_head(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    assert journal.session_header() is None
    journal.write_session_header(workspace_root=str(tmp_path), git_head="abc123")
    assert journal.session_header() == {
        "workspace_root": str(tmp_path),
        "git_head": "abc123",
    }


def test_a_corrupt_header_reads_as_absent_rather_than_raising(tmp_path: Path) -> None:
    from longline.session.tool_journal import header_path

    journal = _journal(tmp_path)
    journal.write_session_header(workspace_root=str(tmp_path), git_head=None)
    header_path(tmp_path, "s1").write_text("{not json", encoding="utf-8")
    assert journal.session_header() is None


def test_workspace_sets_come_from_the_declared_access_modes(tmp_path: Path) -> None:
    """read/write sets, each path carrying the digest the session last recorded.

    "Last recorded", not "first": a file edited twice is identified by where the
    session left it, which is the only revision that can be compared against a
    later reading of the same file.
    """
    journal = _journal(tmp_path)
    read = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Read",
        tool_input={}, workload={"src/a.py": "read"},
    )
    journal.commit(read, outcome="ok", post_state={"src/a.py": "h-read"})
    write = journal.prepare(
        turn_id=1, tool_call_id="tu-2", tool_name="Edit",
        tool_input={}, workload={"src/a.py": "write"},
    )
    journal.commit(write, outcome="ok", post_state={"src/a.py": "h-written"})

    read_set, write_set = workspace_from_records(journal.records())
    assert read_set == {"src/a.py": "h-read"}
    assert write_set == {"src/a.py": "h-written"}


def test_an_uncommitted_operation_contributes_no_post_digest(tmp_path: Path) -> None:
    """It has no post digest -- that is precisely what the interruption destroyed.

    Recording the PRE state as if it were a post state would make the identity
    check compare the world against a revision the session never claimed, and
    every interrupted write would then read as drift.
    """
    journal = _journal(tmp_path)
    journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Edit",
        tool_input={}, workload={"src/a.py": "write"},
    )
    read_set, write_set = workspace_from_records(journal.records())
    assert write_set == {}
    assert read_set == {}


def test_a_file_written_twice_keeps_the_later_digest(tmp_path: Path) -> None:
    """The revision the session LEFT the file at, not the one it first saw.

    Taking the earlier digest would make a file the session itself edited twice
    read as drift on resume -- a false reject caused by the session's own work.
    """
    journal = _journal(tmp_path)
    for digest in ("h-first", "h-second"):
        op = journal.prepare(
            turn_id=1, tool_call_id=f"tu-{digest}", tool_name="Edit",
            tool_input={}, workload={"src/a.py": "write"},
        )
        journal.commit(op, outcome="ok", post_state={"src/a.py": digest})

    assert workspace_from_records(journal.records())[1] == {"src/a.py": "h-second"}


def test_a_resolved_operation_contributes_no_post_digest_either(tmp_path: Path) -> None:
    """Neither verdict can supply the digest the interruption destroyed.

    A verdict record carries no `post_state`, and it must not: the digest would
    have to be taken at resume time, from the very world the identity check is
    about to compare against -- so it would match by construction and say
    nothing. The consequence is real and is recorded as a limit: a file changed
    by an operation whose tool declares no workload stays invisible to the
    identity check even after it is reconciled.
    """
    for status, outcome in (
        (RECONCILED, ReconcileOutcome.APPLIED),
        (ABORTED, ReconcileOutcome.NOT_APPLIED),
        (INDETERMINATE, ReconcileOutcome.UNKNOWN),
    ):
        journal = _journal(tmp_path / status)
        op = journal.prepare(
            turn_id=1, tool_call_id="tu-1", tool_name="Edit",
            tool_input={}, workload={"src/a.py": "write"},
        )
        journal.resolve(op, status=status, outcome=outcome)
        assert workspace_from_records(journal.records())[1] == {}, status
