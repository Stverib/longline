"""The executor is where a tool call becomes journalable.

It is the only layer that holds all three of the things a record needs: the
`tool_call_id` (from the `tool_use` block), the tool, and the moment before
execution. A wrapper around the TOOL cannot do this -- `Tool.execute` is not
given an id -- which is why the record is written here rather than in a
decorator.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from longline.models.content_blocks import ToolUseBlock
from longline.session.tool_journal import (
    COMMITTED,
    EXECUTING,
    PREPARED,
    ToolJournal,
    journal_path,
)
from longline.tools.base import (
    Tool,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    mark_irreversible,
)
from longline.tools.streaming_executor import StreamingToolExecutor
from longline.utils.hashing import sha256_bytes

if TYPE_CHECKING:
    from pathlib import Path


class _Writer(Tool):
    def __init__(self, path: Path, *, fail: bool = False) -> None:
        self._path = path
        self._fail = fail

    def get_name(self) -> str:
        return "Write"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Write", description="", input_schema={})

    def workload(self, tool_input: dict[str, Any]) -> dict[str, str]:
        return {str(self._path): "write"}

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self._path.write_text("new", encoding="utf-8")
        if self._fail:
            return ToolResult(content="Error: nope", is_error=True)
        return ToolResult(content="ok")


class _Bashish(Tool):
    """A tool with a real point of no return: it marks, then it acts."""

    def __init__(self, path: Path, *, marks: int = 1) -> None:
        self._path = path
        self._marks = marks

    def get_name(self) -> str:
        return "Bash"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Bash", description="", input_schema={})

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        for _ in range(self._marks):
            mark_irreversible()
        self._path.write_text("ran", encoding="utf-8")
        return ToolResult(content="ok")


def _executor(tool: Tool, journal: ToolJournal | None) -> StreamingToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)
    return StreamingToolExecutor(registry, journal=journal, turn_id=3)


def _statuses(tmp_path: Path) -> list[str]:
    return [record.status for record in ToolJournal(tmp_path, "s1").records()]


def test_a_completed_call_is_prepared_then_committed(tmp_path: Path) -> None:
    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Writer(tmp_path / "a.txt"), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    assert _statuses(tmp_path) == [PREPARED, COMMITTED]


def test_the_prepared_record_carries_the_call_id_and_the_pre_digest(tmp_path: Path) -> None:
    """This record is the only evidence that a call was issued but not answered."""
    journal = ToolJournal(tmp_path, "s1")
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    executor = _executor(_Writer(target), journal)
    executor.add_tool(ToolUseBlock(id="tu-9", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    prepared = journal.records()[0]
    assert prepared.tool_call_id == "tu-9"
    assert prepared.tool_name == "Write"
    assert prepared.turn_id == 3
    assert prepared.pre_state == {str(target): sha256_bytes(b"old")}
    assert prepared.access == {str(target): "write"}


def test_the_committed_record_carries_the_post_digest(tmp_path: Path) -> None:
    journal = ToolJournal(tmp_path, "s1")
    target = tmp_path / "a.txt"
    executor = _executor(_Writer(target), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    committed = journal.records()[1]
    assert committed.status == COMMITTED
    assert committed.outcome == "ok"
    assert committed.post_state == {str(target): sha256_bytes(b"new")}


def test_an_errored_call_still_commits(tmp_path: Path) -> None:
    """A tool that returned an error DID return.

    Leaving it PREPARED would make every failed call look interrupted, and the
    next resume would reconcile a question that already has an answer.
    """
    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Writer(tmp_path / "a.txt", fail=True), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    assert journal.records()[1].outcome == "error"


def test_a_raising_tool_still_commits(tmp_path: Path) -> None:
    """The executor turns an exception into an error result; the journal must agree.

    The transcript will record an error result for this call, and a journal that
    stayed PREPARED would make the two disagree about what happened.
    """

    class _Boom(_Writer):
        async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
            raise RuntimeError("boom")

    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Boom(tmp_path / "a.txt"), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    assert [record.status for record in journal.records()] == [PREPARED, COMMITTED]


def test_a_tool_that_declares_nothing_journals_with_no_paths(tmp_path: Path) -> None:
    """Bash's record is still the evidence that the call was issued.

    It carries no digests, which is exactly why its reconcile can only ever
    answer UNKNOWN.
    """

    class _Opaque(Tool):
        def get_name(self) -> str:
            return "Bash"

        def get_schema(self) -> ToolSchema:
            return ToolSchema(name="Bash", description="", input_schema={})

        async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
            return ToolResult(content="ok")

    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Opaque(), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Bash", input={"command": "echo x"}))

    asyncio.run(executor.get_results())

    prepared = journal.records()[0]
    assert prepared.tool_name == "Bash"
    assert prepared.access == {}
    assert prepared.pre_state == {}


def test_a_marking_tool_writes_the_middle_state(tmp_path: Path) -> None:
    """Three records, and the middle one is the whole point of this change.

    `PREPARED -> EXECUTING -> COMMITTED` is what lets a resume tell "died before
    the effect was possible" from "died with the effect in flight". With only the
    outer two those are one state, and the only safe reading of that state is the
    pessimistic one.
    """
    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Bashish(tmp_path / "side-effect"), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Bash", input={"command": "echo x"}))

    asyncio.run(executor.get_results())

    assert _statuses(tmp_path) == [PREPARED, EXECUTING, COMMITTED]


def test_the_marker_is_written_once_however_often_the_tool_reaches_for_it(
    tmp_path: Path,
) -> None:
    """A retry loop inside `execute` crosses the same line several times.

    The FIRST crossing is the informative one -- after it the operation is in
    flight -- so the rest would be duplicate lines saying nothing new, at one
    fsync each.
    """
    journal = ToolJournal(tmp_path, "s1")
    executor = _executor(_Bashish(tmp_path / "side-effect", marks=3), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Bash", input={"command": "echo x"}))

    asyncio.run(executor.get_results())

    assert _statuses(tmp_path) == [PREPARED, EXECUTING, COMMITTED]


def test_a_marker_that_cannot_be_written_aborts_the_tool(tmp_path: Path) -> None:
    """The barrier, and the reason this write is NOT best-effort like the others.

    PREPARE and COMMIT swallow their failures: a session directory that cannot be
    written must not be the reason a user's edit does not happen. This write is
    the opposite, because swallowing it produces the one state the recovery path
    reads as "safe to retry" -- a PREPARED with no marker -- for a call that DID
    act. Refusing to run is the correct outcome; running an unrecorded command is
    not.
    """

    class _BrokenMarker(ToolJournal):
        def mark_executing(self, operation_id: str) -> None:
            raise OSError("disk full")

    side_effect = tmp_path / "side-effect"
    executor = _executor(_Bashish(side_effect), _BrokenMarker(tmp_path, "s1"))
    executor.add_tool(ToolUseBlock(id="tu-1", name="Bash", input={"command": "echo x"}))

    results = asyncio.run(executor.get_results())

    assert results[0][1].is_error is True
    assert not side_effect.exists(), "the tool acted despite failing to record that it would"


def test_marking_outside_a_journalled_call_does_nothing(tmp_path: Path) -> None:
    """Every caller that cannot resume runs tools that call this.

    Sub-agents, one-shot `--print`, and the ablation cell all execute `BashTool`
    with no journal, so `mark_irreversible()` has to be a no-op there rather than
    an error or a write to somewhere unexpected.
    """
    executor = _executor(_Bashish(tmp_path / "side-effect"), None)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Bash", input={"command": "echo x"}))

    asyncio.run(executor.get_results())

    assert not journal_path(tmp_path, "s1").exists()
    assert (tmp_path / "side-effect").read_text(encoding="utf-8") == "ran"


def test_no_journal_means_no_file(tmp_path: Path) -> None:
    """Every non-resumable caller passes nothing, and must be unaffected."""
    executor = _executor(_Writer(tmp_path / "a.txt"), None)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))
    asyncio.run(executor.get_results())
    assert not journal_path(tmp_path, "s1").exists()


def test_a_journal_that_cannot_be_written_does_not_stop_the_tool(tmp_path: Path) -> None:
    """Durability is a safety net, not a gate.

    A full disk or a read-only session directory must not be the reason a user's
    edit does not happen.
    """

    class _Broken(ToolJournal):
        def prepare(self, **kwargs: Any) -> str:
            raise OSError("disk full")

    target = tmp_path / "a.txt"
    executor = _executor(_Writer(target), _Broken(tmp_path, "s1"))
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    asyncio.run(executor.get_results())

    assert target.read_text(encoding="utf-8") == "new"


def test_a_failing_commit_does_not_stop_the_tool(tmp_path: Path) -> None:
    """The other half of the same rule: the result still reaches the model."""

    class _BrokenCommit(ToolJournal):
        def commit(self, operation_id: str, **kwargs: Any) -> None:
            raise OSError("disk full")

    journal = _BrokenCommit(tmp_path, "s1")
    executor = _executor(_Writer(tmp_path / "a.txt"), journal)
    executor.add_tool(ToolUseBlock(id="tu-1", name="Write", input={"file_path": "a"}))

    results = asyncio.run(executor.get_results())

    assert results[0][1].is_error is False
    assert [record.status for record in journal.records()] == [PREPARED]
