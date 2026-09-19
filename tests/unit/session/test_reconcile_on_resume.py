"""Turning an orphaned PREPARED into something the model can act on.

`validate_transcript` already repairs an unanswered `tool_use` -- with
`"[Tool result missing due to internal error]"`, which is true for a call that
never ran and FALSE for one whose effect is sitting in the workspace. The
journal is what lets the repair tell the difference, and the difference decides
whether the model retries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from longline.models.content_blocks import ToolResultBlock, ToolUseBlock
from longline.models.messages import AssistantMessage, UserMessage
from longline.session.recovery import TranscriptRepairReport, validate_transcript
from longline.session.tool_journal import (
    ABORTED,
    INDETERMINATE,
    RECONCILE_ABORTED_PREFIX,
    RECONCILE_APPLIED_PREFIX,
    RECONCILE_UNKNOWN_PREFIX,
    RECONCILED,
    ToolJournal,
    reconcile_pending,
)
from longline.tools.base import (
    ReconcileOutcome,
    Tool,
    ToolRegistry,
    ToolResult,
    ToolSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Writes(Tool):
    def __init__(self, outcome: ReconcileOutcome) -> None:
        self._outcome = outcome
        self.asked: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "Writes"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Writes", description="", input_schema={})

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")

    def reconcile(self, tool_input: dict[str, Any]) -> ReconcileOutcome:
        self.asked.append(tool_input)
        return self._outcome


def _registry(tool: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _orphan(tmp_path: Path) -> ToolJournal:
    journal = ToolJournal(tmp_path, "s1")
    journal.prepare(
        turn_id=1,
        tool_call_id="tu-2",
        tool_name="Writes",
        tool_input={"file_path": "a.txt"},
        workload={"a.txt": "write"},
    )
    return journal


@pytest.mark.parametrize(
    ("outcome", "status", "prefix", "is_error"),
    [
        (ReconcileOutcome.APPLIED, RECONCILED, RECONCILE_APPLIED_PREFIX, False),
        (ReconcileOutcome.NOT_APPLIED, ABORTED, RECONCILE_ABORTED_PREFIX, True),
        (ReconcileOutcome.UNKNOWN, INDETERMINATE, RECONCILE_UNKNOWN_PREFIX, True),
    ],
)
def test_each_verdict_becomes_a_distinguishable_result(
    tmp_path: Path,
    outcome: ReconcileOutcome,
    status: str,
    prefix: str,
    is_error: bool,
) -> None:
    """The three verdicts must not collapse into one error string.

    The model's correct next move differs for each: retry an aborted call, do not
    retry an unknown one, and do not retry an applied one. A single "something
    went wrong" text forces it to guess, and guessing wrong in the UNKNOWN case is
    the duplicate this whole mechanism exists to prevent.
    """
    tool = _Writes(outcome)
    journal = _orphan(tmp_path)
    resolved = reconcile_pending(journal, _registry(tool))

    assert len(resolved) == 1
    assert resolved[0].status == status
    assert resolved[0].tool_call_id == "tu-2"
    assert resolved[0].result_text.startswith(prefix)
    assert resolved[0].is_error is is_error
    assert tool.asked == [{"file_path": "a.txt"}]
    assert journal.pending() == []


def test_the_verdict_is_recorded_in_the_journal(tmp_path: Path) -> None:
    """A second resume must not re-decide an operation the first one decided."""
    journal = _orphan(tmp_path)
    reconcile_pending(journal, _registry(_Writes(ReconcileOutcome.APPLIED)))
    assert journal.records()[-1].status == RECONCILED
    assert journal.records()[-1].outcome == "applied"


def test_an_unknown_tool_is_indeterminate_rather_than_an_error(tmp_path: Path) -> None:
    """A tool removed from the registry is not evidence that its call failed."""
    journal = _orphan(tmp_path)
    resolved = reconcile_pending(journal, ToolRegistry())
    assert resolved[0].status == INDETERMINATE
    assert resolved[0].result_text.startswith(RECONCILE_UNKNOWN_PREFIX)


def test_a_tool_whose_reconcile_raises_is_indeterminate(tmp_path: Path) -> None:
    """A broken reconciler must not be able to authorise a retry it did not earn."""

    class _Boom(_Writes):
        def reconcile(self, tool_input: dict[str, Any]) -> ReconcileOutcome:
            raise RuntimeError("boom")

    journal = _orphan(tmp_path)
    assert reconcile_pending(journal, _registry(_Boom(ReconcileOutcome.APPLIED)))[0].status == (
        INDETERMINATE
    )


def test_a_committed_operation_is_not_reconciled(tmp_path: Path) -> None:
    journal = ToolJournal(tmp_path, "s1")
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Writes", tool_input={}, workload={}
    )
    journal.commit(op, outcome="ok", post_state={})
    assert reconcile_pending(journal, _registry(_Writes(ReconcileOutcome.APPLIED))) == []


def test_overrides_replace_the_generic_orphan_repair() -> None:
    """The transcript repair, pointed at the reconciled truth.

    Without this the model is told the call failed when it did not, which is the
    failure mode the journal exists to remove: not a missing answer, a WRONG one.
    """
    messages = [
        UserMessage(content="go"),
        AssistantMessage(content=[ToolUseBlock(id="tu-2", name="Writes", input={})]),
    ]
    report = TranscriptRepairReport()

    repaired = validate_transcript(
        messages,
        report=report,
        result_overrides={"tu-2": (f"{RECONCILE_APPLIED_PREFIX}: Write a.txt", False)},
    )

    block = repaired[-1].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.content.startswith(RECONCILE_APPLIED_PREFIX)
    assert block.is_error is False
    assert report.repaired


def test_an_override_reaches_a_mid_transcript_orphan_too() -> None:
    """The second repair path takes the same override.

    There are two synthesis sites in `validate_transcript` -- the truncated tail
    and a mid-transcript gap -- and an override honoured by only one of them
    would give the same operation two different answers depending on where the
    crash landed.
    """
    messages = [
        AssistantMessage(content=[ToolUseBlock(id="tu-2", name="Writes", input={})]),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="tu-9", content="other", is_error=False)
            ]
        ),
    ]

    repaired = validate_transcript(
        messages, result_overrides={"tu-2": (RECONCILE_ABORTED_PREFIX, True)}
    )

    patched = repaired[1].content[0]
    assert isinstance(patched, ToolResultBlock)
    assert patched.tool_use_id == "tu-2"
    assert patched.content == RECONCILE_ABORTED_PREFIX


def test_an_override_for_an_id_the_transcript_does_not_carry_is_ignored() -> None:
    """A journal from another session must not inject a tool_result."""
    messages = [UserMessage(content="go")]
    repaired = validate_transcript(messages, result_overrides={"tu-9": ("x", True)})
    assert repaired == messages


def test_no_overrides_is_the_old_behaviour() -> None:
    messages = [
        UserMessage(content="go"),
        AssistantMessage(content=[ToolUseBlock(id="tu-1", name="X", input={})]),
    ]
    repaired = validate_transcript(messages)
    block = repaired[-1].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.content == "[Tool result missing due to internal error]"
    assert block.is_error is True
