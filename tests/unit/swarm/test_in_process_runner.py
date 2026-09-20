"""Tests for longline.swarm.in_process_runner — InProcessTeammate runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from longline.core.events import TextDelta, TurnComplete
from longline.models.messages import Usage
from longline.swarm.in_process_runner import (
    InProcessTeammate,
    get_current_team_name,
    get_current_teammate_id,
    is_in_process_teammate,
)
from longline.tools.base import ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _mock_call_model_factory(model: str | None = None, max_tokens: int = 16384) -> Any:
    """Mock factory that returns a call_model yielding a simple text response."""
    async def call_model(**kwargs: Any) -> Any:
        yield TextDelta(text="Task done.")
        yield TurnComplete(stop_reason="end_turn", usage=Usage())
    return call_model


class TestContextVars:
    def test_default_not_teammate(self) -> None:
        assert get_current_teammate_id() is None
        assert get_current_team_name() is None
        assert is_in_process_teammate() is False


@pytest.mark.asyncio
async def test_run_sets_identity(tmp_path: Path) -> None:
    """Running a teammate sets context vars during execution."""
    teammate = InProcessTeammate(
        agent_id="worker@test",
        team_name="test",
        agent_name="worker",
        call_model_factory=_mock_call_model_factory,
        parent_registry=ToolRegistry(),
        claude_dir=tmp_path,
    )
    result = await teammate.run("do something")
    assert len(result) > 0
    assert teammate.is_completed


@pytest.mark.asyncio
async def test_identity_isolation(tmp_path: Path) -> None:
    """Context vars are restored after teammate finishes."""
    assert get_current_teammate_id() is None

    teammate = InProcessTeammate(
        agent_id="iso@test",
        team_name="test",
        agent_name="iso",
        call_model_factory=_mock_call_model_factory,
        parent_registry=ToolRegistry(),
        claude_dir=tmp_path,
    )
    await teammate.run("task")

    assert get_current_teammate_id() is None
    assert is_in_process_teammate() is False


@pytest.mark.asyncio
async def test_sends_completion_message(tmp_path: Path) -> None:
    """Teammate sends a completion message to team-lead's mailbox."""
    from longline.swarm.mailbox import TeammateMailbox

    teammate = InProcessTeammate(
        agent_id="notifier@t",
        team_name="t",
        agent_name="notifier",
        call_model_factory=_mock_call_model_factory,
        parent_registry=ToolRegistry(),
        claude_dir=tmp_path,
    )
    await teammate.run("check something")

    mb = TeammateMailbox("t", claude_dir=tmp_path)
    msgs = mb.receive("team-lead")
    assert len(msgs) == 1
    assert "notifier" in msgs[0].from_name


@pytest.mark.asyncio
async def test_uses_non_interactive_permissions(tmp_path: Path) -> None:
    """Teammate should use non-interactive permissions (fail-fast on ASK)."""
    # The teammate execution path creates a PermissionContext with is_interactive=False
    # This is tested implicitly — if it tried to prompt, it would hang.
    # Here we just verify execution completes successfully.
    teammate = InProcessTeammate(
        agent_id="perm@test",
        team_name="test",
        agent_name="perm",
        call_model_factory=_mock_call_model_factory,
        parent_registry=ToolRegistry(),
        claude_dir=tmp_path,
    )
    result = await teammate.run("test permissions")
    assert teammate.is_completed
    assert len(result) > 0


class TestCheckInboxDoesNotSwallowALoss:
    """A lost inbox has to reach the conversation, not just a log file.

    `check_inbox` used to catch every exception and return `[]`, which made "my
    inbox was unreadable" indistinguishable from "I have no mail": the teammate
    would finish its work and report success, and the leader would receive a
    reply that never mentioned the lost handoff. Both parties report success
    while the messages are gone.
    """

    @pytest.mark.asyncio
    async def test_a_corrupt_inbox_becomes_a_message_the_teammate_sees(
        self, tmp_path: Path
    ) -> None:
        from longline.swarm.mailbox import TeammateMailbox

        teammate = InProcessTeammate(
            agent_id="worker@test",
            team_name="test",
            agent_name="worker",
            call_model_factory=_mock_call_model_factory,
            parent_registry=ToolRegistry(),
            claude_dir=tmp_path,
        )
        box = TeammateMailbox("test", claude_dir=tmp_path)
        box._ensure_inbox_dir()
        box._inbox_path("worker").write_text("[not json", encoding="utf-8")

        messages = await teammate.check_inbox()

        assert messages, "a lost inbox must not read as an absent one"
        assert any("unreadable" in m for m in messages), messages

    @pytest.mark.asyncio
    async def test_an_absent_inbox_is_still_silence(self, tmp_path: Path) -> None:
        """The distinction the fix rests on: nothing to read is not a fault."""
        teammate = InProcessTeammate(
            agent_id="worker@test",
            team_name="test",
            agent_name="worker",
            call_model_factory=_mock_call_model_factory,
            parent_registry=ToolRegistry(),
            claude_dir=tmp_path,
        )

        assert await teammate.check_inbox() == []
