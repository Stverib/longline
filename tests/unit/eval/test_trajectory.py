"""Unit tests for longline/eval/trajectory.py — event-stream extraction."""

from __future__ import annotations

from typing import Any

from longline.core.events import ErrorEvent, TextDelta, ToolUseStart, TurnComplete
from longline.eval.trajectory import Trajectory, extract_trajectory
from longline.models.messages import Usage


async def _stream(*events: Any) -> Any:
    for e in events:
        yield e


async def _extract(*events: Any) -> Trajectory:
    return await extract_trajectory(_stream(*events))


async def test_extracts_tool_calls_tokens_turns() -> None:
    traj = await _extract(
        ToolUseStart(tool_name="Read", tool_id="t1", input={"file_path": "/tmp/a.py"}),
        TurnComplete(stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=20)),
        ToolUseStart(tool_name="Write", tool_id="t2", input={"file_path": "/tmp/a.py", "content": "x"}),
        TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=30, output_tokens=40)),
    )
    assert traj.tool_calls == [
        ("Read", {"file_path": "/tmp/a.py"}),
        ("Write", {"file_path": "/tmp/a.py", "content": "x"}),
    ]
    assert traj.turns == 2
    assert traj.input_tokens == 40
    assert traj.output_tokens == 60
    assert traj.errors == []
    assert traj.text == ""


async def test_accumulates_text_and_errors() -> None:
    traj = await _extract(
        TextDelta(text="hel"),
        TextDelta(text="lo"),
        ErrorEvent(message="boom", is_recoverable=False),
    )
    assert traj.text == "hello"
    assert traj.errors == ["boom"]


def test_trajectory_dataclass_defaults() -> None:
    t = Trajectory()
    assert t.tool_calls == []
    assert t.turns == 0
    assert t.input_tokens == 0
    assert t.output_tokens == 0
    assert t.errors == []
    assert t.text == ""
