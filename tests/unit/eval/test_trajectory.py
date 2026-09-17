"""Unit tests for longline/eval/trajectory.py — event-stream extraction."""

from __future__ import annotations

from typing import Any

import pytest

from longline.core.events import ErrorEvent, TextDelta, ToolResultReady, ToolUseStart, TurnComplete
from longline.eval.trajectory import Trajectory, extract_trajectory
from longline.models.messages import Usage


async def _stream(*events: Any) -> Any:
    for e in events:
        yield e


async def _extract(*events: Any) -> Trajectory:
    return await extract_trajectory(_stream(*events))


async def _extract_with_clock(clock: Any, *events: Any) -> Trajectory:
    return await extract_trajectory(_stream(*events), clock=clock)


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
    assert t.tool_executions == []
    assert t.event_timestamps == {}
    assert t.num_successful_tool_calls == 0
    # `None` means "the transport did not tell us", which is a different fact
    # from "the transport told us it was the model we asked for". See
    # `served_models` -- the two are never collapsed.
    assert t.served_models == []


# --- the served model: what the transport actually ran (Task 9) ---


async def test_served_model_is_recorded_from_the_stream() -> None:
    """The gateway ignores the requested `model` and serves its own.

    A run that records only what it asked for is describing a model that never
    executed. This is the seam that lets the artifact say what actually ran.
    """
    traj = await _extract(
        TurnComplete(stop_reason="end_turn", usage=Usage(), served_model="deepseek-flash"),
    )
    assert traj.served_models == ["deepseek-flash"]


async def test_served_models_dedupes_a_repeated_answer() -> None:
    """Two turns of one conversation answer the same thing; the list is a set."""
    traj = await _extract(
        TurnComplete(stop_reason="tool_use", usage=Usage(), served_model="deepseek-flash"),
        TurnComplete(stop_reason="end_turn", usage=Usage(), served_model="deepseek-flash"),
    )
    assert traj.served_models == ["deepseek-flash"]


async def test_a_mid_conversation_switch_is_visible() -> None:
    """A gateway that fails over between turns must not be silently averaged.

    Keeping BOTH names is the point: a reader has to be able to see that this
    run's numbers came from two different models, because neither a single name
    nor a majority vote would describe it.
    """
    traj = await _extract(
        TurnComplete(stop_reason="tool_use", usage=Usage(), served_model="deepseek-flash"),
        TurnComplete(stop_reason="end_turn", usage=Usage(), served_model="glm-4.6"),
    )
    assert traj.served_models == ["deepseek-flash", "glm-4.6"]


async def test_a_transport_that_says_nothing_yields_no_name() -> None:
    """FAILS ON: inventing the requested model when the stream is silent.

    A default of "deepseek-flash" would make every offline run claim a model it
    never contacted. An empty list is the honest answer, and the mismatch check
    in `runner` keys off exactly this.
    """
    traj = await _extract(
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    )
    assert traj.served_models == []


# --- ToolResultReady capture (Task 1) ---


async def test_captures_tool_results_joined_by_id_not_position() -> None:
    """ToolResultReady has no tool_name and arrives in COMPLETION order.

    Attribution must go through the tool_id -> tool_name map built from
    ToolUseStart events, never by zipping the two lists positionally.
    """
    traj = await _extract(
        ToolUseStart(tool_name="Read", tool_id="t1", input={"file_path": "a"}),
        ToolUseStart(tool_name="Bash", tool_id="t2", input={"command": "ls"}),
        # Bash finished first -> results come back in completion order
        ToolResultReady(tool_id="t2", content="ls out", is_error=False),
        ToolResultReady(tool_id="t1", content="read out", is_error=True),
    )
    assert [e.tool_id for e in traj.tool_executions] == ["t2", "t1"]
    assert [e.tool_name for e in traj.tool_executions] == ["Bash", "Read"]
    assert [e.is_error for e in traj.tool_executions] == [False, True]


async def test_tool_execution_counts() -> None:
    traj = await _extract(
        ToolUseStart(tool_name="Read", tool_id="t1", input={}),
        ToolUseStart(tool_name="Read", tool_id="t2", input={}),
        ToolUseStart(tool_name="Read", tool_id="t3", input={}),
        ToolResultReady(tool_id="t1", content="ok", is_error=False),
        ToolResultReady(tool_id="t2", content="boom", is_error=True),
        ToolResultReady(tool_id="t3", content="ok", is_error=False),
    )
    assert traj.num_tool_calls == 3
    assert traj.num_tool_calls_executed == 3
    assert traj.num_successful_tool_calls == 2


async def test_unexecuted_call_is_not_counted_as_executed() -> None:
    """max_turns abort mid-turn: a ToolUseStart with no ToolResultReady."""
    traj = await _extract(
        ToolUseStart(tool_name="Read", tool_id="t1", input={}),
        ToolUseStart(tool_name="Bash", tool_id="t2", input={}),
        ToolResultReady(tool_id="t1", content="ok", is_error=False),
    )
    assert traj.num_tool_calls == 2
    assert traj.num_tool_calls_executed == 1
    assert traj.num_successful_tool_calls == 1


async def test_unknown_tool_id_is_attributed_as_empty_name() -> None:
    traj = await _extract(ToolResultReady(tool_id="ghost", content="x", is_error=False))
    assert len(traj.tool_executions) == 1
    assert traj.tool_executions[0].tool_name == ""
    assert traj.tool_executions[0].tool_id == "ghost"


async def test_event_timestamps_recorded_via_injected_clock() -> None:
    # One clock sample per event; request_start is sampled once up front.
    ticks = iter([0, 100, 200, 300])

    def clock() -> int:
        return next(ticks)

    traj = await _extract_with_clock(
        clock,
        ToolUseStart(tool_name="Read", tool_id="t1", input={}),
        TurnComplete(stop_reason="tool_use", usage=Usage()),
        ToolResultReady(tool_id="t1", content="ok", is_error=False),
    )
    ts = traj.event_timestamps
    assert set(ts) == {"request_start", "tool_use_start", "turn_1_complete", "tool_result_ready"}
    assert ts["request_start"] == 0
    assert ts["tool_use_start"] == 100
    assert ts["turn_1_complete"] == 200
    assert ts["tool_result_ready"] == 300


async def test_per_tool_latency_from_injected_clock() -> None:
    # 0 = request_start, 1_000 = ToolUseStart, 5_000 = ToolResultReady
    ticks = iter([0, 1_000, 5_000])

    def clock() -> int:
        return next(ticks)

    traj = await _extract_with_clock(
        clock,
        ToolUseStart(tool_name="Bash", tool_id="t1", input={}),
        ToolResultReady(tool_id="t1", content="ok", is_error=False),
    )
    ex = traj.tool_executions[0]
    assert ex.start_ns == 1_000
    assert ex.end_ns == 5_000
    assert ex.duration_ms == pytest.approx(0.004)


async def test_default_clock_is_monotonic_and_populates_timestamps() -> None:
    traj = await _extract(
        ToolUseStart(tool_name="Read", tool_id="t1", input={}),
        ToolResultReady(tool_id="t1", content="ok", is_error=False),
    )
    assert traj.event_timestamps["request_start"] is not None
    assert traj.event_timestamps["tool_result_ready"] >= traj.event_timestamps["tool_use_start"]
    duration = traj.tool_executions[0].duration_ms
    assert duration is not None and duration >= 0.0
