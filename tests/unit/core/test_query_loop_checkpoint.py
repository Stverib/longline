"""Step-level checkpointing: one save per transcript write point.

Production writes one checkpoint per INSTRUCTION (`main.py:809`), so a kill
anywhere inside an instruction leaves nothing. Measured consequence
(Premise P2): the `after_tool` arm's resumed transcript has no record that the
Bash call was issued, and the scripted model re-issues it.

The callback is a callback rather than a `save_session` call because `query_loop`
is a pure async generator that knows nothing about session ids or directories.
It gets told WHEN to save; the caller owns HOW.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from longline.core.events import TextDelta, ToolUseStart, TurnComplete
from longline.core.query_engine import QueryEngine
from longline.core.query_loop import (
    STEP_MODEL_RESPONSE,
    STEP_TOOL_RESULTS,
)
from longline.models.messages import AssistantMessage, Usage, UserMessage
from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema


class _Echo(Tool):
    def get_name(self) -> str:
        return "Echo"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Echo", description="", input_schema={})

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return True

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        return ToolResult(content="echoed")


def _scripted(steps: int):
    """A model callable that issues `steps` tool calls, then answers.

    Mirrors `QueryEngine.make_call_model`'s shape: the engine's attribute is a
    FACTORY returning the callable, so the test's stand-in has to be one too.
    """

    def call(**kwargs: Any):
        async def gen():
            issued = sum(
                1
                for message in kwargs["messages"]
                for block in (message.get("content") or [])
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )
            if issued < steps:
                yield ToolUseStart(tool_name="Echo", tool_id=f"tu-{issued + 1}", input={})
                yield TurnComplete(stop_reason="tool_use", usage=Usage())
            else:
                yield TextDelta(text="done")
                yield TurnComplete(stop_reason="end_turn", usage=Usage())

        return gen()

    return call


def _engine(registry: ToolRegistry, steps: int) -> QueryEngine:
    engine = QueryEngine(client=None, model="m", registry=registry, system_prompt="s")
    engine.make_call_model = lambda model=None, max_tokens=16384: _scripted(steps)
    return engine


def _run(engine: QueryEngine) -> None:
    async def go() -> None:
        async for _ in engine.run_turn(auto_compact=False):
            pass

    asyncio.run(go())


def test_a_checkpoint_is_taken_before_the_tool_body_runs() -> None:
    """The ordering is the whole fix, and it was MEASURED rather than assumed.

    A probe that recorded `len(engine.messages)` at the top of a tool's `execute`
    saw `['UserMessage', 'AssistantMessage']` -- so `query_loop.py`'s
    `messages.append(assistant_msg)` precedes every tool body. That is why a
    checkpoint taken at the model-response write point is enough: the resume path
    never has to reconstruct an assistant message it never saw.
    """
    seen: list[tuple[str, int]] = []
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=1)
    engine.messages.append(UserMessage(content="go"))
    engine.on_step = lambda reason: seen.append((reason, len(engine.messages)))

    _run(engine)

    reasons = [reason for reason, _ in seen]
    assert reasons[:2] == [STEP_MODEL_RESPONSE, STEP_TOOL_RESULTS]
    # Two messages in: the user's and the assistant's, with the tool result not
    # yet appended.
    assert seen[0][1] == 2
    assert seen[1][1] == 3


def test_the_model_response_checkpoint_leaves_the_tool_use_unanswered() -> None:
    """Deliberately, and this is the property `after_tool` recovery rests on.

    An unanswered trailing `tool_use` is exactly what `validate_transcript` knows
    how to repair, and what stops the resumed leg from re-issuing the call. A
    checkpoint placed only after the results came back would leave the transcript
    saying the call was never issued -- which is the defect this whole change
    exists to remove.
    """
    snapshots: list[tuple[str, list[Any]]] = []
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=1)
    engine.messages.append(UserMessage(content="go"))
    engine.on_step = lambda reason: snapshots.append((reason, list(engine.messages)))

    _run(engine)

    model_response = next(snap for reason, snap in snapshots if reason == STEP_MODEL_RESPONSE)
    assert isinstance(model_response[-1], AssistantMessage)
    assert model_response[-1].get_tool_use_blocks()


def test_the_tool_results_checkpoint_is_fully_paired() -> None:
    """The other side of the same coin: once the results are in, they are in.

    A checkpoint taken between the assistant message and its results is the
    recoverable state; one taken after them must not look like it.
    """
    snapshots: list[tuple[str, list[Any]]] = []
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=2)
    engine.messages.append(UserMessage(content="go"))
    engine.on_step = lambda reason: snapshots.append((reason, list(engine.messages)))

    _run(engine)

    for reason, snapshot in snapshots:
        if reason != STEP_TOOL_RESULTS:
            continue
        unanswered = [
            block.id
            for message in snapshot
            if isinstance(message, AssistantMessage)
            for block in message.get_tool_use_blocks()
        ]
        answered = {
            block.tool_use_id
            for message in snapshot
            if isinstance(message, UserMessage) and isinstance(message.content, list)
            for block in message.content
            if hasattr(block, "tool_use_id")
        }
        assert not [i for i in unanswered if i not in answered]


def test_a_raising_on_step_does_not_stop_the_loop() -> None:
    """A checkpoint that cannot be written is logged, not fatal.

    Propagating would turn a full disk into a dead agent -- trading a durability
    guarantee the user did not ask for against one they did.
    """
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=1)

    def boom(reason: str) -> None:
        raise OSError("full disk")

    engine.on_step = boom
    engine.messages.append(UserMessage(content="go"))

    _run(engine)

    assert len(engine.messages) >= 3


def test_no_on_step_is_the_old_behaviour() -> None:
    """Every existing caller passes nothing, and the loop runs to completion.

    Four messages: the instruction, the assistant's tool call, the tool result,
    and the assistant's closing answer.
    """
    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=1)
    engine.messages.append(UserMessage(content="go"))

    _run(engine)

    assert len(engine.messages) == 4


def test_sub_agents_journal_but_do_not_checkpoint() -> None:
    """`submit_messages` runs on the CALLER's transcript, not the engine's.

    Wiring `on_step` there would save the parent's messages while a sub-agent
    mutated a different list -- a checkpoint recording the wrong conversation,
    which is worse than no checkpoint at all.
    """
    engine = QueryEngine(
        client=None, model="m", registry=ToolRegistry(), system_prompt="s"
    )
    engine.make_call_model = lambda model=None, max_tokens=16384: _scripted(steps=0)
    engine.on_step = lambda reason: pytest.fail("a sub-agent turn must not checkpoint")

    async def go() -> None:
        async for _ in engine.submit_messages([UserMessage(content="hi")]):
            pass

    asyncio.run(go())


def test_the_journal_reaches_the_executor_through_the_engine() -> None:
    """`journal` is passed by `run_turn` as well as `submit`.

    The two entry points build the same `query_loop` call, and a journal wired
    into only one of them would silently record nothing for the REPL -- which is
    the only path that can be resumed.
    """
    recorded: list[dict[str, Any]] = []

    class _Spy:
        def prepare(self, **kwargs: Any) -> str:
            recorded.append(kwargs)
            return "op-1"

        def commit(self, operation_id: str, **kwargs: Any) -> None:
            recorded.append({"commit": operation_id})

    registry = ToolRegistry()
    registry.register(_Echo())
    engine = _engine(registry, steps=1)
    engine.tool_journal = _Spy()
    engine.messages.append(UserMessage(content="go"))

    _run(engine)

    assert [row.get("tool_call_id") for row in recorded if "tool_call_id" in row] == ["tu-1"]
    assert {"commit": "op-1"} in recorded
