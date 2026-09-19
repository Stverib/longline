"""Failpoint gates: the sentinel has to outlive the process that wrote it."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from longline.eval.failpoints import (
    AFTER_TOOL,
    BEFORE_MODEL,
    BEFORE_TOOL,
    GATED_FAILPOINTS,
    PARENT_FAILPOINTS,
    FailpointGate,
    GatedModel,
    GatedTool,
    read_sentinel,
    terminate_and_reap,
    truncate_last_line,
    wait_for_sentinel,
    write_sentinel,
)
from longline.tools.base import Tool, ToolResult, ToolSchema


class _EchoTool(Tool):
    """A tool whose only job is to be gated and to record that it ran."""

    def __init__(self, name: str = "Echo", *, delay_s: float = 0.0) -> None:
        self._name = name
        self._delay = delay_s
        self.executions = 0

    def get_name(self) -> str:
        return self._name

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="", input_schema={})

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return True

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self.executions += 1
        if self._delay:
            time.sleep(self._delay)
        return ToolResult(content="echo: executed", is_error=False)


def test_write_sentinel_round_trips_through_disk(tmp_path: Path) -> None:
    write_sentinel(tmp_path, failpoint=BEFORE_MODEL, detail={"model_call_index": 1})
    payload = read_sentinel(tmp_path)
    assert payload is not None
    assert payload["failpoint"] == BEFORE_MODEL
    assert payload["detail"] == {"model_call_index": 1}
    assert payload["pid"] > 0


def test_read_sentinel_reports_none_when_absent(tmp_path: Path) -> None:
    assert read_sentinel(tmp_path) is None


def test_sentinel_survives_the_writer_being_killed(tmp_path: Path) -> None:
    """The whole design rests on this: fsync, not flush.

    A child writes a sentinel and parks. The parent terminates it. If the
    sentinel were only flushed to the OS buffer, this test would be the one
    that fails intermittently -- and an intermittent failpoint is not a
    failpoint.
    """
    script = (
        "import sys, time;"
        "from pathlib import Path;"
        "from longline.eval.failpoints import write_sentinel, block_forever;"
        f"write_sentinel(Path({str(tmp_path)!r}), failpoint='before_model');"
        "block_forever()"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=str(Path.cwd()))
    try:
        payload = wait_for_sentinel(tmp_path, timeout_s=60.0)
        assert payload is not None, "child never wrote the sentinel"
        assert terminate_and_reap(proc) is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert read_sentinel(tmp_path) is not None


def test_terminate_and_reap_reports_true_for_an_already_dead_child() -> None:
    """The boolean answers "is it gone", not "did we kill it".

    This is `_kill_child`'s contract, which returned `proc.returncode is not
    None` before this function existed. A child that exited on its own is just
    as gone, and reporting False here would silently change `SessionResumeRate`
    the moment a kill raced a child that had already stopped.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert terminate_and_reap(proc) is True


def test_terminate_and_reap_kills_a_live_child() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        assert terminate_and_reap(proc) is True
    finally:
        if proc.poll() is None:  # pragma: no cover - terminate_and_reap failed
            proc.kill()
            proc.wait(timeout=10)


def test_terminate_and_reap_accepts_an_explicit_signal() -> None:
    """`_kill_child` has an optional-signal branch, and the extraction has to
    carry it or the two callers stop being equivalent."""
    import signal

    sig = getattr(signal, "SIGTERM", None)
    if sig is None:  # pragma: no cover - Windows has no SIGTERM in the POSIX sense
        return
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        assert terminate_and_reap(proc, signal_num=sig) is True
    finally:
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=10)


def test_wait_for_sentinel_times_out(tmp_path: Path) -> None:
    started = time.monotonic()
    assert wait_for_sentinel(tmp_path, timeout_s=0.3) is None
    assert time.monotonic() - started >= 0.3


def test_gate_triggers_only_at_the_configured_call_index(tmp_path: Path) -> None:
    gate = FailpointGate(tmp_path, BEFORE_MODEL, at_call_index=2)
    assert gate.triggers_model(1) is False
    assert gate.triggers_model(2) is True
    assert gate.triggers_model(3) is False
    assert gate.triggers_tool("Bash") is False


def test_gate_triggers_tool_by_name(tmp_path: Path) -> None:
    gate = FailpointGate(tmp_path, AFTER_TOOL, at_tool_name="Bash")
    assert gate.triggers_tool("Bash") is True
    assert gate.triggers_tool("Edit") is False


def test_a_tool_gate_with_no_tool_name_never_fires(tmp_path: Path) -> None:
    """A tool-named failpoint with an empty name is a case that can never be
    injected, so it must not silently fire on every tool either."""
    gate = FailpointGate(tmp_path, AFTER_TOOL, at_tool_name="")
    assert gate.triggers_tool("Bash") is False
    assert gate.triggers_tool("") is False


def test_disarmed_gate_never_triggers(tmp_path: Path) -> None:
    """The resumed leg installs the SAME wrappers. An armed gate there would
    block the resumed process forever and every case would time out."""
    gate = FailpointGate(tmp_path, BEFORE_MODEL, at_call_index=1, armed=False)
    assert gate.triggers_model(1) is False
    tool_gate = FailpointGate(tmp_path, AFTER_TOOL, at_tool_name="Bash", armed=False)
    assert tool_gate.triggers_tool("Bash") is False


def test_gate_stop_writes_the_sentinel_then_blocks(tmp_path: Path) -> None:
    blocked: list[int] = []
    gate = FailpointGate(tmp_path, BEFORE_MODEL, at_call_index=1, block=lambda: blocked.append(1))
    gate.stop(detail={"model_call_index": 1})
    assert blocked == [1]
    assert gate.reached == 1
    assert read_sentinel(tmp_path) is not None


def test_gated_model_counts_only_while_armed(tmp_path: Path) -> None:
    """The armed-only counter is what makes `after_checkpoint` mean "the first
    model call of the second instruction" without a hand-kept index.

    The generator is drained on every call: `GatedModel.__call__` builds an
    async generator, so nothing runs until it is iterated -- which is exactly
    how `query_loop` uses it.
    """
    served: list[int] = []

    async def _no_events() -> Any:
        for event in ():
            yield event

    class _Inner:
        def __call__(self, **kwargs: Any) -> Any:
            served.append(1)
            return _no_events()

    def _drain(agen: Any) -> None:
        async def _run() -> None:
            async for _ in agen:
                pass

        asyncio.run(_run())

    # `block` must be injected: the DEFAULT parks forever, which is exactly what
    # production wants and exactly what a test must not do. Forgetting it here
    # hangs the suite rather than failing it -- measured, not theorised.
    parked: list[int] = []
    gate = FailpointGate(
        tmp_path,
        BEFORE_MODEL,
        at_call_index=1,
        armed=False,
        block=lambda: parked.append(1),
    )
    model = GatedModel(inner=_Inner(), gate=gate)
    for _ in range(5):
        _drain(model(messages=[]))
    assert model.calls == 0, "a disarmed gate must not count calls"
    assert len(served) == 5, "a disarmed gate must still serve the model"
    assert parked == [], "a disarmed gate must not stop"

    gate.armed = True
    _drain(model(messages=[]))
    assert model.calls == 1, "arming restarts the count at the next call"
    assert parked == [1], "the gate fires on the first ARMED call"

    # And it fires on the first armed call, not the first ever.
    blocked: list[int] = []
    gate2 = FailpointGate(
        tmp_path, BEFORE_MODEL, at_call_index=1, armed=False, block=lambda: blocked.append(1)
    )
    model2 = GatedModel(inner=_Inner(), gate=gate2)
    _drain(model2(messages=[]))
    assert blocked == []
    gate2.armed = True
    _drain(model2(messages=[]))
    assert blocked == [1]


@pytest.mark.asyncio
async def test_gated_tool_records_before_it_parks(tmp_path: Path) -> None:
    """Ordering is load-bearing: journal first, sentinel second.

    If the sentinel were written first, the parent could kill the child in the
    window before the journal entry reached disk, and the run would report a
    clean resume for a side effect that really happened.
    """
    order: list[str] = []
    inner = _EchoTool(delay_s=0.01)
    gate = FailpointGate(
        tmp_path, AFTER_TOOL, at_tool_name="Echo", block=lambda: order.append("block")
    )

    class _Recorder:
        def record(self, **kwargs: Any) -> None:
            order.append("journal")

    gated = GatedTool(
        inner=inner,
        gate=gate,
        journal=_Recorder(),
        artifact_paths=("value.txt",),
        artifact_root=str(tmp_path),
    )
    await gated.execute({"x": 1})
    assert order == ["journal", "block"]
    assert inner.executions == 1


@pytest.mark.asyncio
async def test_before_tool_gate_parks_without_executing_the_tool(tmp_path: Path) -> None:
    inner = _EchoTool()
    gate = FailpointGate(tmp_path, BEFORE_TOOL, at_tool_name="Echo", block=lambda: None)
    gated = GatedTool(inner=inner, gate=gate)
    await gated.execute({})
    assert inner.executions == 0, "the before_tool gate must stop before the tool runs"
    assert read_sentinel(tmp_path) is not None


@pytest.mark.asyncio
async def test_gated_tool_snapshots_artifacts_around_the_execution(tmp_path: Path) -> None:
    """The journal entry must carry the state BEFORE and AFTER, because the
    duplicate test compares a replay's post-state against the first run's."""
    target = tmp_path / "artifact.txt"
    target.write_text("A", encoding="utf-8")
    recorded: list[dict[str, Any]] = []

    class _WritingTool(Tool):
        def get_name(self) -> str:
            return "Writer"

        def get_schema(self) -> ToolSchema:
            return ToolSchema(name="Writer", description="", input_schema={})

        def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
            return True

        async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
            target.write_text("B", encoding="utf-8")
            return ToolResult(content="written")

    class _Journal:
        def record(self, **kwargs: Any) -> None:
            recorded.append(kwargs)

    gated = GatedTool(
        inner=_WritingTool(),
        gate=FailpointGate(tmp_path, AFTER_TOOL, armed=False),
        journal=_Journal(),
        artifact_paths=("artifact.txt",),
        artifact_root=str(tmp_path),
    )
    await gated.execute({})
    assert recorded[0]["pre_state"] != recorded[0]["post_state"]
    assert recorded[0]["pre_state"]["artifact.txt"] != "missing"


def test_gated_tool_forwards_name_schema_and_concurrency() -> None:
    inner = _EchoTool("Custom")
    gated = GatedTool(inner=inner, gate=FailpointGate(Path("."), AFTER_TOOL, at_tool_name="Custom"))
    assert gated.get_name() == "Custom"
    assert gated.get_schema().name == "Custom"
    assert gated.is_concurrency_safe({}) is True


def test_truncate_last_line_cuts_mid_line(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
    removed = truncate_last_line(path)
    assert removed > 0
    text = path.read_text(encoding="utf-8")
    assert text == '{"a": 1}\n{"b": 2}\n{"c"'
    # The surviving prefix must still be parseable line-by-line, which is what
    # load_session relies on.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


@pytest.mark.asyncio
async def test_the_tool_executor_does_not_swallow_a_failpoint(tmp_path: Path) -> None:
    """`FailpointReached` is a BaseException, and this is why.

    `StreamingToolExecutor` converts any `except Exception` raised by a tool
    into an error `ToolResult`, deliberately, so a broken tool does not kill
    the loop. A stop that got converted that way would let the run continue
    past its own failpoint -- the one failure this module cannot tolerate. The
    production path is exercised through the real executor rather than a stub,
    because the stub is what would hide the difference.
    """
    from longline.eval.failpoints import FailpointReached, halt
    from longline.models.content_blocks import ToolUseBlock
    from longline.tools.base import ToolRegistry
    from longline.tools.streaming_executor import StreamingToolExecutor

    registry = ToolRegistry()
    registry.register(_EchoTool())
    executor = StreamingToolExecutor(registry, hooks=None, permission_checker=None)
    gate = FailpointGate(tmp_path, AFTER_TOOL, at_tool_name="Echo", block=halt)
    registry.swap("Echo", GatedTool(inner=_EchoTool(), gate=gate))
    executor.add_tool(ToolUseBlock(id="t1", name="Echo", input={}))
    with pytest.raises(FailpointReached):
        await executor.get_results()
    assert read_sentinel(tmp_path) is not None


def test_failpoint_vocabulary_is_split_into_gated_and_parent() -> None:
    assert set(GATED_FAILPOINTS).isdisjoint(PARENT_FAILPOINTS)
    assert set(GATED_FAILPOINTS) | set(PARENT_FAILPOINTS) == {
        BEFORE_MODEL,
        BEFORE_TOOL,
        AFTER_TOOL,
        "after_checkpoint",
        "truncate_tail",
        "workspace_drift",
    }
