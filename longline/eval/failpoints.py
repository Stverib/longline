"""Failpoints for the loop-resume suite: stop a live agent loop at a named point.

=== What a failpoint is here ===

Six named points at which a real agent loop can be interrupted. Four are stops
the CHILD performs on itself; two are done to the child's leftovers by the
parent after the kill:

| failpoint          | where the process stops                            |
|--------------------|----------------------------------------------------|
| before_model       | entering model call N, before the call is made     |
| before_tool        | entering a tool call, before the tool executes     |
| after_tool         | a tool returned, before its result is recorded     |
| after_checkpoint   | entering model call N of a later instruction       |
| truncate_tail      | (parent) the session JSONL tail is cut mid-line    |
| workspace_drift    | (parent) the fixture files are mutated             |

`before_model` and `after_checkpoint` are the SAME code point (model-call
entry). They are separate names because what they assert about the on-disk
checkpoint differs: `before_model` stops inside the first instruction, where
only the turn-0 checkpoint exists, while `after_checkpoint` stops inside a
SECOND instruction, where the first instruction's completed turn is on disk.

=== Why a gate rather than an exception ===

An exception unwinds the process and gives every `finally` a chance to run.
What this suite measures is a process that stops existing between two
instructions, with no cleanup. So the gate writes a sentinel, fsyncs it, and
then parks forever; the parent kills it. Nothing in the child gets a say.

=== Why the sentinel is a FILE, and why it is fsynced ===

`failpoint_reached` has to be provable from outside the dead process. A byte on
disk that the parent can read after the child is gone is proof; a counter inside
the dead child is not. `flush()` alone would leave the bytes in the process's
buffered stream, and a SIGKILL / TerminateProcess does not flush those -- so the
write is fsynced, and `test_sentinel_survives_the_writer_being_killed` is the
test that fails if someone later "simplifies" that away.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.eval.faults import sha256_file
from longline.tools.base import Tool, ToolResult, ToolSchema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

SENTINEL_NAME = "failpoint.json"

BEFORE_MODEL = "before_model"
BEFORE_TOOL = "before_tool"
AFTER_TOOL = "after_tool"
AFTER_CHECKPOINT = "after_checkpoint"
TRUNCATE_TAIL = "truncate_tail"
WORKSPACE_DRIFT = "workspace_drift"

# The four the child's own gate can stop at.
GATED_FAILPOINTS: tuple[str, ...] = (BEFORE_MODEL, BEFORE_TOOL, AFTER_TOOL, AFTER_CHECKPOINT)
# The two the parent performs on the child's leftovers after the kill.
PARENT_FAILPOINTS: tuple[str, ...] = (TRUNCATE_TAIL, WORKSPACE_DRIFT)
ALL_FAILPOINTS: tuple[str, ...] = GATED_FAILPOINTS + PARENT_FAILPOINTS


class FailpointError(RuntimeError):
    """The failpoint could not be armed as specified (a wiring bug, not a result)."""


# What a `before_tool` gate returns when its `block` returns instead of parking.
# It is not a plausible tool output on purpose: a run that reached here is a
# test-mode run, and a result that looked like real data could be mistaken for
# one.
STOPPED_TOOL_RESULT = "[failpoint] before_tool stopped this call"


class FailpointReached(BaseException):
    """Raised by an in-process `block` so a stop can unwind the run.

    Subclasses `BaseException`, not `Exception`, on purpose. It is not an error
    to be handled -- it is a control-flow signal meaning "this process is about
    to be killed". `StreamingToolExecutor` and several `except Exception` sites
    hold tool and model failures; any of them swallowing this would let the run
    continue past its own failpoint, which is the one failure this module
    cannot tolerate.
    """


def halt() -> None:
    """An in-process `block`: stop the run without parking the process.

    Used by the worker's `no_block` mode, where the phase must return so a unit
    test can assert what it left behind. Production uses `block_forever`.
    """
    raise FailpointReached


# --- the sentinel ---


def sentinel_path(claude_dir: Path) -> Path:
    return Path(claude_dir) / SENTINEL_NAME


def write_sentinel(
    claude_dir: Path,
    *,
    failpoint: str,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    """Write and fsync the sentinel. Returns the path it wrote."""
    if failpoint not in ALL_FAILPOINTS:
        raise FailpointError(f"unknown failpoint {failpoint!r} (known: {sorted(ALL_FAILPOINTS)})")
    path = sentinel_path(claude_dir)
    payload = {
        "failpoint": failpoint,
        "pid": os.getpid(),
        "detail": dict(detail or {}),
    }
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - a filesystem failure, not a result
        raise FailpointError(f"could not write the failpoint sentinel: {exc}") from exc
    return path


def read_sentinel(claude_dir: Path) -> dict[str, Any] | None:
    """The sentinel's payload, or None if absent or unreadable.

    Absent and unreadable are the same answer on purpose: both mean "the parent
    has no proof the child reached the point", and the caller's next move (fail
    the case) is identical either way.
    """
    path = sentinel_path(claude_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def wait_for_sentinel(claude_dir: Path, *, timeout_s: float) -> dict[str, Any] | None:
    """Poll until the child signals, or the timeout expires.

    Returns the payload rather than a bool so the caller can assert WHICH
    failpoint fired: a sentinel for the wrong point is a wiring bug that a
    boolean would report as success.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = read_sentinel(claude_dir)
        if payload is not None:
            return payload
        time.sleep(0.02)
    return None


def block_forever(poll_s: float = 0.05) -> None:
    """Park until the parent kills this process.

    A sleep loop rather than `signal.pause()`: `signal.pause()` does not exist
    on Windows, and on Windows the parent's terminate is `TerminateProcess`,
    which ends the process wherever it happens to be. There is nothing to be
    gained by being interruptible.
    """
    while True:
        time.sleep(poll_s)


# --- the gate ---


@dataclass
class FailpointGate:
    """Decides where to stop, signals the parent, and parks.

    `armed` exists for two reasons. The resumed leg installs the SAME wrappers,
    and an armed gate there would park the resumed process forever -- every case
    would hit the worker timeout and the failure would look like a broken resume
    rather than a broken harness. It also lets the caller choose WHEN counting
    starts: `after_checkpoint` arms the gate only once the first instruction has
    completed, so the stop lands on the second instruction's first model call
    without a hand-kept call index.
    """

    claude_dir: Path
    failpoint: str
    at_call_index: int = 1
    at_tool_name: str = ""
    armed: bool = True
    reached: int = 0
    block: Callable[[], None] = block_forever

    def triggers_model(self, call_index: int) -> bool:
        if not self.armed:
            return False
        if self.failpoint not in (BEFORE_MODEL, AFTER_CHECKPOINT):
            return False
        return call_index == self.at_call_index

    def triggers_tool(self, tool_name: str) -> bool:
        if not self.armed:
            return False
        if self.failpoint not in (BEFORE_TOOL, AFTER_TOOL):
            return False
        return bool(self.at_tool_name) and tool_name == self.at_tool_name

    def stop(self, *, detail: Mapping[str, Any] | None = None) -> None:
        """Signal the parent, then park.

        In production this never returns: `block_forever` parks until the parent
        kills the process. Callers must still treat the call as terminal and
        return immediately afterwards -- see `GatedTool.execute` and
        `GatedModel._serve`. Relying on "the block never returns" would make the
        stop point depend on which `block` was injected, and a stop that fell
        through would run the very thing the failpoint exists to stop before.
        """
        self.reached += 1
        write_sentinel(self.claude_dir, failpoint=self.failpoint, detail=detail)
        self.block()


# --- gated wrappers ---


@dataclass
class GatedModel:
    """Wraps a scripted model and stops at the configured model call.

    Counts its OWN calls, and **only while the gate is armed**. That is what
    lets `after_checkpoint` mean "the first model call after the gate was
    armed": the worker arms the gate only once instruction 1 has completed and
    been saved, so the stop lands on instruction 2's first call without any
    hand-kept call index that would go stale the moment the scenario changed.

    It does NOT read the inner scripted model's counter: the inner model
    increments when it starts producing events, which is AFTER the point this
    failpoint means to stop at.
    """

    inner: Any
    gate: FailpointGate
    calls: int = 0

    def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        if self.gate.armed:
            self.calls += 1
        return self._serve(kwargs)

    async def _serve(self, kwargs: dict[str, Any]) -> AsyncIterator[Any]:
        if self.gate.triggers_model(self.calls):
            self.gate.stop(detail={"model_call_index": self.calls})
            # Unreachable in a real run. Reachable when a test injects a
            # returning `block`, and stopping the response is the correct
            # behaviour there: a model call that fell through would be the very
            # call the failpoint exists to interrupt.
            return
        async for event in self.inner(**kwargs):
            yield event


@dataclass
class GatedTool(Tool):
    """Wraps a production tool: journals its side effect, then optionally stops.

    Two stop points, and the ordering around them is the experiment:

    - `before_tool` stops BEFORE delegating, so the tool never runs.
    - `after_tool` lets the tool run, records the journal entry, and only then
      stops. That ordering is what makes the entry durable across the kill; a
      sentinel written first would open a window in which the parent kills the
      child after the side effect but before the evidence reached disk.

    Subclasses `Tool` rather than duck-typing, so the registry swap is checked
    by the type system -- the same reason `faults.ToolFaultWrapper` and
    `eval_tools.SandboxedTool` do.
    """

    inner: Tool
    gate: FailpointGate
    journal: Any | None = None
    leg: str = "killed"
    artifact_paths: tuple[str, ...] = ()
    artifact_root: str = ""
    calls: int = 0

    def get_name(self) -> str:
        return self.inner.get_name()

    def get_schema(self) -> ToolSchema:
        return self.inner.get_schema()

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return self.inner.is_concurrency_safe(tool_input)

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self.calls += 1
        name = self.get_name()
        if self.gate.triggers_tool(name) and self.gate.failpoint == BEFORE_TOOL:
            self.gate.stop(detail={"tool": name, "tool_call_index": self.calls})
            # Unreachable in a real run; see `GatedModel._serve`. Returning
            # without delegating is what makes `before_tool` mean "the tool
            # never ran" under ANY `block` implementation.
            return ToolResult(content=STOPPED_TOOL_RESULT, is_error=True)

        pre_state = self.snapshot_artifacts()
        result = await self.inner.execute(tool_input)
        post_state = self.snapshot_artifacts()

        if self.journal is not None:
            self.journal.record(
                leg=self.leg,
                tool=name,
                tool_input=dict(tool_input),
                outcome="error" if result.is_error else "ok",
                pre_state=pre_state,
                post_state=post_state,
            )

        if self.gate.triggers_tool(name) and self.gate.failpoint == AFTER_TOOL:
            self.gate.stop(detail={"tool": name, "tool_call_index": self.calls})
        return result

    def snapshot_artifacts(self) -> dict[str, str]:
        """Digest of the declared artifact files, keyed by relative path.

        Only the files the case declares. Hashing the whole sandbox would make
        the journal's `post_state` differ for reasons unrelated to the tool
        under observation (a `.pyc`, a temp file), and a state difference is
        what the duplicate test is built on.
        """
        root = Path(self.artifact_root) if self.artifact_root else Path(".")
        return {rel: sha256_file(root / rel) for rel in self.artifact_paths}


# --- process control ---


def terminate_and_reap(proc: subprocess.Popen[Any], *, grace_s: float = 10.0) -> bool:
    """Terminate a live child and reap it. True when it really ended.

    Shared with `recovery_runner._kill_child`, which waits for its child by a
    fixed delay while this suite waits for a sentinel -- the wait differs, the
    kill does not, so only the kill lives here.
    """
    if proc.poll() is not None:
        return False
    try:
        proc.terminate()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=grace_s)
    except OSError:  # pragma: no cover - the child died between poll and kill
        pass
    return proc.returncode is not None


def truncate_last_line(path: Path) -> int:
    """Cut the file's final line in half. Returns the bytes removed.

    Half a line is the point: a JSONL file whose last line is complete but
    wrong is a different fault from one whose last line was never finished, and
    only the second is what a process dying mid-write produces.

    The cut is placed at the midpoint of the final line so the result is
    neither a valid record nor empty, and the surviving prefix stays
    line-by-line parseable -- which is what `load_session` relies on.
    """
    if not path.is_file():
        raise FailpointError(f"cannot truncate a missing file: {path}")
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    # A trailing newline leaves an empty final element; drop it so "the last
    # line" means the last real record.
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    if not lines:
        raise FailpointError(f"cannot truncate a file with no lines: {path}")
    last = lines[-1]
    lines[-1] = last[: max(1, len(last) // 2)]
    # No trailing newline: a process that died mid-write leaves a half line with
    # nothing after it. Appending one would make the torn record look like a
    # complete (if short) line, which is a different fault.
    out = b"\n".join(lines)
    path.write_bytes(out)
    return len(raw) - len(out)


__all__ = [
    "AFTER_CHECKPOINT",
    "AFTER_TOOL",
    "ALL_FAILPOINTS",
    "BEFORE_MODEL",
    "BEFORE_TOOL",
    "GATED_FAILPOINTS",
    "PARENT_FAILPOINTS",
    "SENTINEL_NAME",
    "STOPPED_TOOL_RESULT",
    "TRUNCATE_TAIL",
    "WORKSPACE_DRIFT",
    "FailpointError",
    "FailpointGate",
    "FailpointReached",
    "GatedModel",
    "GatedTool",
    "block_forever",
    "halt",
    "read_sentinel",
    "sentinel_path",
    "terminate_and_reap",
    "truncate_last_line",
    "wait_for_sentinel",
    "write_sentinel",
]
