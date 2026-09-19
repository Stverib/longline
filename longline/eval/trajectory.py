"""Extract an evaluation trajectory from a QueryEvent stream.

The agent under test is driven by QueryEngine.submit(); the events it yields
are the single source of truth for what the agent actually did. This module
projects them onto a compact `Trajectory` used by both judge layers.

Telemetry captured here (Task 1):
  - every `ToolResultReady`, i.e. every tool that actually EXECUTED, with its
    success/failure flag. Calls that were requested but never dispatched (the
    run aborted mid-turn) are deliberately excluded, so
    `ExecutionSuccessRate = successful / executed` cannot be inflated.
  - the event receipt time (ns) of each event type, from an injectable
    monotonic clock so latency assertions are stable offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from longline.core.events import ErrorEvent, TextDelta, ToolResultReady, ToolUseStart, TurnComplete

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from longline.core.events import QueryEvent

# tool call = (tool_name, input dict), in call order
ToolCall = tuple[str, dict[str, object]]

# Monotonic clock returning nanoseconds. Injected in tests so latency numbers
# are exactly assertable without sleeping.
Clock = "Callable[[], int]"


def _default_clock() -> int:
    """Process-wide monotonic clock in nanoseconds."""
    return time.perf_counter_ns()


@dataclass
class ToolExecution:
    """One tool that was actually dispatched and finished.

    `tool_name` is resolved by joining `ToolResultReady.tool_id` against the
    `ToolUseStart` events — `ToolResultReady` itself carries no tool name. The
    join is by id and never positional: `get_results()` returns results in
    COMPLETION order, which is not call order when tools run concurrently.
    """

    tool_id: str
    tool_name: str
    is_error: bool
    start_ns: int | None = None
    end_ns: int | None = None

    @property
    def duration_ms(self) -> float | None:
        """Wall time of the actual tool execution, in milliseconds."""
        if self.start_ns is None or self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000.0


@dataclass
class Trajectory:
    """What the agent did during one evaluation run."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Cache accounting, kept SEPARATE from `input_tokens` on purpose. The API
    # reports `input_tokens` as the uncached remainder on a provider that
    # implements prompt caching, so a total built from it alone understates the
    # prompt -- measured at ~3x on the deepseek-flash endpoint. Folding these
    # into `input_tokens` would silently redefine a field that every existing
    # raw.jsonl row records; a separate field plus `prompt_tokens` makes the
    # distinction visible instead.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    text: str = ""
    errors: list[str] = field(default_factory=list)
    # Every distinct `TurnComplete.served_model` this stream carried, in first-seen
    # order. A LIST rather than a single name because a gateway can fail over
    # between turns, and a run whose turns were answered by two different models
    # has to be able to say so: picking one -- the first, the last, the majority
    # -- would produce a single confident name for a number that is an average of
    # two. Empty means the transport never told us, which is not the same as
    # "the transport served what we asked for"; `served_model_source` in the run
    # metadata is what distinguishes the two for a reader.
    served_models: list[str] = field(default_factory=list)
    # Task 1 telemetry
    tool_executions: list[ToolExecution] = field(default_factory=list)
    event_timestamps: dict[str, int] = field(default_factory=dict)

    @property
    def num_tool_calls(self) -> int:
        """Tool calls the model requested."""
        return len(self.tool_calls)

    @property
    def num_tool_calls_executed(self) -> int:
        """Tool calls that were actually dispatched to a tool."""
        return len(self.tool_executions)

    @property
    def num_successful_tool_calls(self) -> int:
        """Executed calls that did not report `is_error`."""
        return sum(1 for e in self.tool_executions if not e.is_error)

    @property
    def prompt_tokens(self) -> int:
        """Everything the prompt cost: uncached + written-to-cache + cache hits.

        This is the number a cost or context question wants. `input_tokens`
        alone answers a narrower question ("how much of this prompt was not
        already cached"), and on a caching provider the two differ by a large
        factor.
        """
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


def _error_type(message: str) -> str:
    """Classify an ErrorEvent message into a stable failure category.

    Matches the `failure_attribution` vocabulary in `evals/baselines/README.md`
    where one applies; anything unrecognised falls back to `runtime_error`.
    """
    lowered = message.lower()
    if "max turns" in lowered or "max_turns" in lowered:
        return "max_turns"
    if "gave up after retries" in lowered or "429" in lowered or "529" in lowered:
        return "api_error"
    if "prompt_too_long" in lowered or ("context" in lowered and "overflow" in lowered):
        return "context_overflow"
    return "runtime_error"


async def extract_trajectory(
    stream: AsyncIterator[QueryEvent],
    *,
    clock: Callable[[], int] = _default_clock,
) -> Trajectory:
    """Consume a QueryEvent stream and project it onto a Trajectory.

    `clock` is injectable (monotonic ns) so tests can assert latencies exactly.
    """
    traj = Trajectory()
    traj.event_timestamps["request_start"] = clock()
    # tool_id -> (tool_name, start_ns), filled from ToolUseStart; ToolResultReady
    # has no name of its own and results arrive in completion order.
    pending: dict[str, tuple[str, int]] = {}

    async for event in stream:
        if isinstance(event, ToolUseStart):
            now = clock()
            traj.tool_calls.append((event.tool_name, event.input))
            pending[event.tool_id] = (event.tool_name, now)
            traj.event_timestamps["tool_use_start"] = now
        elif isinstance(event, ToolResultReady):
            now = clock()
            traj.event_timestamps["tool_result_ready"] = now
            name, start_ns = pending.get(event.tool_id, ("", None))
            traj.tool_executions.append(
                ToolExecution(
                    tool_id=event.tool_id,
                    tool_name=name,
                    is_error=event.is_error,
                    start_ns=start_ns,
                    end_ns=now,
                )
            )
        elif isinstance(event, TurnComplete):
            traj.turns += 1
            traj.input_tokens += event.usage.input_tokens
            traj.output_tokens += event.usage.output_tokens
            traj.cache_creation_tokens += event.usage.cache_creation_input_tokens
            traj.cache_read_tokens += event.usage.cache_read_input_tokens
            if event.served_model and event.served_model not in traj.served_models:
                traj.served_models.append(event.served_model)
            traj.event_timestamps[f"turn_{traj.turns}_complete"] = clock()
        elif isinstance(event, TextDelta):
            traj.text += event.text
        elif isinstance(event, ErrorEvent):
            traj.errors.append(event.message)
    return traj


def infer_error_type(traj: Trajectory, *, passed: bool) -> str | None:
    """Best-effort failure category for a finished trajectory.

    An ErrorEvent is always a failure signal from the query loop, even when a
    lenient judge still reports the case as passing — an aborted run that
    happens to satisfy its assertions is not a clean run.
    """
    if traj.errors:
        return _error_type(traj.errors[-1])
    if passed:
        return None
    return None
