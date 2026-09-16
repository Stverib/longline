"""Scripted model streams with fault injection at a chosen call index.

=== What this measures (evals/README.md §5.4, plan §4.4) ===

Five runtime fault classes, each injected at a named point in the model-call
stream:

| fault            | injection point                          |
|------------------|------------------------------------------|
| 429              | first or second model call               |
| 529              | first or second model call               |
| tool_failure     | a named tool's first call returns is_error=true |
| output_truncate  | first response returns stop_reason=max_tokens |
| context_overflow | first response is 413 / prompt_too_long  |

The recovery paths for all five **already exist in `longline/core/query_loop.py`**
(retry of `is_recoverable` errors up to `max_retry=5`, escalation of `max_tokens`
16384 -> 65536, reactive compact on 413 / prompt_too_long). This module only
triggers them; it deliberately does not reimplement any of them.

=== Why the injectors are counters, not booleans ===

`RuntimeRecoveryRate` counts runs in which the fault was **actually injected**,
the recovery path **actually fired**, and the final deterministic judge passed.
A run in which nothing was injected and the task passed anyway is not a
recovery -- it is a plain success wearing the metric's name. The single most
damaging thing this module could do is report 10/10 for a fault that never
happened, so every injector carries a counter and `run_recovery_case` asserts on
it before it will record a success.

=== Two independent proofs per fault ===

An injector counter is a claim made by the same object that is supposed to be
broken. It is checked twice, by two different parties:

1. `InjectionRecord.injected` -- the injector's own count of what it emitted.
2. The trajectory -- what the production loop and executor actually observed on
   the event stream: `ErrorEvent`s seen by us, tool executions that the real
   `StreamingToolExecutor` dispatched and that came back `is_error`.

For the two classes where the second proof is weaker (429/529: the loop
swallows the error and retries, so it never reaches the event stream;
truncate/overflow: every loop in this repo swallows `ErrorEvent`) the record
says which mechanism was used and why, rather than quietly presenting a weak
proof as a strong one. See `InjectionRecord.proof`.

=== The two loops a fault can be injected into ===

`query_engine.py` talks to the model through a callable it owns
(`make_call_model`), so there are two honest ways to stand in for the model:

- `fault_engine(...)` returns a real `QueryEngine` whose `call_model` is
  monkeypatched. This drives **all** of `query_loop`'s recovery logic and is
  what `run_recovery_case` uses.
- `drive_query_loop(events)` runs the scripted events directly through
  `query_loop()` in-process -- literally the production state machine, with no
  engine, no SDK and no sleeps. Used for the raw loop-level protocol tests.

`disable_recovery` in the driver is the acceptance-condition control: it passes
`max_retry=0` / `max_max_output_recovery=0` so the path under test cannot run,
which makes the corresponding test fail. With it unset every argument is
`query_loop`'s own default, so the fixture exercises production behaviour.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from longline.core.events import (
    CompactOccurred,
    ErrorEvent,
    QueryEvent,
    TextDelta,
    ToolResultReady,
    ToolUseStart,
    TurnComplete,
)
from longline.models.messages import Usage
from longline.tools.base import Tool, ToolResult, ToolSchema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence
    from pathlib import Path

# --- fault class names (the vocabulary every report is keyed by) ---

RATE_LIMIT = "429"
OVERLOADED = "529"
TOOL_FAILURE = "tool_failure"
OUTPUT_TRUNCATE = "output_truncate"
CONTEXT_OVERFLOW = "context_overflow"
PROCESS_KILL = "process_kill"

# The five classes that make up RuntimeRecoveryRate.
RUNTIME_FAULTS: tuple[str, ...] = (
    RATE_LIMIT,
    OVERLOADED,
    TOOL_FAILURE,
    OUTPUT_TRUNCATE,
    CONTEXT_OVERFLOW,
)

# The two classes whose fault is reported by a *tool*, not by the model stream.
TOOL_FAULTS: tuple[str, ...] = (TOOL_FAILURE,)

ALL_FAULTS: tuple[str, ...] = (*RUNTIME_FAULTS, PROCESS_KILL)

# The message `stream_response` would produce for a 429/529. It has to contain
# the status code, because `query_engine._build_fault_model` is what turns it
# back into `is_recoverable=True` -- and that reconstruction is itself asserted
# against the real SDK helper in the unit tests.
RATE_LIMIT_MESSAGE = (
    "Error code: 429 - {'type': 'error', 'error': "
    "{'type': 'rate_limit_error', 'message': 'rate limited'}}"
)
OVERLOADED_MESSAGE = (
    "Error code: 529 - {'type': 'error', 'error': "
    "{'type': 'overloaded_error', 'message': 'overloaded'}}"
)
PROMPT_TOO_LONG_MESSAGE = (
    "Error code: 413 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'prompt is too long: 250000 tokens > 200000 maximum'}}"
)

_FAULT_MESSAGES: dict[str, str] = {
    RATE_LIMIT: RATE_LIMIT_MESSAGE,
    OVERLOADED: OVERLOADED_MESSAGE,
    CONTEXT_OVERFLOW: PROMPT_TOO_LONG_MESSAGE,
}

# Marks the synthetic placeholder script. It is never used as an answer: the
# judge (`judge_recovered_answer`) asserts the ANSWER is present, and asserts
# this marker is ABSENT -- so a run that produced only the fault text fails.
PLACEHOLDER_PREFIX = "[fault-injector]"

# The first half of the answer an `output_truncate` case's response manages to
# emit before being cut off. The continuation supplies the rest, so the final
# text equals the answer only when the continuation really ran. It carries no
# placeholder marker of its own -- a real truncation has no way to announce
# itself, and a marker here would pollute the very text being graded.
TRUNCATED_ANSWER_FRAGMENT = "marker=alpha-7"


class InjectionError(RuntimeError):
    """The fault could not be injected as specified (a wiring bug, not a result)."""


# --- injection bookkeeping ---


@dataclass
class InjectionRecord:
    """Proof that an injector fired, and by which mechanism.

    `attempts` is not symmetric across classes on purpose: for the model-stream
    faults it counts *eligible* call indices that were reached, while for the
    tool fault it counts calls that reached the tool. Reporting the number of
    times the wrapper was consulted as if it were the number of faults injected
    would inflate it, so the two are separate fields.
    """

    fault: str
    injected: bool = False
    inject_at_call_index: int = 0
    call_index: int = 0
    attempts: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.notes = list(self.notes)

    @property
    def proof(self) -> str:
        """Which observation actually establishes that a fault occurred.

        Spelled out rather than left to the reader because the strength of the
        evidence differs by class, and a report that presents all six as
        equally proven would be overstating two of them.
        """
        if self.fault in (RATE_LIMIT, OVERLOADED):
            return (
                "injector_emitted_error_event: query_loop consumes the ErrorEvent "
                "internally (it is not re-yielded on recovery) and escalates through "
                "an external sleep hook that counts the retry sleeps it served"
            )
        if self.fault in (OUTPUT_TRUNCATE, CONTEXT_OVERFLOW):
            return (
                "injector_emitted_scripted_response + script_counter_call: "
                "query_loop swallows ErrorEvent on both the compact and the "
                "truncation path, so the fault is established by the scripted "
                "response the model was asked for, and the recovery path by the "
                "scripted summariser call it caused"
            )
        if self.fault in TOOL_FAULTS:
            return (
                "tool_fault_counter + trajectory_tool_executions: the real "
                "StreamingToolExecutor reported an errored execution for the "
                "injected call index, paired with the wrapper's own counter"
            )
        return "injector_counter"


@dataclass
class SleepRecord:
    """Retry sleeps the loop actually performed, captured instead of waited.

    `query_loop` backs off with `asyncio.sleep(min(2.0 * retry_count, 10.0))`
    on every recoverable error. That would make an offline 429 test take four
    real seconds, so the sleep function is injectable and the real one is
    replaced by a recorder. The count is not a convenience: it is the only
    external evidence that the retry path ran, because the loop consumes the
    `ErrorEvent` and never re-yields it.
    """

    durations: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)

    @property
    def count(self) -> int:
        return len(self.durations)

    @property
    def total_seconds(self) -> float:
        return sum(self.durations)


def _event_if_eligible(
    record: InjectionRecord,
    index: int,
    at_indices: Sequence[int],
) -> bool:
    """Whether call `index` is one of the configured injection points."""
    return index in at_indices


def build_injection_events(fault: str, *, answer: str = "") -> list[QueryEvent]:
    """The events one injected call emits, for a call that must not succeed.

    Deliberately returns ONLY the fault (plus, for truncate, the partial text
    the API would have produced). Any answer text here would make a broken run
    pass its judge, which is the failure mode this whole task exists to
    prevent -- see `PLACEHOLDER_PREFIX`.
    """
    if fault in (RATE_LIMIT, OVERLOADED):
        return [ErrorEvent(message=_FAULT_MESSAGES[fault], is_recoverable=True)]
    if fault == OUTPUT_TRUNCATE:
        # The API truncates AFTER emitting the tokens it managed to produce, and
        # the loop SAVES those tokens into the transcript before asking the model
        # to continue. Whatever this emits therefore ends up in the final text,
        # so it must be the first part of the answer itself -- a real truncation
        # cuts an answer in half; it does not append a note saying it was cut.
        #
        # That is also what makes the continuation observable. If this emitted
        # nothing but a marker, the recovered text would be unjudgeable (part
        # fault, part recovery, with no seam); as a half-answer, the final text
        # is the answer only if the continuation actually supplied the rest.
        return [
            TextDelta(text=TRUNCATED_ANSWER_FRAGMENT),
            TurnComplete(stop_reason="max_tokens", usage=Usage(input_tokens=11, output_tokens=3)),
        ]
    if fault == CONTEXT_OVERFLOW:
        return [ErrorEvent(message=_FAULT_MESSAGES[CONTEXT_OVERFLOW], is_recoverable=False)]
    raise InjectionError(f"no scripted fault events for {fault!r}")


# --- the model-stream injector ---

# A fault name for a scripted model that never injects anything. It is not in
# `ALL_FAULTS` on purpose: the tool-failure class carries its fault in the tool,
# and an injector that cannot fail must not be able to contribute to a fault
# class's numerator or denominator.
NO_FAULT = "none"


@dataclass
class ScriptedModelBase:
    """Shared call-counting for every scripted model.

    `record` is a non-optional field with a factory rather than an `| None` one
    filled in by `__post_init__`. The optional form pushed a `None` check into
    every reader, and the ones that forgot it were exactly the counters the
    report depends on -- a counter that can be None is a counter that can
    silently stop counting.
    """

    record: InjectionRecord = field(default_factory=lambda: InjectionRecord(fault=NO_FAULT))
    sleep_record: SleepRecord = field(default_factory=SleepRecord)
    seen_call_indices: list[int] = field(default_factory=list)
    prompts: list[list[dict[str, Any]]] = field(default_factory=list)
    recovered_at_call_index: int | None = None

    def _count_call(self) -> int:
        """Record one model call and return its 1-based index."""
        self.record.call_index += 1
        self.seen_call_indices.append(self.record.call_index)
        return self.record.call_index

    def _note_recovery(self, index: int) -> None:
        """Record the first fault-free call after an injected one."""
        if self.record.injected and self.recovered_at_call_index is None:
            self.recovered_at_call_index = index

    @property
    def injected(self) -> bool:
        return self.record.injected


@dataclass
class ScriptedModel(ScriptedModelBase):
    """A scripted model that answers directly, with no tool calls.

    Used by the fault classes whose fault is carried elsewhere (the tool class)
    or where a direct answer is what the judge expects. It shares the
    injector's call-counting so `retry_count` means the same thing across
    classes, and exposes the same `record` / `prompts` / `sleep_record` surface
    so the runner does not have to branch on which scripted model it holds.
    """

    answer: str = ""

    def __call__(self, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        self.prompts.append(list(kwargs.get("messages", [])))
        return self._serve()

    async def _serve(self) -> AsyncIterator[QueryEvent]:
        self._count_call()
        yield TextDelta(text=self.answer)
        yield TurnComplete(
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=25),
        )


@dataclass
class ToolCallingModel(ScriptedModelBase):
    """A scripted model that calls one tool, then answers from what came back.

    The tool-failure class is the one case where the model has to *react*. The
    agent must see `is_error=true` and decide what to do, and a model script
    that ignores the result and prints the answer anyway would make the fault
    invisible: the case would pass with or without the wrapper, which is the
    definition of a test that measures nothing.

    So the answer is emitted ONLY when the tool result is present and not an
    error. The sequence is:

    - call 1: request the tool;
    - call 2: if the last tool result is an error, call the tool AGAIN (this is
      the adapt-or-retry behaviour the case is about);
    - call 3: emit the answer, which by then is backed by a successful result.

    `saw_tool_error` and `retried_after_error` are the evidence that the agent
    really adapted rather than merely got lucky about call ordering.
    """

    # Every field is defaulted so the ordering constraint a dataclass imposes on
    # base/inherited fields is satisfied. Construction sites always pass the
    # first three explicitly, so a defaulted `tool_name` cannot be reached by
    # accident with an empty name.
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    max_tool_attempts: int = 2
    saw_tool_error: bool = False
    retried_after_error: bool = False

    def __call__(self, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        self.prompts.append(list(kwargs.get("messages", [])))
        return self._serve(kwargs.get("messages", []))

    async def _serve(self, messages: list[dict[str, Any]]) -> AsyncIterator[QueryEvent]:
        index = self._count_call()

        results = _tool_results_in(messages)
        attempts = self._attempts_so_far(messages)

        if not results and attempts == 0:
            # Nothing has been tried yet: ask for the tool.
            yield ToolUseStart(
                tool_name=self.tool_name,
                tool_id=f"tu-{index}",
                input=dict(self.tool_input),
            )
            yield TurnComplete(stop_reason="tool_use", usage=Usage(input_tokens=100, output_tokens=20))
            return

        last_is_error = bool(results) and results[-1]
        if last_is_error:
            self.saw_tool_error = True
        if last_is_error and attempts < self.max_tool_attempts:
            # Adapt: the first attempt failed, so try again. This is the
            # behaviour the "Tool Failure" success condition names.
            self.retried_after_error = True
            yield ToolUseStart(
                tool_name=self.tool_name,
                tool_id=f"tu-{index}",
                input=dict(self.tool_input),
            )
            yield TurnComplete(stop_reason="tool_use", usage=Usage(input_tokens=100, output_tokens=20))
            return

        if last_is_error:
            # Out of attempts with the tool still failing: do NOT invent the
            # answer. Reporting it anyway would let a run pass a judge it never
            # earned, which is exactly what `fault_injected` is there to stop.
            yield TextDelta(text="The tool kept failing, so I could not read the file.")
            yield TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=100, output_tokens=20))
            return

        if self.recovered_at_call_index is None:
            self._note_recovery(index)
        yield TextDelta(text=self.answer)
        yield TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=100, output_tokens=25))

    @staticmethod
    def _attempts_so_far(messages: list[dict[str, Any]]) -> int:
        """How many tool_result blocks the transcript already carries."""
        return sum(1 for m in messages for _ in _tool_result_blocks(m))


def _tool_result_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        b for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


def _tool_results_in(messages: list[dict[str, Any]]) -> list[bool]:
    """`is_error` of every tool_result in order, newest last.

    `ToolResultBlock.to_api_dict()` writes `is_error` only when it is true (the
    API treats absence as false), so a missing key reads as success.
    """
    return [
        bool(block.get("is_error", False))
        for message in messages
        for block in _tool_result_blocks(message)
    ]


@dataclass
class ModelFaultInjector(ScriptedModelBase):
    """A `call_model` that emits scripted responses with a fault at chosen indices.

    The call index is incremented BEFORE the fault decision, so "first or second
    model call" means indices 1 and 2 -- 0 would silently mean "never", which
    would be a fault that never fires and a case that can never be counted.

    Recovery is observed rather than asserted from inside the injector:

    - `recovered_at_call_index` is the first index *after* an injected one that
      produced a fault-free response, i.e. the loop came back and was served.
    - `sleep_record` counts the loop's own back-off calls (429/529), which only
      happen on the recoverable-retry path.
    """

    fault: str = RATE_LIMIT
    answer: str = ""
    at_call_indices: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.fault not in _FAULT_MESSAGES and self.fault != OUTPUT_TRUNCATE:
            raise InjectionError(
                f"ModelFaultInjector cannot inject {self.fault!r} "
                f"(known: {sorted([*_FAULT_MESSAGES, OUTPUT_TRUNCATE])})"
            )
        if not self.at_call_indices or any(i < 1 for i in self.at_call_indices):
            raise InjectionError(
                f"injection indices must be >= 1 (1 = first call), got {self.at_call_indices}"
            )
        # `record.fault` has to come from this injector rather than a factory,
        # which is why the base's default record is replaced here.
        self.record = InjectionRecord(
            fault=self.fault,
            inject_at_call_index=min(self.at_call_indices),
        )

    def __call__(self, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        """Serve one scripted model response. `kwargs` mirrors `stream_response`."""
        self.prompts.append(list(kwargs.get("messages", [])))
        return self._serve()

    async def _serve(self) -> AsyncIterator[QueryEvent]:
        index = self._count_call()

        if _event_if_eligible(self.record, index, self.at_call_indices):
            self.record.injected = True
            self.record.attempts += 1
            for event in build_injection_events(self.fault, answer=self.answer):
                yield event
            return

        self._note_recovery(index)
        yield TextDelta(text=self.answer)
        yield TurnComplete(
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=25),
        )


# --- the tool-fault injector ---


@dataclass
class ToolFaultWrapper(Tool):
    """A `Tool` that returns `is_error=true` on its first call, then delegates.

    Subclasses `Tool` rather than merely imitating it, so a registry swap is
    checked by the type system. Wraps the *production* tool rather than
    replacing it: the wrapper's job is to fail once and then get out of the way,
    so the second call is the real tool doing real work in the sandbox. That is
    what makes "the agent adapted or retried" a real observation -- the retry
    genuinely succeeded.

    The counter is incremented on every call and compared against the number of
    errored executions the trajectory recorded, so the two proofs are checked
    against each other rather than trusted individually.

    `@dataclass` on a `Tool` subclass works because `Tool` has no `__init__` of
    its own: `ABC` supplies none, so the generated one is the only initialiser.
    """

    inner: Tool
    fault_at_call_index: int = 1
    calls: int = 0
    faults_injected: int = 0
    inputs: list[dict[str, Any]] = field(default_factory=list)

    def get_name(self) -> str:
        return self.inner.get_name()

    def get_schema(self) -> ToolSchema:
        return self.inner.get_schema()

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return self.inner.is_concurrency_safe(tool_input)

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self.calls += 1
        self.inputs.append(dict(tool_input))
        if self.calls == self.fault_at_call_index:
            self.faults_injected += 1
            return ToolResult(
                content=(
                    "Error: the injected fault made this tool call fail. "
                    "No data was returned. Retry the call if you still need the value."
                ),
                is_error=True,
            )
        return await self.inner.execute(tool_input)

    @property
    def injected(self) -> bool:
        return self.faults_injected > 0

    def record_into(self, record: InjectionRecord) -> str:
        """Copy this wrapper's counters onto the case's injection record."""
        record.fault = TOOL_FAILURE
        record.injected = self.faults_injected > 0
        record.attempts = self.faults_injected
        record.call_index = self.calls
        record.inject_at_call_index = self.fault_at_call_index
        return "tool_wrapper"


# --- engine-level wiring ---


def fault_registry(
    sandbox: str,
    *,
    tool_name: str,
    fault_at_call_index: int = 1,
    profile: str = "core",
) -> tuple[Any, ToolFaultWrapper]:
    """Build the eval registry with one tool wrapped in a fault injector.

    Returns `(registry, wrapper)`. The wrapper is returned rather than stashed
    inside the registry because the counters are the evidence the case reports,
    and evidence you have to go digging for is evidence that gets skipped.
    """
    from longline.eval.eval_tools import build_eval_registry

    registry = build_eval_registry(sandbox, profile=profile)
    inner = registry.get(tool_name)
    if inner is None:
        raise InjectionError(
            f"tool {tool_name!r} is not in profile {profile!r}; a fault aimed at a "
            "tool the model cannot call would never fire"
        )
    wrapper = ToolFaultWrapper(inner, fault_at_call_index=fault_at_call_index)
    registry.swap(tool_name, wrapper)
    return registry, wrapper


def apply_model_fault(
    engine: Any,
    injector: ScriptedModelBase,
    *,
    patch_sleep: bool = True,
) -> None:
    """Point a REAL `QueryEngine` at the injector instead of the SDK.

    The engine, its registry, its tools and its permission context all stay
    real; only the model transport is replaced. `query_loop` therefore runs its
    production recovery logic unmodified, which is the entire point -- an
    injector that reimplemented the retry would measure the injector.

    `engine.make_call_model` is replaced before any call is made, so the check
    that the replacement is signature-compatible is behavioural rather than
    introspective: `assert_engine_uses_injector` calls the factory and asserts
    that what comes back is the injector.

    `patch_sleep` installs the injector's back-off recorder on the engine, which
    passes it to `query_loop`'s `sleep` parameter. It does NOT rebind
    `asyncio.sleep`: that name lives on the shared `asyncio` module, so
    replacing it is process-global and escapes the run that asked for it --
    which is exactly what happened, and what broke three unrelated tests in
    `tests/unit/tools/agent/test_background_agent.py` when the suites shared a
    process.
    """
    engine.make_call_model = lambda model=None, max_tokens=16384: injector
    if patch_sleep:
        engine.sleep_fn = injector.sleep_record


def assert_engine_uses_injector(engine: Any, injector: ScriptedModelBase) -> None:
    """Fail loudly if the engine would still reach the real SDK.

    A silent miss here is the worst failure mode available: the task would run
    cleanly against the network, never inject anything, probably pass, and be
    recorded as a successful recovery.
    """
    if engine.make_call_model() is not injector:
        raise InjectionError(
            "engine.make_call_model does not return the injector; the fault would "
            "never be injected and the run would be a plain success wearing the "
            "recovery metric's name"
        )


@contextlib.contextmanager
def engine_scope() -> Any:
    """Restore `query_loop`'s module-level state after an injected run.

    Nothing is patched at module scope any more -- the retry back-off is a
    `query_loop` parameter (`sleep`), reached through the engine the caller
    already holds. The context manager is kept because callers that expect to
    bracket a run with it should keep working, and because it is the natural
    place to add any future per-run state.
    """
    yield


# --- direct query_loop driver (offline protocol tests + controls) ---


@dataclass
class LoopOutcome:
    """What one `query_loop` run produced, in the form the assertions need.

    `tool_executions` are `(tool_name, is_error)` pairs as reported by the real
    `StreamingToolExecutor`, not as asserted by the injector.
    """

    events: list[QueryEvent] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    error_events: list[ErrorEvent] = field(default_factory=list)
    turns: int = 0
    compact_events: int = 0
    tool_uses: list[tuple[str, str]] = field(default_factory=list)  # (id, name)
    tool_executions: list[tuple[str, bool]] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.texts)

    @property
    def errored_executions(self) -> int:
        return sum(1 for _, is_error in self.tool_executions if is_error)


async def drive_query_loop(
    injector: Callable[..., Any],
    *,
    messages: list[Any] | None = None,
    registry: Any | None = None,
    auto_compact_fn: Callable[..., Any] | None = None,
    max_turns: int = 10,
    disable_recovery: bool = False,
    disable_reactive_compaction: bool = False,
    sleep: Any | None = None,
) -> LoopOutcome:
    """Run the production `query_loop` in-process over a scripted injector.

    Two independent controls, each turning off exactly the path under test:

    - `disable_recovery=True` passes `max_retry=0` and
      `max_max_output_recovery=0`, which are the retry and truncation budgets.
      It deliberately does NOT touch `max_reactive_compaction`: a control that
      switched off a neighbouring path would make the corresponding fault look
      handled (or not) for a reason unrelated to the path being measured.
    - `disable_reactive_compaction=True` passes `max_reactive_compaction=0`,
      which is the 413 / prompt_too_long compaction arm.

    With both unset, every argument is `query_loop`'s own default, so the run is
    production behaviour.

    `sleep` is passed through to `query_loop`'s back-off parameter. Its default
    is None, which means the loop's own `asyncio.sleep` -- so a 429 retry here
    really does wait out its 2s back-off unless the caller supplies a recorder.
    That is deliberate: shortening a back-off is the caller's decision to make
    explicitly, not something this helper does behind its back.
    """
    from longline.core.query_loop import query_loop
    from longline.models.messages import UserMessage
    from longline.tools.base import ToolRegistry

    msgs = messages if messages is not None else [UserMessage(content="start")]
    reg = registry if registry is not None else ToolRegistry()

    kwargs: dict[str, Any] = {}
    if disable_recovery:
        kwargs["max_retry"] = 0
        kwargs["max_max_output_recovery"] = 0
    if disable_reactive_compaction:
        kwargs["max_reactive_compaction"] = 0
    if sleep is not None:
        kwargs["sleep"] = sleep

    outcome = LoopOutcome(messages=msgs)
    async for event in query_loop(
        messages=msgs,
        system_prompt="test",
        tools=reg,
        call_model=injector,
        max_turns=max_turns,
        auto_compact_fn=auto_compact_fn,
        **kwargs,
    ):
        outcome.events.append(event)
        if isinstance(event, TextDelta):
            outcome.texts.append(event.text)
        elif isinstance(event, ErrorEvent):
            outcome.error_events.append(event)
        elif isinstance(event, TurnComplete):
            outcome.turns += 1
        elif isinstance(event, ToolResultReady):
            name = next((n for i, n in outcome.tool_uses if i == event.tool_id), "")
            outcome.tool_executions.append((name, event.is_error))
        elif isinstance(event, ToolUseStart):
            outcome.tool_uses.append((event.tool_id, event.tool_name))
        elif isinstance(event, CompactOccurred):
            outcome.compact_events += 1
    return outcome


# --- tool-call fingerprints (duplicate detection) ---


def sha256_file(path: Path) -> str:
    """Digest of a file's bytes, or `missing` when it is not there.

    `missing` rather than raising: an absent artifact is a fact about the run
    that a duplicate-check report should be able to state.
    """
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_tool_calls(
    executions: Iterable[tuple[str, dict[str, Any], str]],
) -> list[str]:
    """Stable fingerprints of **completed** tool calls.

    Each entry is `(tool_name, tool_input, result_digest)`. Keying on the result
    as well as the request is what makes the count a statement about a
    *persisted complete tool result* (contract §5.4): the same call re-issued
    against a changed file has a different fingerprint and is not a duplicate.
    """
    digests: list[str] = []
    for name, tool_input, result_digest in executions:
        payload = json.dumps(
            {
                "tool": name,
                "input": tool_input,
                "result": result_digest,
                "phase": "complete",
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digests.append(hashlib.sha256(payload).hexdigest()[:16])
    return digests


def duplicate_persisted_tool_calls(
    executed: Iterable[str],
    restored: Iterable[str],
) -> list[str]:
    """Fingerprints executed in the resumed leg that were already persisted.

    Empty on a clean resume. A non-empty list is the *only* defect this check
    can prove: it says a complete tool result was already on disk and was
    executed again.

    It says nothing about the reverse case. A kill that lands after a tool
    caused an external side effect but before its result was persisted leaves no
    fingerprint to compare against, and this function reports nothing about it.
    That case is classified `ambiguous_side_effect` by `classify_side_effects`,
    and the contract is explicit that the version measured here does **not**
    claim exactly-once for it (`evals/README.md` §5.4): closing it would need a
    durable tool journal with idempotency keys, which is out of scope.
    """
    persisted = set(restored)
    return [fp for fp in executed if fp in persisted]


AMBIGUOUS = "ambiguous_side_effect"
CLEAN_RESUME = "clean_resume"


def classify_side_effects(
    injected_after_side_effect: bool,
    *,
    duplicates: Sequence[str] = (),
) -> str:
    """Name what the resume can and cannot claim about repeated side effects.

    Three outcomes, deliberately ordered so the *weakest* claim wins:

    - `duplicate_persisted_tool_call`: a complete, persisted tool result was
      executed a second time. Definite, and a real defect.
    - `ambiguous_side_effect`: the kill landed inside a tool's side-effect
      window (the tool ran, its result never reached the transcript), so
      whether the effect happened twice cannot be determined from the persisted
      data. **Not counted as a duplicate and not reported as safe.**
    - `clean_resume`: no duplicate was observed AND the kill point was outside
      every side-effect window. Still only a statement about *completed* results.
    """
    if duplicates:
        return "duplicate_persisted_tool_call"
    return AMBIGUOUS if injected_after_side_effect else CLEAN_RESUME


# --- engine factory used by the recovery runner ---


def fault_engine(
    *,
    sandbox: str,
    model: str,
    api_key: str,
    injector: ScriptedModelBase | None = None,
    tool_name: str | None = None,
    tool_fault_at_call_index: int = 1,
    tool_profile: str = "core",
) -> tuple[Any, InjectionRecord]:
    """Build a real `QueryEngine` whose model transport is a fault injector.

    At least one fault must be configured: an engine with no injector would run
    the task cleanly, and a clean run recorded as a recovery is the exact
    laundering of the metric this task exists to prevent.

    The engine is built by `longline.eval.engine_factory.build_engine` (which
    does import `anthropic`), so the API key is a real required argument even
    though no request is ever made.
    """
    if injector is None and tool_name is None:
        raise InjectionError("fault_engine requires a model injector or a tool fault")

    from longline.eval.engine_factory import build_engine

    if tool_name is None:
        engine = build_engine(
            sandbox=sandbox, model=model, api_key=api_key, tool_profile=tool_profile,
        )
        if injector is None:  # pragma: no cover - guarded above
            raise InjectionError("fault_engine requires a model injector or a tool fault")
        # The shared base guarantees a non-optional `record`, so this is the
        # injector's live object rather than a copy that could go stale.
        record = injector.record
    else:
        engine = _build_engine_with_registry(
            sandbox=sandbox, model=model, api_key=api_key, tool_name=tool_name,
            fault_at_call_index=tool_fault_at_call_index, tool_profile=tool_profile,
        )
        record = InjectionRecord(fault=TOOL_FAILURE)

    if injector is not None:
        apply_model_fault(engine, injector)
    engine.injection_record = record
    return engine, record


def _build_engine_with_registry(
    *,
    sandbox: str,
    model: str,
    api_key: str,
    tool_name: str,
    fault_at_call_index: int,
    tool_profile: str,
) -> Any:
    """Same assembly as `build_engine`, but with one tool wrapped.

    Duplicated rather than monkeypatched because `build_engine` takes no
    registry factory: giving it one would widen the production eval factory's
    contract for the sake of a fault-injection helper.
    """
    import anthropic

    from longline.core.query_engine import QueryEngine
    from longline.permissions.gate import PermissionContext, PermissionMode
    from longline.prompts.builder import build_system_prompt

    registry, wrapper = fault_registry(
        sandbox, tool_name=tool_name,
        fault_at_call_index=fault_at_call_index, profile=tool_profile,
    )
    system = "\n\n".join(build_system_prompt(cwd=sandbox, model=model))
    engine = QueryEngine(
        client=anthropic.AsyncAnthropic(api_key=api_key),
        model=model,
        registry=registry,
        system_prompt=system,
        permission_ctx=PermissionContext(mode=PermissionMode.BYPASS, is_interactive=False),
        max_turns=50,
    )
    # Both are set as plain attributes rather than through an accessor: the
    # engine is built here and handed straight back, so nothing else can hold a
    # reference to it between construction and the first use.
    engine.fault_wrapper = wrapper
    return engine
