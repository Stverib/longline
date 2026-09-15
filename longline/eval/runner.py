"""Run evaluation cases and orchestrate judging.

Each case gets a fresh sandbox temp dir; E2E cases copy a named fixture into it
before the agent runs, so runs are reproducible and side-effect-free. Runs are
serial (one QueryEngine at a time) to avoid tripping API rate limits.

Sandbox lifecycle: the temp dir is removed on both the success and the
exception path. Only an explicit ``keep_sandbox_on_failure=True`` leaves a
failed case's sandbox on disk, and the path is recorded in the result so a
human can inspect the scene.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from longline.eval.engine_factory import build_engine
from longline.eval.judges import check_args, check_tools, judge_case
from longline.eval.metrics import Ratio
from longline.eval.trajectory import ToolCall, ToolExecution, extract_trajectory, infer_error_type
from longline.eval.types import EvalCase, ToolCallCase

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from longline.eval.trajectory import Trajectory


@dataclass
class CaseResult:
    """Outcome of a single evaluation case.

    Task 1 additions all carry defaults so existing construction sites and
    tests keep working unchanged.
    """

    case_id: str
    case_type: str
    passed: bool
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    text: str = ""
    errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)
    # --- telemetry (Task 1) ---
    tags: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    variant: str | None = None
    repeat_index: int = 0
    trial: int = 0
    run_id: str | None = None
    error_type: str | None = None
    tool_executions: list[ToolExecution] = field(default_factory=list)
    event_timestamps: dict[str, int] = field(default_factory=dict)
    sandbox: str | None = None
    sandbox_kept: bool = False

    @property
    def num_rounds(self) -> int:
        """Alias for `turns`, under the name the metric contract uses."""
        return self.turns

    @property
    def num_tool_calls(self) -> int:
        """Tool calls the model requested."""
        return len(self.tool_calls)

    @property
    def num_tool_calls_executed(self) -> int:
        """Tool calls actually dispatched to a tool."""
        return len(self.tool_executions)

    @property
    def num_successful_tool_calls(self) -> int:
        """Executed calls that did not report an error."""
        return sum(1 for e in self.tool_executions if not e.is_error)

    @property
    def execution_success_rate(self) -> Ratio:
        """ExecutionSuccessRate: is_error=false executions / executions."""
        return Ratio(self.num_successful_tool_calls, self.num_tool_calls_executed)

    def to_raw_dict(self) -> dict[str, object]:
        """The per-case row written to `raw.jsonl` (metric contract §2.2).

        This is the single source of truth: every aggregate in `summary.json`
        must be recomputable from these rows alone.
        """
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "tags": self.tags,
            "variant": self.variant,
            "repeat_index": self.repeat_index,
            "trial": self.trial,
            "run_id": self.run_id,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "num_rounds": self.num_rounds,
            "num_tool_calls": self.num_tool_calls,
            "num_tool_calls_executed": self.num_tool_calls_executed,
            "num_successful_tool_calls": self.num_successful_tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error_type": self.error_type,
            "errors": self.errors,
            "tool_calls": [t[0] for t in self.tool_calls],
            "tool_calls_with_args": [{"tool_name": n, "input": i} for n, i in self.tool_calls],
            "tool_executions": [
                {
                    "tool_id": e.tool_id,
                    "tool_name": e.tool_name,
                    "is_error": e.is_error,
                    "duration_ms": e.duration_ms,
                }
                for e in self.tool_executions
            ],
            "event_timestamps": self.event_timestamps,
            "judge_detail": self.detail,
        }


def _prepare_sandbox(fixtures_dir: Path, fixture: str | None) -> str:
    """Create a temp sandbox, optionally seeded from a fixture copy.

    On a fixture error the freshly created dir is removed before re-raising, so
    a misconfigured case cannot leak a temp dir on every run.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="longline-eval-"))
    if fixture:
        src = fixtures_dir / fixture
        if not src.is_dir():
            shutil.rmtree(sandbox, ignore_errors=True)
            raise FileNotFoundError(f"fixture not found: {src}")
        shutil.copytree(src, sandbox, dirs_exist_ok=True)
    return str(sandbox)


def _judge_l1(calls: list[ToolCall], case: ToolCallCase) -> dict[str, object]:
    tools_ok = check_tools(calls, case.expect_tools)
    args_ok = check_args(calls, case.expect_args)
    detail: dict[str, object] = {
        "expect_tools": case.expect_tools,
        "expect_args": case.expect_args,
        "tool_subsequence_ok": tools_ok,
        "args_ok": args_ok,
    }
    return detail


def _empty_trajectory() -> Trajectory:
    from longline.eval.trajectory import Trajectory

    return Trajectory()


def _result_from_trajectory(
    case: EvalCase,
    traj: Trajectory,
    *,
    sandbox: str,
    variant: str | None,
    repeat_index: int,
    trial: int,
    run_id: str | None,
) -> CaseResult:
    """Build the CaseResult from a completed trajectory."""
    return CaseResult(
        case_id=case.id,
        case_type="tool_call" if isinstance(case, ToolCallCase) else "e2e",
        passed=False,
        turns=traj.turns,
        input_tokens=traj.input_tokens,
        output_tokens=traj.output_tokens,
        text=traj.text,
        errors=traj.errors,
        tool_calls=traj.tool_calls,
        tags=list(case.tags),
        variant=variant,
        repeat_index=repeat_index,
        trial=trial,
        run_id=run_id,
        tool_executions=traj.tool_executions,
        event_timestamps=traj.event_timestamps,
        sandbox=sandbox,
    )


async def run_case(
    case: EvalCase,
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
    variant: str | None = None,
    repeat_index: int = 0,
    trial: int = 0,
    run_id: str | None = None,
    keep_sandbox_on_failure: bool = False,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> CaseResult:
    """Run one case and return its CaseResult.

    A failure inside the case — a raised exception, or an `ErrorEvent` from the
    query loop — is recorded on the result rather than propagated: the runner
    is expected to walk a whole suite, and a single model/runtime failure must
    not abort the run or vanish from the denominator. The only thing raised out
    of here is a fixture misconfiguration, which cannot be remedied mid-suite.

    build_engine is monkeypatchable so unit tests stay offline; `clock` is
    injectable so latency assertions are deterministic.
    """
    fixture = case.fixture  # both ToolCallCase and E2ECase carry fixture
    sandbox = _prepare_sandbox(fixtures_dir, fixture)
    marked = _MarkingClock(clock)
    result: CaseResult | None = None

    try:
        engine = build_engine(sandbox=sandbox, model=model, api_key=api_key)
        traj = await extract_trajectory(
            engine.submit(case.task, max_turns=case.max_turns), clock=marked,
        )

        result = _result_from_trajectory(
            case, traj, sandbox=sandbox, variant=variant,
            repeat_index=repeat_index, trial=trial, run_id=run_id,
        )

        if isinstance(case, ToolCallCase):
            detail = _judge_l1(traj.tool_calls, case)
            result.detail = detail
            passed = bool(detail["tool_subsequence_ok"]) and bool(detail["args_ok"])
        else:  # E2ECase
            judge_conf = case.judge
            fn = str(judge_conf["fn"])
            args = judge_conf.get("args", {})
            passed = judge_case(fn, Path(sandbox), args)
            result.detail = {"judge_fn": fn, "judge_args": args}

        result.passed = passed
        result.error_type = infer_error_type(traj, passed=passed)
    except BaseException as exc:  # harness failure: record it, never abort the suite
        result = _result_from_trajectory(
            case,
            _empty_trajectory(),
            sandbox=sandbox, variant=variant,
            repeat_index=repeat_index, trial=trial, run_id=run_id,
        )
        result.passed = False
        result.error_type = "runtime_error"
        result.errors = [*result.errors, f"{type(exc).__name__}: {exc}"]
    finally:
        outcome = result
        assert outcome is not None  # assigned on every path above
        if outcome.duration_ms is None:
            outcome.duration_ms = (marked.last() - marked.first()) / 1_000_000.0
        keep = keep_sandbox_on_failure and not outcome.passed
        if keep:
            outcome.sandbox_kept = True
            outcome.errors = [*outcome.errors, f"sandbox kept for inspection: {sandbox}"]
        else:
            shutil.rmtree(sandbox, ignore_errors=True)

    return result


class _MarkingClock:
    """Wraps a clock so the runner can read the first/last tick it observed.

    Keeps `run_case` from having to know how many times the trajectory
    extractor sampled the clock, which is what lets tests inject a plain
    sequence of ticks and still get an exact duration.

    `first()`/`final()` reuse the values the trajectory already sampled rather
    than taking fresh ones, so an injected finite tick sequence is never
    over-consumed. If nothing was sampled (the engine blew up before yielding),
    the clock is read once on each side.
    """

    __slots__ = ("_clock", "_first", "_last")

    def __init__(self, clock: Callable[[], int]) -> None:
        self._clock = clock
        self._first: int | None = None
        self._last: int | None = None

    def __call__(self) -> int:
        now = self._clock()
        if self._first is None:
            self._first = now
        self._last = now
        return now

    def first(self) -> int:
        if self._first is not None:
            return self._first
        return self._clock()

    def last(self) -> int:
        if self._last is not None:
            return self._last
        return self._clock()


async def run_suite(
    cases: Iterable[EvalCase],
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
    variant: str | None = None,
    repeat_index: int = 0,
    run_id: str | None = None,
    keep_sandbox_on_failure: bool = False,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> list[CaseResult]:
    """Run a batch of cases serially.

    Serial by contract (`evals/README.md` §4.6): parallel quality runs trip API
    rate limits and contaminate each other's latency.
    """
    results: list[CaseResult] = []
    for trial, case in enumerate(cases):
        results.append(
            await run_case(
                case,
                model=model,
                api_key=api_key,
                fixtures_dir=fixtures_dir,
                variant=variant,
                repeat_index=repeat_index,
                trial=trial,
                run_id=run_id,
                keep_sandbox_on_failure=keep_sandbox_on_failure,
                clock=clock,
            )
        )
    return results
