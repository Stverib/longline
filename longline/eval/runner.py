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

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.eval.engine_factory import build_engine
from longline.eval.judges import case_passed, judge_case_args, judge_steps
from longline.eval.metrics import Ratio
from longline.eval.trajectory import ToolCall, ToolExecution, extract_trajectory, infer_error_type
from longline.eval.types import ABSTENTION_TAG, EvalCase, ToolCallCase, resolve_fixture

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from longline.eval.trajectory import Trajectory

# Resolves a per-case tool profile; `None` means "use the suite-level default".
ProfileForCase = "Callable[[EvalCase], str]"


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
    # Every distinct model the transport said it served, in first-seen order.
    # Empty means the transport never said; see `model_provenance` for how that
    # is reported, and why it is not silently replaced by the requested model.
    served_models: list[str] = field(default_factory=list)

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

    # --- Task 2: the four tool-calling metrics, denominator by denominator ---
    #
    # Each of these reads its counters out of `detail` and owns its own
    # numerator and denominator. `to_raw_dict()` writes the same counters to
    # raw.jsonl, so an aggregate can be recomputed without this object — which
    # is the contract's definition of a valid number (`evals/README.md` §3).

    def _detail_section(self, section: str) -> dict[str, object]:
        value = self.detail.get(section)
        return value if isinstance(value, dict) else {}

    @property
    def _step_detail(self) -> dict[str, object]:
        return self._detail_section("steps")

    @property
    def _arg_detail(self) -> dict[str, object]:
        return self._detail_section("args")

    @property
    def steps_completed(self) -> bool:
        """True when every expected decision step was satisfied, in order."""
        return bool(self._step_detail.get("all_steps_matched", False))

    @property
    def is_abstention_case(self) -> bool:
        """True for a case whose correct action is to call NO tool.

        BFCL devotes roughly a quarter of its set to this class (240 Irrelevance
        + 882 Live Irrelevance); a suite where every case demands a call rewards
        an agent that always calls something. Such a case declares zero expected
        steps, which makes `all_steps_matched` vacuously true (`all([])`) and
        would therefore pass even when the agent called a pile of irrelevant
        tools. This property is what lets the aggregation apply the real rule.
        """
        return ABSTENTION_TAG in self.tags

    @property
    def abstained(self) -> bool:
        """True when no tool was called at all -- the abstention pass condition."""
        return self.num_tool_calls == 0

    @property
    def abstention_ok(self) -> bool:
        """The pass condition for an abstention case: call nothing, and no extra calls."""
        return self.steps_completed and self.abstained and self.num_extra_tool_calls == 0

    @staticmethod
    def _count(detail: dict[str, object], key: str) -> int:
        """Read an integer counter out of a `judge_detail` section.

        The detail dict is `dict[str, object]` because it also carries lists and
        strings, so the type has to be narrowed before the value is used as a
        ratio numerator or denominator. A missing or non-integer entry reads as
        0, which keeps `Ratio` constructible; a genuinely absent counter shows
        up as an unmeasured ratio rather than a crash mid-report.
        """
        value = detail.get(key, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def num_extra_tool_calls(self) -> int:
        """Calls that matched no expected step (precision's invalid half)."""
        return self._count(self._step_detail, "num_extra_calls")

    @property
    def num_matched_tool_calls(self) -> int:
        """Calls that satisfied a step. Derived, never stored twice."""
        return self.num_tool_calls - self.num_extra_tool_calls

    @property
    def num_arg_checked_calls(self) -> int:
        return self._count(self._arg_detail, "checked_calls")

    @property
    def num_arg_correct_calls(self) -> int:
        return self._count(self._arg_detail, "correct_calls")

    @property
    def num_arg_checked_fields(self) -> int:
        return self._count(self._arg_detail, "checked_fields")

    @property
    def num_arg_correct_fields(self) -> int:
        return self._count(self._arg_detail, "correct_fields")

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
            "served_models": self.served_models,
        }


def served_models_in(items: Iterable[object]) -> list[str]:
    """Every distinct served model across a set of case results, first-seen order.

    Accepts anything carrying a `served_models` list -- a `CaseResult` from an
    ordinary suite, or a `VariantRun` from the multi-agent one -- so the run
    metadata reports provenance through one path regardless of which runner
    produced the rows. A per-runner copy of this loop is a per-runner way for
    the metadata to disagree with the artifact it describes.
    """
    out: list[str] = []
    for item in items:
        for name in getattr(item, "served_models", []) or []:
            if name not in out:
                out.append(name)
    return out


def model_provenance(requested: str, served: list[str]) -> dict[str, object]:
    """Where the run's model name came from, and whether it agrees with the ask.

    A report that prints `model: X` is making a claim about what ran. Three
    cases, and they are not interchangeable:

    - `measured` -- the transport named the model it served. `model_served` is
      evidence. `model_matches_request` is False when the gateway substituted
      a different model, which is a fact a reader of the numbers needs; the
      Anthropic-compatible gateway this project uses ignores the requested
      `model` field entirely and always serves its own.
    - `requested_unverified` -- the transport said nothing. `model_served` is
      empty and `model_matches_request` is None, because "we did not check" is
      not the same as "it matched". Recording the requested name here would be
      a claim with no evidence behind it.
    - a single name that equals the request -- `measured`, matched True.

    A list rather than one name, because a gateway can fail over between turns
    and a run answered by two different models has to be able to say so.
    """
    return {
        "model_requested": requested,
        "model_served": list(served),
        "model_source": "measured" if served else "requested_unverified",
        "model_matches_request": (requested in served) if served else None,
    }


def _prepare_sandbox(fixtures_dir: Path, fixture: str | None, case_id: str = "<unknown>") -> str:
    """Create a temp sandbox, optionally seeded from a fixture copy.

    The fixture name is resolved through `resolve_fixture`, which rejects
    absolute paths and `..` escapes before anything is copied. The runner is
    the last gate before a case's data touches the filesystem, so the check
    lives here as well as at load time: a caller that builds cases
    programmatically never goes through `load_cases`.

    On a fixture error the freshly created dir is removed before re-raising, so
    a misconfigured case cannot leak a temp dir on every run.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="longline-eval-"))
    if fixture:
        src = resolve_fixture(fixtures_dir, fixture, case_id=case_id)
        if not src.is_dir():
            shutil.rmtree(sandbox, ignore_errors=True)
            raise FileNotFoundError(f"fixture not found: {src}")
        # 忽略 __pycache__ / .pytest_cache: 在 fixtures/ 里跑过测试留下的字节码
        # 缓存会被 copytree 原样搬进沙箱. 陈旧的 .pyc 配上比它更旧的源文件时间戳,
        # 会让 Python 认为缓存仍然有效而不重新编译 -- 沙箱里实际执行的可能不是
        # 当前源码. .gitignore 挡得住它们入库, 挡不住 copytree.
        shutil.copytree(src, sandbox, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache",
        ))
    return str(sandbox)


def _judge_l1(calls: list[ToolCall], case: ToolCallCase) -> dict[str, object]:
    """Per-step and per-field detail for a tool-call case.

    Every count the four metrics need is recorded here, separately. The old
    shape fused tool selection and argument correctness into one boolean, which
    is exactly why the legacy `l1_tool_accuracy` could not be split into
    independent denominators (`evals/README.md` §2).

    `matched_calls` is derived rather than stored: it is
    `num_tool_calls - num_extra_calls`, and storing the same fact twice is how
    the two copies eventually disagree.
    """
    steps = judge_steps(calls, case.accepted_tool_steps)
    args = judge_case_args(calls, case.expect_args)
    return {
        "accepted_tool_steps": case.accepted_tool_steps,
        "max_extra_calls": case.max_extra_calls,
        "expect_args": case.expect_args,
        "steps": steps.to_detail(),
        "args": args.to_detail(),
    }


def _tool_case_passed(detail: dict[str, object]) -> bool:
    """Case-level pass: every expected decision step completed AND args correct.

    Note this is *not* the same question as any single metric: the case boolean
    still exists for triage, but every reported number is read off the separate
    counts in `detail`, so extra calls move precision without silently rewriting
    the case outcome.
    """
    steps = detail["steps"]
    args = detail["args"]
    assert isinstance(steps, dict) and isinstance(args, dict)
    return bool(steps["all_steps_matched"]) and bool(args["all_calls_correct"])


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
        served_models=list(traj.served_models),
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
    tool_profile: str = "core",
    forbidden: Iterable[str] = (),
    prompt_variant: str = "baseline",
) -> CaseResult:
    """Run one case and return its CaseResult.

    Failure handling is deliberately **layered**, because the two kinds of
    fault mean different things for the numbers:

    - **Infrastructure faults propagate.** If `build_engine` cannot construct a
      harness, nothing about the case was measured. Recording that as a case
      failure would put an unmeasured case into the denominator and quietly
      depress the success rate. A broken harness must abort loudly so the run
      is discarded, not resumed with a corrupted rate.
    - **Case faults are recorded.** Once the engine exists and the case is
      actually running, an `Exception` is a fact about this case: it is recorded
      on the `CaseResult` with `error_type="runtime_error"` and the suite
      continues. The plan's §5.1 rule — a case whose baseline failed still
      counts in the denominator — requires that such a case be kept in the
      data rather than dropped.
    - **Cancellation propagates.** `KeyboardInterrupt` and
      `asyncio.CancelledError` are statements about the *run*, not the case.
      Recording them would turn Ctrl-C into a case failure and let the suite
      march on, so the recording arm catches `Exception` only and lets these
      through to propagate.

    The sandbox is cleaned on every path, including the propagating ones.

    build_engine is monkeypatchable so unit tests stay offline; `clock` is
    injectable so latency assertions are deterministic.
    """
    fixture = case.fixture  # both ToolCallCase and E2ECase carry fixture
    sandbox = _prepare_sandbox(fixtures_dir, fixture, case_id=case.id)

    # Infra layer: built OUTSIDE the recording try, so a failure here
    # propagates. Only the sandbox is guaranteed cleaned up. BaseException is
    # right here (unlike the recording path below): this arm only cleans up and
    # re-raises, so it cannot swallow a cancellation — it just ensures Ctrl-C
    # during engine construction does not leave a temp dir behind.
    try:
        engine_kwargs: dict[str, Any] = {
            "sandbox": sandbox,
            "model": model,
            "api_key": api_key,
            "tool_profile": tool_profile,
        }
        # Precedence: a ToolCallCase's own forbidden_tools REPLACES the
        # suite-level `forbidden` wholesale, and an EMPTY case-level list
        # means no constraint at case level -- it does not fall back to the
        # suite value. The suite-level param applies only to E2ECase, which
        # carries no forbidden field. Keyword-gated: an empty list is
        # behaviorally a no-op, so pre-existing offline fakes of build_engine
        # (which predate this parameter) keep working unchanged.
        case_forbidden = (
            case.forbidden_tools if isinstance(case, ToolCallCase) else list(forbidden)
        )
        if list(case_forbidden):
            engine_kwargs["forbidden"] = list(case_forbidden)
        # Same keyword-gating as `forbidden`: the default is a no-op, so fakes
        # of build_engine written before this parameter existed keep working.
        if prompt_variant != "baseline":
            engine_kwargs["prompt_variant"] = prompt_variant
        engine = build_engine(**engine_kwargs)
    except BaseException:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise

    marked = _MarkingClock(clock)
    result: CaseResult | None = None

    # Case layer: from here on, failures are recorded, not propagated.
    try:
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
            passed = _tool_case_passed(detail)
        else:  # E2ECase
            passed, check_detail = case_passed(
                case.checks, Path(sandbox), mode=case.checks_mode,
            )
            result.detail = {
                "checks_mode": case.checks_mode,
                "checks": check_detail,
                # Kept for readers written against the single-judge shape:
                # the first check is the artifact assertion in every case that
                # has more than one, and `checks` is the authoritative list.
                "judge_fn": check_detail[0]["fn"] if check_detail else None,
                "judge_args": check_detail[0]["args"] if check_detail else {},
            }

        result.passed = passed
        result.error_type = infer_error_type(traj, passed=passed)
    except Exception as exc:  # case failure: record it, never abort the suite
        # Only Exception is a statement about this case. KeyboardInterrupt and
        # asyncio.CancelledError are statements about the RUN — recording them
        # here would turn Ctrl-C into a "case failure" and let the suite march
        # on, so an operator could never actually stop it. They fall through to
        # the finally (sandbox cleanup) and then propagate.
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
        if outcome is None:
            # Cancellation path: no result to annotate, but the sandbox must
            # still go, so Ctrl-C does not litter temp dirs.
            shutil.rmtree(sandbox, ignore_errors=True)
        else:
            if outcome.duration_ms is None:
                outcome.duration_ms = (marked.last() - marked.first()) / 1_000_000.0
            keep = keep_sandbox_on_failure and not outcome.passed
            if keep:
                outcome.sandbox_kept = True
                outcome.errors = [*outcome.errors, f"sandbox kept for inspection: {sandbox}"]
            else:
                shutil.rmtree(sandbox, ignore_errors=True)

    assert result is not None  # unreachable when cancelled; the raise above wins
    return result


class _MarkingClock:
    """Wraps a clock so the runner can read the first/last tick it observed.

    Keeps `run_case` from having to know how many times the trajectory
    extractor sampled the clock, which is what lets tests inject a plain
    sequence of ticks and still get an exact duration.

    `first()`/`last()` reuse the values the trajectory already sampled rather
    than taking fresh ones, so an injected finite tick sequence is never
    over-consumed. If nothing was sampled (the case raised before yielding),
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
    tool_profile: str = "core",
    forbidden: Iterable[str] = (),
    profile_for_case: Callable[[EvalCase], str] | None = None,
    skip_case_ids: frozenset[str] = frozenset(),
    sink: Callable[[CaseResult], None] | None = None,
    pace_seconds: float = 0.0,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    prompt_variant: str = "baseline",
) -> list[CaseResult]:
    """Run a batch of cases serially.

    Serial by contract (`evals/README.md` §4.6): parallel quality runs trip API
    rate limits and contaminate each other's latency.

    `profile_for_case` lets one suite mix families — a web case and a notebook
    case need different registries, and running either against the wrong one
    would score an unsatisfiable task as a model error.

    `sink`, when given, is called with each result as it completes. The caller
    uses it to append to `raw.jsonl` so an interrupted run keeps what it
    finished -- a paid suite can outlast an account's usage window, and losing
    every completed case to one interruption makes the run unaffordable rather
    than merely slow.

    `skip_case_ids` omits cases already answered by a previous attempt. The
    `trial` index still counts the skipped ones, so a resumed run's rows carry
    the same `trial` a fresh run would have given them.

    `pace_seconds` puts a pause BETWEEN two API calls. The serial contract
    (`evals/README.md` §4.6) keeps quality runs from overlapping, but a gateway
    can still reject back-to-back requests from one account, and a rejected
    request produces a row that measured nothing. The pause is taken only when
    another case is actually about to run, so it never precedes the first call
    and never separates two calls that `skip_case_ids` removed -- the point is
    to keep real calls apart, not to add wall-clock to a resumed run.

    `sleep` is injected so the pacing is exactly assertable without sleeping.
    """
    sleeper = sleep if sleep is not None else asyncio.sleep
    results: list[CaseResult] = []
    # Whether the previous iteration actually ran a case. Pacing separates two
    # real API calls, so it is taken before a case only when one just ran --
    # never before the first, and never across a case `skip_case_ids` removed.
    ran_previous = False
    for trial, case in enumerate(cases):
        if case.id in skip_case_ids:
            continue
        if ran_previous and pace_seconds > 0.0:
            await sleeper(pace_seconds)
        ran_previous = True
        result = await run_case(
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
            tool_profile=(
                profile_for_case(case) if profile_for_case is not None else tool_profile
            ),
            # For a ToolCallCase the case-level forbidden_tools REPLACES this
            # suite-level value (an empty case list is no constraint at all);
            # the suite-level param only reaches E2ECase, via run_case's
            # default. The replacement happens inside run_case.
            forbidden=forbidden,
            prompt_variant=prompt_variant,
        )
        results.append(result)
        if sink is not None:
            sink(result)
    return results
