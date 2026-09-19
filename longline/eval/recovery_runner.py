"""Fault-injection runner: RuntimeRecoveryRate and SessionResumeRate.

=== What this measures (evals/README.md §5.4, plan §4.4) ===

```text
RuntimeRecoveryRate = 五类运行时故障恢复成功数 / 50
SessionResumeRate   = 进程中断后恢复成功数 / 10
```

Six fault classes, ten runs each. Five are injected into the in-process loop
(429, 529, tool failure, output truncate, context overflow); the sixth, Process
Kill, terminates a real subprocess at a saved turn checkpoint and resumes in a
fresh interpreter.

=== What "recovered" means, and what it refuses to mean ===

A run counts as a successful recovery only when **all** of these hold:

1. `fault_injected` -- the injector's own counter says the fault fired, and the
   event stream independently corroborates it (`InjectionRecord.proof` records
   which observation established it, per class);
2. `retry_count > 0` -- the production recovery path actually ran. This is a
   separate condition from (1) on purpose: a line that never fires is a
   different defect from a fault that fires and is not handled;
3. the final deterministic judge passes;
4. for Process Kill only, the checkpoint loads and the transcript validates
   structurally.

`success` is computed from those four, never assigned. The most damaging
possible outcome for this task is a green dataset in which nothing was ever
injected -- so `fault_injected` is a term in the success expression rather than
a field someone is trusted to read.

=== Do not claim exactly-once ===

`duplicate_persisted_tool_calls` counts fingerprints of **already-persisted
complete tool results** that the resumed leg executed again. An empty list is
the only thing this check proves. A kill that lands after a tool produced an
external side effect but before its result reached the transcript leaves nothing
to compare against, and is classified `ambiguous_side_effect` by
`faults.classify_side_effects` -- **not counted as a duplicate, and not reported
as safe**. Proving anything stronger would need a durable tool journal with
idempotency keys; the contract is explicit that this is out of scope
(`evals/README.md` §5.4), and this module does not silently upgrade the claim.

=== Offline vs on-model ===

`run_recovery_case` with the default `model=None` runs the **offline protocol**:
a real `QueryEngine`, real tools, real `query_loop`, real judge, and a scripted
model. That is deterministic and free, and it is what the test suite and
`evals/recovery.jsonl` describe.

Passing a real `model` id runs the same case against the live API. It is
supported and the assertions are identical -- a live run is not trusted more
than an offline one, it is just noisier. Neither path is a substitute for the
other, and a reported number must say which one produced it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.eval.failpoints import terminate_and_reap
from longline.eval.faults import (
    ALL_FAULTS,
    PROCESS_KILL,
    RUNTIME_FAULTS,
    TOOL_FAILURE,
    InjectionError,
    InjectionRecord,
    ModelFaultInjector,
    ScriptedModelBase,
    ToolCallingModel,
    ToolFaultWrapper,
    apply_model_fault,
    assert_engine_uses_injector,
    classify_side_effects,
    duplicate_persisted_tool_calls,
    fingerprint_tool_calls,
    sha256_file,
)
from longline.eval.judges import case_passed
from longline.eval.metrics import Ratio, mean, percentile
from longline.eval.runner import CaseResult, _prepare_sandbox
from longline.eval.types import CaseParseError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from longline.eval.recovery import RecoveryCase

# Root for the run's temp claude_dirs. Always under the system temp dir, never
# under the user's home: `claude_dir` is what `get_sessions_dir()` joins to, and
# a mistake here would write the benchmark's sessions into the operator's real
# `~/.claude`.
CLAUDE_DIR_PREFIX = "longline-recovery-claude-"

# Wall-clock ceiling for one worker phase. A resume that hangs is a failed case
# with a reason, not a suite that never finishes.
WORKER_TIMEOUT_S = 120

# Marker the Process-Kill transcript plants; the resumed leg has to reproduce
# it, so it is part of the fixture contract rather than a magic string in a judge.
CHECKPOINT_VALUE = "alpha-7f3c"

# The raw marker as it was PERSISTED in the checkpoint transcript's tool result.
# Deliberately the unformatted `value=` form rather than the answer string: the
# answer is derived from the fixture file after the resume, and a transcript
# holding the finished answer would let a leg that only replayed its history
# pass a judge meant to test whether it could still read the file
# (`recovery_worker.derive_answer`).
CHECKPOINT_MARKER = f"value={CHECKPOINT_VALUE}"

# The answer a recovered run must produce, and the form the assembled answer
# file is checked against. The worker FORMATS the raw marker into this string.
STABLE_FACT = f"marker={CHECKPOINT_VALUE}"


@dataclass
class RecoveryRun:
    """One fault injection, with every field the contract asks for.

    The six fields in `PER_CASE_FIELDS` are the contract's, by name. The rest
    (`injection_proof`, `offline`, `notes`) exist so a reader can audit *how*
    the verdict was reached rather than only what it was.
    """

    case_id: str
    fault: str
    fault_injected: bool
    retry_count: int
    checkpoint_loaded: bool
    transcript_repaired: bool
    duplicate_persisted_tool_calls: int
    recovery_latency_ms: float
    passed: bool
    success: bool = False
    judge_detail: list[dict[str, Any]] = field(default_factory=list)
    injection_proof: str = ""
    injected_at_call_index: int = 0
    model_call_index: int = 0
    # non-empty only for Process Kill
    side_effect_classification: str = ""
    ambiguous_side_effect: bool = False
    structural_errors: list[str] = field(default_factory=list)
    task_states: dict[str, str] = field(default_factory=dict)
    offline: bool = True
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        """Per-case row, written to `raw.jsonl` and recomputable from it alone."""
        return {
            "case_id": self.case_id,
            "fault": self.fault,
            "fault_injected": self.fault_injected,
            "retry_count": self.retry_count,
            "checkpoint_loaded": self.checkpoint_loaded,
            "transcript_repaired": self.transcript_repaired,
            "duplicate_persisted_tool_calls": self.duplicate_persisted_tool_calls,
            "recovery_latency_ms": self.recovery_latency_ms,
            "passed": self.passed,
            "success": self.success,
            "injection_proof": self.injection_proof,
            "injected_at_call_index": self.injected_at_call_index,
            "model_call_index": self.model_call_index,
            "side_effect_classification": self.side_effect_classification,
            "ambiguous_side_effect": self.ambiguous_side_effect,
            "structural_errors": self.structural_errors,
            "task_states": self.task_states,
            "offline": self.offline,
            "judge_detail": self.judge_detail,
            "notes": self.notes,
        }


# The contract's per-case field list, in one place so a test can assert that the
# row carries all six. A missing one would make summary.json unobtainable from
# raw.jsonl, which is the contract's definition of an invalid number.
PER_CASE_FIELDS: tuple[str, ...] = (
    "fault_injected",
    "retry_count",
    "checkpoint_loaded",
    "transcript_repaired",
    "duplicate_persisted_tool_calls",
    "recovery_latency_ms",
)


def recovery_succeeded(run: RecoveryRun) -> bool:
    """The success predicate, spelled out where it can be read and tested.

    Four independent conditions, all required. `fault_injected` leads because
    every other term is meaningless without it: a run that injected nothing and
    passed the judge is an ordinary success, and counting it as a recovery would
    inflate `RuntimeRecoveryRate` by the share of cases that are simply easy.

    `retry_count` is separate from `fault_injected` rather than folded into it,
    because "the fault never fired" and "the fault fired and the loop gave up"
    are different defects and a single boolean could not tell a reader which
    one it was looking at.
    """
    if not run.fault_injected:
        return False
    if run.retry_count <= 0:
        return False
    if not run.passed:
        return False
    if run.fault == PROCESS_KILL:
        return run.checkpoint_loaded and not run.structural_errors
    return True


@dataclass
class RecoverySummary:
    """The two headline metrics plus their per-class breakdown."""

    runtime_recovery_rate: Ratio
    session_resume_rate: Ratio
    by_fault: dict[str, Ratio]
    recovery_latency_ms: dict[str, float | None]
    failures: list[dict[str, object]] = field(default_factory=list)

    @property
    def runtime_numerator(self) -> int:
        return self.runtime_recovery_rate.numerator

    @property
    def runtime_denominator(self) -> int:
        return self.runtime_recovery_rate.denominator

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_recovery_rate": self.runtime_recovery_rate.to_dict(),
            "session_resume_rate": self.session_resume_rate.to_dict(),
            "by_fault": {k: v.to_dict() for k, v in self.by_fault.items()},
            "recovery_latency_ms": self.recovery_latency_ms,
            "failures": self.failures,
        }


def aggregate_recovery(runs: Sequence[RecoveryRun]) -> RecoverySummary:
    """Collapse per-run rows into the two contract metrics.

    The split is by **fault class**, not by whether the run happened to be
    offline: `RuntimeRecoveryRate` covers the five in-process classes and
    `SessionResumeRate` covers Process Kill alone. Mixing them would make a
    6/10 Process Kill run indistinguishable from a 6/10 truncate run, and the
    contract requires them reported as two numbers.

    A class with zero runs reports ``0/0`` rather than a number, matching
    `metrics.Ratio`: "not measured" must never render as "0%".
    """
    runtime = [r for r in runs if r.fault in RUNTIME_FAULTS]
    resume = [r for r in runs if r.fault == PROCESS_KILL]

    by_fault: dict[str, Ratio] = {}
    for fault in ALL_FAULTS:
        rows = [r for r in runs if r.fault == fault]
        by_fault[fault] = Ratio(sum(1 for r in rows if r.success), len(rows))

    latencies: dict[str, float | None] = {}
    for fault in ALL_FAULTS:
        values = [r.recovery_latency_ms for r in runs if r.fault == fault and r.success]
        latencies[fault] = mean(values) if values else None

    return RecoverySummary(
        runtime_recovery_rate=Ratio(sum(1 for r in runtime if r.success), len(runtime)),
        session_resume_rate=Ratio(sum(1 for r in resume if r.success), len(resume)),
        by_fault=by_fault,
        recovery_latency_ms=latencies,
        failures=[
            {
                "case_id": r.case_id,
                "fault": r.fault,
                "fault_injected": r.fault_injected,
                "retry_count": r.retry_count,
                "passed": r.passed,
                "notes": r.notes,
            }
            for r in runs
            if not r.success
        ],
    )


# --- judging ---


def judge_recovered_answer(checks: list[dict[str, Any]], seed: str) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the deterministic judges to the *answer*, not to a file.

    `E2ECase`'s judges all read the sandbox filesystem, which is right for a
    task whose artifact is a file. Here the artifact is the answer the model
    produced, so the judges are evaluated against a scratch sandbox holding only
    that answer. Reusing `case_passed` rather than writing a second judge
    dispatcher keeps the four check functions (and their mutations) in one place.

    A `passed` of True on an empty answer is impossible: `file_content` requires
    the file to exist, and the file is written only from the model's text.
    """
    payload = {"answer": seed}
    scratch = Path(tempfile.mkdtemp(prefix="recovery-judge-"))
    try:
        (scratch / "answer.txt").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return case_passed(checks, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- the in-process classes ---


def _recovery_checks(case: RecoveryCase) -> list[dict[str, Any]]:
    """The deterministic judges for one case: its own checks, plus the answer."""
    return list(case.checks)


async def _run_runtime_case(
    case: RecoveryCase,
    *,
    model: str | None,
    api_key: str,
    fixtures_dir: Path,
    variant: str | None,
) -> RecoveryRun:
    """One of the five in-process fault classes: inject, drive, judge."""
    sandbox = _prepare_sandbox(fixtures_dir, case.fixture, case_id=case.id)
    started = time.perf_counter()
    try:
        # The fixture the task names is created HERE, before the run, so the
        # agent's tool call finds a real file. The task text carries the path;
        # nothing about its content is staged beside it.
        seed_fixture(Path(sandbox), case=case, value=CHECKPOINT_VALUE)
        # Two injection dialects. A `tool_failure` case has no model-stream fault
        # to script -- the fault is a tool returning `is_error=true` -- so its
        # injector is the tool wrapper and the model is served normally.
        tool_failure = case.fault == TOOL_FAILURE
        tool_wrapper: ToolFaultWrapper | None = None
        injector = _unfaulted_model(case) if tool_failure else ModelFaultInjector(
            fault=case.fault,
            answer=case.answer,
            at_call_indices=tuple(case.inject_at_call_indices),
        )
        engine = _build_fault_engine(
            sandbox=sandbox,
            model=model,
            api_key=api_key,
            injector=None if tool_failure else injector,
            tool_name=case.fault_tool if tool_failure else None,
            tool_profile=case.tool_profile,
        )
        if tool_failure:
            if case.fault_tool is None:
                raise InjectionError(f"{case.id}: tool_failure case needs a tool name")
            tool_wrapper = engine.fault_wrapper
            if tool_wrapper is None:  # pragma: no cover - defensive
                raise InjectionError(f"{case.id}: engine has no fault wrapper")
        else:
            assert_engine_uses_injector(engine, injector)

        # The scripted model drives the engine in BOTH dialects. For the tool
        # class that model contains no fault (the tool does), but it is still
        # what answers -- otherwise `engine.submit` would reach the real SDK,
        # and an "offline" tool-failure case would silently be a live one.
        apply_model_fault(engine, injector, patch_sleep=True)
        if model is None:
            engine = _OfflineEngine(engine)

        # A `context_overflow` case needs an ENTRY-POINT MESSAGE for the loop to
        # send, plus enough history that reactive compact has something to
        # compress: a 413 that arrives on an empty transcript is not a context
        # overflow, and the compact would be a no-op that reads as a failed
        # recovery. `engine.submit()` appends the entry message itself, so only
        # the seed history goes in here.
        if case.seed_history:
            engine.messages.extend(seed_messages(case))

        result = await _drive_engine(
            engine, case, sandbox,
            # A live run lets `submit` build its summariser from the real model.
            # Offline there is no model for it to use, so the scripted one is
            # supplied explicitly rather than poked onto the engine.
            compact_fn=None if model is not None else _scripted_compact_fn(),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _assemble_runtime_run(
            case, result, injector=injector, tool_wrapper=tool_wrapper,
            elapsed_ms=elapsed_ms, variant=variant, offline=model is None,
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def seed_messages(case: RecoveryCase) -> list[Any]:
    """The scripted history a case starts from, as `Message` objects."""
    from longline.models.content_blocks import TextBlock
    from longline.models.messages import AssistantMessage, UserMessage

    out: list[Any] = []
    for entry in case.seed_history:
        if entry["role"] == "assistant":
            out.append(AssistantMessage(content=[TextBlock(text=entry["content"])]))
        else:
            out.append(UserMessage(content=entry["content"]))
    return out


def _scripted_compact_fn() -> Any:
    """A summariser for the reactive-compact path, offline.

    Reactive compact summarises the transcript with a SECOND model call. A live
    run makes that call against the API; offline it has to be scripted, and the
    script deliberately returns real summary text rather than nothing -- an
    empty summary would make the compact produce a shorter transcript for the
    wrong reason, and the case would report a compact that never happened.
    """

    async def _compact_model(**kwargs: Any) -> Any:
        from longline.core.events import TextDelta, TurnComplete
        from longline.models.messages import Usage

        _ = kwargs
        yield TextDelta(text="[compacted: earlier turns summarised for the resume]")
        yield TurnComplete(stop_reason="end_turn", usage=Usage())

    return _compact_model


def seed_fixture(sandbox: Path, *, case: RecoveryCase, value: str) -> Path:
    """Create the file the case's task names, inside the sandbox.

    The path comes from the TASK (via `extract_fixture_path`), not from a field
    beside it, so the file that exists and the file the agent is told to read
    cannot be two different paths. The `<cwd>` placeholder is substituted with
    the sandbox, which is the working directory the engine's tools run in.
    """
    from longline.eval.recovery import extract_fixture_path

    relative = extract_fixture_path(case.task)
    if relative is None:  # pragma: no cover - the loader rejects this
        raise InjectionError(f"{case.id}: task names no fixture path")
    target = sandbox / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{value}\n", encoding="utf-8")
    return target


def _unfaulted_model(case: RecoveryCase) -> ToolCallingModel:
    """The scripted agent for the tool-failure class: it calls the tool, sees
    the error, and retries.

    Not a blank `ScriptedModel` that answers straight away. The success
    condition for this class is "the agent ADAPTS or retries", and an agent that
    never called the tool could satisfy a text judge while proving nothing. The
    tool path is what makes the fault reachable, and the retry is what makes the
    recovery observable.

    The tool path is taken from the case's TASK, so the call the script makes is
    the call the task asked for and not a value smuggled in beside it.

    The input is built for the fault tool: `Bash` takes a `command` and runs it
    with the sandbox as its cwd, while `Read` takes an absolute `file_path`. The
    tool the case names decides which shape is used, and a tool this does not
    know how to call raises rather than producing a call the tool would reject
    -- a rejected call is indistinguishable from a fault in the report.
    """
    from longline.eval.recovery import extract_fixture_path

    relative = extract_fixture_path(case.task)
    if case.fault_tool is None or relative is None:
        raise InjectionError(
            f"{case.id}: a tool_failure case must name both a fault_tool and a "
            "readable path in its task; the scripted agent has nothing to call"
        )
    return ToolCallingModel(
        tool_name=case.fault_tool,
        tool_input=_tool_input_for(case.fault_tool, relative),
        answer=case.answer,
    )


def _tool_input_for(tool_name: str, relative: str) -> dict[str, Any]:
    """The arguments a scripted call to `tool_name` needs, for a relative path.

    Kept to the two tools a recovery case plausibly names. Anything else raises
    instead of guessing: a wrong argument shape would make the tool return an
    error of its own, which the report would then read as the injected fault.
    """
    if tool_name == "Bash":
        # `cat` rather than a bare path: Bash expects a command, and printing the
        # file is what the task asks the agent to do.
        return {"command": f"cat {relative}"}
    if tool_name == "Read":
        # `Read` takes an absolute path and the sandbox is not known here, so the
        # task's `<cwd>` placeholder is what makes this absolute. A relative path
        # here would be resolved against the PROCESS cwd and fail.
        raise InjectionError(
            "Read needs an absolute file_path, which the task's relative path "
            "cannot supply; use Bash for a tool_failure case"
        )
    raise InjectionError(f"no scripted call shape for tool {tool_name!r}")


class _OfflineEngine:
    """Marker wrapper: the engine's model transport is a scripted injector.

    Exists so the run's `offline` flag is a fact about the engine that ran
    rather than about which branch a caller took -- a live run whose model
    argument was dropped would otherwise be reported as offline.
    """

    offline = True

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


async def _drive_engine(
    engine: Any,
    case: RecoveryCase,
    sandbox: str,
    *,
    compact_fn: Any | None = None,
) -> CaseResult:
    """Run the engine and grade the artifact it produced.

    Identical in both modes: `submit()`, then the same `case_passed` dispatch
    every other E2E case uses. The recovery changes what the model *did*, not
    how its output is graded.

    `compact_fn` is passed through to `submit(auto_compact_fn=...)`. It is a
    parameter rather than an attribute poked onto the engine because the engine
    builds its summariser internally, and a monkeypatched attribute would be a
    claim about a call path nobody had verified.
    """
    from longline.eval.trajectory import extract_trajectory

    stream = engine.submit(case.task, max_turns=case.max_turns, auto_compact_fn=compact_fn)
    traj = await extract_trajectory(stream)
    # The agent's final text IS the artifact. Written before judging, so a run
    # that produced no text fails `file_content` rather than being graded
    # against an empty file that happens to exist.
    (Path(sandbox) / "answer.txt").write_text(traj.text, encoding="utf-8")
    passed, detail = case_passed(case.checks, Path(sandbox), mode=case.checks_mode)
    result = CaseResult(
        case_id=case.id, case_type="recovery", passed=passed, tags=list(case.tags),
        turns=traj.turns, input_tokens=traj.input_tokens, output_tokens=traj.output_tokens,
        text=traj.text,
    )
    result.detail = {"checks_mode": case.checks_mode, "checks": detail}
    result.tool_calls = traj.tool_calls
    result.tool_executions = traj.tool_executions
    return result


def _build_fault_engine(
    *,
    sandbox: str,
    model: str | None,
    api_key: str,
    injector: ScriptedModelBase | None,
    tool_name: str | None,
    tool_profile: str,
) -> Any:
    """A real `QueryEngine` with a fault injector attached (see `faults.py`).

    The API key is a real required argument even offline, because the engine
    constructs a real client -- no request is ever made with it, but passing a
    placeholder means the "offline" claim rests on the injector rather than on
    a missing credential.
    """
    from longline.eval.faults import fault_engine

    engine, _record = fault_engine(
        sandbox=sandbox,
        model=model or "offline-recovery",
        api_key=api_key,
        injector=injector,
        tool_name=None if tool_name is None else tool_name,
        tool_profile=tool_profile,
    )
    return engine


def _assemble_runtime_run(
    case: RecoveryCase,
    result: CaseResult,
    *,
    injector: ScriptedModelBase,
    tool_wrapper: ToolFaultWrapper | None,
    elapsed_ms: float,
    variant: str | None,
    offline: bool,
) -> RecoveryRun:
    """Turn the engine run into a `RecoveryRun`, deriving the counts.

    The retry count is the count of calls the model was ASKED for beyond the
    first, which for a retried fault is exactly the number of times the loop
    came back. It is read off the injector's own call counter rather than
    inferred from `turns`, because `turns` counts completed turns and the
    recovery path deliberately rewinds that counter.

    The tool-fault class reports `retry_count` from the wrapper's call count,
    which is the same quantity for that class: the loop retried because the
    model asked again after seeing `is_error=true`.
    """
    notes: list[str] = []
    record = injector.record
    if tool_wrapper is not None:
        record = InjectionRecord(fault=TOOL_FAILURE)
        tool_wrapper.record_into(record)
        retry_count = max(0, tool_wrapper.calls - tool_wrapper.fault_at_call_index)
        corroborated = sum(1 for e in result.tool_executions if e.is_error) >= record.attempts
        if not corroborated:
            notes.append(
                "injector counter and the trajectory's errored executions disagree; "
                "the weaker of the two is used"
            )
            record.injected = record.injected and corroborated
    else:
        retry_count = max(0, injector.record.call_index - 1)

    if not record.injected:
        notes.append(
            "the fault was never injected, so this run is NOT a recovery "
            "regardless of whether the task passed"
        )

    return RecoveryRun(
        case_id=case.id,
        fault=case.fault,
        fault_injected=record.injected,
        retry_count=retry_count,
        checkpoint_loaded=False,
        transcript_repaired=False,
        duplicate_persisted_tool_calls=0,
        recovery_latency_ms=elapsed_ms,
        passed=result.passed,
        judge_detail=_as_check_rows(result.detail.get("checks")),
        injection_proof=record.proof,
        injected_at_call_index=record.inject_at_call_index,
        model_call_index=record.call_index,
        offline=offline,
        notes=notes,
    )


def _as_check_rows(value: Any) -> list[dict[str, Any]]:
    """Narrow `detail["checks"]` to the per-check rows the report writes.

    `detail` is `dict[str, object]` because it also carries strings and ints, so
    the list has to be narrowed before use. A missing or malformed entry yields
    an empty list rather than a crash: a case whose judge never ran is a failed
    case with no detail, which is still recorded.
    """
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


# --- the Process Kill class ---


@dataclass
class WorkerPhaseReport:
    """A worker phase's parsed stdout plus how it ended."""

    report: dict[str, Any]
    returncode: int
    duration_ms: float
    timed_out: bool = False
    stderr_tail: str = ""


def run_worker_phase(
    phase: str,
    spec_path: Path,
    *,
    python: str | None = None,
    timeout_s: int = WORKER_TIMEOUT_S,
) -> WorkerPhaseReport:
    """Run one worker phase in a subprocess and parse its JSON report.

    A non-zero exit is not an exception here: the killed phase is *expected* to
    exit non-zero, and treating that as a runner error would make the fault look
    like a harness failure. The caller decides what a given return code means.

    `PYTHONHASHSEED`/`PYTHONDONTWRITEBYTECODE` are pinned so two runs of the same
    fixture produce byte-identical session files -- without that, a `.pyc` written
    on the first run would make the second run's `claude_dir` differ and the
    "offsets only" assertion would fail for a reason unrelated to the fault.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [python or sys.executable, "-m", "longline.eval.recovery_worker", phase, str(spec_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return WorkerPhaseReport(
            report={},
            returncode=-1,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            timed_out=True,
            stderr_tail=str(exc)[:500],
        )

    report: dict[str, Any] = {}
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                report = parsed
                break

    return WorkerPhaseReport(
        report=report,
        returncode=proc.returncode,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        stderr_tail=proc.stderr[-500:] if proc.stderr else "",
    )


def build_kill_spec(
    *,
    claude_dir: Path,
    session_id: str,
    answer: str,
    task: str = "",
) -> dict[str, Any]:
    """The worker spec for one Process-Kill case.

    Deliberately carries no `value`: the fixture file is written by the prepare
    phase from `CHECKPOINT_VALUE`, and a second copy in the spec would be a
    second source of truth for the thing being recovered.
    """
    return {
        "claude_dir": str(claude_dir),
        "session_id": session_id,
        "answer": answer,
        "task": task,
        "value": CHECKPOINT_VALUE,
    }


async def run_process_kill_case(
    case: RecoveryCase,
    *,
    api_key: str,
    fixtures_dir: Path,
    python: str | None = None,
    kill_signal: int | None = None,
) -> RecoveryRun:
    """Checkpoint in a child process, kill it, then resume in a fresh one.

    Three steps, each producing evidence rather than a verdict:

    1. `prepare` writes the turn checkpoint (transcript + task snapshot) into a
       temp `claude_dir` and proves the stable transcript value survives a
       write/read round trip. Without that proof the judge could pass on a value
       the eval supplied rather than one the resume recovered.
    2. The child is terminated. The "kill" is a real `os.kill` on the child's
       process group, not an exception.
    3. `resume` runs in a NEW interpreter, re-reads the files, repairs the
       transcript through production `validate_transcript()`, restores the task
       snapshot, re-reads the stable value, and prints it -- which the case's
       `command_output_contains` judge then asserts on.

    The `fixtures_dir` argument is accepted for signature symmetry with the
    other classes; this case's starting state is the checkpoint, not a sandbox
    fixture, so nothing is copied from it.
    """
    _ = fixtures_dir
    claude_dir = Path(tempfile.mkdtemp(prefix=CLAUDE_DIR_PREFIX))
    session_id = f"recovery-{case.id}"
    notes: list[str] = []
    started = time.perf_counter()
    try:
        spec_path = claude_dir / "kill_spec.json"
        spec_path.write_text(
            json.dumps(build_kill_spec(
                claude_dir=claude_dir, session_id=session_id,
                answer=case.answer, task=case.task,
            )),
            encoding="utf-8",
        )

        prepared = run_worker_phase("prepare", spec_path, python=python)
        if prepared.returncode != 0 or not prepared.report.get("checkpoint_saved"):
            return _failed_kill_run(
                case, started, [
                    *notes,
                    f"checkpoint was never saved (rc={prepared.returncode}); "
                    f"a kill with no checkpoint is not a resume test: {prepared.stderr_tail}",
                ],
            )

        # Prove the raw marker survived the round trip BEFORE the kill, so the
        # judge cannot later be satisfied by a value that was staged rather than
        # recovered.
        if prepared.report.get("stable_fact") != CHECKPOINT_MARKER:
            return _failed_kill_run(
                case, started, [
                    *notes,
                    "the stable fact did not survive the checkpoint round trip; "
                    "the resume would have nothing honest to recover",
                ],
            )

        killed = _kill_child(prepared, spec_path=spec_path, python=python, signal_num=kill_signal)
        # The checkpoint files must OUTLIVE the child. This is the assertion that
        # makes "kill after a saved checkpoint" a fact rather than a hope: if the
        # transcript were only in memory, the resume below would find nothing.
        session_file = Path(str(prepared.report["session_file"]))
        tasks_file = Path(str(prepared.report["tasks_file"]))
        pre_kill_bytes = session_file.read_bytes() if session_file.is_file() else b""

        resumed = run_worker_phase("resume", spec_path, python=python)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if resumed.returncode != 0:
            return _failed_kill_run(
                case, started,
                [*notes, f"resume worker exited {resumed.returncode}: {resumed.stderr_tail}"],
            )

        report = resumed.report
        checkpoint_loaded = bool(report.get("checkpoint_loaded"))
        structural_errors = [str(e) for e in report.get("structural_errors", [])]
        transcript_repaired = bool(report.get("transcript_repaired"))

        # Duplicate check: compare the fingerprints of tool calls the resumed
        # leg executed against the ones the checkpoint already persisted. Empty
        # means "no ALREADY-PERSISTED complete result was re-executed" and
        # nothing more (see the module docstring).
        persisted = _persisted_tool_call_fingerprints(session_file)
        executed = _executed_tool_call_fingerprints(report.get("stable_fact"))
        duplicates = duplicate_persisted_tool_calls(executed, persisted)

        # The resume phase re-reads a transcript that was written before the
        # kill, so the kill point is outside a tool's side-effect window; the
        # classification still refuses to call the run "safe", only "no
        # duplicate of a persisted result was observed".
        side_effect_class = classify_side_effects(False, duplicates=duplicates)

        passed, detail = _kill_judge(case, report)
        notes.extend(_kill_notes(report, killed, pre_kill_bytes, session_file, tasks_file))

        run = RecoveryRun(
            case_id=case.id,
            fault=PROCESS_KILL,
            # "Injected" for this class means the process really died after a
            # checkpoint really landed on disk.
            fault_injected=bool(pre_kill_bytes) and killed,
            retry_count=1 if checkpoint_loaded else 0,
            checkpoint_loaded=checkpoint_loaded,
            transcript_repaired=transcript_repaired,
            duplicate_persisted_tool_calls=len(duplicates),
            recovery_latency_ms=elapsed_ms,
            passed=passed,
            judge_detail=detail,
            injection_proof=(
                "kill_signal_delivered_to_live_child + checkpoint_bytes_on_disk_pre_kill "
                "+ resume_in_a_fresh_interpreter_read_them_back"
            ),
            structural_errors=structural_errors,
            task_states={str(k): str(v) for k, v in report.get("task_states", {}).items()},
            side_effect_classification=side_effect_class,
            ambiguous_side_effect=side_effect_class == "ambiguous_side_effect",
            offline=True,
            notes=notes,
        )
        run.success = recovery_succeeded(run)
        return run
    finally:
        shutil.rmtree(claude_dir, ignore_errors=True)


def _failed_kill_run(case: RecoveryCase, started: float, notes: list[str]) -> RecoveryRun:
    """A Process-Kill run that never got as far as a recoverable resume."""
    run = RecoveryRun(
        case_id=case.id,
        fault=PROCESS_KILL,
        fault_injected=False,
        retry_count=0,
        checkpoint_loaded=False,
        transcript_repaired=False,
        duplicate_persisted_tool_calls=0,
        recovery_latency_ms=(time.perf_counter() - started) * 1000.0,
        passed=False,
        notes=notes,
    )
    run.success = recovery_succeeded(run)
    return run


def _kill_child(
    prepared: WorkerPhaseReport,
    *,
    spec_path: Path,
    python: str | None,
    signal_num: int | None,
) -> bool:
    """Start a worker, kill it mid-flight, and confirm it stopped.

    The kill is delivered to a live child, not simulated: the process is started,
    confirmed running, signalled with `os.kill` (or `taskkill` on Windows, where
    `os.kill` maps to TerminateProcess and needs no extra ceremony but python's
    `os.kill` is still the portable call), and then reaped. The boolean returned
    says the child really ended, which is the fact the case records.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        [python or sys.executable, "-c", _KILL_HOLD_SCRIPT, str(spec_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    try:
        # Give the child time to actually be running before it is signalled:
        # a kill delivered to a not-yet-started process is not an interruption.
        deadline = time.monotonic() + 10.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
            if time.monotonic() >= deadline:
                break

        # The kill-and-reap sequence is shared with the loop-resume suite's
        # failpoint gate. The two differ in how they WAIT for the child -- a
        # fixed delay here, a sentinel file there -- and not in how they end
        # it, so a platform fix should land in both. This is a pure
        # extraction: `terminate_and_reap` performs the exact sequence that
        # used to be inline, including returning True for a child that had
        # already exited, and `SessionResumeRate` must not move because of it.
        return terminate_and_reap(proc, grace_s=10.0, signal_num=signal_num)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# The child loads its own checkpoint from disk and then parks. Loading through
# the production `load_session()` rather than a copy of it is what makes the
# "checkpoint was really on disk" claim checkable from the child's side too.
_KILL_HOLD_SCRIPT = """
import json, sys, time
from pathlib import Path
from longline.session.storage import load_session

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
loaded = load_session(spec["session_id"], claude_dir=Path(spec["claude_dir"]))
if not loaded:
    sys.exit(3)
print("checkpoint-loaded", flush=True)
while True:
    time.sleep(0.2)
"""


def _persisted_tool_call_fingerprints(session_file: Path) -> list[str]:
    """Fingerprints of complete tool calls the checkpoint persisted.

    Built from the JSONL on disk rather than from an in-memory object, because
    the claim is about what was *durable* at the moment of the kill. An
    in-memory view could include a result that never reached the file, which is
    exactly the distinction the contract draws.
    """
    from longline.session.storage import load_session

    if not session_file.is_file():
        return []
    messages = load_session(session_file.stem, claude_dir=session_file.parent.parent)
    if messages is None:
        return []

    from longline.models.content_blocks import ToolResultBlock, ToolUseBlock
    from longline.models.messages import AssistantMessage, UserMessage

    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for assistant_block in msg.content:
                if isinstance(assistant_block, ToolUseBlock):
                    calls[assistant_block.id] = (assistant_block.name, dict(assistant_block.input))
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for user_block in msg.content:
                if isinstance(user_block, ToolResultBlock):
                    results[user_block.tool_use_id] = json.dumps(
                        user_block.content, sort_keys=True, default=str
                    )

    completed = [
        (name, tool_input, results[tool_use_id])
        for tool_use_id, (name, tool_input) in calls.items()
        if tool_use_id in results
    ]
    return fingerprint_tool_calls(completed)


def _executed_tool_call_fingerprints(stable_fact: Any) -> list[str]:
    """Fingerprints of tool calls the RESUMED leg executed.

    The resumed leg is a plain CLI process that restores a transcript and prints
    a fact; it executes no tools, so this is empty by construction. It is a real
    list rather than a hard-coded `[]` so that wiring a resumed leg which DOES
    re-dispatch tools would change the duplicate count instead of silently
    keeping it at zero.
    """
    _ = stable_fact
    return []


def _kill_judge(case: RecoveryCase, report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Grade the resumed leg's output with the case's declared judges.

    Two things the resumed process derived independently are staged as files so
    the same deterministic judges the rest of the suite uses apply unchanged:

    - `answer.txt` receives the answer the resumed leg DERIVED from the task and
      the fixture (`derived_answer`). It carries nothing the harness knew, so a
      resume that lost either the transcript or the working directory produces
      None here and fails the check rather than passing on a staged value.
    - `stdout.txt` receives the stable fact read back out of the *persisted
      transcript*, which is what `command_output_contains` asserts on.

    Both are quoted into `answer.txt` as JSON so a judge that searches for a
    value cannot match the surrounding scaffolding instead of the value itself.
    """
    derived = report.get("derived_answer")
    marker = str(report.get("mastered_stable_fact") or "")
    payload = {
        "derived_answer": "" if derived is None else str(derived),
        "stdout": marker,
    }
    scratch = Path(tempfile.mkdtemp(prefix="recovery-kill-judge-"))
    try:
        (scratch / "answer.txt").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (scratch / "stdout.txt").write_text(marker, encoding="utf-8")
        return case_passed(case.checks, scratch, mode=case.checks_mode)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _kill_notes(
    report: dict[str, Any],
    killed: bool,
    pre_kill_bytes: bytes,
    session_file: Path,
    tasks_file: Path,
) -> list[str]:
    """Facts about the kill the report should carry, including the weak ones."""
    notes = []
    if not killed:
        notes.append("the child process could not be confirmed terminated")
    if not pre_kill_bytes:
        notes.append("no session bytes were on disk at the moment of the kill")
    else:
        notes.append(f"{len(pre_kill_bytes)} session bytes persisted before the kill")
    if not tasks_file.is_file():
        notes.append("no task snapshot was written, so task restore could not be exercised")
    if report.get("transcript_repaired"):
        notes.append(
            "the resumed transcript needed repair; the checkpoint was not a clean "
            "turn boundary"
        )
    if session_file.is_file() and len(pre_kill_bytes) != len(session_file.read_bytes()):
        notes.append(
            "the session file changed size across the kill; the resumed leg may "
            "have appended to it"
        )
    notes.append(
        "exactly-once is NOT claimed: only an already-persisted complete tool "
        "result is checked for re-execution (evals/README.md §5.4)"
    )
    return notes


# --- entry points ---


async def run_recovery_case(
    case: RecoveryCase,
    *,
    api_key: str,
    fixtures_dir: Path,
    model: str | None = None,
    variant: str | None = None,
    python: str | None = None,
) -> RecoveryRun:
    """Run one fault-injection case and return its fully derived run row.

    `model=None` (the default) means the offline protocol: a scripted model
    stands in for the API and every other component is production code. Passing
    a real model id runs the same case against the live API -- the assertions do
    not change, and the run records which mode produced it.
    """
    if case.fault == PROCESS_KILL:
        return await run_process_kill_case(
            case, api_key=api_key, fixtures_dir=fixtures_dir, python=python,
        )
    run = await _run_runtime_case(
        case, model=model, api_key=api_key, fixtures_dir=fixtures_dir, variant=variant,
    )
    run.success = recovery_succeeded(run)
    return run


async def run_recovery_suite(
    cases: Iterable[RecoveryCase],
    *,
    api_key: str,
    fixtures_dir: Path,
    model: str | None = None,
    python: str | None = None,
) -> list[RecoveryRun]:
    """Run every case serially.

    Serial by contract: parallel runs would contend for the same temp namespace
    and, for the kill class, for process handles, and the latency column would
    then measure the contention.
    """
    runs: list[RecoveryRun] = []
    for case in cases:
        runs.append(
            await run_recovery_case(
                case, api_key=api_key, fixtures_dir=fixtures_dir, model=model, python=python,
            )
        )
    return runs


# --- dataset loading ---


def load_recovery_cases(path: Path, *, fixtures_root: Path | None = None) -> list[RecoveryCase]:
    """Load recovery cases from a JSONL file, one per line.

    Kept here rather than in `types.py` because a recovery case has no fixture
    semantics unless it declares one, and threading a new `type` through the
    shared loader would widen `EvalCase` for every other suite.
    """
    from longline.eval.recovery import RecoveryCase
    from longline.eval.types import validate_fixtures

    cases: list[RecoveryCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseParseError(f"{path}:{lineno}: bad JSON: {exc}") from exc
        if not isinstance(d, dict):
            raise CaseParseError(f"{path}:{lineno}: expected JSON object, got {type(d).__name__}")
        if d.get("type") != "recovery":
            raise CaseParseError(f"{path}:{lineno}: unknown case type {d.get('type')!r}")
        cases.append(RecoveryCase.from_dict(d))

    root = path.parent / "fixtures" if fixtures_root is None else fixtures_root
    validate_fixtures([c for c in cases if c.fixture], root)
    return cases


def _case_is_runtime(case: RecoveryCase) -> bool:
    return case.fault in RUNTIME_FAULTS


def select_fault(cases: Sequence[RecoveryCase], fault: str) -> list[RecoveryCase]:
    """Cases for one fault class, in file order."""
    return [c for c in cases if c.fault == fault]


def load_jsonl_runs(path: Path) -> list[dict[str, Any]]:
    """Read back a `recovery.jsonl`-shaped result file (used by the report)."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def latency_percentiles(runs: Sequence[RecoveryRun]) -> dict[str, float | None]:
    """Mean / p50 / p95 of recovery latency, over successful runs only.

    Over successful runs because a failed recovery's "latency" is the time until
    it gave up, which is a different quantity and would look like a slow
    success.
    """
    values = [r.recovery_latency_ms for r in runs if r.success]
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


__all__ = [
    "CLAUDE_DIR_PREFIX",
    "PER_CASE_FIELDS",
    "STABLE_FACT",
    "RecoveryRun",
    "RecoverySummary",
    "WorkerPhaseReport",
    "aggregate_recovery",
    "build_kill_spec",
    "judge_recovered_answer",
    "latency_percentiles",
    "load_recovery_cases",
    "recovery_succeeded",
    "run_process_kill_case",
    "run_recovery_case",
    "run_recovery_suite",
    "run_worker_phase",
    "select_fault",
    "sha256_file",
]
