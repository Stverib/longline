"""Parent driver for the loop-resume suite: kill a live loop, resume, and judge.

=== One case, end to end ===

1. make a temp `claude_dir`, copy the fixture into a sandbox;
2. `arm` in a child process: real engine, scripted model, gated tools. It
   checkpoints the turn-0 state, runs the instruction the way `main.py` does,
   and parks at the failpoint;
3. wait for the sentinel, then really kill the child;
4. for `truncate_tail`, cut the session file's last line in half; for
   `workspace_drift`, mutate a file the session READ; for
   `workspace_drift_unrelated`, add a file it never touched;
5. `resume` in a NEW interpreter: production recovery path, then the loop runs
   to completion;
6. compare the two legs' journal entries and judge the artifacts.

=== The four layers ===

**State** -- the failpoint really fired and the child really died, the
checkpoint loaded, the transcript is structurally valid and the task snapshot
restored. **Execution** -- no duplicated side effect. **Workspace** -- the
repository is not left broken: the fixture's own test suite still passes, which
is a different statement from "the case's checks passed" (those may only look at
file contents). **Task** -- the case's deterministic judge passed. All four are
required, and `failpoint_reached` sits in FRONT of them: a run whose child never
signalled is not a recovery experiment at all, whatever its judge said.

=== Why the two drift arms are reported separately ===

They answer a different question from "did it recover". One asks whether
dependent drift is CAUGHT; the other asks whether unrelated drift is wrongly
REFUSED. Folding either into the recovery rate would let a detection result move
a number that is supposed to be about recovery, so both are measured, reported,
and kept out of the denominator.

=== The two cells, and why the comparison is an ablation ===

`durability=False` runs the whole suite with the runtime's two new mechanisms
switched off -- no step-level checkpoint, no operation journal, no workspace
header. Everything else is identical: the same harness, cases, judges, scripted
model and instrument. Comparing against a checkout of the pre-change revision
would not be, because this harness IMPORTS the modules the change added.
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

from longline.eval.failpoints import (
    ALL_FAILPOINTS,
    STOPS_IN_INSTRUCTION_TWO,
    TRUNCATE_TAIL,
    WORKSPACE_DRIFT,
    WORKSPACE_DRIFT_UNRELATED,
    FailpointError,
    read_sentinel,
    terminate_and_reap,
    truncate_last_line,
    wait_for_sentinel,
)
from longline.eval.faults import sha256_file
from longline.eval.judges import judge_case
from longline.eval.leg_cost import COST_FIELDS, LegCost, mean_cost
from longline.eval.leg_cost import saving as saving_ratio
from longline.eval.loop_resume import SEED_A_PLACEHOLDER, SEED_B_PLACEHOLDER, SEED_PLACEHOLDER
from longline.eval.loop_resume_worker import JOURNAL_NAME, SESSION_ID
from longline.eval.metrics import Ratio
from longline.eval.side_effect_journal import (
    SideEffectMetrics,
    compute_side_effect_metrics,
    read_journal,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from longline.eval.loop_resume import LoopResumeCase, Scenario

# The failpoints whose result is a DETECTION, not a recovery. Kept separate so
# the headline rate cannot be moved by a capability that answers a different
# question -- one arm asks "is dependent drift caught", the other "is unrelated
# drift wrongly refused", and neither is a statement about recovery.
DETECTION_FAILPOINTS: tuple[str, ...] = (WORKSPACE_DRIFT, WORKSPACE_DRIFT_UNRELATED)

# The arms where drift is SUPPOSED to produce a refusal. Every other arm is the
# false-reject control: a resume the identity check must let through.
RELEVANT_DRIFT_FAILPOINTS: tuple[str, ...] = (WORKSPACE_DRIFT,)

CLAUDE_DIR_PREFIX = "loop-resume-claude-"
SANDBOX_PREFIX = "loop-resume-sandbox-"

# Wall-clock ceiling for one worker phase. A resume that hangs is a failed case
# with a reason, not a suite that never finishes.
WORKER_TIMEOUT_S = 180

# The workspace layer's question: "is the repository left in a working state",
# asked with the fixture's OWN test suite rather than with the case's checks.
# The case's checks say whether the task was done; this says whether the code
# still runs, which a resume that half-applied an edit would break.
WORKSPACE_TEST_ARGS: dict[str, Any] = {
    "command": ["python", "-m", "pytest", "tests/test_calc.py", "-q"],
    "allowed_commands": ["python"],
    "path": "tests/test_calc.py",
    "test": "add",
    "scope": "sandbox",
}


@dataclass
class LoopResumeRun:
    """One failpoint injection, with every layer's verdict kept apart."""

    case_id: str
    failpoint: str
    failpoint_reached: bool
    checkpoint_loaded: bool
    transcript_repaired: bool
    layer_state_ok: bool
    layer_execution_ok: bool
    layer_workspace_ok: bool
    layer_task_ok: bool
    passed: bool
    seed: int = 0
    success: bool = False
    duplicate_side_effects: int = 0
    redundant_re_executions: int = 0
    side_effect_denominator: int = 0
    drift_injected: bool = False
    workspace_drifted: bool = False
    drift_detected: bool = False
    workspace_verdict: str = ""
    workspace_rejected: bool = False
    false_reject: bool = False
    workspace_relevant: list[str] = field(default_factory=list)
    workspace_unrelated: list[str] = field(default_factory=list)
    resume_latency_ms: float = 0.0
    # What this run's two legs cost. `resume_cost` is always present (a refused
    # resume costs zero, which is a fact about it); `restart_cost` is None unless
    # the restart baseline was asked for, because "not measured" and "measured
    # zero" are different claims and a default of zeroes would conflate them.
    resume_cost: LegCost = field(default_factory=LegCost)
    restart_cost: LegCost | None = None
    judge_detail: list[dict[str, Any]] = field(default_factory=list)
    structural_errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    side_effects_by_tool: dict[str, dict[str, int]] = field(default_factory=dict)
    task_states: dict[str, str] = field(default_factory=dict)
    offline: bool = True
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        """Per-case row, recomputable into the summary from `raw.jsonl` alone."""
        return {
            "case_id": self.case_id,
            "failpoint": self.failpoint,
            "seed": self.seed,
            "failpoint_reached": self.failpoint_reached,
            "checkpoint_loaded": self.checkpoint_loaded,
            "transcript_repaired": self.transcript_repaired,
            "layer_state_ok": self.layer_state_ok,
            "layer_execution_ok": self.layer_execution_ok,
            "layer_workspace_ok": self.layer_workspace_ok,
            "layer_task_ok": self.layer_task_ok,
            "passed": self.passed,
            "success": self.success,
            "duplicate_side_effects": self.duplicate_side_effects,
            "redundant_re_executions": self.redundant_re_executions,
            "side_effect_denominator": self.side_effect_denominator,
            "drift_injected": self.drift_injected,
            "workspace_drifted": self.workspace_drifted,
            "drift_detected": self.drift_detected,
            "workspace_verdict": self.workspace_verdict,
            "workspace_rejected": self.workspace_rejected,
            "false_reject": self.false_reject,
            "workspace_relevant": self.workspace_relevant,
            "workspace_unrelated": self.workspace_unrelated,
            "resume_latency_ms": self.resume_latency_ms,
            "resume_cost": self.resume_cost.to_dict(),
            "restart_cost": (
                None if self.restart_cost is None else self.restart_cost.to_dict()
            ),
            "judge_detail": self.judge_detail,
            "structural_errors": self.structural_errors,
            "repairs": self.repairs,
            "tool_errors": self.tool_errors,
            "side_effects_by_tool": self.side_effects_by_tool,
            "task_states": self.task_states,
            "offline": self.offline,
            "notes": self.notes,
        }


# The per-case fields a reader needs to recompute the summary, in one place so
# a test can assert the row carries them all.
PER_CASE_FIELDS: tuple[str, ...] = (
    "failpoint_reached",
    "checkpoint_loaded",
    "transcript_repaired",
    "duplicate_side_effects",
    "redundant_re_executions",
    "resume_latency_ms",
    # The restart-vs-resume comparison is recomputed from the row too, so both
    # legs' costs have to travel in it.
    "resume_cost",
    "restart_cost",
)


def resume_succeeded(run: LoopResumeRun) -> bool:
    """The success predicate, spelled out where it can be read and tested.

    `failpoint_reached` leads, for the same reason `fault_injected` leads in
    `recovery_succeeded`: a run whose child never signalled is an ordinary
    success, and counting it as a recovery would inflate the rate by the share
    of cases that are simply easy.
    """
    if not run.failpoint_reached:
        return False
    if not run.checkpoint_loaded:
        return False
    return (
        run.layer_state_ok
        and run.layer_execution_ok
        and run.layer_workspace_ok
        and run.layer_task_ok
        and run.passed
    )


def workspace_drifted(after: Mapping[str, str], before: Mapping[str, str]) -> bool:
    """Whether the declared artifacts differ between two digest snapshots.

    Compares keys as well as values, so a deleted or newly added artifact counts
    as drift -- a file that vanished is at least as stale a checkpoint as one
    whose contents changed.

    This is a RECORDED FACT about the run, not the workspace layer's verdict:
    every arm that does its job changes an artifact, so "differs from the
    starting state" is what success looks like almost everywhere. The layer
    asks a different question, and the drift arm is the one place this fact is
    the point.
    """
    return dict(after) != dict(before)


@dataclass
class LoopResumeSummary:
    """The headline rates plus the per-failpoint breakdown and side effects."""

    loop_resume_rate: Ratio
    workspace_drift_detection_rate: Ratio
    drift_recall: Ratio
    false_reject_rate: Ratio
    by_failpoint: dict[str, Ratio]
    side_effects: SideEffectMetrics
    failures: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "LoopResumeRate": self.loop_resume_rate.to_dict(),
            "WorkspaceDriftDetectionRate": self.workspace_drift_detection_rate.to_dict(),
            "DriftRecall": self.drift_recall.to_dict(),
            "FalseRejectRate": self.false_reject_rate.to_dict(),
            "by_failpoint": {k: v.to_dict() for k, v in self.by_failpoint.items()},
            "side_effects": self.side_effects.to_row(),
            "failures": self.failures,
        }


def aggregate_loop_resume(runs: Sequence[LoopResumeRun]) -> LoopResumeSummary:
    """Collapse per-run rows into the headline metrics.

    The recovery rate counts every failpoint EXCEPT the detection arm. Mixing
    them would let a production capability that does not exist move a number
    that is supposed to be about recovery.

    Every failpoint gets a key even when it has no runs: a missing key reads as
    "not applicable" and an empty Ratio reads as "not measured", and only the
    second is true.
    """
    recovery = [r for r in runs if r.failpoint not in DETECTION_FAILPOINTS]
    detection = [r for r in runs if r.failpoint in DETECTION_FAILPOINTS]

    # Success is RECOMPUTED here rather than read off `run.success`. The two are
    # set from the same predicate by the runner, so they agree; recomputing
    # means a row that was built without a `success` flag -- which a test or a
    # future caller will do -- cannot silently count as a failure.
    by_failpoint: dict[str, Ratio] = {}
    for failpoint in ALL_FAILPOINTS:
        rows = [r for r in runs if r.failpoint == failpoint]
        by_failpoint[failpoint] = Ratio(sum(1 for r in rows if resume_succeeded(r)), len(rows))

    side_effects = SideEffectMetrics(
        denominator=sum(r.side_effect_denominator for r in runs),
        redundant=sum(r.redundant_re_executions for r in runs),
        duplicated=sum(r.duplicate_side_effects for r in runs),
    )
    for run in runs:
        for tool, counts in run.side_effects_by_tool.items():
            bucket = side_effects.by_tool.setdefault(tool, {"redundant": 0, "duplicated": 0})
            bucket["redundant"] += counts.get("redundant", 0)
            bucket["duplicated"] += counts.get("duplicated", 0)

    return LoopResumeSummary(
        loop_resume_rate=Ratio(
            sum(1 for r in recovery if resume_succeeded(r)), len(recovery)
        ),
        workspace_drift_detection_rate=Ratio(
            sum(1 for r in detection if r.drift_detected), len(detection)
        ),
        # Recall on the arms where drift is SUPPOSED to refuse, and the false
        # reject rate on every arm where it is not. Reported as a pair because
        # either alone is trivially gameable: a detector that always refuses has
        # perfect recall, and one that never refuses has a perfect false-reject
        # rate.
        drift_recall=Ratio(
            sum(
                1
                for r in runs
                if r.failpoint in RELEVANT_DRIFT_FAILPOINTS and r.workspace_rejected
            ),
            sum(1 for r in runs if r.failpoint in RELEVANT_DRIFT_FAILPOINTS),
        ),
        false_reject_rate=Ratio(
            sum(1 for r in runs if r.false_reject),
            sum(1 for r in runs if r.failpoint not in RELEVANT_DRIFT_FAILPOINTS),
        ),
        by_failpoint=by_failpoint,
        side_effects=side_effects,
        failures=[
            {
                "case_id": r.case_id,
                "failpoint": r.failpoint,
                "failpoint_reached": r.failpoint_reached,
                "checkpoint_loaded": r.checkpoint_loaded,
                "layers": {
                    "state": r.layer_state_ok,
                    "execution": r.layer_execution_ok,
                    "workspace": r.layer_workspace_ok,
                    "task": r.layer_task_ok,
                },
                "notes": r.notes,
            }
            for r in runs
            if not resume_succeeded(r)
        ],
    )


@dataclass
class RestartVsResume:
    """What resuming cost against redoing the whole task from scratch.

    The killed leg is deliberately absent from both columns. It is SUNK COST: the
    process died either way, so the work it did before the kill is paid in BOTH
    branches. Charging it to the resume would make resume look worse than the
    alternative it actually competes with -- and the alternative, `restart`, pays
    that work a second time by doing it over.

    What is compared is therefore exactly the decision a crashed session faces:
    redo the task (`restart`) or continue from the checkpoint (`resume`).
    """

    n: int = 0
    restart: dict[str, float] = field(default_factory=dict)
    resume: dict[str, float] = field(default_factory=dict)
    by_failpoint: dict[str, dict[str, Any]] = field(default_factory=dict)

    def saved(self, name: str) -> float | None:
        """Share of the restart's `name` the resume did not redo.

        None, not 0.0, when there was no restart baseline: "not measured" and
        "measured, saved nothing" are different claims, and the whole reason this
        comparison exists is that a 0.0 here is a meaningful result.
        """
        baseline = self.restart.get(name, 0.0)
        if not baseline:
            return None
        return saving_ratio(baseline, self.resume.get(name, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "restart": self.restart,
            "resume": self.resume,
            "saving": {name: self.saved(name) for name in COST_FIELDS},
            "by_failpoint": self.by_failpoint,
        }


def _cost_pair(restart: Sequence[LegCost], resume: Sequence[LegCost]) -> dict[str, Any]:
    """One row of the comparison: the two means and the saving between them."""
    restart_mean = mean_cost(restart)
    resume_mean = mean_cost(resume)
    return {
        "n": len(resume),
        "restart": restart_mean,
        "resume": resume_mean,
        "saving": {
            name: (
                saving_ratio(restart_mean[name], resume_mean[name])
                if restart_mean[name]
                else None
            )
            for name in COST_FIELDS
        },
    }


def restart_vs_resume(runs: Sequence[LoopResumeRun]) -> RestartVsResume:
    """The restart baseline against the resume, per failpoint and overall.

    Only runs that HAVE a baseline take part. Folding an unmeasured run in as
    zeroes would average a real cost with a missing one and report a saving that
    is an artifact of how many baselines happened to run.
    """
    measured = [r for r in runs if r.restart_cost is not None]
    overall = _cost_pair(
        [r.restart_cost for r in measured if r.restart_cost is not None],
        [r.resume_cost for r in measured],
    )

    by_failpoint: dict[str, dict[str, Any]] = {}
    for failpoint in ALL_FAILPOINTS:
        group = [r for r in measured if r.failpoint == failpoint]
        if not group:
            continue
        by_failpoint[failpoint] = _cost_pair(
            [r.restart_cost for r in group if r.restart_cost is not None],
            [r.resume_cost for r in group],
        )

    return RestartVsResume(
        n=len(measured),
        restart=overall["restart"],
        resume=overall["resume"],
        by_failpoint=by_failpoint,
    )


# --- the parent's own actions ---

def _snapshot_artifacts(sandbox: Path, artifacts: Sequence[str]) -> dict[str, str]:
    """Digest the case's declared artifacts, in `GatedTool.snapshot_artifacts`'s shape."""
    return {rel: sha256_file(sandbox / rel) for rel in artifacts}


def apply_seed(sandbox: Path, seed: int) -> None:
    """Make this run's starting state different from its siblings'.

    Varies only what the case's checks and the scripted scenario do not pin: a
    header line in NOTES.md, and the operands in the fixture's own test. The
    Edit's `old_string` and every judge stay identical, so the seed makes the
    INPUT different without making the OUTCOME different.

    That is the honest scope of what a repeat count buys here: "60 runs, 0
    counterexamples" is a statement about 60 fixtures rather than about one
    fixture sixty times. It is still not a distribution, and reporting it as a
    rate would be wrong -- the module docstring of `loop_resume` says so.

    Raises rather than quietly substituting nothing when a placeholder is
    missing: a fixture that lost them would make every repeat byte-identical,
    and the suite would look like it had sampled when it had not.
    """
    notes = sandbox / "NOTES.md"
    test = sandbox / "tests" / "test_calc.py"
    a, b = 2 + seed, 3 + seed
    # b must not be 0, or the buggy `a - b` would already satisfy the test and
    # the arm would pass without the fix -- measured by the `not_contains` on
    # the fixed line, which would then never be exercised.
    if b == 0:  # pragma: no cover - unreachable for the seeds the dataset produces
        raise FailpointError(f"seed {seed} would make the fixture's test vacuous")

    for path, replacements in (
        (notes, {SEED_PLACEHOLDER: str(seed)}),
        (test, {SEED_A_PLACEHOLDER: str(a), SEED_B_PLACEHOLDER: str(b)}),
    ):
        if not path.is_file():
            raise FailpointError(f"the fixture is missing {path.name}")
        text = path.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            if placeholder not in text:
                raise FailpointError(
                    f"{path.name} does not carry {placeholder!r}; without it every "
                    "repeat would start from a byte-identical fixture"
                )
            text = text.replace(placeholder, value)
        path.write_text(text, encoding="utf-8")


def drift_the_workspace(sandbox: Path) -> None:
    """Mutate a file the session has READ, between the kill and the resume.

    `src/calc.py` rather than `NOTES.md`, and the choice is forced rather than
    arbitrary. The dependent set is built from `Tool.workload`, and `Bash`
    declares nothing -- so the file the Bash append writes never enters the write
    set, while the file the `Read` touches does. Drifting a file the session never
    recorded touching would be testing the UNRELATED arm's question by accident.

    Appends rather than overwrites: an appended line leaves every earlier line
    intact, so the file is visibly the product of two writers rather than one that
    merely looks odd. A comment, so the fixture's own test suite still passes
    afterwards -- this arm measures detection, not whether the drift broke
    anything.
    """
    calc = sandbox / "src" / "calc.py"
    if calc.is_file():
        calc.write_text(
            calc.read_text(encoding="utf-8") + "# drifted-by-another-writer\n",
            encoding="utf-8",
        )


def drift_an_unrelated_file(sandbox: Path) -> None:
    """Create a file the session never touched, between the kill and the resume.

    The opposite question to `drift_the_workspace`: a detector that flags
    dependent drift but also refuses this one is not a detector, it is a wall.
    """
    (sandbox / "src" / "UNRELATED.md").write_text(
        "a file another writer added\n", encoding="utf-8"
    )


def _init_workspace_repo(sandbox: Path) -> None:
    """Make the sandbox a git repo, so unrelated drift is detectable at all.

    Without a repo the identity check can still see a DEPENDENT change -- it
    compares hashes it recorded -- but cannot enumerate anything else, so the
    unrelated arm would have nothing to measure and would report a clean
    workspace for a reason that has nothing to do with the runtime.

    Seed FIRST, then commit: the seed's edits belong to the fixture's initial
    state, so the tree starts clean and any later change is genuinely later.
    """
    for args in (
        ("init",),
        ("add", "-A"),
        ("-c", "user.email=eval@longline", "-c", "user.name=eval", "commit", "-m", "fixture"),
    ):
        subprocess.run(
            ["git", *args],
            cwd=sandbox,
            check=True,
            capture_output=True,
            text=True,
        )


def _worker_env() -> dict[str, str]:
    """A child environment that produces byte-identical sessions across runs.

    Without the pins, a `.pyc` written on the first run would make the second
    run's `claude_dir` differ, and a difference that comes from the harness
    would be read as a difference in the thing under test.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent.parent.parent)


def run_worker_phase(
    phase: str,
    spec_path: Path,
    *,
    python: str | None = None,
    timeout_s: int = WORKER_TIMEOUT_S,
) -> tuple[dict[str, Any], int, float]:
    """Run one worker phase in a subprocess and parse its JSON report.

    A non-zero exit is not an exception: the killed phase is EXPECTED to die,
    and treating that as a runner error would make the fault look like a
    harness failure. The caller decides what a return code means.
    """
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                python or sys.executable,
                "-m",
                "longline.eval.loop_resume_worker",
                phase,
                str(spec_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_worker_env(),
            cwd=_repo_root(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {}, -1, (time.perf_counter() - started) * 1000.0

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
    return report, proc.returncode, (time.perf_counter() - started) * 1000.0


def kill_armed_child(
    claude_dir: Path,
    *,
    spec_path: Path,
    python: str | None = None,
) -> bool:
    """Start the armed child, wait for its sentinel, and kill it.

    Returns True only when the child signalled AND is gone. Starting `arm` a
    second time to collect a JSON report is not an option: the killed child IS
    the arm run, and re-running it against the same sandbox would destroy the
    thing being measured. The sentinel and the journal are the evidence, and
    both were designed to cross a process boundary.
    """
    proc = subprocess.Popen(
        [
            python or sys.executable,
            "-m",
            "longline.eval.loop_resume_worker",
            "arm",
            str(spec_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_worker_env(),
        cwd=_repo_root(),
    )
    try:
        if wait_for_sentinel(claude_dir, timeout_s=WORKER_TIMEOUT_S) is None:
            return False
        return terminate_and_reap(proc)
    finally:
        if proc.poll() is None:  # pragma: no cover - terminate_and_reap failed
            proc.kill()
            proc.wait(timeout=10)


# --- one case ---


def _scenario_of(case: LoopResumeCase) -> Scenario:
    """The case's scenario, or a loud failure.

    A run that reached the worker without one would be running some other task
    under this case's name, and nothing in the metrics could tell: the failpoint
    would fire, the layers would pass, and the row would be about a different job.
    """
    if case.scenario is None:
        raise FailpointError(
            f"{case.id}: no scenario. The case's tool sequence travels with the case, "
            "and a run without one would silently be some other task"
        )
    return case.scenario


def _build_spec(
    case: LoopResumeCase,
    *,
    claude_dir: Path,
    sandbox: Path,
    api_key: str,
    durability: bool = True,
) -> dict[str, Any]:
    scenario = _scenario_of(case)
    return {
        "claude_dir": str(claude_dir),
        "sandbox": str(sandbox),
        "session_id": SESSION_ID,
        "failpoint": case.failpoint,
        "failpoint_tool": case.failpoint_tool,
        "at_call_index": 1,
        "api_key": api_key,
        "model": "offline-model",
        "task": case.task.replace("<cwd>", sandbox.as_posix()),
        # The case's own instructions, tool sequence, artifacts and workspace
        # test. The sandbox is substituted here, once, rather than in the worker:
        # two places resolving the same placeholder would eventually disagree.
        "scenario": scenario.to_spec(sandbox),
        "max_turns": case.max_turns,
        # The ablation switch. Off means the runtime records nothing: no step
        # checkpoint, no operation journal, no workspace header. See
        # `loop_resume_worker._attach_durability` for why this, and not a
        # worktree at the pre-change revision, is the honest before/after.
        "durability": durability,
    }


async def run_loop_resume_case(
    case: LoopResumeCase,
    *,
    api_key: str,
    fixtures_dir: Path,
    python: str | None = None,
    durability: bool = True,
    restart_baseline: bool = False,
) -> LoopResumeRun:
    """One failpoint injection, end to end."""
    claude_dir = Path(tempfile.mkdtemp(prefix=CLAUDE_DIR_PREFIX))
    sandbox = Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX))
    notes: list[str] = []
    scenario = _scenario_of(case)
    try:
        if case.fixture:
            shutil.copytree(fixtures_dir / case.fixture, sandbox, dirs_exist_ok=True)
        apply_seed(sandbox, case.seed)
        # After the seed, so the fixture's initial commit contains the seeded
        # content and the tree starts dirty only if the RUNTIME made it dirty.
        _init_workspace_repo(sandbox)

        spec = _build_spec(
            case, claude_dir=claude_dir, sandbox=sandbox, api_key=api_key,
            durability=durability,
        )
        spec_path = claude_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        before = _snapshot_artifacts(sandbox, scenario.artifacts)

        # --- steps 2 and 3: arm in a child, wait for the sentinel, kill it ---
        killed = kill_armed_child(claude_dir, spec_path=spec_path, python=python)
        sentinel = read_sentinel(claude_dir)
        if not killed:
            notes.append("the armed child never signalled its failpoint")

        # --- step 4: the parent-side failpoints ---
        session_file = claude_dir / "sessions" / f"{SESSION_ID}.jsonl"
        drifted = False
        if case.failpoint == TRUNCATE_TAIL and session_file.is_file():
            truncate_last_line(session_file)
        elif case.failpoint == WORKSPACE_DRIFT:
            drift_the_workspace(sandbox)
            drifted = True
        elif case.failpoint == WORKSPACE_DRIFT_UNRELATED:
            drift_an_unrelated_file(sandbox)
            drifted = True

        # --- step 5: resume in a fresh interpreter ---
        resumed, rc, resume_ms = run_worker_phase("resume", spec_path, python=python)
        if rc != 0:
            notes.append(f"the resume worker exited {rc}")

        # --- step 6: the journal comparison and the four layers ---
        entries = read_journal(claude_dir / JOURNAL_NAME)
        metrics = compute_side_effect_metrics(entries)
        after = _snapshot_artifacts(sandbox, scenario.artifacts)
        judge_ok, judge_detail = _judge_case_checks(case, sandbox)
        workspace_ok = bool(judge_case("python_test", sandbox, scenario.workspace_test))
        tool_errors = [str(e) for e in resumed.get("tool_errors", [])]

        # --- step 7: the restart baseline, if this cell asks for one ---
        # Last, and in fresh directories, so it cannot perturb anything above it.
        restart_cost = (
            _run_restart_baseline(
                case,
                api_key=api_key,
                fixtures_dir=fixtures_dir,
                python=python,
                durability=durability,
                notes=notes,
            )
            if restart_baseline
            else None
        )

        run = LoopResumeRun(
            case_id=case.id,
            failpoint=case.failpoint,
            seed=case.seed,
            failpoint_reached=sentinel is not None and killed,
            checkpoint_loaded=bool(resumed.get("checkpoint_loaded")),
            transcript_repaired=bool(resumed.get("transcript_repaired")),
            layer_state_ok=(
                killed
                and bool(resumed.get("layer_state_ok"))
                and not resumed.get("error")
            ),
            layer_execution_ok=metrics.duplicated == 0,
            layer_workspace_ok=workspace_ok,
            layer_task_ok=bool(judge_ok),
            passed=bool(judge_ok),
            duplicate_side_effects=metrics.duplicated,
            redundant_re_executions=metrics.redundant,
            side_effect_denominator=metrics.denominator,
            # Two facts, and the drift arms need them apart. `drift_injected` is
            # what the parent DID; `workspace_drifted` is whether the declared
            # artifacts moved, which is true for every arm that does its job. The
            # dependent arm's mutation lands on a declared artifact so both hold;
            # the unrelated arm's lands deliberately outside that set.
            drift_injected=drifted,
            workspace_drifted=workspace_drifted(after, before),
            drift_detected=bool(resumed.get("workspace_rejected")),
            workspace_verdict=str(resumed.get("workspace_verdict", "")),
            workspace_rejected=bool(resumed.get("workspace_rejected")),
            # The expensive direction of a detection mechanism is the one that
            # blocks work that was safe, so it is measured on EVERY arm -- and
            # every arm except the dependent-drift one is a clean-resume control.
            false_reject=(
                bool(resumed.get("workspace_rejected"))
                and case.failpoint not in RELEVANT_DRIFT_FAILPOINTS
            ),
            workspace_relevant=[str(p) for p in resumed.get("workspace_relevant", [])],
            workspace_unrelated=[str(p) for p in resumed.get("workspace_unrelated", [])],
            resume_latency_ms=resume_ms,
            resume_cost=LegCost.from_dict(resumed.get("cost")),
            restart_cost=restart_cost,
            judge_detail=judge_detail,
            structural_errors=[str(e) for e in resumed.get("structural_errors", [])],
            repairs=[str(e) for e in resumed.get("repairs", [])],
            tool_errors=tool_errors,
            side_effects_by_tool=metrics.by_tool,
            task_states={
                str(k): str(v) for k, v in dict(resumed.get("task_states", {})).items()
            },
            notes=notes,
        )
        run.success = resume_succeeded(run)
        return run
    finally:
        shutil.rmtree(claude_dir, ignore_errors=True)
        shutil.rmtree(sandbox, ignore_errors=True)


def _judge_case_checks(case: LoopResumeCase, sandbox: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the case's own deterministic checks to the resumed sandbox."""
    from longline.eval.judges import case_passed

    return case_passed(case.checks, sandbox, mode=case.checks_mode)


def _run_restart_baseline(
    case: LoopResumeCase,
    *,
    api_key: str,
    fixtures_dir: Path,
    python: str | None,
    durability: bool,
    notes: list[str],
) -> LegCost | None:
    """Redo the case's task from scratch, in a session of its own.

    Its own `claude_dir` AND its own sandbox, built the same way the killed leg's
    were: the baseline has to start from the fixture and the seed, not from
    whatever the killed leg left behind. Sharing either directory would make this
    a measurement of the resume.

    Returns None -- a missing baseline, not a zero one -- when the leg did not
    complete the instructions the resume ran. A restart that stopped early is not
    a cheaper restart, and letting its cost through would understate the baseline
    and inflate the saving.
    """
    claude_dir = Path(tempfile.mkdtemp(prefix=CLAUDE_DIR_PREFIX))
    sandbox = Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX))
    expected = 2 if case.failpoint in STOPS_IN_INSTRUCTION_TWO else 1
    try:
        if case.fixture:
            shutil.copytree(fixtures_dir / case.fixture, sandbox, dirs_exist_ok=True)
        apply_seed(sandbox, case.seed)
        _init_workspace_repo(sandbox)
        spec = _build_spec(
            case, claude_dir=claude_dir, sandbox=sandbox, api_key=api_key,
            durability=durability,
        )
        spec_path = claude_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

        report, rc, _ = run_worker_phase("baseline", spec_path, python=python)
        if rc != 0 or report.get("error") or report.get("instructions_run") != expected:
            notes.append(
                "the restart baseline is not usable: "
                f"rc={rc}, error={report.get('error')!r}, "
                f"instructions={report.get('instructions_run')!r} (wanted {expected})"
            )
            return None
        return LegCost.from_dict(report.get("cost"))
    finally:
        shutil.rmtree(claude_dir, ignore_errors=True)
        shutil.rmtree(sandbox, ignore_errors=True)


async def run_loop_resume_suite(
    cases: Sequence[LoopResumeCase],
    *,
    api_key: str,
    fixtures_dir: Path,
    python: str | None = None,
    durability: bool = True,
    restart_baseline: bool = False,
) -> list[LoopResumeRun]:
    """Run every case serially.

    Serial by contract: parallel runs would contend for the same temp namespace
    and for process handles, and the latency column would then measure the
    contention.
    """
    runs: list[LoopResumeRun] = []
    for case in cases:
        runs.append(
            await run_loop_resume_case(
                case,
                api_key=api_key,
                fixtures_dir=fixtures_dir,
                python=python,
                durability=durability,
                restart_baseline=restart_baseline,
            )
        )
    return runs


__all__ = [
    "DETECTION_FAILPOINTS",
    "PER_CASE_FIELDS",
    "RELEVANT_DRIFT_FAILPOINTS",
    "WORKSPACE_TEST_ARGS",
    "LoopResumeRun",
    "LoopResumeSummary",
    "RestartVsResume",
    "aggregate_loop_resume",
    "apply_seed",
    "drift_an_unrelated_file",
    "drift_the_workspace",
    "kill_armed_child",
    "restart_vs_resume",
    "resume_succeeded",
    "run_loop_resume_case",
    "run_loop_resume_suite",
    "run_worker_phase",
    "workspace_drifted",
]
