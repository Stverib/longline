"""One run of every failpoint, offline, end to end -- real subprocesses."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases
from longline.eval.loop_resume_runner import (
    aggregate_loop_resume,
    run_loop_resume_suite,
)

DATASET = Path("evals/loop_resume.jsonl")
FIXTURES = Path("evals/fixtures")


def _real_state_dir_fingerprint() -> str:
    """Digest of the user's real session directory, if it exists.

    `get_sessions_dir(None)` falls back to `~/.longline`, so a single missed
    `claude_dir` argument would write the benchmark's sessions into the
    operator's real state. This is the test that fails when that happens.
    """
    real = Path.home() / ".longline"
    if not real.exists():
        return "absent"
    digest = hashlib.sha256()
    for path in sorted(real.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(real)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _one_per_failpoint() -> list:
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    return [cases[0] for cases in grouped.values()]


@pytest.fixture(scope="module")
def runs() -> list:
    """One run per failpoint, shared by the assertions below.

    Module-scoped because each case spawns two real subprocesses that each drive
    a real agent loop; re-running them per test would multiply the cost without
    adding a fact.
    """
    import asyncio

    return asyncio.run(
        run_loop_resume_suite(_one_per_failpoint(), api_key="offline", fixtures_dir=FIXTURES)
    )


def _by_failpoint(runs: list, name: str):
    return next(r for r in runs if r.failpoint == name)


def test_one_run_per_failpoint_reports_every_class(runs: list) -> None:
    summary = aggregate_loop_resume(runs)
    assert set(summary.by_failpoint) == set(cases_by_failpoint(load_loop_resume_cases(DATASET)))
    assert len(runs) == len(summary.by_failpoint)


def test_the_failpoint_really_fires_for_every_class(runs: list) -> None:
    """The load-bearing assertion: "recovered" means nothing if nothing fired.

    The sentinel is written by the child and fsynced before it parks, so it is
    proof from outside the dead process rather than the injector's own word.
    """
    for run in runs:
        assert run.failpoint_reached, f"{run.case_id} never reached its failpoint: {run.notes}"
        assert run.checkpoint_loaded, run.case_id


def test_every_non_rejected_arm_leaves_the_repository_working(runs: list) -> None:
    """The workspace layer, asked with the fixture's own test suite.

    The dependent-drift arm is excluded on purpose and not as a convenience: its
    resume is REFUSED, so the bug it was going to fix is still there and the
    fixture's test suite must fail. Asserting it passes would be asserting that
    the refusal did not happen.
    """
    for run in runs:
        if run.workspace_rejected:
            assert not run.layer_workspace_ok, (
                f"{run.case_id} refused the resume and still left a working repo -- "
                "the refusal did not actually stop the run"
            )
            continue
        assert run.layer_workspace_ok, f"{run.case_id} left the repo broken: {run.notes}"


def test_the_after_tool_arm_does_not_replay_its_side_effect(runs: list) -> None:
    """The headline result, and the arm the whole change was built for.

    Before: the kill landed between the Bash append and its result reaching the
    transcript, so the resumed leg had no record the call had been issued and
    issued it again -- 10 of 10 injections duplicated the append.

    The two mechanisms together are what stops it. The step-level checkpoint puts
    the assistant message (carrying the `tool_use`) on disk BEFORE the tool body
    runs, so the resumed transcript knows the call happened; the journal records
    it as PREPARED-without-COMMITTED, so reconciliation can say truthfully that
    its outcome is unknown rather than that it failed.

    The denominator assertion is the negative control that keeps this honest: if
    it is zero, no side effect preceded the gate and none of the rest means
    anything. "No duplicates found" and "the experiment never ran" must not look
    the same.
    """
    run = _by_failpoint(runs, "after_tool")
    assert run.side_effect_denominator > 0, "no side effect preceded the gate"
    assert run.duplicate_side_effects == 0, (
        "the append was replayed and doubled -- the durability fix did not hold"
    )
    assert run.redundant_re_executions == 0, (
        "the resumed leg re-issued the Bash call at all, even without doubling"
    )
    assert run.layer_task_ok, f"the task was not finished: {run.judge_detail}"


def test_the_after_checkpoint_arm_replays_nothing(runs: list) -> None:
    """The POSITIVE control for the execution layer.

    Instruction 1 was completed and persisted before the kill, so the resumed
    leg has nothing to redo. Without this arm, `redundant` could never be zero
    and the metric would have no way to show it recognises a clean resume.
    """
    run = _by_failpoint(runs, "after_checkpoint")
    assert run.side_effect_denominator > 0, "instruction 1 changed no artifacts"
    assert run.redundant_re_executions == 0
    assert run.duplicate_side_effects == 0


def test_the_before_tool_arm_retries_only_the_step_that_never_ran(runs: list) -> None:
    """`before_tool` stops before the tool, so nothing preceded it to duplicate.

    Under instruction-level checkpointing this arm resumed from the turn-0 floor
    and redid the whole instruction. With step-level checkpoints the transcript
    carries the Read that DID complete, so the resumed leg starts at the Bash
    call -- one step re-issued, not three.

    `layer_task_ok` is the assertion that matters here, and it was MISSING: this
    test passed while the arm was scoring 0/10. Every metric it did assert --
    denominator, duplicates, structural state -- is satisfied by a run that
    duplicates nothing because it never got anything done. "Did no harm" and "did
    the job" are different claims, and only one of them was being checked.
    """
    run = _by_failpoint(runs, "before_tool")
    assert run.side_effect_denominator == 0, "nothing state-changing preceded the gate"
    assert run.duplicate_side_effects == 0
    assert run.layer_state_ok, run.structural_errors
    assert run.layer_task_ok, (
        "the resumed leg never retried the call that provably never ran, so the task "
        f"was left unfinished: {run.judge_detail}"
    )


def test_the_dependent_drift_arm_is_refused(runs: list) -> None:
    """The detection half: a file the session READ is changed under the checkpoint."""
    run = _by_failpoint(runs, "workspace_drift")
    assert run.drift_injected, "the parent never mutated the workspace"
    assert run.workspace_rejected, "the runtime resumed onto a stale premise"
    assert run.workspace_verdict == "relevant"
    assert any(p.endswith("calc.py") for p in run.workspace_relevant), run.workspace_relevant


def test_the_unrelated_drift_arm_is_not_refused(runs: list) -> None:
    """The other half, and the one that decides whether the check is usable.

    A detector that refuses whenever the tree is dirty is a wall, not a detector.
    """
    run = _by_failpoint(runs, "workspace_drift_unrelated")
    assert run.drift_injected, "the parent never added its file"
    assert not run.workspace_rejected, (
        f"a file the session never touched blocked the resume: {run.workspace_unrelated}"
    )
    assert run.workspace_verdict == "unrelated"


def test_workspace_detection_stays_out_of_the_recovery_rate(runs: list) -> None:
    """Both drift arms answer a different question from "did it recover"."""
    summary = aggregate_loop_resume(runs)
    assert summary.loop_resume_rate.denominator == 5
    assert summary.workspace_drift_detection_rate.denominator == 2
    assert summary.drift_recall.denominator == 1
    assert summary.false_reject_rate.denominator == 6


def test_the_real_session_directory_is_untouched() -> None:
    import asyncio

    before = _real_state_dir_fingerprint()
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    asyncio.run(
        run_loop_resume_suite(
            [grouped["before_tool"][0]], api_key="offline", fixtures_dir=FIXTURES
        )
    )
    assert _real_state_dir_fingerprint() == before
