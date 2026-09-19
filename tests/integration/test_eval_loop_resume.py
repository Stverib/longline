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
def six_runs() -> list:
    """One run per failpoint, shared by the assertions below.

    Module-scoped because each case spawns two real subprocesses that each
    drive a real agent loop; re-running them per test would multiply the cost
    without adding a fact.
    """
    import asyncio

    return asyncio.run(
        run_loop_resume_suite(_one_per_failpoint(), api_key="offline", fixtures_dir=FIXTURES)
    )


def test_one_run_per_failpoint_reports_every_class(six_runs: list) -> None:
    summary = aggregate_loop_resume(six_runs)
    assert set(summary.by_failpoint) == set(cases_by_failpoint(load_loop_resume_cases(DATASET)))
    assert len(six_runs) == 6


def test_the_failpoint_really_fires_for_every_class(six_runs: list) -> None:
    """The load-bearing assertion: "recovered" means nothing if nothing fired.

    The sentinel is written by the child and fsynced before it parks, so it is
    proof from outside the dead process rather than the injector's own word.
    """
    for run in six_runs:
        assert run.failpoint_reached, f"{run.case_id} never reached its failpoint: {run.notes}"
        assert run.checkpoint_loaded, run.case_id


def test_every_arm_leaves_the_repository_working(six_runs: list) -> None:
    """The workspace layer, asked with the fixture's own test suite."""
    for run in six_runs:
        assert run.layer_workspace_ok, f"{run.case_id} left the repo broken: {run.notes}"


def test_the_after_tool_arm_replays_a_side_effect(six_runs: list) -> None:
    """A NEGATIVE control, not a target value.

    If this arm reports zero redundant re-executions, the gate did not land
    inside the side-effect window and the duplicate metric is measuring
    nothing. "No duplicates found" and "the experiment never ran" must not look
    the same, and this is the test that tells them apart.
    """
    run = next(r for r in six_runs if r.failpoint == "after_tool")
    assert run.side_effect_denominator > 0, "no side effect preceded the gate"
    assert run.redundant_re_executions > 0, "the resumed leg replayed nothing"
    assert run.duplicate_side_effects > 0, "the append was replayed but did not double"


def test_the_after_checkpoint_arm_replays_nothing(six_runs: list) -> None:
    """The POSITIVE control for the execution layer.

    Instruction 1 was completed and persisted before the kill, so the resumed
    leg has nothing to redo. Without this arm, `redundant` could never be zero
    and the metric would have no way to show it recognises a clean resume.
    """
    run = next(r for r in six_runs if r.failpoint == "after_checkpoint")
    assert run.side_effect_denominator > 0, "instruction 1 changed no artifacts"
    assert run.redundant_re_executions == 0
    assert run.duplicate_side_effects == 0


def test_the_turn_zero_arms_all_replay_the_whole_instruction(six_runs: list) -> None:
    """before_model / before_tool / after_tool share one on-disk state.

    The checkpoint is written once per instruction, so a kill anywhere inside
    instruction 1 leaves the transcript with no tool traffic at all -- and the
    resumed leg has no choice but to redo the work. That is the finding, and it
    is why F1 is a control rather than a second experiment.
    """
    for failpoint in ("before_model", "before_tool", "after_tool"):
        run = next(r for r in six_runs if r.failpoint == failpoint)
        assert run.redundant_re_executions == run.side_effect_denominator, (
            f"{failpoint}: {run.redundant_re_executions} replays against "
            f"{run.side_effect_denominator} side effects -- the turn-0 arms no "
            "longer share one on-disk state, so the design note is wrong"
        )


def test_workspace_drift_stays_out_of_the_recovery_rate(six_runs: list) -> None:
    summary = aggregate_loop_resume(six_runs)
    assert summary.loop_resume_rate.denominator == 5
    assert summary.workspace_drift_detection_rate.denominator == 1


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
