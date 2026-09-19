"""The restart-vs-resume comparison: what it averages, and what it refuses to."""

from __future__ import annotations

from longline.eval.failpoints import AFTER_CHECKPOINT, BEFORE_MODEL, BEFORE_TOOL
from longline.eval.leg_cost import LegCost
from longline.eval.loop_resume_runner import LoopResumeRun, restart_vs_resume


def _run(
    failpoint: str, *, resume: LegCost, restart: LegCost | None = None
) -> LoopResumeRun:
    return LoopResumeRun(
        case_id=f"lr-{failpoint}#0",
        failpoint=failpoint,
        failpoint_reached=True,
        checkpoint_loaded=True,
        transcript_repaired=False,
        layer_state_ok=True,
        layer_execution_ok=True,
        layer_workspace_ok=True,
        layer_task_ok=True,
        passed=True,
        resume_cost=resume,
        restart_cost=restart,
    )


def test_runs_without_a_baseline_do_not_take_part() -> None:
    """A run whose restart was never measured must not be averaged in as zeroes.

    Zeroes would make the restart mean smaller than it is, which inflates the
    saving -- the comparison would report a better number the fewer baselines
    happened to run. This is the failure mode that makes the metric worthless
    while leaving it looking healthy.
    """
    runs = [
        _run(BEFORE_TOOL, resume=LegCost(model_calls=3), restart=LegCost(model_calls=4)),
        _run(BEFORE_TOOL, resume=LegCost(model_calls=3), restart=None),
    ]
    comparison = restart_vs_resume(runs)
    assert comparison.n == 1
    assert comparison.restart["model_calls"] == 4.0
    assert comparison.resume["model_calls"] == 3.0


def test_no_baselines_reports_no_saving_rather_than_zero() -> None:
    comparison = restart_vs_resume([_run(BEFORE_TOOL, resume=LegCost(model_calls=3))])
    assert comparison.n == 0
    assert comparison.saved("model_calls") is None
    assert comparison.to_dict()["saving"]["model_calls"] is None


def test_the_saving_is_the_share_of_the_restart_not_redone() -> None:
    runs = [
        _run(BEFORE_TOOL, resume=LegCost(model_calls=2), restart=LegCost(model_calls=4)),
        _run(BEFORE_TOOL, resume=LegCost(model_calls=2), restart=LegCost(model_calls=4)),
    ]
    comparison = restart_vs_resume(runs)
    assert comparison.saved("model_calls") == 0.5


def test_an_arm_that_saves_nothing_reports_zero_not_none() -> None:
    """`before_model` is the arm whose kill lands before anything was persisted,
    so the resume redoes the whole instruction. A 0.0 here is the evidence that
    the metric tracks the kill point instead of being a constant."""
    runs = [
        _run(BEFORE_MODEL, resume=LegCost(model_calls=4), restart=LegCost(model_calls=4))
    ]
    comparison = restart_vs_resume(runs)
    assert comparison.saved("model_calls") == 0.0


def test_by_failpoint_keeps_the_arms_apart() -> None:
    runs = [
        _run(BEFORE_MODEL, resume=LegCost(model_calls=4), restart=LegCost(model_calls=4)),
        _run(
            AFTER_CHECKPOINT,
            resume=LegCost(model_calls=2),
            restart=LegCost(model_calls=6),
        ),
    ]
    by_arm = restart_vs_resume(runs).by_failpoint
    assert set(by_arm) == {BEFORE_MODEL, AFTER_CHECKPOINT}
    assert by_arm[BEFORE_MODEL]["saving"]["model_calls"] == 0.0
    assert by_arm[AFTER_CHECKPOINT]["saving"]["model_calls"] == 1.0 - 2 / 6


def test_a_failpoint_with_no_runs_has_no_row() -> None:
    """An absent arm and an arm that measured zero are different, and a row of
    zeroes for an arm that never ran would be read as the latter."""
    comparison = restart_vs_resume(
        [_run(BEFORE_TOOL, resume=LegCost(), restart=LegCost(model_calls=4))]
    )
    assert set(comparison.by_failpoint) == {BEFORE_TOOL}


def test_the_row_carries_both_legs_costs() -> None:
    """`raw.jsonl` alone has to be enough to recompute the comparison, which is
    what `PER_CASE_FIELDS` exists to guarantee for the other summaries."""
    row = _run(
        BEFORE_TOOL,
        resume=LegCost(model_calls=3, tool_calls=2),
        restart=LegCost(model_calls=4, tool_calls=3),
    ).to_row()
    assert row["resume_cost"]["model_calls"] == 3
    assert row["restart_cost"]["tool_calls"] == 3


def test_a_missing_restart_is_null_in_the_row_not_zeroes() -> None:
    """The row is what a reader recomputes from. A leg that was not measured has
    to read as `null` there, or the same inflation happens one layer down."""
    row = _run(BEFORE_TOOL, resume=LegCost(model_calls=3)).to_row()
    assert row["restart_cost"] is None
