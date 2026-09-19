"""The restart baseline, measured against the resume it is the alternative to."""

from __future__ import annotations

from pathlib import Path

import pytest

from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases
from longline.eval.loop_resume_runner import (
    LoopResumeRun,
    restart_vs_resume,
    run_loop_resume_suite,
)

DATASET = Path("evals/loop_resume.jsonl")
FIXTURES = Path("evals/fixtures")

# The three arms that make the comparison falsifiable, and why each is here:
#
#   before_model      the kill lands before ANY model call, so nothing can have
#                     been persisted and the resume must redo everything. An
#                     arm whose saving is 0.0 is what shows the metric tracks the
#                     kill point instead of being a constant.
#   before_tool       the kill lands after one step was checkpointed, so the
#                     resume must skip exactly that one.
#   after_checkpoint  instruction 1 is complete on disk, and the resume runs
#                     instruction 2 only -- against a restart of BOTH.
ARMS = ("before_model", "before_tool", "after_checkpoint")


@pytest.fixture(scope="module")
def runs() -> list[LoopResumeRun]:
    import asyncio

    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    cases = [grouped[arm][0] for arm in ARMS]
    return asyncio.run(
        run_loop_resume_suite(
            cases,
            api_key="offline",
            fixtures_dir=FIXTURES,
            restart_baseline=True,
        )
    )


def _by_arm(runs: list[LoopResumeRun], arm: str) -> LoopResumeRun:
    return next(r for r in runs if r.failpoint == arm)


def test_every_case_got_a_restart_baseline(runs: list[LoopResumeRun]) -> None:
    """Without this the other tests could pass over an empty comparison.

    `restart_cost=None` is a legitimate per-run outcome (the leg failed), so the
    suite has to say so out loud rather than let the comparison quietly narrow to
    the runs that happened to work.
    """
    for run in runs:
        assert run.restart_cost is not None, f"{run.failpoint}: {run.notes}"


def test_the_restart_measures_the_whole_task_from_scratch(runs: list[LoopResumeRun]) -> None:
    """Instruction 1 is three steps and then an answer: four model calls, three
    tool executions, and no fewer. A baseline that came back cheaper than the task
    is a baseline that did not run the task."""
    run = _by_arm(runs, "before_tool")
    assert run.restart_cost is not None
    assert run.restart_cost.model_calls == 4
    assert run.restart_cost.tool_calls == 3


def test_the_restart_runs_the_same_instructions_the_resume_did(
    runs: list[LoopResumeRun],
) -> None:
    """`after_checkpoint`'s resume runs instruction 2, so its restart runs BOTH.

    Comparing a one-instruction resume against a two-instruction restart would
    report a saving that is entirely an artifact of the restart being a longer
    task. Six model calls is instruction 1's four plus instruction 2's two.
    """
    run = _by_arm(runs, "after_checkpoint")
    assert run.restart_cost is not None
    assert run.restart_cost.model_calls == 6
    assert run.restart_cost.tool_calls == 4


def test_the_saving_tracks_the_kill_point(runs: list[LoopResumeRun]) -> None:
    """The load-bearing assertion: a metric this is not would pass everything else.

    A constant, or a number read off the wrong leg, would survive the structural
    tests above. What it cannot survive is having to be 0.0 on the arm where
    nothing was saved and large on the arm where most of the work was.
    """
    comparison = restart_vs_resume(runs)
    by_arm = comparison.by_failpoint

    assert by_arm["before_model"]["saving"]["model_calls"] == 0.0, (
        "the kill lands before the first model call, so the resume had nothing "
        "to skip -- a saving here would be a fabricated one"
    )
    assert by_arm["before_tool"]["saving"]["model_calls"] == pytest.approx(0.25)
    assert by_arm["after_checkpoint"]["saving"]["model_calls"] == pytest.approx(1 - 2 / 6)


def test_the_resume_never_costs_more_than_the_restart_on_this_script(
    runs: list[LoopResumeRun],
) -> None:
    """A resume that redid MORE than a restart would mean the checkpoint forced
    extra work. That is a possible outcome in general and `saving` is deliberately
    not clamped for it -- so if it happens here, this is where it should surface.
    """
    for run in runs:
        assert run.restart_cost is not None
        assert run.resume_cost.model_calls <= run.restart_cost.model_calls, run.failpoint
        assert run.resume_cost.tool_calls <= run.restart_cost.tool_calls, run.failpoint


def test_token_totals_are_a_function_of_the_model_call_count(
    runs: list[LoopResumeRun],
) -> None:
    """Pins the caveat rather than the finding.

    The scripted model emits a fixed `Usage` per turn, so under this harness
    `input_tokens` is exactly 100 per call and `output_tokens` is 20 per tool turn
    plus 25 for the answer. It therefore carries no fact that `model_calls` does
    not, and no token-efficiency claim may be read off this suite. If the script
    ever changes, this test is the thing that says the caveat moved.
    """
    for run in runs:
        cost = run.resume_cost
        assert cost.input_tokens == 100 * cost.model_calls, run.failpoint
        assert cost.output_tokens == 20 * cost.tool_calls + 25, run.failpoint
