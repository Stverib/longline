"""The loop-resume driver: argument handling and the refusal to report silently.

Kept to the parts that do not need a sweep: parsing, and the check that turns "a
rate over injections that never fired" into a non-zero exit rather than a number.
"""

from __future__ import annotations

from pathlib import Path

from longline.eval.loop_resume_cli import _check_the_runs_are_real, parse_args


def test_the_ablation_is_opt_in() -> None:
    """Durability is ON unless asked otherwise.

    The default has to be the real runtime. A flag that had to be passed to get
    the real behaviour would make the ablation the thing every casual run
    measured, and nobody would notice.
    """
    assert parse_args(["--out", "x"]).durability is True
    assert parse_args(["--out", "x", "--no-durability"]).durability is False


def test_the_label_is_recorded_so_two_cells_cannot_be_confused() -> None:
    args = parse_args(["--out", "x", "--label", "durability-off-ablation"])
    assert args.label == "durability-off-ablation"


def test_the_defaults_point_at_the_committed_dataset() -> None:
    args = parse_args(["--out", "x"])
    assert args.cases == Path("evals/loop_resume.jsonl")
    assert args.fixtures == Path("evals/fixtures")


def test_an_unfired_failpoint_is_reported_rather_than_averaged_away() -> None:
    """The failure mode this exists for: a run whose child never signalled.

    It is not a bad recovery, it is not a recovery at all -- and its four layers
    all read False, so it drags the rate down while looking like a result.
    """

    class _Run:
        def __init__(self, case_id: str, reached: bool) -> None:
            self.case_id = case_id
            self.failpoint_reached = reached

    problems = _check_the_runs_are_real(
        [_Run("good#0", True), _Run("bad#3", False), _Run("worse#7", False)]
    )
    assert len(problems) == 1
    assert "2 run(s) never reached their failpoint" in problems[0]
    assert "bad#3" in problems[0]


def test_a_clean_sweep_reports_nothing() -> None:
    class _Run:
        case_id = "a#0"
        failpoint_reached = True

    assert _check_the_runs_are_real([_Run()]) == []
