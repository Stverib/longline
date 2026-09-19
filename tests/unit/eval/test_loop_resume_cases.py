"""The loop-resume dataset contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from longline.eval.failpoints import ALL_FAILPOINTS, BEFORE_TOOL, WORKSPACE_DRIFT
from longline.eval.loop_resume import (
    LoopResumeCase,
    cases_by_failpoint,
    expand_case,
    load_loop_resume_cases,
)
from longline.eval.types import CaseParseError

DATASET = Path("evals/loop_resume.jsonl")


def _line(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "lr-x",
        "type": "loop_resume",
        "task": "Do the thing in <cwd>/src/.",
        "failpoint": BEFORE_TOOL,
        "failpoint_tool": "Bash",
        "checks": [{"fn": "file_exists", "args": {"path": "NOTES.md"}}],
    }
    base.update(overrides)
    return base


def test_from_dict_reads_the_failpoint_fields() -> None:
    case = LoopResumeCase.from_dict(_line())
    assert case.failpoint == BEFORE_TOOL
    assert case.failpoint_tool == "Bash"
    assert case.repeat == 1


def test_from_dict_rejects_an_unknown_failpoint() -> None:
    with pytest.raises(CaseParseError, match="unknown failpoint"):
        LoopResumeCase.from_dict(_line(failpoint="half_past_the_tool"))


def test_from_dict_requires_a_failpoint() -> None:
    line = _line()
    del line["failpoint"]
    with pytest.raises(CaseParseError, match="failpoint"):
        LoopResumeCase.from_dict(line)


def test_gated_tool_failpoints_require_a_tool_name() -> None:
    """A before_tool/after_tool case with no tool name could never fire, and
    would be recorded as a failed recovery for a reason unrelated to recovery."""
    with pytest.raises(CaseParseError, match="failpoint_tool"):
        LoopResumeCase.from_dict(_line(failpoint=BEFORE_TOOL, failpoint_tool=""))


def test_non_tool_failpoints_must_not_carry_a_tool_name() -> None:
    """A leftover tool name on a model-entry failpoint is a copy-paste bug, and
    silently ignoring it would let a case look tool-gated when it is not."""
    with pytest.raises(CaseParseError, match="not tool-named"):
        LoopResumeCase.from_dict(_line(failpoint="before_model", failpoint_tool="Bash"))


def test_parent_failpoints_require_no_tool_name() -> None:
    case = LoopResumeCase.from_dict(_line(failpoint=WORKSPACE_DRIFT, failpoint_tool=""))
    assert case.failpoint == WORKSPACE_DRIFT


def test_task_must_name_the_cwd_placeholder() -> None:
    """Without it the task is unanswerable offline: the real working directory
    is only known at run time."""
    with pytest.raises(CaseParseError, match="cwd"):
        LoopResumeCase.from_dict(_line(task="Do the thing in src/."))


def test_repeat_must_be_positive() -> None:
    with pytest.raises(CaseParseError, match="repeat"):
        LoopResumeCase.from_dict(_line(repeat=0))


def test_repeat_rejects_a_bool() -> None:
    """`True` is an int in Python, and `repeat=True` would silently mean one run."""
    with pytest.raises(CaseParseError, match="repeat"):
        LoopResumeCase.from_dict(_line(repeat=True))


def test_expand_case_yields_one_per_repeat_with_unique_ids() -> None:
    case = LoopResumeCase.from_dict(_line(repeat=3))
    expanded = expand_case(case)
    assert [c.id for c in expanded] == ["lr-x#0", "lr-x#1", "lr-x#2"]
    assert all(c.repeat == 1 for c in expanded)
    # Nothing but the id may differ.
    assert {c.failpoint for c in expanded} == {BEFORE_TOOL}
    assert {c.failpoint_tool for c in expanded} == {"Bash"}
    assert {c.task for c in expanded} == {case.task}


def test_expand_case_gives_every_repeat_a_distinct_seed() -> None:
    """The seed is what separates "ten runs" from "one run ten times".

    `recovery.py`'s expand_case copies everything but the id, which makes its
    repeats byte-identical. Inheriting that would leave this suite's 60 runs
    standing on one fixture, and its zero-counterexample result would be about
    that fixture rather than about the runtime.
    """
    expanded = expand_case(LoopResumeCase.from_dict(_line(repeat=10)))
    assert [c.seed for c in expanded] == list(range(10))


def test_a_single_run_case_has_seed_zero() -> None:
    assert LoopResumeCase.from_dict(_line()).seed == 0


def test_cases_by_failpoint_groups_every_class() -> None:
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    assert set(grouped) == set(ALL_FAILPOINTS)
    assert all(len(v) > 0 for v in grouped.values())


def test_dataset_runs_ten_per_failpoint() -> None:
    """Ten runs per arm, and one arm per name in the vocabulary.

    The total is DERIVED from `ALL_FAILPOINTS` rather than written as a literal.
    A literal would have to be edited every time an arm is added, which is the
    edit that makes a dataset change look routine -- and the property worth
    pinning is "every arm is present and none was left at one run", not "there
    are exactly sixty".
    """
    cases = load_loop_resume_cases(DATASET)
    grouped = cases_by_failpoint(cases)
    assert set(grouped) == set(ALL_FAILPOINTS)
    assert all(len(v) == 10 for v in grouped.values()), {
        name: len(v) for name, v in grouped.items()
    }
    assert len(cases) == len(ALL_FAILPOINTS) * 10


def test_every_dataset_case_declares_a_tool_for_gated_tool_failpoints() -> None:
    for case in load_loop_resume_cases(DATASET):
        if case.failpoint in ("before_tool", "after_tool"):
            assert case.failpoint_tool, f"{case.id} has no failpoint_tool"


def test_every_dataset_case_names_the_cwd_placeholder() -> None:
    for case in load_loop_resume_cases(DATASET):
        assert "<cwd>" in case.task, f"{case.id} does not name <cwd>"


def test_every_case_that_checks_the_append_can_see_it_double() -> None:
    """The task layer must be able to see a duplicated side effect.

    `contains: fixed-add` alone is satisfied by a file with the line twice, so
    without the `not_contains` an arm reports task success for exactly the outcome
    the execution layer is measuring.

    Written for `after_tool` alone, and that was too narrow: `before_model` and
    `before_tool` scored 0 on `den` -- nothing state-changing happened in the
    killed leg -- which also zeroes `DuplicateSideEffectRate`'s denominator. So
    for those two the task judge was the ONLY thing that could have caught a
    double append, and it could not. A run of the whole matrix came back 10/10 on
    an arm whose judge no input could fail, which is the shape of a number that
    means nothing.
    """
    checked = 0
    for case in load_loop_resume_cases(DATASET):
        notes_checks = [
            check for check in case.checks if check["args"].get("path") == "NOTES.md"
        ]
        for check in notes_checks:
            assert "not_contains" in check["args"], (
                f"{case.id} checks NOTES.md content without a duplicate guard"
            )
            checked += 1
    # Guards the sweep itself: a dataset whose paths changed would make the loop
    # vacuous and green, which is the failure this test exists to prevent.
    assert checked >= 5, f"only {checked} cases asserted NOTES.md content"


def test_the_workspace_drift_case_does_not_assert_notes_content() -> None:
    """Drift appends to NOTES.md. A content assertion there would make the arm
    fail because of the drift itself, conflating detection with task success."""
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    paths = {c["args"].get("path") for c in grouped["workspace_drift"][0].checks}
    assert "NOTES.md" not in paths


def test_the_after_checkpoint_case_expects_instruction_twos_artifact() -> None:
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    paths = {c["args"].get("path") for c in grouped["after_checkpoint"][0].checks}
    assert "REPORT.md" in paths


def test_loader_rejects_a_foreign_case_type(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"id":"a","type":"e2e","task":"t","checks":[]}\n', encoding="utf-8")
    with pytest.raises(CaseParseError, match="unknown case type"):
        load_loop_resume_cases(path, fixtures_root=tmp_path)
