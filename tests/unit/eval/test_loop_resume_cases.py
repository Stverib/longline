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


def test_cases_by_failpoint_groups_every_class() -> None:
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    assert set(grouped) == set(ALL_FAILPOINTS)
    assert all(len(v) > 0 for v in grouped.values())


def test_dataset_runs_ten_per_failpoint() -> None:
    cases = load_loop_resume_cases(DATASET)
    grouped = cases_by_failpoint(cases)
    assert set(grouped) == set(ALL_FAILPOINTS)
    assert all(len(v) == 10 for v in grouped.values())
    assert len(cases) == 60


def test_every_dataset_case_declares_a_tool_for_gated_tool_failpoints() -> None:
    for case in load_loop_resume_cases(DATASET):
        if case.failpoint in ("before_tool", "after_tool"):
            assert case.failpoint_tool, f"{case.id} has no failpoint_tool"


def test_every_dataset_case_names_the_cwd_placeholder() -> None:
    for case in load_loop_resume_cases(DATASET):
        assert "<cwd>" in case.task, f"{case.id} does not name <cwd>"


def test_the_after_tool_case_checks_that_the_append_did_not_double() -> None:
    """The task layer must be able to see a duplicated side effect.

    `contains: fixed-add` alone is satisfied by a file with the line twice, so
    without the `not_contains` the after_tool arm would report task success for
    exactly the outcome the execution layer is measuring.
    """
    grouped = cases_by_failpoint(load_loop_resume_cases(DATASET))
    checks = grouped["after_tool"][0].checks
    notes_checks = [
        c for c in checks
        if c["args"].get("path") == "NOTES.md"
    ]
    assert notes_checks, "the after_tool case does not check NOTES.md"
    assert "not_contains" in notes_checks[0]["args"]


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
