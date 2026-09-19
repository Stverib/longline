"""The loop-resume dataset contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from longline.eval.failpoints import (
    AFTER_CHECKPOINT,
    ALL_FAILPOINTS,
    BEFORE_TOOL,
    WORKSPACE_DRIFT,
)
from longline.eval.loop_resume import (
    LoopResumeCase,
    cases_by_failpoint,
    expand_case,
    load_loop_resume_cases,
)
from longline.eval.types import CaseParseError

DATASET = Path("evals/loop_resume.jsonl")


def _scenario(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "followups": [],
        "steps": [[{"tool": "Bash", "input": {"command": "echo x >> NOTES.md"}}]],
        "artifacts": ["NOTES.md"],
        "workspace_test": {"command": ["python", "-m", "pytest"], "path": "tests/t.py"},
    }
    base.update(overrides)
    return base


def _line(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "lr-x",
        "type": "loop_resume",
        "task": "Do the thing in <cwd>/src/.",
        "failpoint": BEFORE_TOOL,
        "failpoint_tool": "Bash",
        "checks": [{"fn": "file_exists", "args": {"path": "NOTES.md"}}],
        "scenario": _scenario(),
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


def test_a_case_without_a_scenario_is_refused() -> None:
    """Before item 6 the tool sequence was a module constant, so every case in the
    suite was the same task with a different place to die -- a matrix with one
    row. A case that carried no scenario would silently get that back."""
    line = _line()
    del line["scenario"]
    with pytest.raises(CaseParseError, match="scenario"):
        LoopResumeCase.from_dict(line)


def test_a_gate_that_could_never_fire_is_refused() -> None:
    """`failpoint_tool` names a tool the task has to actually call.

    Otherwise the gate never fires, and the run is recorded as a failed recovery
    for a reason that has nothing to do with recovery -- the defect class this
    suite exists to catch, and one that would be baked into the dataset."""
    line = _line(scenario=_scenario(steps=[[{"tool": "Read", "input": {}}]]))
    with pytest.raises(CaseParseError, match="could never fire"):
        LoopResumeCase.from_dict(line)


def test_an_instruction_two_failpoint_needs_a_second_instruction() -> None:
    """The kill has to land somewhere. On a one-instruction task these arms have
    nothing to stop in, and would report a failure for a reason unrelated to
    recovery."""
    line = _line(failpoint=AFTER_CHECKPOINT, failpoint_tool="")
    with pytest.raises(CaseParseError, match="nothing for it to stop in"):
        LoopResumeCase.from_dict(line)


def test_step_lists_must_match_the_instruction_count() -> None:
    with pytest.raises(CaseParseError, match="step lists"):
        LoopResumeCase.from_dict(
            _line(scenario=_scenario(followups=["and then"]))
        )


def test_a_scenario_needs_at_least_one_artifact() -> None:
    """With no artifacts the journal digests nothing, `den` is zero on every arm,
    and no run can fail on a duplicated execution. That is the shape of a number
    that means nothing, so it is refused at load time rather than reported."""
    with pytest.raises(CaseParseError, match="artifacts"):
        LoopResumeCase.from_dict(_line(scenario=_scenario(artifacts=[])))


def test_a_scenario_needs_a_workspace_test() -> None:
    """It is a different question from the case's checks: not "did the checks
    pass" but "is the repository still working"."""
    with pytest.raises(CaseParseError, match="workspace_test"):
        LoopResumeCase.from_dict(_line(scenario=_scenario(workspace_test={})))


def test_a_step_without_a_tool_or_input_is_refused() -> None:
    with pytest.raises(CaseParseError, match="tool"):
        LoopResumeCase.from_dict(_line(scenario=_scenario(steps=[[{"input": {}}]])))
    with pytest.raises(CaseParseError, match="input"):
        LoopResumeCase.from_dict(_line(scenario=_scenario(steps=[[{"tool": "Read"}]])))


def test_expand_case_keeps_the_scenario() -> None:
    """The repeats are runs of the SAME task. Dropping the scenario on expansion
    would make every repeat fail at spec-build time -- or worse, not."""
    case = LoopResumeCase.from_dict(_line(repeat=2))
    expanded = expand_case(case)
    assert all(c.scenario == case.scenario for c in expanded)


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


# --- dataset-level shape, now that the scenario is data rather than code ---


def test_every_dataset_case_carries_a_scenario() -> None:
    for case in load_loop_resume_cases(DATASET):
        assert case.scenario is not None, case.id
        assert case.scenario.artifacts, case.id
        assert case.scenario.steps[0], case.id


def test_the_after_tool_arm_gates_on_the_step_whose_side_effect_gets_replayed() -> None:
    """The `after_tool` arm gates on Bash, and the FIRST Bash is the append, so
    the killer leg's only state-changing execution is the append itself -- exactly
    the side effect the resumed leg used to replay.

    If someone reorders the sequence so a read-only command comes first, the
    denominator drops to zero and the metric silently measures nothing while still
    reporting 10/10. This is the assertion that stands between those two.
    """
    case = cases_by_failpoint(load_loop_resume_cases(DATASET))["after_tool"][0]
    assert case.scenario is not None
    steps = list(case.scenario.steps[0])
    tools = [str(s["tool"]) for s in steps]
    assert tools[0] == "Read", "a read must not be the gated step"
    first_bash = tools.index("Bash")
    assert ">>" in str(steps[first_bash]["input"]["command"]), (
        "the gated Bash call has to be the one that changes a file"
    )
    assert "Edit" in tools


def test_the_artifacts_cover_every_file_the_scenario_mutates() -> None:
    """The journal digests the declared artifacts only. A file the scenario writes
    but does not declare is invisible to every side-effect metric -- the run would
    look clean because nothing was being watched."""
    for case in load_loop_resume_cases(DATASET):
        assert case.scenario is not None
        declared = set(case.scenario.artifacts)
        for group in case.scenario.steps:
            for step in group:
                tool = str(step["tool"])
                target = str(
                    step["input"].get("file_path") or step["input"].get("path") or ""
                )
                if tool in ("Edit", "Write") and target:
                    assert target in declared, f"{case.id}: {target} is not an artifact"
