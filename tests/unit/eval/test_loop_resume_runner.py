"""The parent driver: the four layers, and the failpoint_reached guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from longline.eval.failpoints import (
    AFTER_TOOL,
    ALL_FAILPOINTS,
    BEFORE_MODEL,
    WORKSPACE_DRIFT,
)
from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases
from longline.eval.loop_resume_runner import (
    PER_CASE_FIELDS,
    LoopResumeRun,
    aggregate_loop_resume,
    resume_succeeded,
    workspace_drifted,
)
from longline.eval.side_effect_journal import KILLED, RESUMED, SideEffectEntry

DATASET = Path("evals/loop_resume.jsonl")
FIXTURES = Path("evals/fixtures")


def _run(**overrides: Any) -> LoopResumeRun:
    base: dict[str, Any] = {
        "case_id": "lr-x#0",
        "failpoint": BEFORE_MODEL,
        "failpoint_reached": True,
        "checkpoint_loaded": True,
        "transcript_repaired": False,
        "layer_state_ok": True,
        "layer_execution_ok": True,
        "layer_workspace_ok": True,
        "layer_task_ok": True,
        "passed": True,
    }
    base.update(overrides)
    return LoopResumeRun(**base)


def test_success_requires_all_four_layers() -> None:
    assert resume_succeeded(_run()) is True
    for field_name in (
        "layer_state_ok",
        "layer_execution_ok",
        "layer_workspace_ok",
        "layer_task_ok",
    ):
        assert resume_succeeded(_run(**{field_name: False})) is False, field_name


def test_success_requires_the_failpoint_to_have_been_reached() -> None:
    """A run that never reached its kill point is an ordinary success, not a
    recovery, and counting it would inflate the rate by how easy the task is."""
    assert resume_succeeded(_run(failpoint_reached=False)) is False


def test_success_requires_a_loaded_checkpoint() -> None:
    assert resume_succeeded(_run(checkpoint_loaded=False)) is False


def test_success_requires_the_judge_not_just_the_layers() -> None:
    assert resume_succeeded(_run(passed=False)) is False


def test_workspace_drifted_detects_a_changed_digest() -> None:
    assert workspace_drifted({"a": "AAA"}, {"a": "BBB"}) is True


def test_workspace_drifted_is_false_for_identical_digests() -> None:
    assert workspace_drifted({"a": "AAA"}, {"a": "AAA"}) is False


def test_workspace_drifted_detects_a_deleted_or_added_file() -> None:
    assert workspace_drifted({"a": "missing"}, {"a": "AAA"}) is True
    assert workspace_drifted({}, {"a": "AAA"}) is True


def test_aggregate_excludes_workspace_drift_from_the_headline() -> None:
    """workspace_drift is a detection arm. The runtime has no workspace
    identity check, so folding it into the recovery rate would move the
    headline for a reason that has nothing to do with recovery."""
    runs = [
        _run(case_id="a", failpoint=BEFORE_MODEL),
        _run(case_id="b", failpoint=AFTER_TOOL),
        _run(case_id="c", failpoint=WORKSPACE_DRIFT, drift_detected=False),
    ]
    summary = aggregate_loop_resume(runs)
    assert summary.loop_resume_rate.denominator == 2
    assert summary.loop_resume_rate.numerator == 2
    assert summary.workspace_drift_detection_rate.denominator == 1
    assert summary.workspace_drift_detection_rate.numerator == 0
    assert summary.workspace_drift_detection_rate.value == 0.0


def test_aggregate_reports_per_failpoint_even_when_empty() -> None:
    summary = aggregate_loop_resume([_run()])
    assert set(summary.by_failpoint) == set(ALL_FAILPOINTS)
    assert summary.by_failpoint[WORKSPACE_DRIFT].denominator == 0
    assert summary.by_failpoint[WORKSPACE_DRIFT].value is None


def test_aggregate_lists_every_failure_with_its_reason() -> None:
    summary = aggregate_loop_resume(
        [
            _run(case_id="ok"),
            _run(case_id="bad", layer_task_ok=False, passed=False, notes=["judge said no"]),
        ]
    )
    assert [f["case_id"] for f in summary.failures] == ["bad"]
    assert summary.failures[0]["notes"] == ["judge said no"]
    layers = summary.failures[0]["layers"]
    assert isinstance(layers, dict)
    assert layers["task"] is False


def test_aggregate_sums_side_effects_across_runs() -> None:
    runs = [
        _run(
            case_id="a",
            duplicate_side_effects=1,
            redundant_re_executions=2,
            side_effect_denominator=2,
        ),
        _run(
            case_id="b",
            duplicate_side_effects=0,
            redundant_re_executions=1,
            side_effect_denominator=1,
        ),
    ]
    summary = aggregate_loop_resume(runs)
    assert summary.side_effects.duplicated == 1
    assert summary.side_effects.redundant == 3
    assert summary.side_effects.denominator == 3


def test_aggregate_merges_the_per_tool_breakdown() -> None:
    runs = [
        _run(case_id="a", side_effects_by_tool={"Bash": {"redundant": 1, "duplicated": 1}}),
        _run(case_id="b", side_effects_by_tool={"Bash": {"redundant": 1, "duplicated": 0}}),
    ]
    summary = aggregate_loop_resume(runs)
    assert summary.side_effects.by_tool["Bash"] == {"redundant": 2, "duplicated": 1}


def test_apply_seed_makes_two_runs_start_from_different_fixtures(tmp_path: Path) -> None:
    """The seed must actually reach the sandbox, not just the case object."""
    import shutil

    from longline.eval.loop_resume_runner import apply_seed

    texts = []
    for seed in (0, 7):
        sandbox = tmp_path / f"s{seed}"
        shutil.copytree(FIXTURES / "resume_repo", sandbox)
        apply_seed(sandbox, seed)
        texts.append(
            (
                (sandbox / "NOTES.md").read_text(encoding="utf-8"),
                (sandbox / "tests" / "test_calc.py").read_text(encoding="utf-8"),
            )
        )
    assert texts[0] != texts[1], "two seeds produced a byte-identical fixture"

    notes, test = texts[1]
    assert "7" in notes
    assert "add(9, 10)" in test, test


def test_apply_seed_leaves_the_scenarios_edit_target_intact(tmp_path: Path) -> None:
    """The Edit's `old_string` must survive seeding, or the arm's fix step
    silently stops applying and the duplicate metric measures the wrong tool."""
    import shutil

    from longline.eval.loop_resume_runner import apply_seed
    from longline.eval.loop_resume_worker import SCENARIO_ONE

    sandbox = tmp_path / "s"
    shutil.copytree(FIXTURES / "resume_repo", sandbox)
    apply_seed(sandbox, 3)
    calc = (sandbox / "src" / "calc.py").read_text(encoding="utf-8")
    old_string = SCENARIO_ONE[2]["input"]["old_string"]
    assert old_string in calc


def test_apply_seed_keeps_the_fixtures_test_failing_before_the_fix(tmp_path: Path) -> None:
    """The workspace layer runs this test. If the buggy code already passed it,
    that layer would be vacuous for every seed."""
    import shutil
    import subprocess
    import sys

    from longline.eval.loop_resume_runner import apply_seed

    sandbox = tmp_path / "s"
    shutil.copytree(FIXTURES / "resume_repo", sandbox)
    apply_seed(sandbox, 2)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_calc.py", "-q"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0, "the fixture's test passes before the fix"


def test_apply_seed_raises_when_a_placeholder_is_gone(tmp_path: Path) -> None:
    """A fixture that lost its placeholders would make every repeat identical,
    and the suite would look like it had sampled when it had not."""
    import shutil

    import pytest

    from longline.eval.failpoints import FailpointError
    from longline.eval.loop_resume_runner import apply_seed

    sandbox = tmp_path / "s"
    shutil.copytree(FIXTURES / "resume_repo", sandbox)
    (sandbox / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    with pytest.raises(FailpointError, match="does not carry"):
        apply_seed(sandbox, 0)


def test_the_committed_fixture_carries_every_seed_placeholder() -> None:
    for name in ("NOTES.md", "tests/test_calc.py"):
        text = (FIXTURES / "resume_repo" / name).read_text(encoding="utf-8")
        assert "<seed" in text, f"{name} carries no seed placeholder"


def test_every_row_carries_the_per_case_fields() -> None:
    row = _run().to_row()
    for name in PER_CASE_FIELDS:
        assert name in row, name


def test_dataset_expands_to_sixty_runs_ten_per_failpoint() -> None:
    cases = load_loop_resume_cases(DATASET)
    grouped = cases_by_failpoint(cases)
    assert len(cases) == 60
    assert all(len(v) == 10 for v in grouped.values())


def test_side_effect_metrics_come_from_the_journal_not_the_transcript() -> None:
    """A guard against a future "simplification" that reads the transcript:
    the whole point is that the transcript cannot see this."""
    from longline.eval.side_effect_journal import compute_side_effect_metrics

    entries = [
        SideEffectEntry(
            seq=1,
            leg=KILLED,
            tool="Bash",
            input_fp="fp",
            outcome="ok",
            pre_state={"f": "A"},
            post_state={"f": "B"},
        ),
        SideEffectEntry(
            seq=1,
            leg=RESUMED,
            tool="Bash",
            input_fp="fp",
            outcome="ok",
            pre_state={"f": "B"},
            post_state={"f": "C"},
        ),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.duplicated == 1
