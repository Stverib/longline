"""Unit tests for longline/eval/cli.py — argument parsing + wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from longline.eval import cli


def test_parse_known_args_defaults(tmp_path: Path) -> None:
    ns = cli.parse_args(["--case-file", str(tmp_path / "c.jsonl")])
    assert ns.type == "all"
    assert ns.model == "claude-sonnet-4-20250514"
    assert ns.case_file == str(tmp_path / "c.jsonl")
    assert ns.max_cases is None
    assert ns.out_dir is not None


def test_parse_args_type_filter() -> None:
    ns = cli.parse_args(["--type", "e2e", "--model", "claude-haiku-4-5-20251001", "--max-cases", "3"])
    assert ns.type == "e2e"
    assert ns.model == "claude-haiku-4-5-20251001"
    assert ns.max_cases == 3


def test_split_cases_by_type() -> None:
    from longline.eval.types import E2ECase, ToolCallCase

    cases = [
        ToolCallCase(id="a", task="t"),
        E2ECase(id="b", task="t"),
    ]
    tc, e2e = cli.split_cases(cases)
    assert [c.id for c in tc] == ["a"]
    assert [c.id for c in e2e] == ["b"]


# --- Task 1: new flags, backwards compatible with the legacy ones ---


def test_legacy_flags_are_unchanged() -> None:
    ns = cli.parse_args(["--type", "all", "--fixtures-dir", "f", "--out-dir", "o", "--md"])
    assert ns.type == "all"
    assert ns.fixtures_dir == "f"
    assert ns.out_dir == "o"
    assert ns.md is True


def test_new_flag_defaults() -> None:
    ns = cli.parse_args([])
    assert ns.suite is None
    assert ns.variant is None
    assert ns.repeats == 1
    assert ns.run_id is None
    assert ns.keep_sandbox_on_failure is False


def test_repeats_zero_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--repeats", "0"])


def test_new_flags_parsed() -> None:
    ns = cli.parse_args([
        "--suite", "e2e", "--variant", "candidate", "--repeats", "3",
        "--run-id", "run-42", "--keep-sandbox-on-failure",
    ])
    assert ns.suite == "e2e"
    assert ns.variant == "candidate"
    assert ns.repeats == 3
    assert ns.run_id == "run-42"
    assert ns.keep_sandbox_on_failure is True


def test_apply_suite_sets_type_and_default_case_file() -> None:
    ns = cli.parse_args([])
    cli._apply_suite(ns)
    assert ns.type == "all"
    assert Path(ns.case_file).name == "tool_calls.jsonl"


def test_apply_suite_e2e_switches_case_file() -> None:
    ns = cli.parse_args(["--suite", "e2e"])
    cli._apply_suite(ns)
    assert ns.type == "e2e"
    assert Path(ns.case_file).name == "e2e.jsonl"


def test_apply_suite_explicit_case_file_wins() -> None:
    # The explicit-flag check reads the argv the parser was given.
    argv = ["--suite", "e2e", "--case-file", "custom.jsonl"]
    ns = cli.parse_args(argv)
    cli._apply_suite(ns, argv)
    assert ns.case_file == "custom.jsonl"
    assert ns.type == "e2e"


def test_apply_suite_explicit_type_wins() -> None:
    argv = ["--suite", "tool_calls", "--type", "e2e"]
    ns = cli.parse_args(argv)
    cli._apply_suite(ns, argv)
    assert ns.type == "e2e"
    assert Path(ns.case_file).name == "tool_calls.jsonl"


def test_apply_suite_legacy_type_only_is_untouched() -> None:
    ns = cli.parse_args(["--type", "tool_call"])
    cli._apply_suite(ns)
    assert ns.type == "tool_call"


def test_apply_suite_none_is_noop() -> None:
    ns = cli.parse_args([])
    before = ns.type
    cli._apply_suite(ns)
    assert ns.type == before


def test_split_cases_by_tag() -> None:
    from longline.eval.types import E2ECase

    cases = [
        E2ECase(id="a", task="t", tags=["blind"]),
        E2ECase(id="b", task="t", tags=["instruction_following"]),
        E2ECase(id="c", task="t"),
    ]
    assert [c.id for c in cli._select_by_tag(cases, "blind")] == ["a"]
    assert [c.id for c in cli._select_by_tag(cases, None)] == ["a", "b", "c"]


def test_default_run_id_is_filesystem_safe_and_unique() -> None:
    a = cli.make_run_id("claude-sonnet-4-20250514", "e2e")
    b = cli.make_run_id("claude-sonnet-4-20250514", "e2e")
    assert "/" not in a and ":" not in a
    assert a.startswith("claude-sonnet-4-20250514_e2e_")
    assert len(a.split("_")) >= 3
    assert a != b  # microsecond + process entropy


# --- run directory layout (evals/README.md §3) ---


def _fake_results() -> list[Any]:
    from longline.eval.runner import CaseResult
    from longline.eval.trajectory import ToolExecution

    return [
        CaseResult(
            case_id="e2e-001", case_type="e2e", passed=True, turns=2,
            input_tokens=10, output_tokens=5, duration_ms=12.5,
            tags=["create"], variant="baseline", repeat_index=0, trial=0,
            tool_executions=[ToolExecution("t1", "Write", False, 0, 1_000_000)],
            tool_calls=[("Write", {"file_path": "a"})],
        ),
        CaseResult(
            case_id="e2e-002", case_type="e2e", passed=False, turns=3,
            input_tokens=20, output_tokens=7, duration_ms=30.0,
            tags=["create"], variant="baseline", repeat_index=0, trial=1,
            error_type="max_turns",
        ),
    ]


def test_write_run_dir_produces_contract_filenames(tmp_path: Path) -> None:
    results = _fake_results()
    run_dir = tmp_path / "run-1"
    metadata = cli.run_metadata(
        run_id="run-1", suite="e2e", variant="baseline",
        model="m", case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
    )
    cli._write_run_dir(run_dir=run_dir, results=results, metadata=metadata)

    assert (run_dir / "raw.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_raw_jsonl_is_recomputable_source_of_truth(tmp_path: Path) -> None:
    results = _fake_results()
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=results,
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant=None, model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    rows = [json.loads(x) for x in (run_dir / "raw.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["case_id"] == "e2e-001"
    assert rows[0]["duration_ms"] == 12.5
    assert rows[0]["num_tool_calls_executed"] == 1
    assert rows[0]["num_successful_tool_calls"] == 1
    # The numerator/denominator of the summary must be derivable from these rows.
    assert sum(1 for r in rows if r["passed"]) == 1
    assert len(rows) == 2


def test_summary_json_carries_metadata_and_ratios(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=_fake_results(),
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant="candidate", model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=3,
        ),
    )
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    for key in ("run_id", "suite", "variant", "model", "git_sha", "started_at",
                "platform", "python_version", "case_file_sha256",
                "repeat_index", "repeats_completed"):
        assert key in summary["metadata"], key
    metrics = summary["metrics"]
    assert metrics["l2_pass1"]["numerator"] == 1
    assert metrics["l2_pass1"]["denominator"] == 2
    assert metrics["l2_pass1"]["value"] == 0.5
    assert len(metrics["l2_pass1"]["ci95_wilson"]) == 2
    assert metrics["latency_ms"]["p50"] == pytest.approx(21.25)


def test_report_md_prints_numerator_over_denominator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=_fake_results(),
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant=None, model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    md = (run_dir / "report.md").read_text("utf-8")
    assert "1/2" in md
    assert "## By category" in md
    assert "max_turns" in md


def test_jsonl_results_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    original = _fake_results()
    cli._write_run_dir(
        run_dir=run_dir, results=original,
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant="baseline", model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    reloaded = cli._load_jsonl_results(run_dir / "raw.jsonl")
    assert [r.case_id for r in reloaded] == ["e2e-001", "e2e-002"]
    assert [r.passed for r in reloaded] == [True, False]
    assert reloaded[0].duration_ms == 12.5
    assert reloaded[1].error_type == "max_turns"


def test_paired_delta_rejects_unaligned_runs() -> None:
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()[:1]
    with pytest.raises(ValueError, match="not aligned"):
        paired_report_delta(baseline, candidate)


def test_paired_delta_on_aligned_runs() -> None:
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    candidate[1].passed = True
    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs"] == 2
    assert out["duration_ms"]["per_case"][0]["case_id"] == "e2e-001"


def test_paired_delta_excludes_unmeasured_duration_not_zero() -> None:
    """A None duration means "not measured" and must NOT be counted as 0.0.

    Coercing it to zero would drag the paired mean toward zero while looking
    like a real observation.
    """
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()   # durations 12.5 and 30.0
    candidate = _fake_results()
    baseline[1].duration_ms = None  # case e2e-002 was never timed

    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs"] == 2
    assert out["n_pairs_duration"] == 1
    assert out["n_pairs_duration_excluded"] == 1
    # Only the timed pair survives, and its delta is the real one.
    assert out["duration_ms"]["per_case"] == [{"case_id": "e2e-001", "delta": 0.0}]
    assert out["duration_ms"]["n_pairs"] == 1


def test_paired_delta_unmeasured_duration_does_not_bias_the_mean() -> None:
    """The excluded pair must not pull the mean toward zero."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    # candidate case 0 is 100ms slower; case 1 was never timed on either side
    candidate[0].duration_ms = 112.5
    baseline[1].duration_ms = None
    candidate[1].duration_ms = None

    out = paired_report_delta(baseline, candidate)
    # With the None coerced to 0.0 the mean would be (100.0 + 0.0) / 2 = 50.0.
    assert out["duration_ms"]["mean"] == pytest.approx(100.0)
    assert out["n_pairs_duration_excluded"] == 1


def test_paired_delta_all_durations_unmeasured_is_not_zero() -> None:
    """If nothing was timed, the duration delta is unmeasured — not 0.0."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    for r in [*baseline, *candidate]:
        r.duration_ms = None

    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs_duration"] == 0
    assert out["n_pairs_duration_excluded"] == 2
    assert out["duration_ms"]["mean"] is None
    assert out["duration_ms"]["per_case"] == []


def test_paired_delta_turns_unaffected_by_missing_duration() -> None:
    """`turns` is always known, so it is never excluded."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    for r in [*baseline, *candidate]:
        r.duration_ms = None

    out = paired_report_delta(baseline, candidate)
    assert out["turns"]["n_pairs"] == 2
