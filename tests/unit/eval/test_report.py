"""Unit tests for longline/eval/report.py — aggregation and rendering."""

from __future__ import annotations

import json

import pytest

from longline.eval.report import aggregate, render_markdown
from longline.eval.runner import CaseResult
from longline.eval.trajectory import ToolExecution


def _res(cid: str, ctype: str, passed: bool, turns: int = 0, **kw: object) -> CaseResult:
    """Build a CaseResult; `num_tool_calls`/`num_successful_tool_calls` are
    materialized as synthetic ToolExecution entries, since the counts are
    derived from the executions list rather than stored directly."""
    n_calls = int(kw.pop("num_tool_calls", 0))  # type: ignore[arg-type]
    n_ok = int(kw.pop("num_successful_tool_calls", 0))  # type: ignore[arg-type]
    execs = [
        ToolExecution(tool_id=f"t{i}", tool_name="Read", is_error=i >= n_ok)
        for i in range(n_calls)
    ]
    return CaseResult(
        case_id=cid, case_type=ctype, passed=passed,
        turns=turns, input_tokens=100, output_tokens=200,
        text="", errors=[], tool_calls=[("Read", {}) for _ in range(n_calls)],
        tool_executions=execs, **kw,  # type: ignore[arg-type]
    )


def test_aggregate_l1_tool_accuracy() -> None:
    results = [
        _res("a", "tool_call", True),
        _res("b", "tool_call", True),
        _res("c", "tool_call", False),
    ]
    rep = aggregate(results)
    assert rep.l1_tool_accuracy == 2 / 3
    assert rep.total_cases == 3


def test_aggregate_l2_pass1() -> None:
    results = [
        _res("x", "e2e", True),
        _res("y", "e2e", False),
    ]
    rep = aggregate(results)
    assert rep.l2_pass1 == 0.5


def test_aggregate_averages() -> None:
    results = [
        _res("a", "tool_call", True, turns=2),
        _res("b", "tool_call", True, turns=4),
    ]
    rep = aggregate(results)
    assert rep.avg_turns == 3.0
    assert rep.avg_input_tokens == 100.0
    assert rep.avg_output_tokens == 200.0


def test_aggregate_with_no_l2_keeps_pass1_none() -> None:
    results = [_res("a", "tool_call", True)]
    rep = aggregate(results)
    assert rep.l2_pass1 is None


def test_render_markdown_contains_metrics() -> None:
    results = [
        _res("a", "tool_call", True, turns=1),
        _res("b", "e2e", True, turns=2),
    ]
    rep = aggregate(results)
    md = render_markdown(rep)
    assert "Tool-call accuracy" in md
    assert "E2E pass@1" in md
    assert "turns" in md.lower()


# --- Task 1: numerator/denominator reporting ---


def test_aggregate_keeps_ratios_with_numerator_and_denominator() -> None:
    results = [
        _res("a", "tool_call", True),
        _res("b", "tool_call", True),
        _res("c", "tool_call", False),
        _res("x", "e2e", True),
        _res("y", "e2e", False),
        _res("z", "e2e", False),
        _res("w", "e2e", False),
    ]
    rep = aggregate(results)
    assert rep.l1_ratio.numerator == 2
    assert rep.l1_ratio.denominator == 3
    assert rep.l2_ratio.numerator == 1
    assert rep.l2_ratio.denominator == 4
    # legacy float fields stay identical
    assert rep.l1_tool_accuracy == 2 / 3
    assert rep.l2_pass1 == 0.25


def test_aggregate_zero_denominator_ratio_is_unmeasured_not_zero() -> None:
    rep = aggregate([_res("a", "tool_call", True)])
    assert rep.l2_ratio.denominator == 0
    assert rep.l2_ratio.value is None
    assert rep.l2_pass1 is None


def test_render_markdown_prints_numerator_and_denominator() -> None:
    results = [
        _res("a", "tool_call", True),
        _res("b", "tool_call", True),
        _res("c", "tool_call", False),
    ]
    md = render_markdown(aggregate(results))
    assert "2/3" in md
    assert "95% Wilson CI" in md


def test_render_markdown_zero_denominator_says_not_measured() -> None:
    md = render_markdown(aggregate([_res("a", "tool_call", True)]))
    assert "not measured" in md.lower()


def test_render_markdown_handles_zero_turns_without_crashing() -> None:
    md = render_markdown(aggregate([]))
    assert "n/a" in md


def test_aggregate_tool_execution_and_latency_stats() -> None:
    results = [
        _res("a", "tool_call", True, num_tool_calls=2, num_successful_tool_calls=1),
        _res("b", "tool_call", True, num_tool_calls=4, num_successful_tool_calls=3),
    ]
    rep = aggregate(results)
    assert rep.tool_execution_rate.numerator == 4
    assert rep.tool_execution_rate.denominator == 6
    assert rep.mean_duration_ms is None
    assert rep.p50_duration_ms is None


def test_aggregate_percentiles_from_durations() -> None:
    results = [_res(f"c{i}", "e2e", True, duration_ms=float(i)) for i in range(1, 11)]
    rep = aggregate(results)
    assert rep.p50_duration_ms == pytest.approx(5.5)
    assert rep.p95_duration_ms == pytest.approx(9.55)
    assert rep.mean_duration_ms == pytest.approx(5.5)


def test_report_to_dict_is_json_serializable() -> None:
    results = [_res("a", "e2e", True, duration_ms=1.5)]
    rep = aggregate(results)
    payload = json.loads(json.dumps(rep.to_dict()))
    assert payload["l2_pass1"]["numerator"] == 1
    assert payload["l2_pass1"]["denominator"] == 1
    assert payload["l2_pass1"]["ci95_wilson"][0] > 0.0
    assert payload["latency_ms"]["p50"] == 1.5


def test_report_to_dict_zero_denominator_is_null() -> None:
    payload = aggregate([_res("a", "tool_call", True)]).to_dict()
    assert payload["l2_pass1"]["value"] is None
    assert payload["l2_pass1"]["denominator"] == 0


def test_render_markdown_renders_pp_delta_not_percent() -> None:
    baseline = aggregate([_res("a", "e2e", True), _res("b", "e2e", True)])
    candidate = aggregate([_res("a", "e2e", True), _res("b", "e2e", False)])
    md = render_markdown(candidate, baseline=baseline, baseline_label="run-a")
    assert "pp" in md
    assert "-50.0 pp" in md


def test_render_markdown_no_baseline_no_pp_line() -> None:
    md = render_markdown(aggregate([_res("a", "e2e", True)]))
    assert "pp" not in md
