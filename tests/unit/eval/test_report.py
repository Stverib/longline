"""Unit tests for longline/eval/report.py — aggregation and rendering."""

from __future__ import annotations

from longline.eval.report import aggregate, render_markdown
from longline.eval.runner import CaseResult


def _res(cid: str, ctype: str, passed: bool, turns: int = 0) -> CaseResult:
    return CaseResult(
        case_id=cid, case_type=ctype, passed=passed,
        turns=turns, input_tokens=100, output_tokens=200,
        text="", errors=[], tool_calls=[],
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
