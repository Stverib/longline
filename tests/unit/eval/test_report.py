"""Unit tests for longline/eval/report.py — aggregation and rendering."""

from __future__ import annotations

import json
from typing import Any

import pytest

from longline.eval.metrics import Ratio
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


# --- Task 2: four tool-calling metrics in summary.json / report.md ---


def _tool_res(
    cid: str,
    *,
    tags: list[str] | None = None,
    calls: int = 1,
    extra: int = 0,
    steps_ok: bool = True,
    arg_calls: tuple[int, int] = (0, 0),
    arg_fields: tuple[int, int] = (0, 0),
) -> CaseResult:
    """A tool_call CaseResult carrying the Task 2 detail counters."""
    return CaseResult(
        case_id=cid,
        case_type="tool_call",
        passed=steps_ok and arg_calls[0] == arg_calls[1],
        tags=list(tags if tags is not None else ["blind"]),
        tool_calls=[("Read", {}) for _ in range(calls)],
        detail={
            "steps": {"all_steps_matched": steps_ok, "num_extra_calls": extra},
            "args": {
                "correct_calls": arg_calls[0], "checked_calls": arg_calls[1],
                "correct_fields": arg_fields[0], "checked_fields": arg_fields[1],
            },
        },
    )


class TestFourMetricsInReport:
    def test_blind_cases_only_in_selection_denominator(self) -> None:
        rep = aggregate([
            _tool_res("b1", tags=["blind"], steps_ok=True),
            _tool_res("b2", tags=["blind"], steps_ok=False),
            _tool_res("i1", tags=["instruction-following"], steps_ok=True),
        ])
        assert rep.tool_selection_case_accuracy == Ratio(1, 2)
        # instruction-following 单独展示,不混进主数字
        assert rep.instruction_following_case_accuracy == Ratio(1, 1)

    def test_precision_counts_extra_calls_in_denominator(self) -> None:
        rep = aggregate([_tool_res("b1", calls=4, extra=1)])
        assert rep.tool_call_precision == Ratio(3, 4)

    def test_argument_metrics_are_separate(self) -> None:
        rep = aggregate([_tool_res("b1", arg_calls=(2, 3), arg_fields=(5, 9))])
        assert rep.argument_call_accuracy == Ratio(2, 3)
        assert rep.argument_field_accuracy == Ratio(5, 9)

    def test_summary_dict_has_all_four_with_denominators(self) -> None:
        rep = aggregate([_tool_res("b1", calls=2, extra=1, arg_calls=(1, 1), arg_fields=(2, 3))])
        d = rep.to_dict()
        tc = d["tool_calling"]
        assert set(tc) >= {  # type: ignore[arg-type]
            "tool_selection_case_accuracy", "tool_call_precision",
            "argument_call_accuracy", "argument_field_accuracy",
            "execution_success_rate", "instruction_following_case_accuracy",
        }
        for key in ("tool_selection_case_accuracy", "tool_call_precision",
                    "argument_call_accuracy", "argument_field_accuracy"):
            block = tc[key]  # type: ignore[index]
            assert {"numerator", "denominator", "value", "ci95_wilson"} <= set(block)

    def test_markdown_renders_all_four_rows(self) -> None:
        rep = aggregate([_tool_res("b1", calls=2, extra=1, arg_calls=(1, 1), arg_fields=(2, 3))])
        md = render_markdown(rep)
        for label in ("ToolSelectionCaseAccuracy", "ToolCallPrecision",
                      "ArgumentCallAccuracy", "ArgumentFieldAccuracy",
                      "ExecutionSuccessRate", "InstructionFollowingCaseAccuracy"):
            assert label in md
        assert "(1/2)" in md  # selection
        assert "(1/2)" in md  # precision — same fraction, different denominator source

    def test_markdown_omits_section_when_no_tool_cases(self) -> None:
        rep = aggregate([_res("e", "e2e", True)])
        md = render_markdown(rep)
        assert "ToolSelectionCaseAccuracy" not in md


# --- Task 4: the compression A/B section -----------------------------------


def _compression_summary(**overrides: Any) -> Any:
    """A CompressionSummary with two eligible cases and one excluded."""
    from longline.eval.compression_runner import aggregate_compression

    runs = []
    for cid, base_ok, cand_ok, before, after, retained in (
        ("cc-1", True, True, 1000, 500, 5),
        ("cc-2", True, False, 1200, 600, 3),
        ("cc-3", False, True, 900, 450, 0),
    ):
        run = _fake_compression_run(cid, base_ok, cand_ok, before, after, retained)
        runs.append(run)
    summary = aggregate_compression(runs)
    for key, value in overrides.items():
        setattr(summary, key, value)
    return summary


def _fake_compression_run(case_id: str, base_ok: bool, cand_ok: bool,
                          before: int, after: int, retained: int) -> Any:
    from longline.eval.compression import KeyFact
    from longline.eval.compression_runner import CompactEvidence, CompressionRun
    from longline.eval.runner import CaseResult

    base = CaseResult(case_id=case_id, case_type="compression", passed=base_ok,
                      variant="compression_off")
    cand = CaseResult(case_id=case_id, case_type="compression", passed=cand_ok,
                      variant="compression_on")
    evidence = CompactEvidence(
        messages_before=21, messages_after=9, tokens_before=before, tokens_after=after,
        summariser_calls=1, summary="s", compacted=True,
    )
    cand.detail["compression"] = evidence.to_detail()
    facts = [
        KeyFact(id=f"A{i}", kind="file-path", statement="s", probe="p?",
                check={"fn": "file_exists", "args": {"path": "x"}},
                answer=f"v{i}")
        for i in range(1, 6)
    ]
    lost = [f"A{i}" for i in range(retained + 1, 6)] if base_ok else []
    return CompressionRun(
        case_id=case_id, baseline=base, candidate=cand,
        key_facts=facts, retained_fact_ids=[f"A{i}" for i in range(1, retained + 1)],
        lost_fact_ids=lost, num_facts=5, retained_facts=retained,
        excluded_from_denominator=not base_ok,
        exclusion_reason=None if base_ok else "baseline_failed",
        evidence=evidence,
    )


class TestCompressionReport:
    def test_section_absent_without_compression_results(self) -> None:
        md = render_markdown(aggregate([_res("e", "e2e", True)]))
        assert "Compression" not in md

    def test_render_includes_all_four_metrics(self) -> None:
        md = render_markdown(aggregate([]), compression=_compression_summary())
        for label in ("CompressionRatio", "KeyInfoRetention",
                      "PostCompressionSuccessRate", "SuccessDeltaPP"):
            assert label in md, f"missing metric row: {label}"

    def test_token_counts_are_labelled_estimated(self) -> None:
        """Contract §5.3: the report must say the tokens are ESTIMATED."""
        md = render_markdown(aggregate([]), compression=_compression_summary())
        assert "estimated" in md.lower()

    def test_success_delta_is_in_percentage_points_not_percent(self) -> None:
        md = render_markdown(aggregate([]), compression=_compression_summary())
        assert "pp" in md
        # "下降 3%" is forbidden; a relative percent must not appear at all.
        assert "%" not in md.split("SuccessDeltaPP")[1].split("|")[2]

    def test_baseline_exclusion_is_reported_not_hidden(self) -> None:
        """The excluded case must be visible, with its reason."""
        md = render_markdown(aggregate([]), compression=_compression_summary())
        assert "cc-3" in md
        assert "baseline_failed" in md

    def test_lost_facts_are_named_per_case(self) -> None:
        """The plan's acceptance condition: trace WHICH fact was lost."""
        md = render_markdown(aggregate([]), compression=_compression_summary())
        assert "cc-2" in md
        assert "A4" in md and "A5" in md  # retained=3 -> A4/A5 lost in cc-2

    def test_no_section_when_summary_is_none(self) -> None:
        md = render_markdown(aggregate([]), compression=None)
        assert "CompressionRatio" not in md


# --- Task 6: the latency section -------------------------------------------


def _latency_summary(*, reduction: float | None = 0.25) -> Any:
    """A two-case `LatencySummary` built by hand, so no clock is involved."""
    from longline.eval.latency_runner import (
        DEFAULT_TIME_SCALE,
        CaseLatency,
        LatencySummary,
    )

    def metrics(start: float, turn: float, overlap: float) -> dict[str, dict[str, float | None]]:
        def one(v: float) -> dict[str, float | None]:
            return {f"{k}_{s}": v for k in
                    ("tool_start_latency_ms", "turn_latency_ms", "overlap_time_ms")
                    for s in ("mean", "p50", "p95")}

        return {"buffered": one(start), "streaming": one(start * (1 - (reduction or 0)))}

    cases = [
        CaseLatency(
            case_id="lat-001", note="single tool", samples_per_arm=40,
            metrics=metrics(40.0, 190.0, 0.0), reduction=reduction,
            reduction_per_sample=[reduction or 0.0] * 40,
            reduction_mean=reduction, reduction_p50=reduction,
            lifetime_samples=40, overlap_samples=0,
        ),
        CaseLatency(
            case_id="lat-002", note="three tools", samples_per_arm=40,
            metrics=metrics(80.0, 130.0, 50.0), reduction=reduction,
            reduction_per_sample=[reduction or 0.0] * 40,
            reduction_mean=reduction, reduction_p50=reduction,
            lifetime_samples=40, overlap_samples=0,
        ),
    ]
    return LatencySummary(
        samples_per_arm=40, warmups_per_arm=5, time_scale=DEFAULT_TIME_SCALE, cases=cases,
    )


class TestLatencyReportSection:
    def test_section_renders_both_arms_per_case(self) -> None:
        md = render_markdown(aggregate([]), latency=_latency_summary())
        assert "## Streaming tool latency" in md
        for case_id in ("lat-001", "lat-002"):
            assert case_id in md
        assert "buffered" in md and "streaming" in md

    def test_metrics_are_mean_p50_and_p95(self) -> None:
        """Contract §5.5: the report口径 is mean / p50 / p95 plus a paired delta."""
        md = render_markdown(aggregate([]), latency=_latency_summary())
        header = md.split("### Paired delta")[0]
        assert "mean" in header and "p50" in header and "p95" in header

    def test_reduction_is_a_ratio_not_percentage_points(self) -> None:
        """The specific error the contract's §4.3 rule exists to prevent.

        Checked on the RENDERED VALUE CELLS, not on the section text: the
        section's own note mentions the word "pp" precisely to say it is not
        used here, so scanning the whole block would fail on the explanation
        rather than on a number.
        """
        md = render_markdown(aggregate([]), latency=_latency_summary())
        assert "ratio of durations" in md
        delta_section = md.split("### Paired delta per case")[1]
        delta_rows = [line for line in delta_section.splitlines() if line.startswith("| lat-")]
        assert delta_rows
        for row in delta_rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            # Columns: case, reduction, mean of per-sample ratios, p50, counts, note.
            for cell in cells[1:4]:
                assert cell.endswith("%"), f"a duration ratio rendered as {cell!r}"
                assert "pp" not in cell

    def test_the_sample_count_and_excluded_warmups_are_printed(self) -> None:
        """A latency mean without its n and its excluded warmups is not reproducible."""
        md = render_markdown(aggregate([]), latency=_latency_summary())
        assert "40" in md  # samples per arm
        assert "warmup" in md.lower()

    def test_the_time_scale_is_stated(self) -> None:
        """A scaled run is not reproducible without the factor it was run at."""
        from longline.eval.latency_runner import DEFAULT_TIME_SCALE

        md = render_markdown(aggregate([]), latency=_latency_summary())
        assert "scale" in md.lower()
        assert f"{DEFAULT_TIME_SCALE:g}" in md

    def test_an_unmeasured_reduction_says_so(self) -> None:
        """FAILS ON: a None reduction rendered as 0%, which reads as "no effect"."""
        md = render_markdown(aggregate([]), latency=_latency_summary(reduction=None))
        assert "not measured" in md

    def test_no_section_when_summary_is_none(self) -> None:
        md = render_markdown(aggregate([]), latency=None)
        assert "Streaming tool latency" not in md


def _multi_agent_summary(*, mean_speedup: float | None) -> Any:
    """A one-case `MultiAgentSummary` built by hand, so nothing is measured."""
    from longline.eval.metrics import Ratio
    from longline.eval.multi_agent_runner import MultiAgentSummary

    both = Ratio(numerator=1, denominator=1)
    return MultiAgentSummary(
        group="controlled",
        num_cases=1,
        eligible_cases=1,
        excluded_cases=0,
        single_success_rate=both,
        multi_success_rate=both,
        single_wall_time_ms=16.9,
        multi_wall_time_ms=146.3,
        mean_speedup=mean_speedup,
        mean_token_overhead=2.3333,
        single_tokens={"input_tokens": 1, "output_tokens": 1, "total_tokens": 74520,
                       "child_tokens": 0},
        multi_tokens={"input_tokens": 1, "output_tokens": 1, "total_tokens": 248400,
                      "child_tokens": 198720},
        single_tool_calls=5,
        multi_tool_calls=5,
        agent_counts=[5],
    )


class TestSpeedupIsARatioNotAPercent:
    """`Speedup` has parity at 1.00x, so it cannot share `_fmt_pct_ratio`.

    `_fmt_pct_ratio` renders `value * 100` as a signed percent, which is correct
    for a quantity that already IS a relative change (`reduction`,
    `TokenOverhead`) and wrong for this one. Shipped once: an offline run
    measured `single / multi = 0.1155` and the report printed `+11.6%`, which
    reads as an 11.6% improvement when the multi arm was in fact 8.7x SLOWER.
    """

    def test_a_slowdown_is_never_rendered_with_a_plus_sign(self) -> None:
        """FAILS ON: the shipped bug -- 0.1155 rendered as '+11.6%'."""
        from longline.eval.report import _fmt_speedup

        rendered = _fmt_speedup(0.1155)
        assert not rendered.startswith("+"), rendered
        assert "slower" in rendered

    def test_a_slowdown_states_the_factor(self) -> None:
        """FAILS ON: a bare '0.12x' a reader has to invert in their head."""
        from longline.eval.report import _fmt_speedup

        assert _fmt_speedup(0.5) == "0.50x (2.0x slower)"

    def test_parity_is_named(self) -> None:
        """FAILS ON: 1.0 rendered as '+100.0%', which reads as a doubling."""
        from longline.eval.report import _fmt_speedup

        assert _fmt_speedup(1.0) == "1.00x (parity)"

    def test_an_improvement_says_faster(self) -> None:
        from longline.eval.report import _fmt_speedup

        assert _fmt_speedup(2.0) == "2.00x faster"

    def test_unmeasured_is_not_zero(self) -> None:
        """FAILS ON: a None speedup rendered as '0.00x (parity)'."""
        from longline.eval.report import _fmt_speedup

        assert _fmt_speedup(None) == "n/a"

    def test_the_two_formatters_disagree_on_the_same_value(self) -> None:
        """The regression in one line: the same number, two meanings.

        This is the test that would have caught it without knowing which suite
        it came from. `0.1155` as a relative change is a small positive change;
        as a duration ratio it is a large regression.
        """
        from longline.eval.report import _fmt_pct_ratio, _fmt_speedup

        assert _fmt_pct_ratio(0.1155) == "+11.6%"
        assert _fmt_speedup(0.1155) == "0.12x (8.7x slower)"

    def test_the_multi_agent_section_uses_the_ratio_formatter(self) -> None:
        """FAILS ON: the section calling `_fmt_pct_ratio` on a mean speedup."""
        summary = _multi_agent_summary(mean_speedup=0.1155)
        md = render_markdown(aggregate([]), multi_agent={"controlled": summary})
        assert "0.12x (8.7x slower)" in md
        assert "+11.6%" not in md


class TestTokenEfficiency:
    """Three ratios, three denominators, and None where nothing was measured.

    Reported next to the pass rate rather than instead of it: a pass@1 that
    rises while tokens-per-success rises faster is not an improvement. The
    feedback that prompted these named a fourth, `cost_per_success`, but its
    definition (`total_tokens / passed_cases`) is the same quantity as
    TokensPerSuccessfulCase, and this repo has no price table -- a currency
    figure would be invented rather than measured, so it is carried under the
    token name.
    """

    def _tok(
        self, cid: str, *, passed: bool, in_tok: int, out_tok: int, executed: int,
    ) -> CaseResult:
        return CaseResult(
            case_id=cid, case_type="e2e", passed=passed,
            input_tokens=in_tok, output_tokens=out_tok,
            tool_executions=[
                ToolExecution(tool_id=f"t{i}", tool_name="Read", is_error=False)
                for i in range(executed)
            ],
        )

    def test_each_metric_uses_its_own_denominator(self) -> None:
        rep = aggregate([
            self._tok("a", passed=True, in_tok=100, out_tok=10, executed=2),
            self._tok("b", passed=False, in_tok=300, out_tok=30, executed=6),
        ])

        assert rep.total_tokens == 440
        assert rep.tokens_per_case == 220.0            # 440 / 2 cases
        assert rep.tokens_per_successful_case == 440.0  # 440 / 1 passed case
        assert rep.input_tokens_per_tool_call == 50.0   # 400 / 8 executed calls

    def test_an_empty_run_is_unmeasured_not_zero(self) -> None:
        rep = aggregate([])

        assert rep.tokens_per_case is None
        assert rep.tokens_per_successful_case is None
        assert rep.input_tokens_per_tool_call is None

    def test_nothing_passed_leaves_the_per_success_ratio_unmeasured(self) -> None:
        rep = aggregate([self._tok("a", passed=False, in_tok=10, out_tok=1, executed=1)])

        assert rep.tokens_per_successful_case is None
        assert rep.tokens_per_case == 11.0

    def test_no_tool_call_executed_leaves_that_ratio_unmeasured(self) -> None:
        """A run that executed nothing has no per-call cost to report."""
        rep = aggregate([self._tok("a", passed=True, in_tok=10, out_tok=1, executed=0)])

        assert rep.input_tokens_per_tool_call is None

    def test_the_summary_dict_exposes_them_together(self) -> None:
        rep = aggregate([self._tok("a", passed=True, in_tok=10, out_tok=1, executed=1)])
        payload = rep.to_dict()["token_efficiency"]

        assert payload == {
            "tokens_per_case": 11.0,
            "tokens_per_successful_case": 11.0,
            "input_tokens_per_tool_call": 10.0,
        }

    def test_the_markdown_reports_them(self) -> None:
        rep = aggregate([self._tok("a", passed=True, in_tok=10, out_tok=1, executed=1)])
        md = render_markdown(rep)

        assert "Token efficiency" in md
        assert "11.0 tok/case" in md


class TestDedicatedToolPreference:
    """DedicatedToolPreferenceRate: did the agent reach for the dedicated tool?

    The denominator is `dedicated + equivalent-Bash`, so a Bash call the
    equivalence table cannot judge is excluded from BOTH sides. That is what
    keeps the rate from moving when the table is edited.
    """

    def _with_calls(self, cid: str, calls: list[tuple[str, dict[str, Any]]]) -> CaseResult:
        return CaseResult(case_id=cid, case_type="e2e", passed=True, tool_calls=calls)

    def test_counts_the_substitution(self) -> None:
        from longline.eval.report import report_tool_preference

        got = report_tool_preference([self._with_calls("a", [
            ("Read", {"file_path": "x.py"}),
            ("Grep", {"pattern": "p"}),
            ("Bash", {"command": "cat y.py"}),   # should have been Read
            ("Bash", {"command": "pytest -q"}),  # neutral: not a missed Read
        ])])

        assert got.numerator == 2
        assert got.denominator == 3

    def test_a_neutral_bash_call_moves_neither_side(self) -> None:
        from longline.eval.report import report_tool_preference

        got = report_tool_preference([self._with_calls("a", [
            ("Read", {"file_path": "x.py"}),
            ("Bash", {"command": "git status"}),
            ("Bash", {"command": "pytest -q"}),
        ])])

        assert (got.numerator, got.denominator) == (1, 1)

    def test_unmeasured_when_nothing_had_an_alternative(self) -> None:
        from longline.eval.report import report_tool_preference

        got = report_tool_preference([self._with_calls("a", [
            ("Bash", {"command": "git status"}),
        ])])

        assert got.value is None

    def test_write_and_notebook_edit_are_not_in_the_numerator(self) -> None:
        """Neither has a Bash equivalent, so counting them would inflate the
        rate with calls that never had a choice."""
        from longline.eval.report import report_tool_preference

        got = report_tool_preference([self._with_calls("a", [
            ("Write", {"file_path": "x.py"}),
            ("NotebookEdit", {"notebook_path": "n.ipynb"}),
            ("Bash", {"command": "cat y.py"}),
        ])])

        assert (got.numerator, got.denominator) == (0, 1)

    def test_aggregate_exposes_it(self) -> None:
        rep = aggregate([self._with_calls("a", [
            ("Read", {"file_path": "x.py"}),
            ("Bash", {"command": "cat y.py"}),
        ])])

        assert rep.dedicated_tool_preference.numerator == 1
        assert rep.dedicated_tool_preference.denominator == 2
        assert rep.to_dict()["tool_calling"]["dedicated_tool_preference"]["value"] == 0.5


class TestStabilitySection:
    """The report must surface the stability split, not just the pass rate."""

    def _row(self, cid: str, repeat: int, passed: bool, calls: list[str]) -> CaseResult:
        return CaseResult(
            case_id=cid, case_type="e2e", passed=passed, repeat_index=repeat,
            tool_calls=[(n, {}) for n in calls],
        )

    def test_aggregate_carries_the_split(self) -> None:
        rep = aggregate([
            self._row("stable", 0, True, ["Read"]),
            self._row("stable", 1, True, ["Read"]),
            self._row("mixed", 0, True, ["Read"]),
            self._row("mixed", 1, False, ["Bash"]),
            self._row("broken", 0, False, ["Bash"]),
            self._row("broken", 1, False, ["Bash"]),
        ])
        kinds = {s.case_id: s.kind for s in rep.stability}

        assert kinds == {"stable": "stable", "mixed": "mixed", "broken": "always_fail"}

    def test_markdown_names_the_cause_of_each_mixed_case(self) -> None:
        rep = aggregate([
            self._row("mixed", 0, True, ["Glob", "Write", "Write"]),
            self._row("mixed", 1, False, ["Glob", "Write", "Write"]),
        ])
        md = render_markdown(rep)

        assert "## Run stability" in md
        assert "| mixed case | cause | first divergence |" in md
        assert "| mixed | content_driven | - |" in md

    def test_markdown_reports_first_action_consistency(self) -> None:
        rep = aggregate([
            self._row("c", 0, True, ["Read"]),
            self._row("c", 1, True, ["Grep"]),
        ])
        md = render_markdown(rep)

        assert "First-action consistency" in md

    def test_summary_dict_exposes_the_stability_block(self) -> None:
        rep = aggregate([
            self._row("c", 0, True, ["Read"]),
            self._row("c", 1, True, ["Read"]),
        ])
        block = rep.to_dict()["stability"]

        assert block["cases"][0]["case_id"] == "c"
        assert block["first_action_consistency"] == {
            "stable": 1, "mixed": 0, "highly_unstable": 0,
        }
        assert block["redundant_actions"] == {"repeated_reads": 0, "read_after_write": 0}
        assert block["notebook_edit_substitution"] == 0


class TestCacheAwareTokenEfficiency:
    """The efficiency metrics price the WHOLE prompt, not the uncached part.

    On a provider that reports most of the prompt as a cache hit, `input_tokens`
    alone understates the prompt by a large factor -- measured at ~3x on the
    deepseek-flash endpoint. `total_input_tokens` keeps reporting the API's own
    field so historical rows stay readable; `total_prompt_tokens` is what a cost
    question wants, and it is what the per-case figures now use.
    """

    def _cached(
        self, cid: str, *, passed: bool, uncached: int, cached: int, executed: int,
    ) -> CaseResult:
        return CaseResult(
            case_id=cid, case_type="e2e", passed=passed,
            input_tokens=uncached, output_tokens=0,
            cache_read_tokens=cached,
            tool_executions=[
                ToolExecution(tool_id=f"t{i}", tool_name="Read", is_error=False)
                for i in range(executed)
            ],
        )

    def test_prompt_tokens_include_cache_hits(self) -> None:
        rep = aggregate([
            self._cached("a", passed=True, uncached=100, cached=900, executed=1),
            self._cached("b", passed=True, uncached=200, cached=800, executed=1),
        ])

        assert rep.total_input_tokens == 300       # the API's own field
        assert rep.total_prompt_tokens == 2000      # 300 uncached + 1800 cached
        assert rep.tokens_per_case == 1000.0        # over the real prompt
        assert rep.total_cached_tokens == 1700      # 900 + 800
        assert rep.input_tokens_per_tool_call == 1000.0  # 2000 prompt / 2 calls

    def test_a_row_without_cache_fields_keeps_its_old_numbers(self) -> None:
        """The change is additive: pre-existing raw.jsonl rows are unaffected."""
        rep = aggregate([CaseResult(
            case_id="old", case_type="e2e", passed=True,
            input_tokens=500, output_tokens=50,
        )])

        assert rep.total_prompt_tokens == 500
        assert rep.tokens_per_case == 550.0         # 500 + 50 output

    def test_creation_tokens_count_too(self) -> None:
        """A prompt written to cache is still a prompt that was sent."""
        rep = aggregate([CaseResult(
            case_id="a", case_type="e2e", passed=True,
            input_tokens=10, output_tokens=0, cache_creation_tokens=90,
        )])

        assert rep.total_prompt_tokens == 100
        assert rep.tokens_per_case == 100.0

    def test_the_summary_dict_carries_both_totals(self) -> None:
        rep = aggregate([self._cached("a", passed=True, uncached=10, cached=90, executed=1)])
        payload = rep.to_dict()

        assert payload["total_input_tokens"] == 10
        assert payload["total_prompt_tokens"] == 100

    def test_the_markdown_names_both(self) -> None:
        """A reader must be able to see the cached part, not just the total."""
        rep = aggregate([self._cached("a", passed=True, uncached=10, cached=90, executed=1)])
        md = render_markdown(rep)

        assert "cached" in md


def _pair_arm(name: str, *, passed: bool, ms: float, tokens: int) -> Any:
    """One `VariantRun` with a real ledger carrying `tokens`."""
    from longline.eval.child_usage import LEADER, TurnUsage, UsageLedger
    from longline.eval.multi_agent_runner import VariantRun

    ledger = UsageLedger(spawned=[LEADER])
    ledger.record(TurnUsage(agent=LEADER, input_tokens=tokens, output_tokens=0, tool_calls=1))
    return VariantRun(
        variant=name, passed=passed, duration_ms=ms, ledger=ledger,
        accounts={}, subtask_verdicts={},
    )


def _runs_by_category(outcomes: dict[str, list[bool]]) -> list[Any]:
    """One `MultiAgentRun` per verdict, tagged with its category.

    The speedup of run N is `N`, which is arbitrary but distinct: per-case
    ratios that repeated themselves would make a mean and a P50 agree by
    accident, and the block would then pass whether or not it computed two
    different things.
    """
    from longline.eval.multi_agent_runner import MultiAgentRun

    runs = []
    index = 0
    for category, verdicts in outcomes.items():
        for verdict in verdicts:
            index += 1
            runs.append(
                MultiAgentRun(
                    case_id=f"{category}-{index}",
                    group="controlled",
                    workers=2,
                    num_subtasks=2,
                    single=_pair_arm("single", passed=verdict, ms=1000.0, tokens=100),
                    multi=_pair_arm("multi", passed=verdict, ms=1000.0 / index, tokens=300),
                    expected_paths=[],
                    category=category,
                )
            )
    return runs


def _summary_over(runs: list[Any]) -> Any:
    from longline.eval.multi_agent_runner import aggregate_multi_agent

    return aggregate_multi_agent(runs)


class TestSpeedupBlock:
    """Mean AND P50, from ONE list of ratios.

    The mean is kept for comparability with the frozen offline report, which
    publishes `mean over per-case ratios`; switching the only figure to P50
    would make the two runs unreadable against each other. P50 is added because
    a mean of per-case ratios is dragged by whichever case happened to be small.
    """

    def test_both_figures_come_from_the_same_sample(self) -> None:
        """FAILS ON: computing P50 over a different (e.g. filtered) list.

        A mean and a median over two different samples look fine in a report and
        are not comparable to each other, which is the only thing the pair is
        for. The outlier is here to pull them apart so the test can tell.
        """
        from longline.eval.metrics import speedup_block

        block = speedup_block([1.0, 2.0, 3.0, 100.0])

        assert block["speedup_mean"] == pytest.approx(26.5)
        assert block["speedup_p50"] == pytest.approx(2.5)
        assert block["n"] == 4

    def test_no_samples_is_not_a_zero_speedup(self) -> None:
        from longline.eval.metrics import speedup_block

        block = speedup_block([])

        assert block["speedup_mean"] is None
        assert block["speedup_p50"] is None
        assert block["n"] == 0


class TestSuccessPer1KTokens:
    def test_a_zero_denominator_is_none_not_zero(self) -> None:
        """FAILS ON: returning 0.0, which reads as "succeeded never".

        That is a different and much worse claim than "there was nothing to
        divide by" -- a case that made no model calls has no rate at all.
        """
        from longline.eval.metrics import success_per_1k_tokens

        assert success_per_1k_tokens(successes=0, total_tokens=0) is None
        assert success_per_1k_tokens(successes=3, total_tokens=0) is None

    def test_the_rate_is_per_thousand(self) -> None:
        from longline.eval.metrics import success_per_1k_tokens

        assert success_per_1k_tokens(successes=3, total_tokens=1500) == pytest.approx(2.0)


class TestPairReport:
    def test_categories_are_reported_separately(self) -> None:
        """A pooled average is a fact about the corpus mix, not the architecture.

        The three categories are different tasks, so one mean over all 18
        answers "how did this particular 6/6/6 split do" -- a question nobody
        asked, whose answer changes the moment the corpus is rebalanced.
        """
        from longline.eval.multi_agent import (
            CATEGORY_ANALYSIS,
            CATEGORY_DEPENDENT,
            CATEGORY_MODIFICATION,
        )

        summary = _summary_over(_runs_by_category({
            CATEGORY_ANALYSIS: [True, True, False],
            CATEGORY_MODIFICATION: [True, False, False],
            CATEGORY_DEPENDENT: [True, True, True],
        }))

        def rate(category: str) -> float:
            block = summary.by_category[category]["success_rate"]["multi_agent"]
            return float(block["value"])

        assert rate(CATEGORY_ANALYSIS) == pytest.approx(2 / 3)
        assert rate(CATEGORY_MODIFICATION) == pytest.approx(1 / 3)
        assert rate(CATEGORY_DEPENDENT) == pytest.approx(1.0)

    def test_a_multi_category_summary_refuses_to_publish_a_pooled_figure(self) -> None:
        """FAILS ON: a summary that emits a cross-category number anyway.

        `is_pooled` is what the report renders on: when it is True the headline
        row is withheld and only the per-category tables appear. The JSON keeps
        `pooled: None` as the machine-readable statement of the same refusal, so
        a consumer cannot read a pooled value out of the raw artifact either.
        """
        from longline.eval.multi_agent import CATEGORY_ANALYSIS, CATEGORY_MODIFICATION

        summary = _summary_over(_runs_by_category({
            CATEGORY_ANALYSIS: [True, False],
            CATEGORY_MODIFICATION: [True],
        }))

        assert summary.is_pooled is True
        assert summary.to_dict()["pooled"] is None, (
            "no pooled multi-category figure may be emitted at all"
        )

    def test_a_single_category_summary_still_publishes_its_figures(self) -> None:
        """The frozen `multi_agent` suite is one category, and must not regress.

        Refusing EVERY pooled figure would take the frozen baseline's headline
        away with it, and that report is the thing the new numbers are read
        against.
        """
        from longline.eval.multi_agent import CATEGORY_ANALYSIS

        summary = _summary_over(_runs_by_category({CATEGORY_ANALYSIS: [True, False]}))

        assert summary.is_pooled is False
        pooled = summary.to_dict()["pooled"]
        assert pooled is not None
        assert pooled["speedup"]["speedup_p50"] is not None

    def test_each_category_block_carries_its_own_speedup_pair(self) -> None:
        """Per-category, not the group's figures repeated under three headings."""
        from longline.eval.multi_agent import CATEGORY_ANALYSIS, CATEGORY_MODIFICATION

        summary = _summary_over(_runs_by_category({
            CATEGORY_ANALYSIS: [True, False],
            CATEGORY_MODIFICATION: [True, False, False],
        }))

        analysis = summary.by_category[CATEGORY_ANALYSIS]["speedup"]
        modification = summary.by_category[CATEGORY_MODIFICATION]["speedup"]

        assert analysis["n"] == 2
        assert modification["n"] == 3
        assert analysis["speedup_mean"] != modification["speedup_mean"]

    def test_every_block_reports_success_per_1k_tokens(self) -> None:
        """The corpus-level question is "how much does a success cost"."""
        from longline.eval.multi_agent import CATEGORY_ANALYSIS

        summary = _summary_over(_runs_by_category({CATEGORY_ANALYSIS: [True, False]}))

        block = summary.by_category[CATEGORY_ANALYSIS]

        assert block["success_per_1k_tokens"]["multi_agent"] is not None
        assert block["success_per_1k_tokens"]["single_agent"] is not None

    def test_the_report_withholds_the_headline_when_categories_are_pooled(self) -> None:
        """The refusal has to reach the Markdown, not only the dataclass."""
        from longline.eval.multi_agent import CATEGORY_ANALYSIS, CATEGORY_MODIFICATION

        summary = _summary_over(_runs_by_category({
            CATEGORY_ANALYSIS: [True],
            CATEGORY_MODIFICATION: [True],
        }))

        md = render_markdown(aggregate([]), multi_agent={"pair": summary})

        assert "withheld" in md
        assert CATEGORY_ANALYSIS in md
        assert CATEGORY_MODIFICATION in md
