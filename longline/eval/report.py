"""Aggregate CaseResults into an EvalReport and render markdown.

Report shape follows `evals/README.md`: every proportion carries its numerator,
denominator, value and 95% Wilson CI; latency carries mean/p50/p95; success-rate
differences are expressed in **percentage points**, never as a relative percent.

`report.md` is presentation only — it must never be the sole source of a
number. Everything here is recomputable from `raw.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from longline.eval.metrics import Ratio, mean, paired_delta, percentage_points, percentile

if TYPE_CHECKING:
    from longline.eval.runner import CaseResult


@dataclass
class GroupSummary:
    """Per-category / per-variant roll-up (contract §5.1 "report by category")."""

    num_cases: int
    passed: Ratio
    avg_turns: float | None = None
    avg_tool_calls: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_duration_ms: float | None = None
    failure_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_cases": self.num_cases,
            "passed": self.passed.to_dict(),
            "avg_turns": self.avg_turns,
            "avg_tool_calls": self.avg_tool_calls,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_duration_ms": self.avg_duration_ms,
            "failure_types": self.failure_types,
        }


@dataclass
class EvalReport:
    total_cases: int
    # Legacy float fields, kept so existing callers/readers keep working.
    l1_tool_accuracy: float | None = None
    l2_pass1: float | None = None
    avg_turns: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    per_case: list[dict[str, object]] = field(default_factory=list)
    # --- metric-contract shapes (Task 1) ---
    l1_ratio: Ratio = field(default_factory=lambda: Ratio(0, 0))
    l2_ratio: Ratio = field(default_factory=lambda: Ratio(0, 0))
    tool_execution_rate: Ratio = field(default_factory=lambda: Ratio(0, 0))
    mean_duration_ms: float | None = None
    p50_duration_ms: float | None = None
    p95_duration_ms: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_category: dict[str, GroupSummary] = field(default_factory=dict)
    by_variant: dict[str, GroupSummary] = field(default_factory=dict)
    variant: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, object]:
        """JSON-ready summary. Every rate keeps numerator/denominator/CI."""
        return {
            "total_cases": self.total_cases,
            "variant": self.variant,
            "l1_tool_accuracy": self.l1_ratio.to_dict(),
            "l2_pass1": self.l2_ratio.to_dict(),
            "tool_execution_rate": self.tool_execution_rate.to_dict(),
            "avg_turns": self.avg_turns,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": {
                "mean": self.mean_duration_ms,
                "p50": self.p50_duration_ms,
                "p95": self.p95_duration_ms,
            },
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "by_variant": {k: v.to_dict() for k, v in self.by_variant.items()},
            "per_case": self.per_case,
        }


def _category_key(result: CaseResult) -> str:
    """The category a case is bucketed under: first tag, else its case type."""
    if result.tags:
        return result.tags[0]
    return result.case_type


def _summarize_group(results: list[CaseResult]) -> GroupSummary:
    durations = [r.duration_ms for r in results if r.duration_ms is not None]
    return GroupSummary(
        num_cases=len(results),
        passed=Ratio.fraction(r.passed for r in results),
        avg_turns=mean([float(r.turns) for r in results]),
        avg_tool_calls=mean([float(r.num_tool_calls) for r in results]),
        avg_input_tokens=mean([float(r.input_tokens) for r in results]),
        avg_output_tokens=mean([float(r.output_tokens) for r in results]),
        avg_duration_ms=mean(durations),
        failure_types=_failure_type_counts(results),
    )


def _failure_type_counts(results: list[CaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        if r.passed or r.error_type is None:
            continue
        counts[r.error_type] = counts.get(r.error_type, 0) + 1
    return counts


def aggregate(results: list[CaseResult], *, variant: str | None = None) -> EvalReport:
    """Summarize a batch of CaseResults by layer, category, variant and latency."""
    l1 = [r for r in results if r.case_type == "tool_call"]
    l2 = [r for r in results if r.case_type == "e2e"]

    l1_ratio = Ratio.fraction(r.passed for r in l1)
    l2_ratio = Ratio.fraction(r.passed for r in l2)

    executed = sum(r.num_tool_calls_executed for r in results)
    successful = sum(r.num_successful_tool_calls for r in results)

    durations = [r.duration_ms for r in results if r.duration_ms is not None]

    per_case: list[dict[str, object]] = [
        {
            "id": r.case_id,
            "type": r.case_type,
            "tags": r.tags,
            "variant": r.variant,
            "repeat_index": r.repeat_index,
            "passed": r.passed,
            "duration_ms": r.duration_ms,
            "turns": r.turns,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "num_tool_calls": r.num_tool_calls,
            "num_successful_tool_calls": r.num_successful_tool_calls,
            "error_type": r.error_type,
            "tool_calls": [t[0] for t in r.tool_calls],  # tool-name trajectory for failure triage
            "errors": r.errors,
            "detail": r.detail,
        }
        for r in results
    ]

    categories: dict[str, list[CaseResult]] = {}
    for r in results:
        categories.setdefault(_category_key(r), []).append(r)

    variants: dict[str, list[CaseResult]] = {}
    for r in results:
        variants.setdefault(r.variant or "unspecified", []).append(r)

    return EvalReport(
        total_cases=len(results),
        l1_tool_accuracy=l1_ratio.value,
        l2_pass1=l2_ratio.value,
        avg_turns=mean([float(r.turns) for r in results]),
        avg_input_tokens=mean([float(r.input_tokens) for r in results]),
        avg_output_tokens=mean([float(r.output_tokens) for r in results]),
        per_case=per_case,
        l1_ratio=l1_ratio,
        l2_ratio=l2_ratio,
        tool_execution_rate=Ratio(successful, executed),
        mean_duration_ms=mean(durations),
        p50_duration_ms=percentile(durations, 50) if durations else None,
        p95_duration_ms=percentile(durations, 95) if durations else None,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        by_category={k: _summarize_group(v) for k, v in categories.items()},
        by_variant={k: _summarize_group(v) for k, v in variants.items()},
        variant=variant,
    )


def _fmt_pct(ratio: Ratio) -> str:
    """`66.7% (2/3)` — a percentage is meaningless without its denominator."""
    if ratio.value is None:
        return "not measured (0/0)"
    return f"{ratio.value * 100:.1f}% ({ratio.numerator}/{ratio.denominator})"


def _fmt_ci(ratio: Ratio) -> str:
    lo, hi = ratio.ci95_wilson()
    if ratio.denominator == 0:
        return "n/a"
    return f"[{lo * 100:.1f}-{hi * 100:.1f}%]"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def render_markdown(
    report: EvalReport,
    *,
    baseline: EvalReport | None = None,
    baseline_label: str | None = None,
) -> str:
    """Render the report as a compact markdown table + summary lines.

    Every rate prints its numerator/denominator next to the percentage. If a
    `baseline` report is supplied, success-rate differences are rendered in
    **percentage points** (`-3.0 pp`), never as a relative percent.
    """
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- **Total cases:** {report.total_cases}",
        f"- **Tool-call accuracy (L1):** {_fmt_pct(report.l1_ratio)} 95% Wilson CI {_fmt_ci(report.l1_ratio)}",
        f"- **E2E pass@1 (L2):** {_fmt_pct(report.l2_ratio)} 95% Wilson CI {_fmt_ci(report.l2_ratio)}",
        f"- **Tool execution success:** {_fmt_pct(report.tool_execution_rate)}",
        "- **Averages:** "
        f"turns={_fmt_num(report.avg_turns, 2)}, "
        f"in_tok={_fmt_num(report.avg_input_tokens, 0)}, "
        f"out_tok={_fmt_num(report.avg_output_tokens, 0)}",
        "- **Latency (ms):** "
        f"mean={_fmt_ms(report.mean_duration_ms)}, "
        f"p50={_fmt_ms(report.p50_duration_ms)}, "
        f"p95={_fmt_ms(report.p95_duration_ms)}",
        "- **Tokens:** "
        f"input={report.total_input_tokens}, output={report.total_output_tokens}, "
        f"total={report.total_tokens}",
    ]

    if baseline is not None:
        label = baseline_label or "baseline"
        delta = percentage_points(baseline.l2_ratio, report.l2_ratio)
        l1_delta = percentage_points(baseline.l1_ratio, report.l1_ratio)
        b_l2, c_l2 = baseline.l2_ratio.value, report.l2_ratio.value
        if delta is not None and b_l2 is not None and c_l2 is not None:
            lines.append(
                f"- **E2E pass@1 vs {label}:** {delta:+.1f} pp "
                f"({b_l2 * 100:.1f}% -> {c_l2 * 100:.1f}%)"
            )
        if l1_delta is not None:
            lines.append(f"- **Tool-call accuracy vs {label}:** {l1_delta:+.1f} pp")

    if report.by_category:
        lines += [
            "",
            "## By category",
            "",
            "| category | pass | turns | tool_calls | in_tok | out_tok | p50 ms | failure types |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, group in sorted(report.by_category.items()):
            ratio = Ratio(group.passed.numerator, group.passed.denominator)
            failures = ", ".join(f"{k}:{v}" for k, v in sorted(group.failure_types.items())) or "-"
            lines.append(
                f"| {name} | {_fmt_pct(ratio)} | {_fmt_num(group.avg_turns, 2)} | "
                f"{_fmt_num(group.avg_tool_calls, 2)} | "
                f"{_fmt_num(group.avg_input_tokens, 0)} | "
                f"{_fmt_num(group.avg_output_tokens, 0)} | "
                f"{_fmt_ms(group.avg_duration_ms)} | {failures} |"
            )

    lines += [
        "",
        "| case | type | passed | ms | turns | in_tok | out_tok | error_type |",
        "|------|------|--------|----|-------|--------|---------|------------|",
    ]
    for c in report.per_case:
        lines.append(
            f"| {c['id']} | {c['type']} | {c['passed']} | "
            f"{_fmt_ms(_as_float(c['duration_ms']))} | "
            f"{c['turns']} | {c['input_tokens']} | {c['output_tokens']} | "
            f"{c['error_type'] or '-'} |"
        )
    return "\n".join(lines)


def _as_float(value: object) -> float | None:
    """Coerce a per_case row field back to float for the ms formatter."""
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt_num(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def paired_report_delta(
    baseline: list[CaseResult], candidate: list[CaseResult]
) -> dict[str, Any]:
    """Per-case paired deltas between two aligned runs.

    Cases are matched by (case_id, repeat_index) and both runs must cover the
    same set; a mismatch raises rather than silently dropping cases, because a
    silently-shrunk denominator is a wrong number that looks right.
    """
    def _index(results: list[CaseResult]) -> dict[tuple[str, int], CaseResult]:
        return {(r.case_id, r.repeat_index): r for r in results}

    b_idx = _index(baseline)
    c_idx = _index(candidate)
    if set(b_idx) != set(c_idx):
        only_b = sorted(set(b_idx) - set(c_idx))
        only_c = sorted(set(c_idx) - set(b_idx))
        raise ValueError(
            f"paired runs are not aligned: only in baseline={only_b}, only in candidate={only_c}"
        )
    keys = sorted(b_idx)
    durations = paired_delta(
        [b_idx[k].duration_ms or 0.0 for k in keys],
        [c_idx[k].duration_ms or 0.0 for k in keys],
        baseline_ids=[k[0] for k in keys],
        candidate_ids=[k[0] for k in keys],
    )
    turns = paired_delta(
        [float(b_idx[k].turns) for k in keys],
        [float(c_idx[k].turns) for k in keys],
    )
    return {
        "n_pairs": len(keys),
        "duration_ms": durations.to_dict(),
        "turns": turns.to_dict(),
    }
