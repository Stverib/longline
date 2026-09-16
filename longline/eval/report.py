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

from longline.eval.metrics import (
    Ratio,
    mean,
    paired_delta,
    percentage_points,
    percentile,
    reliability_ratio,
)
from longline.eval.types import E2E_CATEGORY_TAGS

if TYPE_CHECKING:
    from longline.eval.compression_runner import CompressionSummary
    from longline.eval.latency_runner import LatencySummary
    from longline.eval.multi_agent_runner import MultiAgentSummary
    from longline.eval.runner import CaseResult

# Case tags that decide the ToolSelectionCaseAccuracy denominator. A blind case
# is one whose task text does not name or hint at the expected tool; an
# instruction-following case names it outright and is therefore a regression
# check on following orders, not a measurement of tool *selection*.
BLIND_TAG = "blind"
INSTRUCTION_FOLLOWING_TAG = "instruction-following"


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
    # pass^k (tau-bench, arXiv:2406.12045): the fraction of E2E tasks on which
    # EVERY repeat passed. Reported next to `l2_ratio` because the two answer
    # different questions -- `l2_ratio` is the mean success rate, `l2_pass_k` is
    # the reliability of that mean. Three runs of 1/1/1 and 1/0/1 are identical
    # under the former (0.667) and opposite under the latter (1.0 vs 0.0).
    l2_pass_k: Ratio = field(default_factory=lambda: Ratio(0, 0))
    l2_pass_k_k: int = 0
    l2_repeats: int = 0
    # Pass rate by how many tool calls the case actually needed. Diagnostic
    # only: the count never enters a pass condition (plan §4.1).
    by_tool_call_bucket: dict[str, GroupSummary] = field(default_factory=dict)
    tool_execution_rate: Ratio = field(default_factory=lambda: Ratio(0, 0))
    # --- Task 2: the four tool-calling metrics, each with its own denominator ---
    # `tool_selection_case_accuracy` is the resume-facing number and excludes
    # instruction-following cases by construction (see `aggregate`).
    tool_selection_case_accuracy: Ratio = field(default_factory=lambda: Ratio(0, 0))
    tool_call_precision: Ratio = field(default_factory=lambda: Ratio(0, 0))
    argument_call_accuracy: Ratio = field(default_factory=lambda: Ratio(0, 0))
    argument_field_accuracy: Ratio = field(default_factory=lambda: Ratio(0, 0))
    # Reported separately, never folded into the number above.
    instruction_following_case_accuracy: Ratio = field(default_factory=lambda: Ratio(0, 0))
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
            # Reliability, not accuracy: the fraction of tasks passing ALL k
            # repeats. `l2_pass_k_k` records which k this was computed at, since
            # a pass^1 and a pass^3 are not the same claim.
            "l2_pass_k": self.l2_pass_k.to_dict(),
            "l2_pass_k_k": self.l2_pass_k_k,
            "l2_repeats": self.l2_repeats,
            "tool_execution_rate": self.tool_execution_rate.to_dict(),
            # Task 2: four metrics, four independent denominators.
            "tool_calling": {
                "tool_selection_case_accuracy": self.tool_selection_case_accuracy.to_dict(),
                "tool_call_precision": self.tool_call_precision.to_dict(),
                "argument_call_accuracy": self.argument_call_accuracy.to_dict(),
                "argument_field_accuracy": self.argument_field_accuracy.to_dict(),
                "execution_success_rate": self.tool_execution_rate.to_dict(),
                "instruction_following_case_accuracy": (
                    self.instruction_following_case_accuracy.to_dict()
                ),
            },
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
        # Diagnostic stratification, not a pass condition (plan §4.1).
        "by_tool_call_bucket": {k: v.to_dict() for k, v in self.by_tool_call_bucket.items()},
            "by_variant": {k: v.to_dict() for k, v in self.by_variant.items()},
            "per_case": self.per_case,
        }


def _category_key(result: CaseResult) -> str:
    """The category a case is bucketed under.

    Prefers membership in `E2E_CATEGORY_TAGS` over tag *position*. Taking
    `tags[0]` happened to give the right answer only because the case data
    lists the category first (`["retrieval", "answer"]`), so any author who
    reordered a case's tags would silently move it to a different bucket in
    the report without changing a thing about the case. Selecting on the
    explicit category tag removes that dependence on ordering; the first-tag
    and case-type fallbacks remain for tool-call cases and ad-hoc runs whose
    tags are not E2E categories.
    """
    for tag in result.tags:
        if tag in E2E_CATEGORY_TAGS:
            return tag
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


def _tool_call_buckets(results: list[CaseResult]) -> dict[str, GroupSummary]:
    """Pass rate bucketed by how many tool calls the case actually needed.

    The contract (plan §4.1) requires the long-chain category's final result and
    its call count to be reported SEPARATELY: the call count is diagnostic and
    must never decide pass/fail. This is that split -- the number of calls is
    the bucketing key, never a component of the pass condition.

    The field standard for multi-step credit assignment is a per-class
    breakdown (OSWorld and WebArena break down by domain, GAIA by difficulty
    level) rather than per-step scoring, which is what this does. Bucketing also
    converts an admitted weak spot into a reported stratification: it shows
    whether success degrades as the chain lengthens, which a single aggregate
    success rate hides.

    Empty buckets are omitted rather than reported as 0%, so a bucket of zero
    cases cannot be misread as a 0% success rate.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        if r.case_type != "e2e":
            continue
        n = r.num_tool_calls
        # Coarse, human-readable bands; the boundaries are a presentation
        # choice, stated here so the report can name them.
        if n <= 2:
            label = "0-2 calls"
        elif n <= 5:
            label = "3-5 calls"
        elif n <= 9:
            label = "6-9 calls"
        else:
            label = "10+ calls"
        buckets.setdefault(label, []).append(r)
    return {label: _summarize_group(v) for label, v in buckets.items()}


def category_metrics(
    results: list[CaseResult],
    category: str,
    *,
    case_type: str | None = "e2e",
) -> GroupSummary:
    """Per-category aggregates, over the cases whose tags name this category.

    The contract (§5.1) asks for each E2E category's success rate, average
    rounds, tool calls, tokens, duration and failure types. `aggregate` already
    buckets by **first tag**, which is right for the reporting layout but wrong
    for a suite where a case can legitimately carry more than one tag: a case
    tagged `["file-ops", "create"]` would then be invisible to a `create` query,
    and one tagged in the other order would appear twice across two queries.

    So this helper selects on *any* tag and ignores the bucket layout, which
    makes the per-category numbers independent of tag order. `case_type` keeps
    an E2E query from silently picking up tool-call cases that happen to share
    a tag; pass None to search every case type.
    """
    selected = [
        r for r in results
        if category in r.tags and (case_type is None or r.case_type == case_type)
    ]
    return _summarize_group(selected)


def _failure_type_counts(results: list[CaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        if r.passed or r.error_type is None:
            continue
        counts[r.error_type] = counts.get(r.error_type, 0) + 1
    return counts


def _tool_calling_totals(
    results: list[CaseResult],
) -> dict[str, Ratio]:
    """The four tool-calling ratios, summed over tool-call cases.

    Reading order matters here: `select` counts **cases**, the other three count
    **calls** or **fields**. They are summed separately on purpose — a single
    loop that accumulated one "score" per case is what fused the legacy metric
    and made its denominator unrecoverable.
    """
    blind = [r for r in results if BLIND_TAG in r.tags]
    instruction = [r for r in results if INSTRUCTION_FOLLOWING_TAG in r.tags]

    # Denominator: blind tool-call cases only (contract §5.2 exclusion rule).
    #
    # Abstention cases need a different rule from ordinary cases. An ordinary
    # case passes when every expected step was matched. An abstention case
    # expects NO step, so `all_steps_matched` is vacuously true for it and would
    # pass even when the agent called a pile of irrelevant tools. `abstention_ok`
    # is the real condition: call nothing, and no extra calls.
    select_num = sum(
        1 for r in blind
        if (r.abstention_ok if r.is_abstention_case else r.steps_completed)
    )
    # Denominator: EVERY requested call, extras included.
    calls = sum(r.num_tool_calls for r in results)
    matched = sum(r.num_matched_tool_calls for r in results)
    # Denominator: matched calls whose arguments the case declares.
    arg_calls_checked = sum(r.num_arg_checked_calls for r in results)
    arg_calls_ok = sum(r.num_arg_correct_calls for r in results)
    # Denominator: declared argument fields that were actually checked.
    arg_fields_checked = sum(r.num_arg_checked_fields for r in results)
    arg_fields_ok = sum(r.num_arg_correct_fields for r in results)

    return {
        "tool_selection_case_accuracy": Ratio(select_num, len(blind)),
        "tool_call_precision": Ratio(matched, calls),
        "argument_call_accuracy": Ratio(arg_calls_ok, arg_calls_checked),
        "argument_field_accuracy": Ratio(arg_fields_ok, arg_fields_checked),
        "instruction_following_case_accuracy": Ratio(
            sum(1 for r in instruction if r.steps_completed), len(instruction)
        ),
    }


def _repeat_outcomes(results: list[CaseResult]) -> tuple[list[int], list[int]]:
    """Group results by case id into (successes per case, trials per case).

    A 3-repeat suite appends three CaseResults per case, differing only in
    `repeat_index`. Collapsing them here is what makes pass^k computable: the
    mean success rate (pass@1) over those three runs cannot tell an agent that
    passed 1/1/1 from one that passed 1/0/1, while pass^3 can.
    """
    trials: dict[str, list[bool]] = {}
    for r in results:
        trials.setdefault(r.case_id, []).append(bool(r.passed))
    successes = [sum(1 for ok in v if ok) for v in trials.values()]
    counts = [len(v) for v in trials.values()]
    return successes, counts


def aggregate(results: list[CaseResult], *, variant: str | None = None) -> EvalReport:
    """Summarize a batch of CaseResults by layer, category, variant and latency."""
    l1 = [r for r in results if r.case_type == "tool_call"]
    l2 = [r for r in results if r.case_type == "e2e"]

    l1_ratio = Ratio.fraction(r.passed for r in l1)
    l2_ratio = Ratio.fraction(r.passed for r in l2)

    # pass^k over the E2E set. With repeats == 1 this is just the pass rate
    # again, which is why it is only meaningful on a multi-repeat run.
    e2e_successes, e2e_trials = _repeat_outcomes(l2)
    k = min(e2e_trials) if e2e_trials else 0
    e2e_pass_k = reliability_ratio(e2e_successes, e2e_trials, k) if k >= 1 else Ratio(0, 0)

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

    tool_metrics = _tool_calling_totals(l1)

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
        l2_pass_k=e2e_pass_k,
        l2_pass_k_k=k,
        l2_repeats=max(e2e_trials) if e2e_trials else 0,
        tool_execution_rate=Ratio(successful, executed),
        tool_selection_case_accuracy=tool_metrics["tool_selection_case_accuracy"],
        tool_call_precision=tool_metrics["tool_call_precision"],
        argument_call_accuracy=tool_metrics["argument_call_accuracy"],
        argument_field_accuracy=tool_metrics["argument_field_accuracy"],
        instruction_following_case_accuracy=tool_metrics["instruction_following_case_accuracy"],
        mean_duration_ms=mean(durations),
        p50_duration_ms=percentile(durations, 50) if durations else None,
        p95_duration_ms=percentile(durations, 95) if durations else None,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        by_category={k: _summarize_group(v) for k, v in categories.items()},
        by_tool_call_bucket=_tool_call_buckets(l2),
        by_variant={k: _summarize_group(v) for k, v in variants.items()},
        variant=variant,
    )


def _fmt_pct(ratio: Ratio) -> str:
    """`66.7% (2/3)` — a percentage is meaningless without its denominator."""
    if ratio.value is None:
        return "not measured (0/0)"
    return f"{ratio.value * 100:.1f}% ({ratio.numerator}/{ratio.denominator})"


def _fmt_ci(ratio: Ratio) -> str:
    """`[10.8-60.3%]`, or `n/a` when the ratio was not measured."""
    ci = ratio.ci95_wilson()
    if ci is None:
        return "n/a"
    lo, hi = ci
    return f"[{lo * 100:.1f}-{hi * 100:.1f}%]"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def _metric_row(label: str, ratio: Ratio, definition: str) -> str:
    """One markdown row: value, CI, and the fraction the value came from.

    The definition column carries the numerator/denominator wording so a reader
    can tell which of the four ratios they are looking at without opening the
    contract — the four names are similar and the whole point of the split is
    that they are *not* interchangeable.
    """
    return (
        f"| {label} | {_fmt_pct(ratio)} | {_fmt_ci(ratio)} | "
        f"{definition} = {ratio.numerator}/{ratio.denominator} |"
    )


def render_markdown(
    report: EvalReport,
    *,
    baseline: EvalReport | None = None,
    baseline_label: str | None = None,
    compression: CompressionSummary | None = None,
    latency: LatencySummary | None = None,
    multi_agent: dict[str, MultiAgentSummary] | None = None,
) -> str:
    """Render the report as a compact markdown table + summary lines.

    Every rate prints its numerator/denominator next to the percentage. If a
    `baseline` report is supplied, success-rate differences are rendered in
    **percentage points** (`-3.0 pp`), never as a relative percent.

    `compression` adds the Task 4 section, `latency` the Task 6 one and
    `multi_agent` the Task 7 one. They are separate arguments rather than fields
    on `EvalReport` because none of those suites' metrics compose with the E2E
    ones: compression has its own denominators (facts, eligible cases), latency's
    headline number is a ratio of two durations, and multi-agent's is a ratio of
    durations plus a ratio of token counts. Folding any of them in would invite
    someone to average across suites.

    `multi_agent` is keyed by GROUP (`controlled` / `exploratory`) and rendered
    as one section per group. Contract §5.6 requires those to be reported
    separately -- a controlled case pre-declares its subtasks so both arms do the
    same work, while an exploratory case's coordinator decomposes freely, so a
    pooled number would be comparing different work under one heading.
    """
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- **Total cases:** {report.total_cases}",
        f"- **Tool-call accuracy (L1):** {_fmt_pct(report.l1_ratio)} 95% Wilson CI {_fmt_ci(report.l1_ratio)}",
        f"- **E2E pass@1 (L2):** {_fmt_pct(report.l2_ratio)} 95% Wilson CI {_fmt_ci(report.l2_ratio)}",
        # Reliability alongside accuracy. `pass@1` is the mean success rate;
        # `pass^k` is the fraction of tasks that passed EVERY repeat. A suite
        # whose runs alternate pass/fail has a high mean and a low pass^k.
        (
            f"- **E2E pass^{report.l2_pass_k_k} (reliability, {report.l2_repeats} runs/case):** "
            f"{_fmt_pct(report.l2_pass_k)}"
            if report.l2_repeats > 1
            else "- **E2E pass^k:** n/a (needs --repeats >= 2)"
        ),
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

    if report.tool_selection_case_accuracy.denominator or report.tool_call_precision.denominator:
        lines += [
            "",
            "## Tool calling (four metrics, independent denominators)",
            "",
            "| metric | value | 95% Wilson CI | numerator / denominator |",
            "|---|---|---|---|",
            _metric_row(
                "ToolSelectionCaseAccuracy (blind only)",
                report.tool_selection_case_accuracy, "完成全部决策步骤的盲测用例 / 盲测用例",
            ),
            _metric_row(
                "ToolCallPrecision",
                report.tool_call_precision, "匹配有效步骤的调用 / 全部工具调用",
            ),
            _metric_row(
                "ArgumentCallAccuracy",
                report.argument_call_accuracy, "参数整体正确的调用 / 需校验参数的调用",
            ),
            _metric_row(
                "ArgumentFieldAccuracy",
                report.argument_field_accuracy, "正确参数字段 / 被检查参数字段",
            ),
            _metric_row(
                "ExecutionSuccessRate",
                report.tool_execution_rate, "is_error=false 的执行 / 实际执行",
            ),
            _metric_row(
                "InstructionFollowingCaseAccuracy (separate, not in the number above)",
                report.instruction_following_case_accuracy, "显式指定工具的用例 / 该类用例",
            ),
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

    if compression is not None:
        lines += _compression_lines(compression)

    if latency is not None:
        lines += _latency_lines(latency)

    if multi_agent:
        lines += _multi_agent_lines(multi_agent)

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


def _compression_lines(summary: CompressionSummary) -> list[str]:
    """The Task 4 section: four metrics, four denominators, plus the fact trace.

    Two things this section deliberately does NOT do:

    - It does not render `SuccessDeltaPP` as a relative percent. The contract
      forbids "down 3%" for a 3-point drop, so the unit is printed as `pp` and
      the two underlying rates are shown next to it.
    - It does not drop the excluded cases. A case whose baseline failed leaves
      the degradation denominator (contract §5.3) but is listed here with its
      reason, because a silently-shrunk denominator is a wrong number that
      looks right.
    """
    ratio = summary.compression_ratio
    ratio_text = "not measured" if ratio is None else f"{ratio * 100:.1f}%"

    lines = [
        "",
        "## Context compression (paired A/B, estimated tokens)",
        "",
        "> Token counts are **estimated** via `estimate_messages_tokens()`, not a "
        "tokenizer's count. `CompressionRatio` is therefore a ratio between two "
        "estimates of the same region, which is what makes it usable despite the "
        "absolute value under-counting (system prompt and tool schemas excluded).",
        "",
        "| metric | value | 95% Wilson CI | numerator / denominator |",
        "|---|---|---|---|",
        f"| CompressionRatio | {ratio_text} | n/a (a ratio, not a proportion) | "
        f"mean of {len(summary.compression_ratio_per_case)} per-case ratios |",
        _metric_row(
            "KeyInfoRetention", summary.key_info_retention,
            "压缩后仍能正确使用的关键事实 / 关键事实总数",
        ),
        _metric_row(
            "PostCompressionSuccessRate", summary.post_compression_success_rate,
            "压缩后任务成功数 / 可计入分母的压缩任务数",
        ),
        (
            f"| SuccessDeltaPP | {summary.success_delta_pp:+.1f} pp | n/a | "
            f"candidate {_fmt_pct(summary.post_compression_success_rate)} - "
            f"baseline {_fmt_pct(summary.baseline_success_rate)} |"
            if summary.success_delta_pp is not None
            else "| SuccessDeltaPP | not measured | n/a | one side unmeasured |"
        ),
        "",
        f"- **Cases:** {summary.num_cases} total, {summary.eligible_cases} eligible, "
        f"{summary.excluded_cases} excluded (baseline failed)",
        f"- **Mean estimated tokens:** before="
        f"{_fmt_num(summary.mean_tokens_before, 1)}, after="
        f"{_fmt_num(summary.mean_tokens_after, 1)} "
        f"(paired mean delta={_fmt_num(summary.paired_tokens.mean, 1)})",
        "",
        "### Per-case fact trace",
        "",
        "Each fact is scored by a follow-up question whose answer IS the fact, "
        "never by searching the summary text: *the summary contains the path* is "
        "not *the agent can still use the path*.",
        "",
        "| case | baseline | candidate | ratio | retained | lost facts | excluded |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary.per_case:
        case_id = str(row["case_id"])
        per_case_ratio = summary.compression_ratio_per_case.get(case_id)
        ratio_cell = "n/a" if per_case_ratio is None else f"{per_case_ratio * 100:.1f}%"
        # `lost_fact_ids` is `object` on the row dict, so the narrowing is
        # explicit: a non-list value reads as "no facts lost" rather than
        # exploding the renderer, which runs after a paid suite has finished.
        lost_value = row["lost_fact_ids"]
        lost = [str(f) for f in lost_value] if isinstance(lost_value, list) else []
        lost_cell = ", ".join(lost) if lost else "-"
        excluded = "yes" if row["excluded_from_denominator"] else "-"
        if row["exclusion_reason"]:
            excluded = f"yes ({row['exclusion_reason']})"
        lines.append(
            f"| {case_id} | {row['baseline_passed']} | {row['candidate_passed']} | "
            f"{ratio_cell} | {row['retained_facts']}/{row['num_facts']} | "
            f"{lost_cell} | {excluded} |"
        )
    return lines


def _latency_lines(summary: LatencySummary) -> list[str]:
    """The Task 6 section: streaming vs buffered, per case, with the paired delta.

    Three things this section is careful about:

    - **`LatencyReduction` is a ratio of two durations, not a `pp` difference.**
      The contract's `pp` unit (§4.3) belongs to success rates; a 20 ms saving on
      a 200 ms turn is 10%, and calling it "10 pp" would invent a numerator and
      denominator that do not exist. So the unit is printed as a percent *change*
      and the two underlying durations are shown next to it.
    - **The paired delta is per case**, as §4.2 requires of any A/B latency
      comparison: the mean of the two arms can agree while individual cases move
      in opposite directions, and only the per-case column shows that.
    - **The warmup count and the sample count are printed.** A latency mean
      without its sample size and its excluded warmups is not reproducible.
    """
    lines = [
        "",
        "## Streaming tool latency (paired A/B, scripted stream)",
        "",
        "> `LatencyReduction` is `(buffered - streaming) / buffered` on "
        "**durations**, so it is a ratio, **not** a difference in percentage "
        "points (contract §5.5 / §4.3). `pp` is reserved for success rates.",
        "",
        f"- **Samples:** {summary.samples_per_arm} per arm per case, "
        f"{summary.warmups_per_arm} warmup pairs excluded",
        f"- **Time scale:** {summary.time_scale:g}x (declared case durations x this)",
        "- **Units:** milliseconds",
        "",
        "| case | arm | ToolStartLatency mean | p50 | p95 | TurnLatency "
        "mean | OverlapTime mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in summary.cases:
        for arm in ("buffered", "streaming"):
            m = case.metrics[arm]
            lines.append(
                f"| {case.case_id} | {arm} | "
                f"{_fmt_ms(m['tool_start_latency_ms_mean'])} | "
                f"{_fmt_ms(m['tool_start_latency_ms_p50'])} | "
                f"{_fmt_ms(m['tool_start_latency_ms_p95'])} | "
                f"{_fmt_ms(m['turn_latency_ms_mean'])} | "
                f"{_fmt_ms(m['overlap_time_ms_mean'])} |"
            )
    lines += [
        "",
        "### Paired delta per case (ToolStartLatency)",
        "",
        "| case | reduction (ratio of durations) | mean of per-sample ratios | "
        "p50 of per-sample ratios | samples streaming faster / slower | case note |",
        "|---|---|---|---|---|---|",
    ]
    for case in summary.cases:
        reduction = "not measured" if case.reduction is None else f"{case.reduction * 100:+.1f}%"
        lines.append(
            f"| {case.case_id} | {reduction} | "
            f"{_fmt_pct_ratio(case.reduction_mean)} | "
            f"{_fmt_pct_ratio(case.reduction_p50)} | "
            f"{case.lifetime_samples} / {case.overlap_samples} | {case.note} |"
        )
    return lines


def _fmt_pct_ratio(value: float | None) -> str:
    """A ratio rendered as a signed percent change; `pp` is never used here."""
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _multi_agent_lines(summaries: dict[str, MultiAgentSummary]) -> list[str]:
    """The Task 7 section: one block per group, never one pooled number.

    Three things this section is careful about:

    - **`Speedup` and `TokenOverhead` are ratios, not `pp` differences.** The
      contract's `pp` unit (§4.3) belongs to success rates; a 2x speedup is
      `+100%` of a duration ratio, and calling it "100 pp" would invent a
      numerator and denominator that do not exist. So they are printed as signed
      percent changes with their units named.
    - **The child tokens are shown next to every total.** The §5.6 red line is
      that the leader's usage alone is not the run's usage, so a reader must be
      able to see how much of the fan-out's cost came from its children -- and a
      bug that dropped them would show as `child=0` rather than as a total that
      merely looks small.
    - **The excluded cases are listed, not dropped.** A case whose accounting did
      not reconcile leaves the ratio denominators but stays visible with its
      reason, because a silently-shrunk denominator is a wrong number that looks
      right -- the same rule the compression section follows.
    """
    lines: list[str] = [
        "",
        "## Single-agent vs multi-agent (paired A/B)",
        "",
        "> `Speedup` is `single_wall_time / multi_wall_time` and `TokenOverhead` is "
        "`(multi - single) / single` -- both **ratios**, never differences in "
        "percentage points (contract §5.6 / §4.3). `pp` is reserved for success "
        "rates. Tokens include **every** sub-agent, not only the leader.",
    ]

    for group, summary in sorted(summaries.items()):
        lines += [
            "",
            f"### Group: `{group}`",
            "",
            (
                "> Controlled cases pre-declare their independent subtasks, so both "
                "variants do the same work. Exploratory cases let the coordinator "
                "decompose freely and are **not** comparable with them."
                if group == "controlled"
                else "> Exploratory cases let the coordinator decompose freely; the "
                "two variants may not have done the same work, so this group is "
                "reported on its own and never merged with the controlled number."
            ),
            "",
            f"- **Cases:** {summary.num_cases} total, {summary.eligible_cases} eligible, "
            f"{summary.excluded_cases} excluded (a variant did not complete or its "
            "token accounting did not reconcile)",
            f"- **Agent counts (multi arm):** {summary.agent_counts or 'n/a'}",
            "",
            "| metric | value | 95% Wilson CI | numerator / denominator |",
            "|---|---|---|---|",
            _metric_row(
                "SuccessRate single_agent",
                summary.single_success_rate,
                "判分通过的单 Agent 用例 / 用例总数",
            ),
            _metric_row(
                "SuccessRate multi_agent",
                summary.multi_success_rate,
                "判分通过的多 Agent 用例 / 用例总数",
            ),
            (
                f"| Speedup | {_fmt_pct_ratio(summary.mean_speedup)} "
                f"(ratio of durations) | n/a | mean over "
                f"{summary.eligible_cases} per-case `single / multi` ratios |"
            ),
            (
                f"| TokenOverhead | {_fmt_pct_ratio(summary.mean_token_overhead)} "
                f"(ratio) | n/a | mean over {summary.eligible_cases} per-case "
                "`(multi - single) / single` ratios |"
            ),
            "",
            "| variant | WallClockTime | in tok | out tok | total tok | child tok | tool calls |",
            "|---|---|---|---|---|---|---|",
            f"| single_agent | {_fmt_ms(summary.single_wall_time_ms)} | "
            f"{summary.single_tokens['input_tokens']} | "
            f"{summary.single_tokens['output_tokens']} | "
            f"{summary.single_tokens['total_tokens']} | "
            f"{summary.single_tokens['child_tokens']} | {summary.single_tool_calls} |",
            f"| multi_agent | {_fmt_ms(summary.multi_wall_time_ms)} | "
            f"{summary.multi_tokens['input_tokens']} | "
            f"{summary.multi_tokens['output_tokens']} | "
            f"{summary.multi_tokens['total_tokens']} | "
            f"{summary.multi_tokens['child_tokens']} | {summary.multi_tool_calls} |",
            "",
            "### Per-case (this group)",
            "",
            "| case | workers | single pass | multi pass | single ms | multi ms | "
            "speedup | single tok | multi tok | multi child tok | excluded |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in summary.per_case:
            single = row["single"]
            multi = row["multi"]
            assert isinstance(single, dict) and isinstance(multi, dict)
            single_ms = _as_float(single["duration_ms"])
            multi_ms = _as_float(multi["duration_ms"])
            speedup = (
                None
                if not single_ms or not multi_ms
                else single_ms / multi_ms
            )
            excluded = "yes" if row["excluded_from_denominator"] else "-"
            if row["exclusion_reason"]:
                excluded = f"yes ({row['exclusion_reason']})"
            lines.append(
                f"| {row['case_id']} | {row['workers']} | {single['passed']} | "
                f"{multi['passed']} | {_fmt_ms(single_ms)} | {_fmt_ms(multi_ms)} | "
                f"{_fmt_pct_ratio(speedup)} | {single['total_tokens']} | "
                f"{multi['total_tokens']} | {multi['child_tokens']} | {excluded} |"
            )
    return lines


def _fmt_num(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def paired_report_delta(
    baseline: list[CaseResult], candidate: list[CaseResult]
) -> dict[str, Any]:
    """Per-case paired deltas between two aligned runs.

    Cases are matched by (case_id, repeat_index) and both runs must cover the
    same set; a mismatch raises rather than silently dropping cases, because a
    silently-shrunk denominator is a wrong number that looks right.

    Unmeasured values are handled by **exclusion, not coercion**. A
    `duration_ms` of None means "not measured" — substituting 0.0 would drag
    the paired mean toward zero while looking like a real observation, which is
    the exact failure mode `Ratio.value` and `percentile` are built to avoid. A
    pair is dropped only when one of its two durations is missing; the dropped
    count is reported as `n_pairs_duration_excluded` so the denominator
    difference is visible rather than implied.

    `turns` needs no such handling: it is a plain int that is always known.
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

    timed = [
        k for k in keys
        if b_idx[k].duration_ms is not None and c_idx[k].duration_ms is not None
    ]
    durations = paired_delta(
        [b_idx[k].duration_ms for k in timed],  # type: ignore[misc]
        [c_idx[k].duration_ms for k in timed],  # type: ignore[misc]
        baseline_ids=[k[0] for k in timed],
        candidate_ids=[k[0] for k in timed],
    )
    turns = paired_delta(
        [float(b_idx[k].turns) for k in keys],
        [float(c_idx[k].turns) for k in keys],
    )
    return {
        "n_pairs": len(keys),
        "n_pairs_duration": len(timed),
        "n_pairs_duration_excluded": len(keys) - len(timed),
        "duration_ms": durations.to_dict(),
        "turns": turns.to_dict(),
    }
