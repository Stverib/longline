"""Aggregate CaseResults into an EvalReport and render markdown.

Report shape (these numbers are what the resume quantifies):
  - Layer 1 tool-call accuracy (passed tool_call cases / total)
  - Layer 2 E2E pass@1 (passed e2e cases / total)
  - Average turns and token cost across all cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from longline.eval.runner import CaseResult


@dataclass
class EvalReport:
    total_cases: int
    l1_tool_accuracy: float | None = None  # pass rate of tool_call cases
    l2_pass1: float | None = None          # pass rate of e2e cases
    avg_turns: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    per_case: list[dict[str, object]] = field(default_factory=list)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: list[CaseResult]) -> EvalReport:
    """Summarize a batch of CaseResults by layer and by averages."""
    l1 = [r for r in results if r.case_type == "tool_call"]
    l2 = [r for r in results if r.case_type == "e2e"]

    l1_acc = sum(r.passed for r in l1) / len(l1) if l1 else None
    l2_pass = sum(r.passed for r in l2) / len(l2) if l2 else None

    per_case = [
        {
            "id": r.case_id,
            "type": r.case_type,
            "passed": r.passed,
            "turns": r.turns,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "tool_calls": [t[0] for t in r.tool_calls],  # tool-name trajectory for failure triage
            "errors": r.errors,
            "detail": r.detail,
        }
        for r in results
    ]

    return EvalReport(
        total_cases=len(results),
        l1_tool_accuracy=l1_acc,
        l2_pass1=l2_pass,
        avg_turns=_mean([float(r.turns) for r in results]),
        avg_input_tokens=_mean([float(r.input_tokens) for r in results]),
        avg_output_tokens=_mean([float(r.output_tokens) for r in results]),
        per_case=per_case,
    )


def render_markdown(report: EvalReport) -> str:
    """Render the report as a compact markdown table + summary lines."""
    l1_line = (
        f"- **Tool-call accuracy (L1):** {report.l1_tool_accuracy * 100:.1f}%"
        if report.l1_tool_accuracy is not None
        else "- **Tool-call accuracy (L1):** n/a"
    )
    l2_line = (
        f"- **E2E pass@1 (L2):** {report.l2_pass1 * 100:.1f}%"
        if report.l2_pass1 is not None
        else "- **E2E pass@1 (L2):** n/a"
    )
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- **Total cases:** {report.total_cases}",
        l1_line,
        l2_line,
        "- **Averages:** "
        f"turns={report.avg_turns:.2f}, "
        f"in_tok={report.avg_input_tokens:.0f}, "
        f"out_tok={report.avg_output_tokens:.0f}",
        "",
        "| case | type | passed | turns | in_tok | out_tok |",
        "|------|------|--------|-------|--------|---------|",
    ]
    for c in report.per_case:
        lines.append(
            f"| {c['id']} | {c['type']} | {c['passed']} | "
            f"{c['turns']} | {c['input_tokens']} | {c['output_tokens']} |"
        )
    return "\n".join(lines)
