"""Deterministic judges for the evaluation suite.

Layer 1 (tool calls): `judge_steps()` maps the agent's calls onto the case's
    accepted decision steps, and `judge_case_args()` decides argument
    correctness both per field and per call. Together they expose the
    numerator/denominator of all four tool-calling metrics
    (`evals/README.md` §5.2) — nothing is collapsed into a single boolean,
    because a fused boolean cannot carry an independent denominator.
    `check_tools()` / `check_args()` remain as the legacy boolean wrappers.
Layer 2 (E2E): judge_case() dispatches named, side-effect-limited checks run
    against a sandbox directory. All pass/fail is computed with re, the
    filesystem, and subprocess exit codes — never an LLM.

Paths in judge args are relative to the sandbox directory and resolved here.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from longline.eval.trajectory import ToolCall


# --- Layer-1 decision-step matching ---


@dataclass(frozen=True)
class StepMatch:
    """How one call sequence lines up with a case's accepted decision steps.

    Two index lists, deliberately kept separate:

    - `step_indices[i]` is the call index that satisfied step `i`, or None if
      that step was never satisfied. `step_indices` is therefore the per-step
      detail the contract asks to be emitted, not a pass/fail summary.
    - `extra_call_indices` are the calls that matched no step. They are the
      ToolCallPrecision denominator's "invalid" half, so they must be reported
      rather than discarded.
    """

    step_indices: list[int | None]
    matched_call_indices: list[int]
    extra_call_indices: list[int]

    @property
    def all_steps_matched(self) -> bool:
        """ToolSelectionCaseAccuracy's per-case outcome (contract §5.2)."""
        return all(i is not None for i in self.step_indices)

    @property
    def num_extra_calls(self) -> int:
        return len(self.extra_call_indices)

    @property
    def num_matched_calls(self) -> int:
        return len(self.matched_call_indices)

    def exceeded_extra_budget(self, max_extra_calls: int) -> bool:
        """True when unmatched calls outnumber the case's `max_extra_calls`.

        This never flips `all_steps_matched`: extra calls are visible as a
        *rate* (ToolCallPrecision), not as a hidden rewrite of the case result.
        """
        return self.num_extra_calls > max_extra_calls

    def to_detail(self) -> dict[str, object]:
        return {
            "step_indices": self.step_indices,
            "matched_call_indices": self.matched_call_indices,
            "extra_call_indices": self.extra_call_indices,
            "all_steps_matched": self.all_steps_matched,
            "num_extra_calls": self.num_extra_calls,
        }


def judge_steps(
    calls: list[ToolCall],
    accepted_tool_steps: list[list[str]],
) -> StepMatch:
    """Match calls against steps, IN ORDER, greedily from the left.

    Each step consumes the earliest not-yet-consumed call whose tool name is in
    that step's candidate set. This generalises the legacy ordered-subsequence
    check to steps that accept several plausible tools: `[["Glob", "Grep"]]`
    passes for either, whereas `check_tools` could only name one.

    Matching is monotonic — a call already spent on step `i` cannot also satisfy
    step `i+1`. That is what keeps a single `Read` from silently satisfying a
    two-step plan that wanted "find it, then read it".

    Calls that match no step are collected in `extra_call_indices` instead of
    being ignored; the contract requires them to be counted against precision.
    """
    step_indices: list[int | None] = [None] * len(accepted_tool_steps)
    consumed: set[int] = set()
    matched: list[int] = []

    # Ordered subsequence matching with alternatives, one step at a time.
    #
    # `cursor` is the earliest call index a step may still consume. It advances
    # only when a step actually matches, so a step that finds nothing does not
    # sabotage the steps after it — (Bash, Read) against (Grep, Read) is one
    # wrong call, not two, and the Read still satisfies step 1.
    #
    # It is NOT reset per step, which is what keeps order meaningful: in
    # (Read, Grep) against (Grep, Read), Grep lands on step 0 and the Read
    # that preceded it can no longer be reached.
    cursor = 0
    for si, accepted in enumerate(accepted_tool_steps):
        accepted_set = set(accepted)
        for ci in range(cursor, len(calls)):
            if calls[ci][0] not in accepted_set:
                continue
            step_indices[si] = ci
            consumed.add(ci)
            matched.append(ci)
            cursor = ci + 1
            break

    extras = [ci for ci in range(len(calls)) if ci not in consumed]
    return StepMatch(
        step_indices=step_indices,
        matched_call_indices=matched,
        extra_call_indices=extras,
    )


# --- Layer-1 argument checking ---


@dataclass(frozen=True)
class ArgCheckResult:
    """Argument correctness at two granularities, from one pass.

    `correct_calls` / `checked_calls` is ArgumentCallAccuracy's ratio;
    `correct_fields` / `checked_fields` is ArgumentFieldAccuracy's. They are
    reported together but computed independently — a call with one wrong field
    of two moves the field ratio and not the call ratio, which is precisely the
    distinction the fused legacy boolean destroyed.
    """

    correct_calls: int
    checked_calls: int
    correct_fields: int
    checked_fields: int

    @property
    def all_calls_correct(self) -> bool:
        """Legacy-compatible view: every checked call had every field correct."""
        return self.correct_calls == self.checked_calls

    def per_tool_detail(self) -> dict[str, dict[str, int]]:
        return self._detail

    _detail: dict[str, dict[str, int]]

    def to_detail(self) -> dict[str, object]:
        return {
            "correct_calls": self.correct_calls,
            "checked_calls": self.checked_calls,
            "correct_fields": self.correct_fields,
            "checked_fields": self.checked_fields,
            "all_calls_correct": self.all_calls_correct,
            "per_tool": self._detail,
        }


def judge_case_args(
    calls: list[ToolCall],
    expect_args: dict[str, dict[str, str]],
) -> ArgCheckResult:
    """Check declared argument regexes and return per-field *and* per-call counts.

    Denominator rule (contract §5.2): only **matched calls whose arguments the
    case declares** are checked. A tool that the case says nothing about, and a
    declared tool the agent never called, both contribute to neither numerator
    nor denominator. A declared tool that was never called is a missing *step*,
    which `judge_steps` already scores — charging it here as well would let one
    mistake depress two independent metrics.

    Per tool, the **best** call is the one that satisfies the most declared
    fields. A retry that fixes the arguments therefore reads as "the arguments
    were ultimately right", instead of double-charging the first attempt.
    """
    correct_calls = 0
    checked_calls = 0
    correct_fields = 0
    checked_fields = 0
    detail: dict[str, dict[str, int]] = {}

    for tool_name, arg_pats in expect_args.items():
        candidates = [ti for name, ti in calls if name == tool_name]
        if not candidates:
            # Declared but never called: a step-level miss, not an arg miss.
            detail[tool_name] = {
                "correct_fields": 0, "checked_fields": 0, "best_call_correct": 0, "calls": 0,
            }
            continue

        best_correct = 0
        for name, tool_input in calls:
            if name != tool_name:
                continue
            hits = sum(
                1 for arg, pat in arg_pats.items() if _match_arg(tool_input.get(arg), pat)
            )
            best_correct = max(best_correct, hits)

        n_fields = len(arg_pats)
        checked_calls += 1
        checked_fields += n_fields
        correct_fields += best_correct
        if best_correct == n_fields:
            correct_calls += 1
        detail[tool_name] = {
            "correct_fields": best_correct,
            "checked_fields": n_fields,
            "best_call_correct": int(best_correct == n_fields),
            "calls": len(candidates),
        }

    return ArgCheckResult(
        correct_calls=correct_calls,
        checked_calls=checked_calls,
        correct_fields=correct_fields,
        checked_fields=checked_fields,
        _detail=detail,
    )


def check_tools(calls: list[ToolCall], expect_tools: list[str]) -> bool:
    """True if expect_tools appears, in order, as a subsequence of calls.

    Legacy boolean wrapper (each tool is a one-candidate step); new code should
    read `judge_steps()` so extra calls and per-step detail survive.
    """
    return judge_steps(calls, [[t] for t in expect_tools]).all_steps_matched


def check_args(calls: list[ToolCall], expect_args: dict[str, dict[str, str]]) -> bool:
    """True if for every tool in expect_args, at least one call with that tool
    name matches ALL the given regexes on its named arguments.

    Argument values are stringified before matching; missing args fail the check.
    """
    return judge_case_args(calls, expect_args).all_calls_correct


def _match_arg(value: object, pattern: str) -> bool:
    if value is None:
        return False
    return re.search(pattern, str(value), re.IGNORECASE) is not None


# --- Layer-2 deterministic judges ---
# Each takes (sandbox: Path, args: dict[str, Any]) and returns bool.

def judge_file_content(sandbox: Path, args: dict[str, Any]) -> bool:
    path = sandbox / str(args["path"])
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if "contains" in args and re.search(str(args["contains"]), text, re.IGNORECASE) is None:
        return False
    return not ("not_contains" in args
                and re.search(str(args["not_contains"]), text, re.IGNORECASE) is not None)


def judge_file_exists(sandbox: Path, args: dict[str, Any]) -> bool:
    return (sandbox / str(args["path"])).is_file()


def judge_command_ok(sandbox: Path, args: dict[str, Any]) -> bool:
    proc = subprocess.run(
        str(args["command"]),
        shell=True,
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0


def judge_command_output_contains(sandbox: Path, args: dict[str, Any]) -> bool:
    proc = subprocess.run(
        str(args["command"]),
        shell=True,
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return re.search(str(args["contains"]), proc.stdout, re.IGNORECASE) is not None


_JUDGES: dict[str, Any] = {
    "file_content": judge_file_content,
    "file_exists": judge_file_exists,
    "command_ok": judge_command_ok,
    "command_output_contains": judge_command_output_contains,
}


def judge_case(fn_name: str, sandbox: Path, args: dict[str, Any]) -> bool:
    """Dispatch a named judge against a sandbox directory."""
    fn = _JUDGES.get(fn_name)
    if fn is None:
        raise ValueError(f"unknown judge fn: {fn_name!r} (known: {sorted(_JUDGES)})")
    return bool(fn(sandbox, args))
