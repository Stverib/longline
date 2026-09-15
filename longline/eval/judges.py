"""Deterministic judges for the evaluation suite.

Layer 1 (tool calls): check_tools() verifies an ordered-subsequence match of
    expected tools; check_args() verifies regexes against the input of each
    tool call.
Layer 2 (E2E): judge_case() dispatches named, side-effect-limited checks run
    against a sandbox directory. All pass/fail is computed with re, the
    filesystem, and subprocess exit codes — never an LLM.

Paths in judge args are relative to the sandbox directory and resolved here.
"""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from longline.eval.trajectory import ToolCall


def check_tools(calls: list[ToolCall], expect_tools: list[str]) -> bool:
    """True if expect_tools appears, in order, as a subsequence of calls.

    Extra intermediate calls are allowed; the expected tools must each be
    called at least once and in the given relative order.
    """
    it = iter(c[0] for c in calls)
    return all(expected in it for expected in expect_tools)  # in consumes up to the match


def check_args(calls: list[ToolCall], expect_args: dict[str, dict[str, str]]) -> bool:
    """True if for every tool in expect_args, at least one call with that tool
    name matches ALL the given regexes on its named arguments.

    Argument values are stringified before matching; missing args fail the check.
    """
    for tool_name, arg_pats in expect_args.items():
        matched_tool = False
        for name, tool_input in calls:
            if name != tool_name:
                continue
            if all(_match_arg(tool_input.get(arg), pat) for arg, pat in arg_pats.items()):
                matched_tool = True
                break
        if not matched_tool:
            return False
    return True


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
