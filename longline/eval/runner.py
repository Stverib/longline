"""Run evaluation cases and orchestrate judging.

Each case gets a fresh sandbox temp dir; E2E cases copy a named fixture into it
before the agent runs, so runs are reproducible and side-effect-free. Runs are
serial (one QueryEngine at a time) to avoid tripping API rate limits.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from longline.eval.engine_factory import build_engine
from longline.eval.judges import check_args, check_tools, judge_case
from longline.eval.trajectory import ToolCall, extract_trajectory
from longline.eval.types import EvalCase, ToolCallCase

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class CaseResult:
    """Outcome of a single evaluation case."""

    case_id: str
    case_type: str
    passed: bool
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    text: str = ""
    errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)


def _prepare_sandbox(fixtures_dir: Path, fixture: str | None) -> str:
    """Create a temp sandbox, optionally seeded from a fixture copy."""
    sandbox = Path(tempfile.mkdtemp(prefix="longline-eval-"))
    if fixture:
        src = fixtures_dir / fixture
        if not src.is_dir():
            raise FileNotFoundError(f"fixture not found: {src}")
        shutil.copytree(src, sandbox, dirs_exist_ok=True)
    return str(sandbox)


def _judge_l1(calls: list[ToolCall], case: ToolCallCase) -> dict[str, object]:
    tools_ok = check_tools(calls, case.expect_tools)
    args_ok = check_args(calls, case.expect_args)
    detail: dict[str, object] = {
        "expect_tools": case.expect_tools,
        "expect_args": case.expect_args,
        "tool_subsequence_ok": tools_ok,
        "args_ok": args_ok,
    }
    return detail


async def run_case(
    case: EvalCase,
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
) -> CaseResult:
    """Run one case and return its CaseResult.

    build_engine is monkeypatchable so unit tests stay offline.
    """
    fixture = case.fixture  # both ToolCallCase and E2ECase carry fixture
    sandbox = _prepare_sandbox(fixtures_dir, fixture)

    engine = build_engine(sandbox=sandbox, model=model, api_key=api_key)
    traj = await extract_trajectory(
        engine.submit(case.task, max_turns=case.max_turns)
    )

    result = CaseResult(
        case_id=case.id,
        case_type="tool_call" if isinstance(case, ToolCallCase) else "e2e",
        passed=False,
        turns=traj.turns,
        input_tokens=traj.input_tokens,
        output_tokens=traj.output_tokens,
        text=traj.text,
        errors=traj.errors,
        tool_calls=traj.tool_calls,
    )

    if isinstance(case, ToolCallCase):
        detail = _judge_l1(traj.tool_calls, case)
        result.detail = detail
        passed = bool(detail["tool_subsequence_ok"]) and bool(detail["args_ok"])
    else:  # E2ECase
        judge_conf = case.judge
        fn = str(judge_conf["fn"])
        args = judge_conf.get("args", {})
        passed = judge_case(fn, Path(sandbox), args)
        result.detail = {"judge_fn": fn, "judge_args": args}

    result.passed = passed
    return result


async def run_suite(
    cases: Iterable[EvalCase],
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
) -> list[CaseResult]:
    """Run a batch of cases serially."""
    results: list[CaseResult] = []
    for case in cases:
        results.append(await run_case(case, model=model, api_key=api_key, fixtures_dir=fixtures_dir))
    return results
