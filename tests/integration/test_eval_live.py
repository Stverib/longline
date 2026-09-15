"""Integration test — eval suite drives the real API end-to-end.

Skips when ANTHROPIC_API_KEY is absent (mirrors test_query_loop.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from longline.eval.report import EvalReport, aggregate
from longline.eval.runner import run_case
from longline.eval.types import E2ECase, ToolCallCase

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


skip_no_key = pytest.mark.skipif(_api_key() is None, reason="No API key available")

FIXTURES = PROJECT_ROOT / "evals" / "fixtures"
MODEL = "claude-sonnet-4-20250514"


@skip_no_key
class TestEvalLive:
    async def test_tool_call_case_runs_end_to_end(self) -> None:
        # 建一个空临时 sandbox;任务即创建一个文件,确保离线也能成立(无 fixture 依赖)
        case = ToolCallCase(
            id="live-tc",
            task="创建文件 hello.txt,内容写 'world'",
            expect_tools=["Write"],
            expect_args={"Write": {"file_path": r"hello\.txt"}},
            max_turns=4,
        )
        result = await run_case(case, model=MODEL, api_key=_api_key() or "", fixtures_dir=FIXTURES)
        _skip_on_api_error(result.errors)
        assert result.case_id == "live-tc"
        assert result.turns >= 1  # 至少一轮

    async def test_e2e_case_runs_end_to_end(self) -> None:
        case = E2ECase(
            id="live-e2e",
            task="创建文件 hello.txt,内容写 'world'",
            judge={"fn": "file_content", "args": {"path": "hello.txt", "contains": "world"}},
            max_turns=4,
        )
        result = await run_case(case, model=MODEL, api_key=_api_key() or "", fixtures_dir=FIXTURES)
        _skip_on_api_error(result.errors)
        # 目标是确定性判分;这里不强制 passed(模型可能没建好),只验证链路与报告不崩
        rep: EvalReport = aggregate([result])
        assert rep.total_cases == 1
        assert result.detail.get("judge_fn") == "file_content"


def _skip_on_api_error(errors: list[str]) -> None:
    """Skip the test on transient API/network failures (not logic bugs).

    Mirrors _check_for_connection_error in tests/integration/test_query_loop.py.
    """
    transient = ("Connection error", "502", "503", "529", "overloaded", "rate limit")
    for msg in errors:
        if any(tok in msg for tok in transient):
            pytest.skip(f"Transient API/network error, skipping: {msg}")
