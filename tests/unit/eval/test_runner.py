"""Unit tests for longline/eval/runner.py — case runner + suite orchestrator.

Runs offline: build_engine is monkeypatched to a fake that yields scripted
events, so no API key is required.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from longline.core.events import TextDelta, ToolUseStart, TurnComplete
from longline.eval.runner import run_case, run_suite
from longline.eval.types import E2ECase, ToolCallCase
from longline.models.messages import Usage

if TYPE_CHECKING:
    import pytest


class FakeTool:
    def get_name(self) -> str:
        return "Bash"


def _fake_engine_factory(events: list[Any]) -> Any:
    """Bootstrap a fake build_engine replacement via monkeypatch."""

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        registry = SimpleNamespace(list_tools=lambda: [FakeTool()])
        system = "test"

        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                for event in events:
                    yield event

        return SimpleNamespace(
            registry=registry,
            system_prompt=system,
            model=model,
            submit=_FakeEngine().submit,
        )

    return _build_engine


async def test_run_tool_call_case_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Read", tool_id="t", input={"file_path": "/tmp/a.py"}),
        TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=3)),
    ]))
    case = ToolCallCase(
        id="tc-001", task="read a file", expect_tools=["Read"],
        expect_args={"Read": {"file_path": r"a\.py"}},
    )
    result = await run_case(case, model="m", api_key="k", fixtures_dir=Path("does-not-exist"))
    assert result.passed is True
    assert result.turns == 1
    assert result.input_tokens == 5
    assert result.output_tokens == 3


async def test_run_tool_call_case_fails_on_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Grep", tool_id="t", input={"pattern": "x"}),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-002", task="grep then read", expect_tools=["Read"])
    result = await run_case(case, model="m", api_key="k", fixtures_dir=Path("does-not-exist"))
    assert result.passed is False
    assert result.detail["tool_subsequence_ok"] is False


async def test_run_e2e_case_copies_fixture_and_judges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    fixtures = tmp_path / "fixtures"
    (fixtures / "simple_repo").mkdir(parents=True)
    (fixtures / "simple_repo" / "README.md").write_text("hello fixture\n", encoding="utf-8")

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TextDelta(text="done"),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))

    case = E2ECase(
        id="e2e-001", task="do the thing", fixture="simple_repo",
        judge={"fn": "file_content", "args": {"path": "README.md", "contains": "fixture"}},
    )
    result = await run_case(case, model="m", api_key="k", fixtures_dir=fixtures)
    assert result.passed is True
    assert result.text == "done"


async def test_run_suite_runs_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Read", tool_id="t", input={}),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    cases = [
        ToolCallCase(id="a", task="t1", expect_tools=["Read"]),
        ToolCallCase(id="b", task="t2", expect_tools=["Read"]),
    ]
    results = await run_suite(cases, model="m", api_key="k", fixtures_dir=Path("x"))
    assert len(results) == 2
    assert results[0].case_id == "a"
    assert results[1].case_id == "b"
