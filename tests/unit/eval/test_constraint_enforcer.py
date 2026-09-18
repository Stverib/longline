"""Program-level constraint enforcement. See longline/eval/constraint_enforcer.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from longline.eval.constraint_enforcer import strip_forbidden
from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema


class FakeTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    # These tests only exercise registry membership, never schema export, so a
    # null schema is honest about what the fake supports.
    def get_schema(self) -> ToolSchema:
        return None  # type: ignore[return-value]

    async def execute(self, tool_input: dict[str, object]) -> ToolResult:
        raise NotImplementedError


def test_strip_forbidden_removes_declared_tools() -> None:
    reg = ToolRegistry()
    for n in ("Bash", "Grep", "Read"):
        reg.register(FakeTool(n))
    strip_forbidden(reg, ["Bash"])
    assert [t.get_name() for t in reg.list_tools()] == ["Grep", "Read"]


def test_strip_forbidden_ignores_unknown_names() -> None:
    reg = ToolRegistry()
    reg.register(FakeTool("Grep"))
    strip_forbidden(reg, ["Bash"])  # absent from this profile: vacuously satisfied
    assert [t.get_name() for t in reg.list_tools()] == ["Grep"]


def test_strip_forbidden_returns_removed_names() -> None:
    reg = ToolRegistry()
    reg.register(FakeTool("Grep"))
    assert strip_forbidden(reg, ["Bash", "Grep"]) == ["Grep"]


def test_tool_call_case_accepts_forbidden_tools_field() -> None:
    from longline.eval.types import ToolCallCase

    case = ToolCallCase(
        id="x",
        task="t",
        accepted_tool_steps=[["Read"]],
        forbidden_tools=["Bash"],
    )
    assert case.forbidden_tools == ["Bash"]


def test_tool_call_case_forbidden_tools_defaults_to_empty() -> None:
    from longline.eval.types import ToolCallCase

    case = ToolCallCase(id="x", task="t", expect_tools=["Read"])
    assert case.forbidden_tools == []


def test_from_dict_parses_forbidden_tools() -> None:
    from longline.eval.types import ToolCallCase

    case = ToolCallCase.from_dict({
        "type": "tool_call",
        "id": "x",
        "task": "t",
        "expect_tools": ["Read"],
        "forbidden_tools": ["Bash", "WebSearch"],
    })
    assert case.forbidden_tools == ["Bash", "WebSearch"]


def test_from_dict_rejects_non_list_forbidden_tools() -> None:
    from longline.eval.types import CaseParseError, ToolCallCase

    with pytest.raises(CaseParseError):
        ToolCallCase.from_dict({
            "type": "tool_call",
            "id": "x",
            "task": "t",
            "expect_tools": ["Read"],
            "forbidden_tools": "Bash",
        })


def test_build_registry_strips_declared_tools(tmp_path: Path) -> None:
    """A declared forbidden tool is absent from the registry build_eval_registry makes."""
    from longline.eval.eval_tools import build_eval_registry

    reg = build_eval_registry(str(tmp_path), profile="core", forbidden=["Grep"])
    names = [t.get_name() for t in reg.list_tools()]
    assert "Grep" not in names
    assert "Read" in names
    # The schema export the API request is built from must agree with the
    # registry: a dead entry behind a list_tools-shaped check would surface
    # exactly here.
    assert "Grep" not in [s["name"] for s in reg.get_api_schemas()]


def test_build_registry_without_forbidden_is_unchanged(tmp_path: Path) -> None:
    from longline.eval.eval_tools import build_eval_registry

    reg = build_eval_registry(str(tmp_path), profile="core")
    names = [t.get_name() for t in reg.list_tools()]
    assert names == ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]


async def test_run_case_applies_forbidden_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case's forbidden list reaches the registry the agent runs against."""
    import longline.eval.runner as mod
    from longline.core.events import TurnComplete
    from longline.models.messages import Usage

    captured: dict[str, object] = {}

    def _fake(
        *,
        sandbox: str,
        model: str,
        api_key: str,
        tool_profile: str = "core",
        forbidden: list[str] | tuple[()] = (),
    ) -> object:
        captured["forbidden"] = list(forbidden)
        registry = SimpleNamespace(list_tools=lambda: [])
        events = [TurnComplete(stop_reason="end_turn", usage=Usage())]

        async def submit(user_input: str, **kwargs: object) -> object:
            for event in events:
                yield event

        return SimpleNamespace(
            registry=registry, system_prompt="t", model=model, submit=submit,
        )

    monkeypatch.setattr(mod, "build_engine", _fake)
    from longline.eval.types import ToolCallCase

    case = ToolCallCase(id="c", task="t", expect_tools=["Read"], forbidden_tools=["Bash"])
    result = await mod.run_case(
        case, model="m", api_key="k", fixtures_dir=Path("does-not-exist"),
    )
    assert captured["forbidden"] == ["Bash"]
    assert result is not None
