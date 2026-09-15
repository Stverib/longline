"""Unit tests for longline/eval/eval_tools.py — offline tool profiles.

这些替身的唯一职责是让「工具选择」可测:schema 必须与生产工具逐字段一致,
否则模型看到的选项就和线上不同,选择率也就不是线上那个数字.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from longline.eval.engine_factory import EVAL_TOOL_NAMES
from longline.eval.eval_tools import (
    ALL_EVAL_TOOL_NAMES,
    WEB_FAMILY,
    EvalWebFetchTool,
    EvalWebSearchTool,
    build_tool_profile,
)
from longline.tools.notebook.notebook_edit_tool import NotebookEditTool
from longline.tools.task_tools.task_tools import TaskStore
from longline.tools.web_fetch.web_fetch_tool import WebFetchTool
from longline.tools.web_search.web_search_tool import WebSearchTool


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _schema_dict(tool: Any) -> dict[str, Any]:
    s = tool.get_schema()
    return {"name": s.name, "description": s.description, "input_schema": s.input_schema}


class TestWebStandInsMatchProductionSchema:
    """schema 一致性是这一节存在的理由,必须逐字段比对,不是抽查."""

    def test_web_search_schema_identical_to_production(self) -> None:
        assert _schema_dict(EvalWebSearchTool()) == _schema_dict(WebSearchTool())

    def test_web_fetch_schema_identical_to_production(self) -> None:
        assert _schema_dict(EvalWebFetchTool()) == _schema_dict(WebFetchTool())

    def test_names_match_production(self) -> None:
        assert EvalWebSearchTool().get_name() == WebSearchTool().get_name() == "WebSearch"
        assert EvalWebFetchTool().get_name() == WebFetchTool().get_name() == "WebFetch"


class TestWebStandInsAreOffline:
    def test_web_search_returns_canned_success_without_network(self) -> None:
        r = _run(EvalWebSearchTool().execute({"query": "python asyncio"}))
        assert r.is_error is False
        assert "python asyncio" in r.content

    def test_web_search_is_deterministic(self) -> None:
        tool = EvalWebSearchTool()
        a = _run(tool.execute({"query": "x"}))
        b = _run(tool.execute({"query": "x"}))
        assert a.content == b.content

    def test_web_search_without_query_is_an_error(self) -> None:
        # 与生产工具同样的入参校验:真实执行失败要能被 ExecutionSuccessRate 看见.
        assert _run(EvalWebSearchTool().execute({})).is_error is True

    def test_web_fetch_returns_canned_page(self) -> None:
        r = _run(EvalWebFetchTool().execute({"url": "https://example.com/docs"}))
        assert r.is_error is False
        assert "https://example.com/docs" in r.content

    def test_web_fetch_without_url_is_an_error(self) -> None:
        assert _run(EvalWebFetchTool().execute({})).is_error is True

    def test_no_network_module_is_imported(self) -> None:
        # 离线是硬约束:替身模块不得引入 httpx 之类的客户端.
        import longline.eval.eval_tools as mod

        src = open(mod.__file__, encoding="utf-8").read()  # noqa: SIM115
        for banned in ("httpx", "requests", "urllib.request", "socket"):
            assert banned not in src, f"eval_tools must stay offline, found {banned!r}"


class TestProfiles:
    def test_core_profile_is_the_existing_registry(self) -> None:
        assert build_tool_profile("core") == EVAL_TOOL_NAMES

    def test_web_profile_adds_both_web_tools(self) -> None:
        assert set(WEB_FAMILY) <= set(build_tool_profile("web"))

    def test_all_profile_covers_every_family(self) -> None:
        names = set(build_tool_profile("all"))
        assert set(EVAL_TOOL_NAMES) <= names
        assert set(ALL_EVAL_TOOL_NAMES) <= names

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown tool profile"):
            build_tool_profile("nope")

    def test_all_eval_tool_names_lists_every_family(self) -> None:
        assert {"WebSearch", "WebFetch", "NotebookEdit"} <= set(ALL_EVAL_TOOL_NAMES)
        assert {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop"} <= set(
            ALL_EVAL_TOOL_NAMES
        )


class TestNotebookAndTaskToolsAreTheProductionOnes:
    def test_notebook_tool_name(self) -> None:
        assert NotebookEditTool().get_name() == "NotebookEdit"

    def test_task_tools_share_an_injectable_store(self) -> None:
        # 用例必须能在自己的沙箱里放一个 store,避免任务状态跨用例泄漏.
        from longline.eval.eval_tools import build_eval_registry

        store = TaskStore()
        registry = build_eval_registry("/tmp/sandbox", profile="task", task_store=store)
        created = _run(registry.get("TaskCreate").execute({"subject": "s"}))  # type: ignore[union-attr]
        assert created.is_error is False
        assert [t.subject for t in store.list_all()] == ["s"]
