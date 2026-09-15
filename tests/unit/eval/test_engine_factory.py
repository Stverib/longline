"""Unit tests for longline/eval/engine_factory.py — engine assembly for eval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from longline.core.query_engine import QueryEngine
from longline.eval.engine_factory import EVAL_TOOL_NAMES, build_engine

if TYPE_CHECKING:
    from pathlib import Path


def test_engine_uses_minimal_toolset(tmp_path: Path) -> None:
    engine = build_engine(
        sandbox=str(tmp_path),
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )
    assert isinstance(engine, QueryEngine)
    registered = {t.get_name() for t in engine.registry.list_tools()}
    assert registered == set(EVAL_TOOL_NAMES)
    # 沙箱路径进入 system prompt,供模型知道工作目录
    assert str(tmp_path) in engine.system_prompt


def test_eval_tool_names_complete() -> None:
    # 必须包含 Layer-1 用例所依赖的读/搜/写工具
    assert {"Read", "Grep"} <= set(EVAL_TOOL_NAMES)
    assert {"Write", "Edit", "Bash"} <= set(EVAL_TOOL_NAMES)


class TestToolProfiles:
    """`--tool-profile` 让一个用例能跑在「会用到的那族工具确实存在」的注册表上。"""

    def test_default_profile_is_unchanged_core_set(self, tmp_path: Path) -> None:
        engine = build_engine(
            sandbox=str(tmp_path), model="m", api_key="k",
        )
        assert {t.get_name() for t in engine.registry.list_tools()} == set(EVAL_TOOL_NAMES)

    def test_web_profile_registers_offline_web_stand_ins(self, tmp_path: Path) -> None:
        engine = build_engine(
            sandbox=str(tmp_path), model="m", api_key="k", tool_profile="web",
        )
        names = {t.get_name() for t in engine.registry.list_tools()}
        assert {"WebSearch", "WebFetch"} <= names
        # 同一 schema,不同实现:不能把联网的生产工具塞进评测注册表.
        assert type(engine.registry.get("WebSearch")).__name__ == "EvalWebSearchTool"

    def test_task_profile_registers_all_five_task_tools(self, tmp_path: Path) -> None:
        engine = build_engine(
            sandbox=str(tmp_path), model="m", api_key="k", tool_profile="task",
        )
        names = {t.get_name() for t in engine.registry.list_tools()}
        assert {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop"} <= names

    def test_notebook_profile_registers_notebook_edit(self, tmp_path: Path) -> None:
        engine = build_engine(
            sandbox=str(tmp_path), model="m", api_key="k", tool_profile="notebook",
        )
        assert "NotebookEdit" in {t.get_name() for t in engine.registry.list_tools()}

    def test_core_profile_has_no_extra_families(self, tmp_path: Path) -> None:
        engine = build_engine(
            sandbox=str(tmp_path), model="m", api_key="k", tool_profile="core",
        )
        names = {t.get_name() for t in engine.registry.list_tools()}
        assert names.isdisjoint({"WebSearch", "WebFetch", "NotebookEdit", "TaskCreate"})

    def test_unknown_profile_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown tool profile"):
            build_engine(sandbox=str(tmp_path), model="m", api_key="k", tool_profile="bogus")
