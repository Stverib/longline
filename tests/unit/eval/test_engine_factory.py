"""Unit tests for longline/eval/engine_factory.py — engine assembly for eval."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
