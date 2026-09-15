"""Unit tests for longline/eval/types.py — case models + JSONL loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from longline.eval.types import CaseParseError, E2ECase, ToolCallCase, load_cases

if TYPE_CHECKING:
    from pathlib import Path


def test_load_tool_call_case(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "tc-001", "type": "tool_call", "task": "x", '
        '"expect_tools": ["Read"], "expect_args": {"Read": {"file_path": "a\\\\.py"}}, '
        '"max_turns": 5, "tags": ["a"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(f)
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, ToolCallCase)
    assert c.expect_tools == ["Read"]
    assert c.expect_args == {"Read": {"file_path": "a\\.py"}}


def test_load_e2e_case(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "e2e-001", "type": "e2e", "task": "t", '
        '"fixture": "simple_repo", '
        '"judge": {"fn": "file_content", "args": '
        '{"path": "README.md", "contains": "hello"}}, '
        '"max_turns": 8, "tags": ["x"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(f)
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, E2ECase)
    assert c.fixture == "simple_repo"
    assert c.judge == {"fn": "file_content", "args": {"path": "README.md", "contains": "hello"}}


def test_load_unknown_type_raises(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text('{"id": "z", "type": "nope", "task": "t"}\n', encoding="utf-8")
    with pytest.raises(CaseParseError):
        load_cases(f)


def test_load_bad_json_line_raises(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text('{"id": "z", "type": "tool_call"\n', encoding="utf-8")
    with pytest.raises(CaseParseError):
        load_cases(f)


def test_load_missing_required_field_raises(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    # tool_call 缺 expect_tools
    f.write_text('{"id": "tc", "type": "tool_call", "task": "t", "expect_args": {}}\n', encoding="utf-8")
    with pytest.raises(CaseParseError):
        load_cases(f)
