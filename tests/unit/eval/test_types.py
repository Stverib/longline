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


class TestAcceptedToolSteps:
    """expect_tools 的按步骤候选集合扩展(契约 §5.2)."""

    def test_expect_tools_maps_to_single_candidate_steps(self) -> None:
        # 旧字段必须继续可用:每个工具变成「只有一个候选」的一步.
        c = ToolCallCase.from_dict(
            {"id": "c", "type": "tool_call", "task": "t", "expect_tools": ["Grep", "Read"]}
        )
        assert c.expect_tools == ["Grep", "Read"]
        assert c.accepted_tool_steps == [["Grep"], ["Read"]]

    def test_accepted_tool_steps_parsed(self) -> None:
        c = ToolCallCase.from_dict(
            {
                "id": "c", "type": "tool_call", "task": "t",
                "accepted_tool_steps": [["Glob", "Grep"], ["Read"]],
                "max_extra_calls": 1,
            }
        )
        assert c.accepted_tool_steps == [["Glob", "Grep"], ["Read"]]
        assert c.max_extra_calls == 1

    def test_accepted_tool_steps_do_not_backfill_expect_tools(self) -> None:
        # 新字段是权威来源;expect_tools 是派生视图,不能反向捏造.
        c = ToolCallCase.from_dict(
            {"id": "c", "type": "tool_call", "task": "t", "accepted_tool_steps": [["Glob"]]}
        )
        assert c.expect_tools == []

    def test_both_fields_set_raises(self) -> None:
        # 两个来源互相矛盾,必须显式报错而不是静默择一.
        with pytest.raises(CaseParseError, match="expect_tools"):
            ToolCallCase.from_dict({
                "id": "c", "type": "tool_call", "task": "t",
                "expect_tools": ["Read"], "accepted_tool_steps": [["Read"]],
            })

    def test_neither_field_raises(self) -> None:
        with pytest.raises(CaseParseError, match="accepted_tool_steps"):
            ToolCallCase.from_dict({"id": "c", "type": "tool_call", "task": "t"})

    def test_empty_step_raises(self) -> None:
        # 一个没有任何候选工具的空步骤永远无法被满足,是数据错误.
        with pytest.raises(CaseParseError, match="empty"):
            ToolCallCase.from_dict({
                "id": "c", "type": "tool_call", "task": "t", "accepted_tool_steps": [[]],
            })

    def test_non_list_accepted_tool_steps_raises(self) -> None:
        with pytest.raises(CaseParseError):
            ToolCallCase.from_dict({
                "id": "c", "type": "tool_call", "task": "t", "accepted_tool_steps": "Read",
            })

    def test_negative_max_extra_calls_raises(self) -> None:
        with pytest.raises(CaseParseError, match="max_extra_calls"):
            ToolCallCase.from_dict({
                "id": "c", "type": "tool_call", "task": "t",
                "accepted_tool_steps": [["Read"]], "max_extra_calls": -1,
            })

    def test_max_extra_calls_defaults_to_zero(self) -> None:
        c = ToolCallCase.from_dict(
            {"id": "c", "type": "tool_call", "task": "t", "accepted_tool_steps": [["Read"]]}
        )
        assert c.max_extra_calls == 0

    def test_blind_rationale_parsed(self) -> None:
        # 盲测用例必须自带「为什么不泄漏」的可机读说明(见 tool_selection.jsonl).
        c = ToolCallCase.from_dict({
            "id": "c", "type": "tool_call", "task": "t",
            "accepted_tool_steps": [["Read"]],
            "blind_rationale": "任务只描述目标,未提任何手段。",
        })
        assert c.blind_rationale == "任务只描述目标,未提任何手段。"

    def test_blind_rationale_defaults_to_none(self) -> None:
        c = ToolCallCase.from_dict(
            {"id": "c", "type": "tool_call", "task": "t", "accepted_tool_steps": [["Read"]]}
        )
        assert c.blind_rationale is None

    def test_direct_construction_from_expect_tools_derives_steps(self) -> None:
        # 不走 from_dict 的构造路径同样必须归一化:否则 accepted_tool_steps
        # 为空会被 judge_steps 读成「没有期望步骤」,全场静默通过.
        c = ToolCallCase(id="c", task="t", expect_tools=["Read", "Write"])
        assert c.accepted_tool_steps == [["Read"], ["Write"]]

    def test_direct_construction_prefers_explicit_steps(self) -> None:
        c = ToolCallCase(id="c", task="t", accepted_tool_steps=[["Glob", "Grep"]])
        assert c.accepted_tool_steps == [["Glob", "Grep"]]
        assert c.expect_tools == []

    def test_round_trip_through_jsonl_keeps_new_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "cases.jsonl"
        f.write_text(
            '{"id": "ts-001", "type": "tool_call", "task": "t", '
            '"accepted_tool_steps": [["Glob", "Grep"], ["Read"]], "max_extra_calls": 1, '
            '"blind_rationale": "why", "tags": ["blind"]}\n',
            encoding="utf-8",
        )
        c = load_cases(f)[0]
        assert isinstance(c, ToolCallCase)
        assert c.accepted_tool_steps == [["Glob", "Grep"], ["Read"]]
        assert c.max_extra_calls == 1
        assert c.blind_rationale == "why"
