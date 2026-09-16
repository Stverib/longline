"""Unit tests for longline/eval/types.py — case models + JSONL loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from longline.eval.types import (
    CaseParseError,
    E2ECase,
    ToolCallCase,
    load_cases,
    resolve_fixture,
)

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


class TestE2EChecks:
    """E2ECase 的复合 checks 与单 judge 兼容(契约 §5.1)."""

    def test_legacy_single_judge_becomes_a_one_element_checks_list(self) -> None:
        c = E2ECase.from_dict(
            {"id": "e", "type": "e2e", "task": "t",
             "judge": {"fn": "file_exists", "args": {"path": "a.txt"}}}
        )
        assert c.checks == [{"fn": "file_exists", "args": {"path": "a.txt"}}]
        assert c.num_checks == 1
        # 旧字段本身保留,方便按 judge_fn 读 detail 的旧读者.
        assert c.judge == {"fn": "file_exists", "args": {"path": "a.txt"}}

    def test_checks_mode_defaults_to_all(self) -> None:
        c = E2ECase.from_dict(
            {"id": "e", "type": "e2e", "task": "t",
             "judge": {"fn": "file_exists", "args": {"path": "a"}}}
        )
        assert c.checks_mode == "all"

    def test_multiple_checks_are_parsed_in_order(self) -> None:
        c = E2ECase.from_dict({
            "id": "e", "type": "e2e", "task": "t",
            "checks": [
                {"fn": "file_exists", "args": {"path": "a"}},
                {"fn": "file_content", "args": {"path": "a", "contains": "x"}},
            ],
        })
        assert [k["fn"] for k in c.checks] == ["file_exists", "file_content"]
        assert c.num_checks == 2

    def test_checks_do_not_backfill_the_legacy_judge_field(self) -> None:
        # judge 是「输入侧派生视图」,不能从 checks 反向捏造 —— 否则两个来源
        # 会各自漂移,读者分不清数字来自哪一个.
        c = E2ECase.from_dict({
            "id": "e", "type": "e2e", "task": "t",
            "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
        })
        assert c.judge == {}

    def test_both_judge_and_checks_raises(self) -> None:
        with pytest.raises(CaseParseError, match="not both"):
            E2ECase.from_dict({
                "id": "e", "type": "e2e", "task": "t",
                "judge": {"fn": "file_exists", "args": {"path": "a"}},
                "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
            })

    def test_neither_judge_nor_checks_raises(self) -> None:
        with pytest.raises(CaseParseError, match="requires"):
            E2ECase.from_dict({"id": "e", "type": "e2e", "task": "t"})

    def test_empty_checks_list_raises(self) -> None:
        # 空 checks 在 all() 下恒为真,是一条永远通过的用例.
        with pytest.raises(CaseParseError, match="non-empty"):
            E2ECase.from_dict({"id": "e", "type": "e2e", "task": "t", "checks": []})

    def test_check_without_a_string_fn_raises(self) -> None:
        with pytest.raises(CaseParseError, match="string 'fn'"):
            E2ECase.from_dict({"id": "e", "type": "e2e", "task": "t", "checks": [{"args": {}}]})

    def test_bad_checks_mode_raises(self) -> None:
        with pytest.raises(CaseParseError, match="checks_mode"):
            E2ECase.from_dict({
                "id": "e", "type": "e2e", "task": "t",
                "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
                "checks_mode": "most",
            })

    def test_any_mode_is_accepted(self) -> None:
        c = E2ECase.from_dict({
            "id": "e", "type": "e2e", "task": "t",
            "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
            "checks_mode": "any",
        })
        assert c.checks_mode == "any"

    def test_direct_construction_with_a_judge_normalises(self) -> None:
        # 不走 from_dict 的构造路径同样必须归一化,否则 checks 为空,
        # 下游可能读成「没有断言」而放过一切.
        c = E2ECase(id="e", task="t", judge={"fn": "file_exists", "args": {"path": "a"}})
        assert c.checks == [{"fn": "file_exists", "args": {"path": "a"}}]

    def test_direct_construction_prefers_explicit_checks(self) -> None:
        c = E2ECase(
            id="e", task="t",
            checks=[{"fn": "file_exists", "args": {"path": "a"}}],
        )
        assert c.judge == {}
        assert c.num_checks == 1

    def test_direct_construction_rejects_a_bad_mode(self) -> None:
        with pytest.raises(CaseParseError, match="checks_mode"):
            E2ECase(id="e", task="t", checks=[{"fn": "file_exists"}], checks_mode="nope")

    def test_category_tag_helper(self) -> None:
        c = E2ECase.from_dict({
            "id": "e", "type": "e2e", "task": "t", "tags": ["code", "bugfix"],
            "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
        })
        assert c.category_tag() == "code"

    def test_category_tag_is_none_without_a_category(self) -> None:
        c = E2ECase.from_dict({
            "id": "e", "type": "e2e", "task": "t", "tags": ["ad-hoc"],
            "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
        })
        assert c.category_tag() is None


class TestFixtureResolution:
    """fixture 解析后必须仍位于 fixtures 根目录内(契约 §8.4)."""

    def test_valid_name_resolves_under_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "fixtures"
        (root / "repo").mkdir(parents=True)
        assert resolve_fixture(root, "repo") == root / "repo"

    def test_parent_traversal_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "fixtures"
        root.mkdir()
        (tmp_path / "outside").mkdir()
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(root, "../outside", case_id="e2e-x")

    def test_absolute_path_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "fixtures"
        root.mkdir()
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(root, str(tmp_path / "outside"), case_id="e2e-x")

    def test_nested_traversal_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "fixtures"
        (root / "repo").mkdir(parents=True)
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(root, "repo/../../outside", case_id="e2e-x")

    def test_dot_names_are_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "fixtures"
        root.mkdir()
        for name in ("", ".", ".."):
            with pytest.raises(CaseParseError):
                resolve_fixture(root, name, case_id="e2e-x")

    def test_loader_validates_every_fixture(self, tmp_path: Path) -> None:
        cases = tmp_path / "cases.jsonl"
        cases.write_text(
            json.dumps({
                "id": "e", "type": "e2e", "task": "t", "fixture": "../outside",
                "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
            }) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "fixtures").mkdir()
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            load_cases(cases)

    def test_loader_accepts_a_fixture_inside_the_root(self, tmp_path: Path) -> None:
        (tmp_path / "fixtures" / "repo").mkdir(parents=True)
        cases = tmp_path / "cases.jsonl"
        cases.write_text(
            json.dumps({
                "id": "e", "type": "e2e", "task": "t", "fixture": "repo",
                "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
            }) + "\n",
            encoding="utf-8",
        )
        assert len(load_cases(cases)) == 1

    def test_explicit_fixtures_root_overrides_the_default(self, tmp_path: Path) -> None:
        (tmp_path / "elsewhere" / "repo").mkdir(parents=True)
        cases = tmp_path / "cases.jsonl"
        cases.write_text(
            json.dumps({
                "id": "e", "type": "e2e", "task": "t", "fixture": "repo",
                "checks": [{"fn": "file_exists", "args": {"path": "a"}}],
            }) + "\n",
            encoding="utf-8",
        )
        assert len(load_cases(cases, fixtures_root=tmp_path / "elsewhere")) == 1
