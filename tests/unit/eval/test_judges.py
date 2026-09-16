"""Unit tests for longline/eval/judges.py — deterministic judges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from longline.eval.judges import (
    check_args,
    check_tools,
    judge_case,
    judge_case_args,
    judge_steps,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLayer1:
    def test_check_tools_subsequence_ok(self) -> None:
        calls = [("Grep", {}), ("Read", {}), ("Write", {})]
        assert check_tools(calls, ["Grep", "Read"]) is True

    def test_check_tools_out_of_order_fails(self) -> None:
        calls = [("Read", {}), ("Grep", {})]
        assert check_tools(calls, ["Grep", "Read"]) is False

    def test_check_tools_missing_fails(self) -> None:
        calls = [("Read", {})]
        assert check_tools(calls, ["Read", "Write"]) is False

    def test_check_args_regex_match(self) -> None:
        calls = [("Read", {"file_path": "/tmp/src/config.py", "offset": 2})]
        assert check_args(calls, {"Read": {"file_path": r"config\.py"}}) is True

    def test_check_args_regex_mismatch_fails(self) -> None:
        calls = [("Read", {"file_path": "/tmp/other.py"})]
        assert check_args(calls, {"Read": {"file_path": r"config\.py"}}) is False

    def test_check_args_empty_expectation_trivially_ok(self) -> None:
        calls: list[tuple[str, dict]] = [("Read", {})]
        assert check_args(calls, {}) is True


class TestLayer2Judges:
    def test_file_content_contains(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello world\n", encoding="utf-8")
        assert judge_case("file_content", tmp_path, {"path": "README.md", "contains": "hello"}) is True

    def test_file_content_not_contains(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
        assert judge_case("file_content", tmp_path, {"path": "a.txt", "not_contains": "absent"}) is True

    def test_file_content_missing_file_fails(self, tmp_path: Path) -> None:
        assert judge_case("file_content", tmp_path, {"path": "nope.txt", "contains": "x"}) is False

    def test_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("", encoding="utf-8")
        assert judge_case("file_exists", tmp_path, {"path": "x.py"}) is True
        assert judge_case("file_exists", tmp_path, {"path": "missing.py"}) is False

    def test_command_ok_exit_zero(self, tmp_path: Path) -> None:
        assert judge_case("command_ok", tmp_path, {"command": "exit 0"}) is True

    def test_command_ok_exit_nonzero(self, tmp_path: Path) -> None:
        assert judge_case("command_ok", tmp_path, {"command": "exit 2"}) is False

    def test_command_output_contains(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_output_contains", tmp_path, {"command": "echo success", "contains": "success"}
        ) is True

    def test_unknown_judge_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            judge_case("bogus", tmp_path, {})


class TestStepMatch:
    """judge_steps:把一次调用序列映射到「哪些调用匹配了哪一步」.

    这是四个指标里 ToolSelectionCaseAccuracy 和 ToolCallPrecision 的共同底座:
    调用与步骤的对应关系必须单独可查,不能再塌缩成一个布尔值.
    """

    def test_all_steps_matched_in_order(self) -> None:
        calls = [("Glob", {}), ("Read", {}), ("Read", {})]
        m = judge_steps(calls, [["Glob", "Grep"], ["Read"]])
        assert m.step_indices == [0, 1]
        assert m.matched_call_indices == [0, 1]
        assert m.extra_call_indices == [2]
        assert m.all_steps_matched is True
        assert m.num_extra_calls == 1
        assert m.exceeded_extra_budget(0) is True
        assert m.exceeded_extra_budget(1) is False

    def test_alternative_tool_satisfies_step(self) -> None:
        # 一步接受多个工具:用 Grep 而不是 Glob 同样算选对.
        m = judge_steps([("Grep", {}), ("Read", {})], [["Glob", "Grep"], ["Read"]])
        assert m.all_steps_matched is True
        assert m.num_extra_calls == 0

    def test_out_of_order_is_not_matched(self) -> None:
        # 顺序是契约的一部分(对应旧 check_tools 的子序列语义).
        # call 1 的 Grep 满足了第 0 步;第 1 步要的 Read 只在它之前出现过,
        # 已经无法再消费,因此整条计划不成立.
        calls = [("Read", {}), ("Grep", {})]
        m = judge_steps(calls, [["Grep"], ["Read"]])
        assert m.step_indices == [1, None]
        assert m.all_steps_matched is False
        assert m.matched_call_indices == [1]
        assert m.extra_call_indices == [0]
        # 同样两次调用,顺序反过来就成立 —— 差别只在顺序上.
        assert judge_steps(calls, [["Read"], ["Grep"]]).all_steps_matched is True

    def test_a_failed_step_does_not_consume_later_calls(self) -> None:
        """匹配失败的步骤不能吞掉它后面的调用.

        (Bash, Read) 对 (Grep, Read):Bash 是那一次额外调用,Read 仍然满足第 1 步.
        如果实现「没匹配上也把游标推过去」,这里会被算成两次错误,precision 偏低.
        """
        m = judge_steps(
            [("Bash", {"command": "ls"}), ("Read", {})],
            [["Glob", "Grep"], ["Read"]],
        )
        assert m.step_indices == [None, 1]
        assert m.all_steps_matched is False
        assert m.extra_call_indices == [0]
        assert m.matched_call_indices == [1]

    def test_later_step_cannot_be_consumed_by_an_earlier_call(self) -> None:
        # 单次 Read 不能同时满足「先找到再读」的两步计划.
        m = judge_steps([("Read", {}), ("Glob", {})], [["Read"], ["Read"]])
        assert m.step_indices == [0, None]
        assert m.all_steps_matched is False

    def test_missing_step_leaves_it_unmatched(self) -> None:
        m = judge_steps([("Glob", {})], [["Glob", "Grep"], ["Read"]])
        assert m.step_indices == [0, None]
        assert m.all_steps_matched is False
        assert m.matched_call_indices == [0]

    def test_greedy_match_does_not_skip_a_satisfiable_step(self) -> None:
        # ["Read","Write"] 的备选是 Read 或 Write;从左边贪心匹配第一步到 Read,
        # 第二步(Bash)仍能匹配到后面的 Bash.若实现用了「最长匹配」等其它策略,
        # 这里的索引会漂移.
        m = judge_steps(
            [("Read", {}), ("Bash", {}), ("Write", {})],
            [["Read", "Write"], ["Bash"]],
        )
        assert m.step_indices == [0, 1]
        assert m.extra_call_indices == [2]

    def test_no_steps_means_nothing_to_match(self) -> None:
        m = judge_steps([("Read", {})], [])
        assert m.all_steps_matched is True
        assert m.matched_call_indices == []
        assert m.num_extra_calls == 1

    def test_empty_calls(self) -> None:
        m = judge_steps([], [["Read"]])
        assert m.step_indices == [None]
        assert m.all_steps_matched is False
        assert m.exceeded_extra_budget(0) is False


class TestJudgeCaseArgs:
    """judge_case_args:按工具聚合的参数字段级判定.

    四个指标里 ArgumentCallAccuracy 与 ArgumentFieldAccuracy 共用本结构:
    前者看「整次调用是否全对」,后者看「字段对了几个」.两者必须能分别算出来.
    """

    def test_all_fields_correct(self) -> None:
        calls = [("Read", {"file_path": "/tmp/src/config.py"})]
        r = judge_case_args(calls, {"Read": {"file_path": r"config\.py"}})
        assert r.checked_fields == 1
        assert r.correct_fields == 1
        assert r.checked_calls == 1
        assert r.correct_calls == 1

    def test_field_level_partial_credit_is_separate_from_call_level(self) -> None:
        # 一次调用里 1 个字段对,1 个字段错:字段准确率不是 0,调用准确率是 0.
        # 这正是旧实现(合成一个布尔)丢掉的信息.
        calls = [("Write", {"file_path": "notes.txt", "content": "wrong"})]
        r = judge_case_args(calls, {"Write": {"file_path": r"notes\.txt", "content": "hello"}})
        assert (r.correct_fields, r.checked_fields) == (1, 2)
        assert (r.correct_calls, r.checked_calls) == (0, 1)

    def test_tool_never_called_is_not_in_the_denominator(self) -> None:
        # 契约 §5.2:分母是「需要校验参数的匹配调用数」.没有调用就没有调用可校验.
        r = judge_case_args([("Bash", {})], {"Read": {"file_path": r".*"}})
        assert (r.correct_calls, r.checked_calls) == (0, 0)

    def test_untested_tool_is_not_in_the_denominator(self) -> None:
        # 用例没有声明 Bash 的参数期望,就不该把 Bash 调用算进参数分母.
        r = judge_case_args([("Bash", {"command": "ls"})], {})
        assert (r.correct_calls, r.checked_calls) == (0, 0)

    def test_only_the_best_call_per_tool_is_checked(self) -> None:
        # 同名工具的多次调用取最好的一次:一次写错参数后重写正确,
        # 是「参数最终传对了」,不该因为第一次的失误扣两次分.
        calls = [("Write", {"content": "wrong"}), ("Write", {"content": "hello"})]
        r = judge_case_args(calls, {"Write": {"content": "hello"}})
        assert (r.correct_calls, r.checked_calls) == (1, 1)
        assert (r.correct_fields, r.checked_fields) == (1, 1)

    def test_missing_arg_counts_as_a_wrong_field(self) -> None:
        calls = [("Read", {})]
        r = judge_case_args(calls, {"Read": {"file_path": r".*"}})
        assert (r.correct_fields, r.checked_fields) == (0, 1)

    def test_check_args_still_agrees_with_the_new_result(self) -> None:
        calls = [("Read", {"file_path": "/tmp/other.py"})]
        pats = {"Read": {"file_path": r"config\.py"}}
        assert check_args(calls, pats) is judge_case_args(calls, pats).all_calls_correct
        assert check_args(calls, pats) is False

    def test_undeclared_tool_does_not_move_check_args(self) -> None:
        # 与 test_untested_tool_is_not_in_the_denominator 同一条规则在布尔层的表现.
        assert check_args([("Bash", {})], {"Read": {"file_path": r".*"}}) is True
