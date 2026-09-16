"""Unit tests for longline/eval/judges.py — deterministic judges."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from longline.eval.judges import (
    case_passed,
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

    # --- Anchored patterns are LINE-anchored, not whole-file anchored. ---
    #
    # Regression guard. `judge_file_content` used to run `re.search` without
    # `re.MULTILINE`, so `^`/`$` bound to the whole string and an assertion like
    # `^PORT = 3000$` only matched a file whose first and last characters were
    # exactly that line. 25 assertions across 15 E2E cases were dead that way:
    # a perfectly correct artifact scored zero, and nothing noticed because no
    # test required a correct artifact to PASS — only that fixtures were not
    # pre-satisfied.

    def test_file_content_anchored_pattern_matches_an_inner_line(self, tmp_path: Path) -> None:
        """`^PORT = 3000$` must match line 2 of a multi-line file."""
        (tmp_path / "config.py").write_text('HOST = "x"\nPORT = 3000\nDEBUG = True\n', encoding="utf-8")
        assert judge_case(
            "file_content", tmp_path, {"path": "config.py", "contains": "^PORT = 3000$"}
        ) is True

    def test_file_content_anchored_pattern_rejects_a_non_matching_inner_line(self, tmp_path: Path) -> None:
        """The same assertion must still FAIL when the line differs."""
        (tmp_path / "config.py").write_text('HOST = "x"\nPORT = 8080\nDEBUG = True\n', encoding="utf-8")
        assert judge_case(
            "file_content", tmp_path, {"path": "config.py", "contains": "^PORT = 3000$"}
        ) is False

    def test_file_content_header_row_of_a_table_matches(self, tmp_path: Path) -> None:
        """Markdown-table rows: every anchored row must match, not just the first."""
        (tmp_path / "report.md").write_text(
            "| stage | total |\n| alpha | 10 |\n| beta | 15 |\n", encoding="utf-8"
        )
        for row in (r"^\| stage \| total \|$", r"^\| alpha \| 10 \|$", r"^\| beta \| 15 \|$"):
            assert judge_case("file_content", tmp_path, {"path": "report.md", "contains": row}) is True, row

    def test_file_content_not_contains_anchored_pattern_is_effective(self, tmp_path: Path) -> None:
        """`not_contains` with an anchor must actually detect the line.

        Without MULTILINE the inner regex could not match, so the negated check
        was unconditionally True — the assertion was not merely loose, it was
        inert, and every artifact passed it.
        """
        (tmp_path / "a.txt").write_text("Status\nReviewed\nDone\n", encoding="utf-8")
        assert judge_case(
            "file_content", tmp_path, {"path": "a.txt", "not_contains": "^Reviewed$"}
        ) is False
        (tmp_path / "b.txt").write_text("Status\nPending\n", encoding="utf-8")
        assert judge_case(
            "file_content", tmp_path, {"path": "b.txt", "not_contains": "^Reviewed$"}
        ) is True

    def test_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("", encoding="utf-8")
        assert judge_case("file_exists", tmp_path, {"path": "x.py"}) is True
        assert judge_case("file_exists", tmp_path, {"path": "missing.py"}) is False

    def test_command_ok_exit_zero(self, tmp_path: Path) -> None:
        # 命令现在是参数列表 + allowlist,不再是 shell 字符串(见 TestLayer2CommandSecurity).
        assert judge_case(
            "command_ok", tmp_path,
            {"command": ["python", "-c", "pass"], "allowed_commands": ["python"]},
        ) is True

    def test_command_ok_exit_nonzero(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_ok", tmp_path,
            {"command": ["python", "-c", "raise SystemExit(2)"], "allowed_commands": ["python"]},
        ) is False

    def test_command_output_contains(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_output_contains", tmp_path,
            {"command": ["python", "-c", "print('success')"], "contains": "success",
             "allowed_commands": ["python"]},
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

class TestLayer2CommandSecurity:
    """命令判分必须走参数列表且受 allowlist 约束(契约 §8.4).

    旧实现是 `subprocess.run(str(args["command"]), shell=True, ...)`,命令字符串
    直接来自 `evals/*.jsonl`。任何能改那个文件的人(包括未来的 Agent)就能在宿主
    上执行任意命令。下面这些测试钉住的就是那条通道已经关闭。
    """

    def test_argv_list_is_accepted(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_ok", tmp_path,
            {"command": ["python", "-c", "raise SystemExit(0)"], "allowed_commands": ["python"]},
        ) is True

    def test_nonzero_exit_fails(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_ok", tmp_path,
            {"command": ["python", "-c", "raise SystemExit(2)"], "allowed_commands": ["python"]},
        ) is False

    def test_shell_substitution_is_not_expanded(self, tmp_path: Path) -> None:
        """`$(...)`/反引号/`%VAR%` 不再被 shell 展开,而是原样传给被调程序.

        这是旧实现最直接的漏洞:`shell=True` 下命令字符串里任何位置都能塞进
        `$(touch pwned)`,由 cmd.exe / /bin/sh 展开执行。现在没有 shell 参与,
        整串作为 argv 的一个元素传给 python,而 python 只是把它当字符串打印。
        """
        for payload in ("$(echo hi)", "`echo hi`", "%PATH%", "&&", "|"):
            proc = judge_case(
                "command_output_contains", tmp_path,
                {
                    "command": ["python", "-c", "import sys; print(repr(sys.argv[1]))", payload],
                    "contains": re.escape(payload),
                    "allowed_commands": ["python"],
                },
            )
            assert proc is True, f"{payload!r} was not passed through literally"
        assert not (tmp_path / "hi").exists()

    def test_metacharacter_after_a_valid_program_is_not_a_new_command(self, tmp_path: Path) -> None:
        """`&&` 后面的内容不能变成第二条命令 —— 它只是被调程序的一个参数.

        这里的判据是**副作用**,不是退出码:python 会忽略 `-c <code>` 之后多余的
        argv,所以进程正常退出(returncode 0)。真正说明问题的是 `&&` 右侧的
        `touch pwned` 没有被执行 —— 用户目录里没有多出文件。
        """
        marker = tmp_path / "pwned"
        judge_case(
            "command_ok", tmp_path,
            {
                "command": ["python", "-c", "pass", "&&", "touch", "pwned"],
                "allowed_commands": ["python"],
            },
        )
        assert not marker.exists(), "a second command was executed"

    def test_undeclared_command_is_rejected(self, tmp_path: Path) -> None:
        # 没有声明 allowlist 就把命令交给系统 = allowlist 形同虚设.
        with pytest.raises(ValueError, match="allowed_commands"):
            judge_case("command_ok", tmp_path, {"command": ["python", "-c", "pass"]})

    def test_command_outside_the_allowlist_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not in the case's declared"):
            judge_case(
                "command_ok", tmp_path,
                {"command": ["curl", "http://x"], "allowed_commands": ["python"]},
            )

    def test_path_shaped_program_is_rejected(self, tmp_path: Path) -> None:
        # `./evil` 放在 fixture 里就能绕过 allowlist 的名字检查,必须按形状拒绝.
        for prog in ("./evil", "/usr/bin/python", "C:\\tmp\\python.exe"):
            with pytest.raises(ValueError, match="bare executable name"):
                judge_case(
                    "command_ok", tmp_path,
                    {"command": [prog], "allowed_commands": ["python"]},
                )

    def test_basename_matching_allows_versioned_interpreter(self, tmp_path: Path) -> None:
        # 声明 python 就该放行 python3.12 / python.exe,但不放行 pythonx.
        assert judge_case(
            "command_ok", tmp_path,
            {"command": ["python3.12", "-c", "pass"], "allowed_commands": ["python"]},
        ) is True
        with pytest.raises(ValueError):
            judge_case(
                "command_ok", tmp_path,
                {"command": ["pythonx", "-c", "pass"], "allowed_commands": ["python"]},
            )

    def test_string_command_is_shlex_split_not_shell_interpreted(self, tmp_path: Path) -> None:
        assert judge_case(
            "command_output_contains", tmp_path,
            {"command": "python -c print(42)", "contains": "42", "allowed_commands": ["python"]},
        ) is True

    def test_empty_command_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty command"):
            judge_case("command_ok", tmp_path, {"command": [], "allowed_commands": ["python"]})


class TestJsonValue:
    def test_top_level_key(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text('{"app": "demo"}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["app"], "equals": "demo"}
        ) is True

    def test_nested_key_and_index(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text(
            '{"servers": [{"port": 8000}, {"port": 9000}]}', encoding="utf-8"
        )
        assert judge_case(
            "json_value", tmp_path,
            {"path": "c.json", "key_path": ["servers", 1, "port"], "equals": 9000},
        ) is True

    def test_wrong_value_fails(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text('{"port": 3000}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["port"], "equals": 8000}
        ) is False

    def test_int_and_string_are_not_interchangeable(self, tmp_path: Path) -> None:
        # `"8000"` 不是 `8000`:宽松比较会放过任何严格下游都会拒绝的产物.
        (tmp_path / "c.json").write_text('{"port": "8000"}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["port"], "equals": 8000}
        ) is False

    def test_float_and_int_compare_equal(self, tmp_path: Path) -> None:
        # JSON 只有一个数字类型,写 80.0 并没有改变值.
        (tmp_path / "c.json").write_text('{"n": 80.0}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["n"], "equals": 80}
        ) is True

    def test_bool_is_not_one(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text('{"flag": true}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["flag"], "equals": 1}
        ) is False

    def test_unparseable_json_fails_without_raising(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text("{not json", encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["a"], "equals": 1}
        ) is False

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        assert judge_case(
            "json_value", tmp_path, {"path": "nope.json", "key_path": ["a"], "equals": 1}
        ) is False

    def test_missing_path_fails(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text('{"a": 1}', encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": ["b"], "equals": 1}
        ) is False

    def test_out_of_range_index_fails(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text("[1]", encoding="utf-8")
        assert judge_case(
            "json_value", tmp_path, {"path": "c.json", "key_path": [5], "equals": 1}
        ) is False


class TestLineSetEquals:
    def test_exact_set_matches_regardless_of_order(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("b\na\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "f.txt", "equals": ["a", "b"]}
        ) is True

    def test_extra_line_fails(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "f.txt", "equals": ["a", "b"]}
        ) is False

    def test_missing_line_fails(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "f.txt", "equals": ["a", "b"]}
        ) is False

    def test_blank_lines_and_whitespace_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("\n  a  \n\n b \n\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "f.txt", "equals": ["a", "b"]}
        ) is True

    def test_not_contains_rejects_junk(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\nDEBUG: leftover\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path,
            {"path": "f.txt", "equals": ["a", "b", "DEBUG: leftover"], "not_contains": "DEBUG"},
        ) is False

    def test_duplicates_allowed_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\na\nb\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "f.txt", "equals": ["a", "b"]}
        ) is True

    def test_duplicates_rejected_when_disallowed(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\na\nb\n", encoding="utf-8")
        assert judge_case(
            "line_set_equals", tmp_path,
            {"path": "f.txt", "equals": ["a", "b"], "duplicates_allowed": False},
        ) is False

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        assert judge_case(
            "line_set_equals", tmp_path, {"path": "nope.txt", "equals": []}
        ) is False


class TestDirectorySnapshot:
    def test_exact_file_set(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path, {"path": ".", "equals": ["a.txt", "sub/b.txt"]}
        ) is True

    def test_stray_file_fails_exact_mode(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "scratch.tmp").write_text("junk", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path, {"path": ".", "equals": ["a.txt"]}
        ) is False

    def test_files_contains_is_the_weaker_form(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "scratch.tmp").write_text("junk", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path,
            {"path": ".", "files_contains": ["a.txt"], "files_exact": False},
        ) is True

    def test_missing_required_file_fails(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path,
            {"path": ".", "files_contains": ["a.txt", "b.txt"], "files_exact": False},
        ) is False

    def test_min_size_rejects_an_empty_file(self, tmp_path: Path) -> None:
        """`file_exists` 挡不住的「创建了但没写内容」必须被尺寸下限挡下."""
        (tmp_path / "out.txt").write_text("", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path,
            {"path": ".", "files_contains": ["out.txt"], "files_exact": False,
             "min_sizes": {"out.txt": 1}},
        ) is False
        (tmp_path / "out.txt").write_text("done", encoding="utf-8")
        assert judge_case(
            "directory_snapshot", tmp_path,
            {"path": ".", "files_contains": ["out.txt"], "files_exact": False,
             "min_sizes": {"out.txt": 1}},
        ) is True

    def test_directories_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "empty_dir").mkdir()
        assert judge_case(
            "directory_snapshot", tmp_path, {"path": ".", "equals": ["a.txt"]}
        ) is True

    def test_missing_root_fails(self, tmp_path: Path) -> None:
        assert judge_case(
            "directory_snapshot", tmp_path, {"path": "nope", "equals": []}
        ) is False


class TestPythonTest:
    def test_passing_test_succeeds(self, tmp_path: Path) -> None:
        (tmp_path / "test_ok.py").write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")
        assert judge_case(
            "python_test", tmp_path,
            {"command": ["python", "-m", "pytest", "test_ok.py", "-q"],
             "allowed_commands": ["python"]},
        ) is True

    def test_failing_test_fails(self, tmp_path: Path) -> None:
        (tmp_path / "test_bad.py").write_text("def test_a():\n    assert 1 == 2\n", encoding="utf-8")
        assert judge_case(
            "python_test", tmp_path,
            {"command": ["python", "-m", "pytest", "test_bad.py", "-q"],
             "allowed_commands": ["python"]},
        ) is False

    def test_argv_only_no_shell(self, tmp_path: Path) -> None:
        # 与 command_ok 共用同一条受控路径:命令字符串必须仍被 allowlist 拦下.
        with pytest.raises(ValueError, match="not in the case's declared"):
            judge_case(
                "python_test", tmp_path,
                {"command": ["rm", "-rf", "/"], "allowed_commands": ["python"]},
            )

    def test_declared_metadata_is_carried_in_args(self, tmp_path: Path) -> None:
        # path/test/scope 只作报告用,不改变判定;它们必须存在且不报错.
        (tmp_path / "test_ok.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
        assert judge_case(
            "python_test", tmp_path,
            {"command": ["python", "-m", "pytest", "-q"], "allowed_commands": ["python"],
             "path": "test_ok.py", "test": "a", "scope": "sandbox"},
        ) is True


class TestCasePassed:
    """复合 checks 默认全部通过才算成功(契约 §5.1)."""

    def test_all_checks_must_pass(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        passed, detail = case_passed(
            [
                {"fn": "file_exists", "args": {"path": "a.txt"}},
                {"fn": "file_content", "args": {"path": "a.txt", "contains": "hello"}},
            ],
            tmp_path,
        )
        assert passed is True
        assert [d["passed"] for d in detail] == [True, True]

    def test_one_failing_check_fails_the_case(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        passed, detail = case_passed(
            [
                {"fn": "file_exists", "args": {"path": "a.txt"}},
                {"fn": "file_content", "args": {"path": "a.txt", "contains": "nope"}},
            ],
            tmp_path,
        )
        assert passed is False
        # 逐条明细必须保留,失败报告要能指出是哪一条断言断的.
        assert [d["passed"] for d in detail] == [True, False]
        assert detail[1]["fn"] == "file_content"

    def test_empty_checks_list_raises_instead_of_passing_vacuously(self, tmp_path: Path) -> None:
        # 以前这里钉的是「空列表在 all() 下为真」. 那是个陷阱: E2ECase 从
        # from_dict 进来时确实会被拒, 但直接构造 (测试、程序化调用方) 不走
        # loader, 于是 case_passed([], ...) 返回 (True, []) -- 一个没有任何
        # 断言的用例被报成「通过」. 静默恒真正是这套判分器一直在防的模式,
        # 所以现在改成显式报错.
        with pytest.raises(ValueError, match="no checks"):
            case_passed([], tmp_path)

    def test_any_mode_passes_on_one_hit(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text('{"k": 1}', encoding="utf-8")
        passed, _ = case_passed(
            [
                {"fn": "file_exists", "args": {"path": "a.yaml"}},
                {"fn": "file_exists", "args": {"path": "a.json"}},
            ],
            tmp_path,
            mode="any",
        )
        assert passed is True

    def test_any_mode_still_fails_when_nothing_hits(self, tmp_path: Path) -> None:
        passed, _ = case_passed(
            [{"fn": "file_exists", "args": {"path": "a.yaml"}}], tmp_path, mode="any"
        )
        assert passed is False

    def test_a_raising_check_is_recorded_not_propagated(self, tmp_path: Path) -> None:
        """一条坏 check 不能中止整轮 40 条 —— 记成失败并留下原因."""
        passed, detail = case_passed(
            [
                {"fn": "no_such_judge", "args": {}},
                {"fn": "file_exists", "args": {"path": "missing"}},
            ],
            tmp_path,
        )
        assert passed is False
        assert detail[0]["error"] is not None
        assert "unknown judge fn" in str(detail[0]["error"])

    def test_bad_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checks_mode"):
            case_passed([], tmp_path, mode="most")
