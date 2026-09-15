"""Unit tests for longline/eval/judges.py — deterministic judges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from longline.eval.judges import check_args, check_tools, judge_case

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
