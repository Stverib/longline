"""Unit tests for longline/eval/cli.py — argument parsing + wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from longline.eval import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_known_args_defaults(tmp_path: Path) -> None:
    ns = cli.parse_args(["--case-file", str(tmp_path / "c.jsonl")])
    assert ns.type == "all"
    assert ns.model == "claude-sonnet-4-20250514"
    assert ns.case_file == str(tmp_path / "c.jsonl")
    assert ns.max_cases is None
    assert ns.out_dir is not None


def test_parse_args_type_filter() -> None:
    ns = cli.parse_args(["--type", "e2e", "--model", "claude-haiku-4-5-20251001", "--max-cases", "3"])
    assert ns.type == "e2e"
    assert ns.model == "claude-haiku-4-5-20251001"
    assert ns.max_cases == 3


def test_split_cases_by_type() -> None:
    from longline.eval.types import E2ECase, ToolCallCase

    cases = [
        ToolCallCase(id="a", task="t"),
        E2ECase(id="b", task="t"),
    ]
    tc, e2e = cli.split_cases(cases)
    assert [c.id for c in tc] == ["a"]
    assert [c.id for c in e2e] == ["b"]
