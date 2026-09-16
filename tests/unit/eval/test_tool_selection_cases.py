"""Contract tests for evals/tool_selection.jsonl — the 60-case tool set.

This file is the executable form of `evals/README.md` §5.2 and §8.1. It fails
on the things a reviewer would otherwise have to notice by hand:

- the family breakdown and the 48/12 blind split,
- every case's expectation being satisfiable against a real tool profile,
- no blind task text naming or hinting at a tool,
- no judge being trivially true (an empty `expect_args` is not a check).

The leakage checks are literal scans; the Chinese hint list is a heuristic and
is documented as one in `test_leakage.py_` — see `leakage.py`'s docstring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from longline.eval.eval_tools import ALL_EVAL_TOOL_NAMES, build_tool_profile
from longline.eval.types import ToolCallCase, load_cases
from tests.unit.eval.leakage import (
    BLIND_TAG,
    INSTRUCTION_FOLLOWING_TAG,
    blind_cases,
    find_hint_leaks,
    find_tool_name_leaks,
    instruction_following_cases,
)

CASE_FILE = Path(__file__).resolve().parents[3] / "evals" / "tool_selection.jsonl"

# Family tag -> required case count (plan §4.2). The plan's table sums to 60,
# which is the whole set: 48 blind + 12 instruction-following. The instruction
# half carries family tags too, so each family's total is split across both
# halves rather than the instruction cases being an extra 12 on top.
FAMILY_COUNTS: dict[str, int] = {
    "read-write-edit": 16,
    "glob-grep": 12,
    "bash": 6,
    "web": 8,
    "notebook": 4,
    "task": 6,
    "multi": 8,
}

EXPECTED_INSTRUCTION_FOLLOWING = 12
# 8 abstention cases (the correct action is to call no tool) were added after the
# original 60. They carry no family tag -- they exercise no tool family -- so
# FAMILY_COUNTS is unchanged and the family sum stays 60.
EXPECTED_ABSTENTION = 8
EXPECTED_BLIND = 48 + EXPECTED_ABSTENTION  # 56
EXPECTED_TOTAL = sum(FAMILY_COUNTS.values()) + EXPECTED_ABSTENTION  # 68


@pytest.fixture(scope="module")
def cases() -> list[ToolCallCase]:
    loaded = load_cases(CASE_FILE)
    assert all(isinstance(c, ToolCallCase) for c in loaded)
    return loaded  # type: ignore[return-value]


class TestFileShape:
    def test_case_file_exists_and_loads(self) -> None:
        assert CASE_FILE.is_file(), f"missing case file: {CASE_FILE}"

    def test_ids_are_unique(self, cases: list[ToolCallCase]) -> None:
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_total_matches_the_contract(self, cases: list[ToolCallCase]) -> None:
        """68 = 56 blind + 12 instruction-following.

        The blind half is 48 tool cases + 8 abstention cases. The abstention
        cases carry no family tag, so the family table below still sums to 60.
        """
        assert len(cases) == EXPECTED_TOTAL

    def test_exact_family_breakdown(self, cases: list[ToolCallCase]) -> None:
        """Family counts are over the 60 tool cases (plan §4.2's table sums to 60).

        The instruction half carries family tags too, so the split of a family
        between blind and instruction-following is visible here rather than
        being hidden by an off-by-twelve total. Abstention cases are excluded by
        construction: they carry no family tag.
        """
        counts: dict[str, int] = {family: 0 for family in FAMILY_COUNTS}
        for c in cases:
            for family in FAMILY_COUNTS:
                if family in c.tags:
                    counts[family] += 1
        assert counts == FAMILY_COUNTS

    def test_abstention_cases_require_calling_nothing(self, cases: list[ToolCallCase]) -> None:
        """The abstention class is the one BFCL devotes ~25% of its set to.

        A suite where every case requires a call rewards an agent that always
        calls something. Each abstention case must therefore declare an EMPTY
        `accepted_tool_steps` (call nothing is the pass condition) and must not
        be counted in any tool family.
        """
        abstention = [c for c in cases if "abstention" in c.tags]
        assert len(abstention) == EXPECTED_ABSTENTION
        for c in abstention:
            assert c.accepted_tool_steps == [], f"{c.id}: abstention case must expect no tool step"
            assert not (set(c.tags) & set(FAMILY_COUNTS)), f"{c.id}: abstention case carries a family tag"
            assert BLIND_TAG in c.tags, f"{c.id}: abstention case must be blind (no tool named)"

    def test_each_family_contributes_to_both_halves(self, cases: list[ToolCallCase]) -> None:
        # 每个族都必须有盲测样本,否则它的选择率没有分母.
        for family in FAMILY_COUNTS:
            blind_in_family = [c for c in blind_cases(cases) if family in c.tags]
            assert blind_in_family, f"family {family} has no blind case"

    def test_instruction_cases_cover_every_family(self, cases: list[ToolCallCase]) -> None:
        covered = {
            family
            for c in instruction_following_cases(cases)
            for family in FAMILY_COUNTS
            if family in c.tags
        }
        assert covered == set(FAMILY_COUNTS), f"instruction set misses {set(FAMILY_COUNTS) - covered}"

    def test_blind_and_instruction_split(self, cases: list[ToolCallCase]) -> None:
        assert len(blind_cases(cases)) == EXPECTED_BLIND
        assert len(instruction_following_cases(cases)) == EXPECTED_INSTRUCTION_FOLLOWING

    def test_exactly_one_kind_per_case(self, cases: list[ToolCallCase]) -> None:
        # 一条用例不能同时进主数字和回归集.
        for c in cases:
            is_blind = BLIND_TAG in c.tags
            is_instr = INSTRUCTION_FOLLOWING_TAG in c.tags
            assert is_blind != is_instr, f"{c.id} must be exactly one of blind/instruction"

    def test_every_blind_case_has_a_rationale(self, cases: list[ToolCallCase]) -> None:
        # 可机读的「为什么不泄漏」说明是人工复核的抓手.
        for c in blind_cases(cases):
            assert c.blind_rationale, f"{c.id} is blind but has no blind_rationale"
            assert len(c.blind_rationale.strip()) >= 8, f"{c.id} rationale is too thin"

    def test_instruction_cases_name_the_tool(self, cases: list[ToolCallCase]) -> None:
        # 反过来:这一类*必须*点名工具,否则它就不是指令跟随用例.
        for c in instruction_following_cases(cases):
            assert find_tool_name_leaks(c.task), (
                f"{c.id} is tagged instruction-following but its task names no tool"
            )


class TestNoLeakageInBlindCases:
    def test_no_blind_task_contains_a_registered_tool_name(
        self, cases: list[ToolCallCase]
    ) -> None:
        offenders = {
            c.id: find_tool_name_leaks(c.task) for c in blind_cases(cases)
            if find_tool_name_leaks(c.task)
        }
        assert not offenders, f"blind cases name a tool outright: {offenders}"

    def test_no_blind_task_contains_a_chinese_hint_word(
        self, cases: list[ToolCallCase]
    ) -> None:
        """Heuristic check — a green result here is necessary, not sufficient."""
        offenders = {
            c.id: find_hint_leaks(c.task) for c in blind_cases(cases)
            if find_hint_leaks(c.task)
        }
        assert not offenders, f"blind cases hint at a tool: {offenders}"

    def test_hint_word_list_is_actually_populated(self) -> None:
        # 防止有人把词表清空让测试变绿.
        from tests.unit.eval.leakage import TOOL_HINT_WORDS

        assert len(TOOL_HINT_WORDS) >= 5
        assert sum(len(v) for v in TOOL_HINT_WORDS.values()) >= 25


class TestExpectationsAreSatisfiable:
    def test_every_expected_tool_exists_in_a_profile(self, cases: list[ToolCallCase]) -> None:
        known = set(ALL_EVAL_TOOL_NAMES)
        unknown: dict[str, list[str]] = {}
        for c in cases:
            bad = sorted({t for step in c.accepted_tool_steps for t in step} - known)
            if bad:
                unknown[c.id] = bad
        assert not unknown, f"cases expect tools no profile registers: {unknown}"

    def test_web_cases_need_the_web_profile(self, cases: list[ToolCallCase]) -> None:
        # 用例的可接受工具必须真的在对应 profile 里,否则期望无法被满足.
        for c in cases:
            if "web" in c.tags:
                names = set(build_tool_profile("web"))
                wanted = {t for step in c.accepted_tool_steps for t in step}
                assert wanted <= names, f"{c.id} expects {wanted - names} missing from the web profile"

    def test_task_cases_need_the_task_profile(self, cases: list[ToolCallCase]) -> None:
        for c in cases:
            if "task" in c.tags:
                names = set(build_tool_profile("task"))
                wanted = {t for step in c.accepted_tool_steps for t in step}
                assert wanted <= names, f"{c.id} expects {wanted - names} missing from the task profile"

    def test_notebook_cases_need_the_notebook_profile(self, cases: list[ToolCallCase]) -> None:
        for c in cases:
            if "notebook" in c.tags:
                names = set(build_tool_profile("notebook"))
                wanted = {t for step in c.accepted_tool_steps for t in step}
                assert wanted <= names, f"{c.id} expects {wanted - names} missing from the notebook profile"


class TestJudgesAreNotTriviallyTrue:
    def test_every_case_declares_args_for_at_least_one_step_tool(
        self, cases: list[ToolCallCase]
    ) -> None:
        """A case with no `expect_args` checks only tool *name*.

        That is legitimate for a name-only decision, but if it is the norm the
        argument metrics have no denominator at all. Requiring a majority keeps
        ArgumentCallAccuracy real (contract §5.2 wants a non-empty denominator).
        """
        with_args = [c for c in cases if c.expect_args]
        assert len(with_args) >= 40, (
            f"only {len(with_args)}/60 cases declare expect_args; "
            "the argument metrics would have almost no denominator"
        )

    def test_expect_args_keys_are_reachable(self, cases: list[ToolCallCase]) -> None:
        # 参数期望挂在一个用例永远不会调用的工具上 = 恒真的空断言.
        for c in cases:
            expected = {t for step in c.accepted_tool_steps for t in step}
            unreachable = sorted(set(c.expect_args) - expected)
            assert not unreachable, f"{c.id} declares args for un-expected tools {unreachable}"

    def test_all_arg_patterns_compile(self, cases: list[ToolCallCase]) -> None:
        for c in cases:
            for pats in c.expect_args.values():
                for pat in pats.values():
                    re.compile(pat)  # raises on a bad pattern

    def test_multi_candidate_steps_are_actually_used(self, cases: list[ToolCallCase]) -> None:
        # 契约允许一个决策点有多个合理工具;至少要真的用到这个能力.
        multi = [c for c in cases if any(len(step) > 1 for step in c.accepted_tool_steps)]
        assert len(multi) >= 8, f"only {len(multi)} cases accept more than one tool per step"

    def test_multi_tool_cases_have_at_least_two_steps(self, cases: list[ToolCallCase]) -> None:
        for c in cases:
            if "multi" in c.tags:
                assert len(c.accepted_tool_steps) >= 2, f"{c.id} is multi but has one step"


class TestFixtureIsolation:
    def test_referenced_fixtures_exist(self, cases: list[ToolCallCase]) -> None:
        fixtures_dir = CASE_FILE.parent / "fixtures"
        for c in cases:
            if c.fixture:
                assert (fixtures_dir / c.fixture).is_dir(), f"{c.id}: missing fixture {c.fixture}"

    def test_shared_fixture_simple_repo_is_not_modified(self) -> None:
        """`simple_repo` is the legacy suite's fixture and must stay byte-identical.

        `evals/tool_calls.jsonl` has 30 cases against it, two of which assert on
        `src/version.py`'s exact contents. Adding a file or editing one to make a
        new case more convenient silently changes what the legacy cases measure,
        so new cases get their own fixture (`tool_repo`) instead.
        """
        fixture = CASE_FILE.parent / "fixtures" / "simple_repo"
        on_disk = sorted(
            p.relative_to(fixture).as_posix() for p in fixture.rglob("*") if p.is_file()
        )
        assert on_disk == [
            "README.md",
            "data/notes.txt",
            "last.txt",
            "src/config.py",
            "src/main.py",
            "src/version.py",
        ], f"simple_repo changed; the legacy suite depends on its exact contents: {on_disk}"

    def test_max_turns_is_sane(self, cases: list[ToolCallCase]) -> None:
        for c in cases:
            assert 1 <= c.max_turns <= 20, f"{c.id} max_turns={c.max_turns}"

    def test_raw_jsonl_shape_of_one_case(self) -> None:
        """Each case must round-trip through JSON with its Task 2 fields intact."""
        line = json.dumps({
            "id": "x", "type": "tool_call", "task": "t",
            "accepted_tool_steps": [["Glob", "Grep"], ["Read"]],
            "max_extra_calls": 1, "blind_rationale": "why",
        })
        d: dict[str, Any] = json.loads(line)
        c = ToolCallCase.from_dict(d)
        assert c.max_extra_calls == 1
