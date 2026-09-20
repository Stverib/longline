"""Tests for the paired-benefit corpus generator.

`evals/tools/gen_pair_cases.py` is loaded by file location rather than by
import: `evals/tools` is not a package (the other generators there are run as
scripts), and adding an `__init__.py` to reach this one would change how those
two are invoked.

=== What these tests are for ===

The corpus is generated rather than hand-authored because every case needs TWO
sibling fixture trees that are byte-identical, and a human keeping 36 trees in
sync is a promise rather than a mechanism. `assert_fixtures_identical` already
checks a case's two trees; it says nothing about a rerun matching the previous
rerun, and it cannot see whether the six analysis cases are six different tasks
or one task six times -- which is exactly the defect `multi_agent.jsonl` has
(18 controlled rows sharing one `task` string).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from longline.eval.multi_agent import (
    CATEGORY_ANALYSIS,
    assert_fixtures_identical,
    load_multi_agent_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_generator() -> Any:
    """Import the generator by file location.

    The module is registered in `sys.modules` BEFORE `exec_module`, which is
    not optional here: `@dataclass` resolves `cls.__module__` through
    `sys.modules`, so a module executed without being registered raises
    `AttributeError: 'NoneType' object has no attribute '__dict__'` on the first
    dataclass it defines.
    """
    path = PROJECT_ROOT / "evals" / "tools" / "gen_pair_cases.py"
    spec = importlib.util.spec_from_file_location("gen_pair_cases", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _digest_tree(root: Path) -> dict[str, str]:
    """`relative posix path -> sha256` for every file under `root`."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


class TestGeneratorIsDeterministic:
    def test_two_runs_produce_identical_corpora(self, tmp_path: Path) -> None:
        """Idempotence is what keeps a regenerated corpus reviewable as a diff.

        `assert_fixtures_identical` proves a case's two SIBLING trees match. It
        says nothing about run 2 matching run 1, and a generator that embedded a
        timestamp or iterated a set would still pass that check while producing
        a diff every time it ran.
        """
        first = gen.generate(tmp_path / "a")
        second = gen.generate(tmp_path / "b")

        assert _digest_tree(first.fixtures_dir) == _digest_tree(second.fixtures_dir)
        assert first.cases_file.read_bytes() == second.cases_file.read_bytes()

    def test_regenerating_over_an_existing_corpus_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        """Running twice into ONE directory must not accumulate or drift."""
        gen.generate(tmp_path)
        before = _digest_tree(tmp_path / "fixtures")

        gen.generate(tmp_path)

        assert _digest_tree(tmp_path / "fixtures") == before


class TestSiblingFixturesAreIdentical:
    def test_every_case_passes_the_identity_check(self, tmp_path: Path) -> None:
        """The loader's own check, run on every generated case.

        Both trees come out of one `build_fixture` call with the same arguments,
        so this is a regression on the generator rather than the thing that
        makes the trees match. A generator that wrote a case id, a path or an
        index into only one of the two trees would be caught here.
        """
        root = gen.generate(tmp_path)
        cases = load_multi_agent_cases(
            root.cases_file, fixtures_root=root.fixtures_dir
        )

        assert cases, "the generator produced no cases"
        for case in cases:
            assert_fixtures_identical(case, root.fixtures_dir)


class TestAnalysisCategory:
    def test_there_are_six_analysis_cases(self, tmp_path: Path) -> None:
        root = gen.generate(tmp_path)
        cases = load_multi_agent_cases(
            root.cases_file, fixtures_root=root.fixtures_dir
        )

        counts = Counter(c.category for c in cases)
        assert counts[CATEGORY_ANALYSIS] == 6

    def test_no_two_analysis_cases_share_a_task_string(self, tmp_path: Path) -> None:
        """Six clones of one prompt is the defect this corpus replaces.

        `multi_agent.jsonl`'s 18 controlled rows carry one identical `task`
        string. Folding them into a category would make that category's average
        a fact about one template, and this assertion is what stops the new
        corpus from being the same mistake with a different filename.
        """
        root = gen.generate(tmp_path)
        cases = [
            c for c in load_multi_agent_cases(
                root.cases_file, fixtures_root=root.fixtures_dir
            )
            if c.category == CATEGORY_ANALYSIS
        ]

        tasks = [c.task for c in cases]
        assert len(set(tasks)) == len(tasks), "analysis cases share a task string"

    def test_analysis_subtasks_never_share_a_write_target(self, tmp_path: Path) -> None:
        """The loader rejects overlap; this checks the generator does not emit it."""
        root = gen.generate(tmp_path)
        cases = [
            c for c in load_multi_agent_cases(
                root.cases_file, fixtures_root=root.fixtures_dir
            )
            if c.category == CATEGORY_ANALYSIS
        ]

        for case in cases:
            targets = [s.writes for s in case.subtasks]
            assert len(set(targets)) == len(targets), case.id

    def test_each_analysis_case_declares_at_least_two_subtasks(
        self, tmp_path: Path
    ) -> None:
        """Below two there is nothing to fan out, and the loader would reject it."""
        root = gen.generate(tmp_path)
        cases = [
            c for c in load_multi_agent_cases(
                root.cases_file, fixtures_root=root.fixtures_dir
            )
            if c.category == CATEGORY_ANALYSIS
        ]

        for case in cases:
            assert len(case.subtasks) >= 2, case.id

    def test_the_fixtures_carry_the_facts_the_subtasks_ask_about(
        self, tmp_path: Path
    ) -> None:
        """A subtask asking about a module the fixture does not contain is a
        case nobody can pass, and it would fail as a model error rather than as
        a corpus error.
        """
        root = gen.generate(tmp_path)
        cases = [
            c for c in load_multi_agent_cases(
                root.cases_file, fixtures_root=root.fixtures_dir
            )
            if c.category == CATEGORY_ANALYSIS
        ]

        for case in cases:
            tree = root.fixtures_dir / case.fixture_single
            module_names = {
                p.stem for p in (tree / "modules").glob("*.py")
            }
            assert module_names, f"{case.id}: fixture has no modules/"
            for subtask in case.subtasks:
                named = [
                    name for name in module_names
                    if name in subtask.instruction
                ]
                assert named, (
                    f"{case.id}/{subtask.id}: the instruction names no module "
                    f"present in the fixture ({sorted(module_names)})"
                )
