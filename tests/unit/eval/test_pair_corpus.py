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
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from longline.eval.multi_agent import (
    CATEGORY_ANALYSIS,
    CATEGORY_MODIFICATION,
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


def _cases_in(root: Any, category: str) -> list[Any]:
    """Every case of one category, asserting the category is not empty.

    The `assert` is the point. A `for case in cases:` loop over an empty list
    passes without executing a single assertion, so a test that loads a
    category and forgot to check it exists would go green the moment the
    generator stopped emitting it -- reporting success for the exact failure it
    was written to catch.
    """
    cases = [
        c for c in load_multi_agent_cases(
            root.cases_file, fixtures_root=root.fixtures_dir
        )
        if c.category == category
    ]
    assert cases, f"the {category} category must not be empty"
    return cases


class TestModificationCategory:
    def test_there_are_six_modification_cases(self, tmp_path: Path) -> None:
        root = gen.generate(tmp_path)
        cases = load_multi_agent_cases(
            root.cases_file, fixtures_root=root.fixtures_dir
        )

        counts = Counter(c.category for c in cases)
        assert counts[CATEGORY_MODIFICATION] == 6

    def test_no_hidden_test_is_reachable_from_a_fixture(self, tmp_path: Path) -> None:
        """A hidden test living in the fixture is not hidden.

        The runner copies the fixture into a temp sandbox and runs the agent
        there, so anything under that tree is readable by the model. A case
        whose answer is readable measures reading, and it would report a pass
        rate no live run could reproduce.
        """
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_MODIFICATION)

        assert cases, "the modification category must not be empty"
        for case in cases:
            for side in ("single", "multi"):
                tree = root.fixtures_dir / "pair" / f"{case.id}_{side}"
                assert tree.is_dir(), f"{case.id}: missing {side} fixture"
                assert not list(tree.rglob("test_*.py")), (
                    f"{case.id}: a hidden test is reachable from the {side} fixture"
                )
                assert not list(tree.rglob("*hidden*")), (
                    f"{case.id}: a hidden test is reachable from the {side} fixture"
                )

    def test_every_modification_case_declares_a_hidden_test(self, tmp_path: Path) -> None:
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_MODIFICATION)

        for case in cases:
            assert case.hidden_test, f"{case.id}: no hidden_test declared"
            path = root.fixtures_dir / case.hidden_test
            assert path.is_file(), f"{case.id}: hidden test missing at {path}"

    def test_every_modification_case_grades_with_python_test(self, tmp_path: Path) -> None:
        """The hidden test has to actually be run, not merely declared."""
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_MODIFICATION)

        for case in cases:
            fns = [check["fn"] for check in case.checks]
            assert "python_test" in fns, f"{case.id}: judges are {fns}"

    def test_the_hidden_test_names_the_modules_the_case_touches(
        self, tmp_path: Path
    ) -> None:
        """A hidden test for the wrong modules would grade the wrong work."""
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_MODIFICATION)

        for case in cases:
            tree = root.fixtures_dir / "pair" / f"{case.id}_single"
            module_names = sorted(p.stem for p in (tree / "modules").glob("*.py"))
            assert module_names, f"{case.id}: fixture has no modules/"
            source = (root.fixtures_dir / case.hidden_test).read_text(encoding="utf-8")
            for name in module_names:
                assert name in source, (
                    f"{case.id}: the hidden test never mentions module {name!r}"
                )

    def test_the_hidden_test_is_not_satisfied_by_the_untouched_fixture(
        self, tmp_path: Path
    ) -> None:
        """The judge must FAIL before the work is done.

        A test that passes on the starting tree grades nothing: every run would
        score, including one that did not touch the repository, and the pass
        rate would be a property of the fixture rather than of the agent. This
        is the same mistake `TestFixturesAreNotPreSatisfied` guards against in
        the E2E suite.
        """
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_MODIFICATION)

        for case in cases:
            tree = root.fixtures_dir / "pair" / f"{case.id}_single"
            proc = _run_hidden_test(tree, root.fixtures_dir / case.hidden_test)
            assert proc.returncode != 0, (
                f"{case.id}: the hidden test PASSES on the untouched fixture, so "
                "it grades nothing"
            )

    def test_the_hidden_test_passes_once_the_work_is_done(self, tmp_path: Path) -> None:
        """The other half: the judge must be PASSABLE.

        "Fails on the untouched tree" is not sufficient evidence that a judge
        measures the work. A test with an ImportError fails on every tree,
        including a correct one, and the case would report 0% forever while
        looking like a hard task rather than a broken judge.
        """
        root = gen.generate(tmp_path)
        for case in _cases_in(root, CATEGORY_MODIFICATION):
            tree = root.fixtures_dir / "pair" / f"{case.id}_single"
            for module in sorted((tree / "modules").glob("*.py")):
                module.write_text(
                    module.read_text(encoding="utf-8")
                    + f'\n\ndef describe() -> str:\n    return "the {module.stem} module"\n',
                    encoding="utf-8",
                )
            proc = _run_hidden_test(tree, root.fixtures_dir / case.hidden_test)
            assert proc.returncode == 0, (
                f"{case.id}: the hidden test cannot be satisfied even by a "
                f"correct implementation\n{proc.stdout}\n{proc.stderr}"
            )


def _run_hidden_test(tree: Path, hidden_test: Path) -> subprocess.CompletedProcess[str]:
    """Run a case's hidden judge inside a fixture tree, then clean up after it.

    The copy is removed afterwards so the fixture tree goes back to being
    byte-identical to its sibling -- a leaked `test_hidden.py` would make the
    NEXT assertion about "the fixture contains no hidden test" fail, and the
    failure would point at the wrong test.
    """
    target = tree / hidden_test.name
    target.write_bytes(hidden_test.read_bytes())
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", hidden_test.name, "-q", "-p", "no:cacheprovider"],
            cwd=tree, capture_output=True, text=True, check=False,
        )
    finally:
        target.unlink()
        shutil.rmtree(tree / ".pytest_cache", ignore_errors=True)


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
        cases = _cases_in(root, CATEGORY_ANALYSIS)

        tasks = [c.task for c in cases]
        assert len(set(tasks)) == len(tasks), "analysis cases share a task string"

    def test_analysis_subtasks_never_share_a_write_target(self, tmp_path: Path) -> None:
        """The loader rejects overlap; this checks the generator does not emit it."""
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_ANALYSIS)

        for case in cases:
            targets = [s.writes for s in case.subtasks]
            assert len(set(targets)) == len(targets), case.id

    def test_each_analysis_case_declares_at_least_two_subtasks(
        self, tmp_path: Path
    ) -> None:
        """Below two there is nothing to fan out, and the loader would reject it."""
        root = gen.generate(tmp_path)
        cases = _cases_in(root, CATEGORY_ANALYSIS)

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
        cases = _cases_in(root, CATEGORY_ANALYSIS)

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
