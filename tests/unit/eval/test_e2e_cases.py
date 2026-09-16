"""Contract tests for `evals/e2e.jsonl` — the 40-case E2E benchmark.

This file is the executable form of `evals/README.md` §5.1 and §8. The things a
reviewer would otherwise have to check by hand, and that a silent data edit
would otherwise break:

- 40 cases, exactly 8 in each of the 5 categories (plan §4.1).
- Every judge named in the file exists, and every `checks` entry is well-formed.
- Every referenced fixture exists and stays inside `evals/fixtures/`.
- **No judge is trivially true.** Every case's own checks are run against a
  fresh copy of its fixture *before* the agent would run: a case whose checks
  already pass on the starting state measures nothing.
- **Every judge has a demonstrated failing variant.** `MUTATIONS` below pairs a
  correct artifact with a deliberately broken one for each judge; the test
  asserts the judge accepts the first and rejects the second. A judge with no
  failing variant is unverified, so this is what makes "judges can fail" a fact
  rather than a claim.

The mutation corpus is deliberately per-*judge* rather than per-case: what needs
proving is that the judge function rejects a broken artifact, and one broken
artifact is enough for a given judge. The per-case starting-state check is the
other half, and it is what catches a case that picked the wrong assertion.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest

from longline.eval.judges import case_passed, judge_case
from longline.eval.types import (
    E2E_CATEGORY_TAGS,
    CaseParseError,
    E2ECase,
    load_cases,
    resolve_fixture,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_FILE = PROJECT_ROOT / "evals" / "e2e.jsonl"
FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"

EXPECTED_TOTAL = 40
EXPECTED_PER_CATEGORY = 8

# Fixtures the E2E suite is allowed to reference — the base-fixture vocabulary,
# not the full contents of `evals/fixtures/`. `notebook_repo` and
# `workspace_repo` live in that directory for the tool-selection suite; listing
# only what E2E actually uses keeps this a check on *this* dataset rather than a
# mirror of the directory.
#
# `simple_repo` is deliberately shared with the legacy tool-call suite and its
# file list is frozen (see tests/unit/eval/test_tool_selection_cases.py).
KNOWN_FIXTURES = (
    "buggy_repo",
    "longchain_repo",
    "longchain_repo_buggy",
    "project_repo",
    "project_repo_broken",
    "project_repo_buggy",
    "retrieval_repo",
    "simple_repo",
    "tool_repo",
    "workspace_repo",
)


@pytest.fixture(scope="module")
def cases() -> list[E2ECase]:
    loaded = load_cases(CASE_FILE)
    assert all(isinstance(c, E2ECase) for c in loaded)
    return loaded  # type: ignore[return-value]


def _sandbox_from_fixture(tmp_path: Path, fixture: str | None) -> Path:
    """Materialise a case's starting state exactly as the runner does."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True)
    if fixture:
        shutil.copytree(FIXTURES_DIR / fixture, sandbox, dirs_exist_ok=True)
    return sandbox


class TestFileShape:
    def test_case_file_exists_and_loads(self) -> None:
        assert CASE_FILE.is_file(), f"missing case file: {CASE_FILE}"

    def test_ids_are_unique(self, cases: list[E2ECase]) -> None:
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_total_is_forty(self, cases: list[E2ECase]) -> None:
        assert len(cases) == EXPECTED_TOTAL

    def test_exactly_eight_per_category(self, cases: list[E2ECase]) -> None:
        """每个类别严格 8 条(计划 §4.1 的验收条件).

        用 Counter 而不是逐个 assert,是为了让失败信息直接显示「哪一类少了几条」,
        而不是只说 `8 != 7`。
        """
        counts = dict.fromkeys(E2E_CATEGORY_TAGS, 0)
        for c in cases:
            found = [t for t in c.tags if t in E2E_CATEGORY_TAGS]
            assert len(found) == 1, f"{c.id}: expected exactly one category tag, got {found}"
            counts[found[0]] += 1
        assert counts == dict.fromkeys(E2E_CATEGORY_TAGS, EXPECTED_PER_CATEGORY)

    def test_categories_sum_to_the_whole_set(self, cases: list[E2ECase]) -> None:
        # 每个类别一次、每条用例一次,分类不能有遗漏也不能重复计数.
        tagged = sum(1 for c in cases for t in c.tags if t in E2E_CATEGORY_TAGS)
        assert tagged == EXPECTED_TOTAL

    def test_every_case_has_at_least_one_check(self, cases: list[E2ECase]) -> None:
        for c in cases:
            assert c.checks, f"{c.id} has no checks (would pass vacuously)"
            assert c.num_checks >= 1

    def test_checks_mode_is_all_everywhere(self, cases: list[E2ECase]) -> None:
        """契约 §5.1:默认全部通过才算成功.

        `any` 模式在本数据集中一条都没用到,这是有意的 —— 它只留给真正互斥的成功
        形态。用到了就必须在这里显式说明理由,否则不该出现。
        """
        for c in cases:
            assert c.checks_mode == "all", f"{c.id} uses checks_mode={c.checks_mode!r}"

    def test_check_fns_are_known(self, cases: list[E2ECase]) -> None:
        from longline.eval.judges import _JUDGES

        for c in cases:
            for check in c.checks:
                assert check["fn"] in _JUDGES, f"{c.id}: unknown judge {check['fn']!r}"

    def test_all_checks_are_well_formed(self, cases: list[E2ECase]) -> None:
        for c in cases:
            for check in c.checks:
                assert isinstance(check.get("fn"), str), f"{c.id}: check without a fn"
                args = check.get("args", {})
                assert isinstance(args, dict), f"{c.id}: check args must be a dict, got {args!r}"


class TestJudgeArgsAreComplete:
    """Each judge's required args, checked against the data file.

    A judge called without its required key raises at run time, and `case_passed`
    records that as a failed case — so a missing arg would silently turn into a
    permanent zero rather than a loud data error. These assertions move that
    failure to load time.
    """

    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "file_content": ("path",),
        "file_exists": ("path",),
        "json_value": ("path", "key_path", "equals"),
        "line_set_equals": ("path", "equals"),
        "directory_snapshot": ("path",),
        "command_ok": ("command", "allowed_commands"),
        "command_output_contains": ("command", "contains", "allowed_commands"),
        "python_test": ("command", "allowed_commands"),
    }

    def test_required_args_present(self, cases: list[E2ECase]) -> None:
        for c in cases:
            for check in c.checks:
                for key in self.REQUIRED[check["fn"]]:
                    assert key in check.get("args", {}), (
                        f"{c.id}: {check['fn']} is missing required arg {key!r}"
                    )

    def test_command_judges_declare_an_allowlist(self, cases: list[E2ECase]) -> None:
        """契约 §8.4:只允许评测文件中声明的受控命令."""
        for c in cases:
            for check in c.checks:
                if check["fn"] in ("command_ok", "command_output_contains", "python_test"):
                    allowed = check["args"].get("allowed_commands")
                    assert allowed, f"{c.id}: command judge with no allowed_commands"
                    assert all(isinstance(a, str) for a in allowed)

    def test_command_judges_use_an_argument_list(self, cases: list[E2ECase]) -> None:
        """契约 §8.4:命令必须是参数列表,不能是任意 shell 字符串.

        字符串形式走 shlex 拆分,但列表形式才是数据文件该有的写法 —— 它不可能
        被读成「交给 shell 一行字」。
        """
        for c in cases:
            for check in c.checks:
                if check["fn"] in ("command_ok", "command_output_contains", "python_test"):
                    command = check["args"]["command"]
                    assert isinstance(command, list), f"{c.id}: command must be a list"
                    assert all(isinstance(a, str) for a in command)

    def test_command_programs_are_in_their_own_allowlist(self, cases: list[E2ECase]) -> None:
        for c in cases:
            for check in c.checks:
                if check["fn"] in ("command_ok", "command_output_contains", "python_test"):
                    argv0 = check["args"]["command"][0]
                    allowed = check["args"]["allowed_commands"]
                    assert argv0 in allowed, (
                        f"{c.id}: command program {argv0!r} is not in allowed_commands {allowed}"
                    )

    def test_no_command_uses_shell_metacharacters_as_syntax(self, cases: list[E2ECase]) -> None:
        # 参数里出现这些字符本身不违规(它们只是数据),但作为**独立元素**出现
        # 说明作者还在按 shell 语法思考,应该改成真正的 argv。
        for c in cases:
            for check in c.checks:
                if check["fn"] not in ("command_ok", "command_output_contains", "python_test"):
                    continue
                argv = check["args"]["command"]
                for token in argv:
                    assert token not in {"&&", "||", ";", "|", ">", ">>", "<"}, (
                        f"{c.id}: {token!r} is shell syntax, not an argument"
                    )


class TestFixtureContainment:
    def test_referenced_fixtures_exist(self, cases: list[E2ECase]) -> None:
        for c in cases:
            if c.fixture:
                assert (FIXTURES_DIR / c.fixture).is_dir(), f"{c.id}: missing fixture {c.fixture}"

    def test_known_fixture_vocabulary(self, cases: list[E2ECase]) -> None:
        """One base fixture per job; a new directory has to be added here on purpose.

        The 40 cases are built from a handful of shared bases rather than 40
        private trees, so the fixture set stays reviewable. Pinning the list
        means an accidental `"fixture": "tool_rep"` typo fails here rather than
        at run time.
        """
        used = {c.fixture for c in cases if c.fixture}
        assert used <= set(KNOWN_FIXTURES), f"unknown fixtures referenced: {used - set(KNOWN_FIXTURES)}"

    def test_parent_traversal_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(FIXTURES_DIR, "../secrets", case_id="e2e-x")

    def test_absolute_path_is_rejected(self) -> None:
        outside = PROJECT_ROOT / "pyproject.toml"
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(FIXTURES_DIR, str(outside), case_id="e2e-x")

    def test_deep_traversal_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(FIXTURES_DIR, "simple_repo/../../pyproject.toml", case_id="e2e-x")

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        """`resolve()` 会解开符号链接,所以「先 resolve 再比前缀」能挡住链接逃逸."""
        root = tmp_path / "fixtures"
        root.mkdir()
        (root / "real").mkdir()
        link = root / "escape"
        try:
            link.symlink_to(tmp_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available on this platform")
        with pytest.raises(CaseParseError, match="escapes the fixtures root"):
            resolve_fixture(root, "escape", case_id="e2e-x")

    def test_dot_and_empty_names_are_rejected(self) -> None:
        for name in ("", ".", ".."):
            with pytest.raises(CaseParseError):
                resolve_fixture(FIXTURES_DIR, name, case_id="e2e-x")

    def test_legitimate_name_resolves(self) -> None:
        assert resolve_fixture(FIXTURES_DIR, "simple_repo", case_id="x") == FIXTURES_DIR / "simple_repo"


class TestFixturesAreNotPreSatisfied:
    """The starting state must NOT already satisfy the case's own judges.

    This is the "judge 不得恒真" half of contract §8.2 that can be checked
    mechanically: run every case's checks against a fresh copy of its fixture
    (i.e. what the agent starts from) and require at least one to fail. If they
    all pass, the case is measuring nothing — it would report success for an
    agent that did nothing at all.
    """

    def test_no_case_passes_on_its_untouched_fixture(
        self, cases: list[E2ECase], tmp_path: Path
    ) -> None:
        offenders: dict[str, list[str]] = {}
        for c in cases:
            sandbox = _sandbox_from_fixture(tmp_path / c.id, c.fixture)
            passed, detail = case_passed(c.checks, sandbox, mode=c.checks_mode)
            if passed:
                offenders[c.id] = [d["fn"] for d in detail]
        assert not offenders, (
            "these cases already pass before the agent runs, so they measure "
            f"nothing: {offenders}"
        )


# --- mutation corpus -------------------------------------------------------
#
# One entry per judge: `(args, correct_builder, broken_builder)`. The correct
# builder produces an artifact the judge must accept; the broken builder makes
# the smallest change that still violates the assertion, which the judge must
# reject. Both write into a fresh tmp dir.
#
# This is the plan's Task 3 acceptance criterion ("所有 judges 对正确 fixture
# 通过、对至少一个错误变体失败") as a test rather than a one-off script. A judge
# that accepts its mutation is unverified, so the test fails and names it.

def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# judge name -> (args, correct, broken)
Mutation = tuple[dict[str, Any], Any, Any]

MUTATIONS: dict[str, Mutation] = {
    "file_exists": (
        {"path": "out.txt"},
        lambda r: _write(r, "out.txt", "done\n"),
        lambda r: None,
    ),
    "file_content": (
        {"path": "README.md", "contains": "Project: zero"},
        lambda r: _write(r, "README.md", "# repo\n\nProject: zero\n"),
        # Broken: the line the task asked for is simply missing.
        lambda r: _write(r, "README.md", "# repo\n"),
    ),
    "json_value": (
        {"path": "config.json", "key_path": ["port"], "equals": 8000},
        lambda r: _write(r, "config.json", '{"port": 8000}\n'),
        # Broken: right-looking, but a string where an int was required.
        lambda r: _write(r, "config.json", '{"port": "8000"}\n'),
    ),
    "line_set_equals": (
        {"path": "files.txt", "equals": ["a.txt", "b.txt"]},
        lambda r: _write(r, "files.txt", "a.txt\nb.txt\n"),
        # Broken: one line is correct, the other is a plausible near-miss.
        lambda r: _write(r, "files.txt", "a.txt\nmain.py\n"),
    ),
    "directory_snapshot": (
        {"path": ".", "equals": ["index.md", "README.md"]},
        lambda r: (_write(r, "index.md", "# Index\n"), _write(r, "README.md", "# r\n")),
        # Broken: index.md was never created.
        lambda r: _write(r, "README.md", "# r\n"),
    ),
    "command_ok": (
        {
            "command": ["python", "-c", "import target; target.check()"],
            "allowed_commands": ["python"],
        },
        lambda r: _write(r, "target.py", "def check():\n    return True\n"),
        # Broken: the module the command imports does not define what it calls,
        # so the command exits non-zero. (A no-op "broken" builder would leave
        # the correct artifact in place and the judge would rightly pass it.)
        lambda r: _write(r, "target.py", "def something_else():\n    return True\n"),
    ),
    "command_output_contains": (
        {
            "command": ["python", "-c", "from multi import mul; print(mul(2, 3))"],
            "contains": "6",
            "allowed_commands": ["python"],
        },
        lambda r: _write(r, "multi.py", "def mul(a, b):\n    return a * b\n"),
        lambda r: _write(r, "multi.py", "def mul(a, b):\n    return a + b\n"),
    ),
    "python_test": (
        {
            "command": ["python", "-m", "pytest", "test_target.py", "-q"],
            "allowed_commands": ["python"],
            "path": "test_target.py",
        },
        lambda r: (
            _write(r, "target.py", "def add(a, b):\n    return a + b\n"),
            _write(r, "test_target.py", "def test_add():\n    from target import add\n\n    assert add(1, 2) == 3\n"),
        ),
        lambda r: (
            _write(r, "target.py", "def add(a, b):\n    return a - b\n"),
            _write(r, "test_target.py", "def test_add():\n    from target import add\n\n    assert add(1, 2) == 3\n"),
        ),
    ),
}


# A wrong-but-runnable variant for the correct builder of `python_test` above;
# `command_ok`'s mutation is "the fix did not land", modelled by an exit code.
def _command_ok_broken_command() -> dict[str, Any]:
    return {"command": ["python", "-c", "raise SystemExit(3)"], "allowed_commands": ["python"]}


class TestJudgeMutations:
    """Every judge must pass its correct artifact and fail its broken one."""

    @pytest.mark.parametrize("judge_name", sorted(MUTATIONS))
    def test_correct_artifact_passes(self, judge_name: str, tmp_path: Path) -> None:
        args, correct, _ = MUTATIONS[judge_name]
        root = tmp_path / "correct"
        root.mkdir()
        correct(root)
        assert judge_case(judge_name, root, args) is True

    @pytest.mark.parametrize("judge_name", sorted(MUTATIONS))
    def test_broken_artifact_fails(self, judge_name: str, tmp_path: Path) -> None:
        args, _, broken = MUTATIONS[judge_name]
        root = tmp_path / "broken"
        root.mkdir()
        broken(root)
        assert judge_case(judge_name, root, args) is False, (
            f"{judge_name} accepted its deliberately broken variant — the judge "
            "cannot fail, so it verifies nothing"
        )

    def test_command_ok_fails_on_a_nonzero_exit(self, tmp_path: Path) -> None:
        """`command_ok` 的「正确产物」是退出码 0,所以它的错误变体是退出码非 0.

        这一条单独写,是因为它的 mutation 不是「文件内容坏掉」而是命令本身失败,
        `MUTATIONS` 里那个 no-op broken(留下正确产物)表达不了它。
        """
        root = tmp_path / "s"
        root.mkdir()
        assert judge_case(
            "command_ok", root,
            {"command": ["python", "-c", "pass"], "allowed_commands": ["python"]},
        ) is True
        assert judge_case("command_ok", root, _command_ok_broken_command()) is False

    def test_every_registered_judge_has_a_mutation(self) -> None:
        """新的 judge 必须同时补上 mutation,否则「能失败」没有被证明过."""
        from longline.eval.judges import _JUDGES

        missing = sorted(set(_JUDGES) - set(MUTATIONS))
        assert not missing, f"these judges have no demonstrated failing variant: {missing}"


class TestEveryCaseJudgeIsCoveredByAMutation:
    """No case may use a judge the mutation corpus does not vouch for."""

    def test_all_case_judges_are_in_the_mutation_corpus(self, cases: list[E2ECase]) -> None:
        used = {check["fn"] for c in cases for check in c.checks}
        uncovered = sorted(used - set(MUTATIONS))
        assert not uncovered, f"cases use judges with no mutation proof: {uncovered}"

    def test_judge_usage_matches_the_contract(self, cases: list[E2ECase]) -> None:
        """契约 §5.1 点名的四个新 judge 必须真的被用上.

        数据文件里没有它们,新增判分器就只是代码,不是指标的一部分。
        """
        used = {check["fn"] for c in cases for check in c.checks}
        for required in ("json_value", "line_set_equals", "python_test", "directory_snapshot"):
            assert required in used, f"contract §5.1 names {required!r}, but no case uses it"

    def test_category_judge_families(self, cases: list[E2ECase]) -> None:
        """每一类用的判分手段要和计划 §4.1 的表对得上."""
        by_cat: dict[str, set[str]] = {t: set() for t in E2E_CATEGORY_TAGS}
        for c in cases:
            cat = next(t for t in c.tags if t in E2E_CATEGORY_TAGS)
            by_cat[cat] |= {check["fn"] for check in c.checks}

        # 文件操作:文件存在、内容、目录状态
        assert {"file_content", "file_exists", "directory_snapshot"} <= by_cat["file-ops"]
        # 代码任务:测试命令、函数输出、静态内容
        assert "python_test" in by_cat["code"]
        assert "file_content" in by_cat["code"]
        # 检索任务:答案落在产物文件里
        assert "file_content" in by_cat["retrieval"]
        # 多工具:最终产物 + 关键中间状态
        assert {"file_content", "line_set_equals"} <= by_cat["multi-tool"]
        # 长链路:最终产物
        assert "file_content" in by_cat["long-chain"]


class TestLongChainCasesDoNotScoreToolCalls:
    """契约 §5.1 红线:长链路用例的 ≥5 次工具调用只作诊断,不代替结果判分.

    E2E 的 `checks` 里没有、也不该有任何「调用了某个工具」的判分器 ——
    调用次数只有在报告层作为诊断指标出现。
    """

    def test_long_chain_checks_are_result_only(self, cases: list[E2ECase]) -> None:
        result_judges = {
            "file_content", "file_exists", "json_value",
            "line_set_equals", "directory_snapshot", "python_test",
            "command_ok", "command_output_contains",
        }
        for c in cases:
            if "long-chain" not in c.tags:
                continue
            for check in c.checks:
                assert check["fn"] in result_judges, (
                    f"{c.id}: {check['fn']!r} is not a final-state judge"
                )

    def test_long_chain_cases_assert_more_than_the_artifact(self, cases: list[E2ECase]) -> None:
        """长链路用例必须同时钉住最终产物**和**关键中间状态(计划 §4.1 表).

        只断言最终产物时,一个把答案抄进 report.md 而没有真正处理数据的 Agent
        也能通过。用 `line_set_equals` / `directory_snapshot` / 第二个文件的
        内容断言来固定中间状态。
        """
        for c in cases:
            if "long-chain" not in c.tags:
                continue
            fns = [check["fn"] for check in c.checks]
            assert len(fns) >= 2, f"{c.id}: long-chain case with a single check"
            assert len(set(fns)) >= 2 or fns.count("file_content") >= 2, (
                f"{c.id}: long-chain case checks only one kind of assertion"
            )


class TestSandboxIsolation:
    def test_no_case_writes_absolute_paths_outside_its_sandbox(self, cases: list[E2ECase]) -> None:
        """判分参数里的 path 必须是相对路径 —— 绝对路径会绕过沙箱副本."""
        for c in cases:
            for check in c.checks:
                path = check["args"].get("path")
                if path is None:
                    continue
                assert not Path(str(path)).is_absolute(), f"{c.id}: absolute judge path {path!r}"
                assert ".." not in Path(str(path)).parts, f"{c.id}: judge path escapes sandbox"

    def test_max_turns_is_sane(self, cases: list[E2ECase]) -> None:
        for c in cases:
            assert 1 <= c.max_turns <= 20, f"{c.id} max_turns={c.max_turns}"

    def test_long_chain_cases_get_more_turns(self, cases: list[E2ECase]) -> None:
        # max_turns 要够完成任务,又不会高到白烧额度.
        for c in cases:
            if "long-chain" in c.tags:
                assert c.max_turns >= 10, f"{c.id}: long-chain case with only {c.max_turns} turns"


class TestFixtureShape:
    """The base fixtures the 40 cases are built from.

    These are structural assertions, not a freeze on every byte: the point is
    that each fixture actually contains the things its cases need, so a case
    asserting on a file cannot pass vacuously against an empty directory.
    """

    def test_simple_repo_is_untouched(self) -> None:
        """`simple_repo` 由已退役的 `tool_calls.jsonl` 与旧测试共用,文件列表冻结."""
        fixture = FIXTURES_DIR / "simple_repo"
        on_disk = sorted(p.relative_to(fixture).as_posix() for p in fixture.rglob("*") if p.is_file())
        assert on_disk == [
            "README.md",
            "data/notes.txt",
            "last.txt",
            "src/config.py",
            "src/main.py",
            "src/version.py",
        ]

    def test_project_repo_has_a_working_test_suite(self) -> None:
        assert (FIXTURES_DIR / "project_repo" / "pytest.ini").is_file()
        assert (FIXTURES_DIR / "project_repo" / "src" / "projectkit" / "stats.py").is_file()

    def test_project_repo_buggy_has_the_median_bug(self) -> None:
        text = (FIXTURES_DIR / "project_repo_buggy" / "src" / "projectkit" / "stats.py").read_text(
            encoding="utf-8"
        )
        assert "BUG" in text

    def test_project_repo_broken_does_not_import(self) -> None:
        source = (FIXTURES_DIR / "project_repo_broken" / "src" / "projectkit" / "report.py").read_text(
            encoding="utf-8"
        )
        with pytest.raises(SyntaxError):
            compile(source, "report.py", "exec")

    def test_longchain_repo_defines_done(self) -> None:
        text = (FIXTURES_DIR / "longchain_repo" / "docs" / "done_definition.md").read_text(
            encoding="utf-8"
        )
        assert "ROLLUP_VERSION" in text
        assert "| stage | total |" in text

    def test_retrieval_answers_are_not_guessable_from_the_readme(self) -> None:
        """检索用例的答案必须只在语料里,不能写在 README 里被一眼看到."""
        readme = (FIXTURES_DIR / "retrieval_repo" / "README.md").read_text(encoding="utf-8")
        for answer in ("3170", "us-east", "Option B", "45", "RET-4021"):
            assert answer not in readme, f"{answer!r} leaks the answer into the README"


class TestCommandJudgingResidualRisk:
    """What the command allowlist does NOT protect against.

    These tests pin the *limits* of the control, so nobody reads the allowlist
    as a sandbox. A `python -c "..."` command is still arbitrary code running
    as the current user; the allowlist only decides whether the eval file is
    allowed to ask for it. The real defence is that `evals/*.jsonl` is a
    reviewed, version-controlled artifact — not the allowlist.
    """

    def test_an_allowed_program_can_still_run_arbitrary_code(self, tmp_path: Path) -> None:
        marker = tmp_path / "side_effect"
        judge_case(
            "command_ok", tmp_path,
            {
                "command": ["python", "-c", f"open({str(marker)!r}, 'w').write('x')"],
                "allowed_commands": ["python"],
            },
        )
        assert marker.exists(), (
            "the allowlist is not a sandbox; if this ever stops writing, the "
            "threat model in the docstring needs rewriting, not just this test"
        )

    def test_the_allowlist_blocks_the_program_not_the_arguments(self, tmp_path: Path) -> None:
        # 这条用例是为了把「控制的是什么」写死在测试里:被拒的是程序名.
        with pytest.raises(ValueError, match="not in the case's declared"):
            judge_case(
                "command_ok", tmp_path,
                {"command": ["sh", "-c", "true"], "allowed_commands": ["python"]},
            )

    def test_no_case_in_the_dataset_declares_a_general_purpose_shell(self, cases: list[E2ECase]) -> None:
        """数据集本身不应出现 shell 解释器作为受控命令."""
        banned = {"sh", "bash", "zsh", "cmd", "powershell", "pwsh"}
        for c in cases:
            for check in c.checks:
                allowed = check["args"].get("allowed_commands") or []
                bad = {a for a in allowed if Path(str(a)).name.lower() in banned}
                assert not bad, f"{c.id} declares a shell as an allowed command: {bad}"
