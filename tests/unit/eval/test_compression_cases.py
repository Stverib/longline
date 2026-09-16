"""Contract tests for `evals/compression.jsonl` -- the 20-case compression set.

This file is the executable form of `evals/README.md` §5.3. The things a
reviewer would otherwise have to check by hand, and that a silent data edit
would otherwise break:

- 20 cases, each with 8-12 history turns, exactly 5 key facts, and a
  continuation task.
- The five fact **kinds** the plan names are all present, and every case covers
  a fact whose kind has an answer that is not a path (see below).
- Each fact carries a `check` that is a *question whose answer is the fact* --
  not a keyword search over a summary. The `probe` field is the half that makes
  the metric mean "the agent can still use the fact".
- No case's checks already pass on its untouched fixture: a compression case
  whose artifact assertion holds before the agent runs measures nothing.

The fixture-shape tests at the bottom pin the multi-agent toolkit, because the
continuation tasks are written against it and a renamed function would turn
every artifact check into a permanent zero.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from longline.eval.compression import (
    FACT_KINDS,
    CompressionCase,
    load_compression_cases,
)
from longline.eval.judges import _JUDGES, case_passed
from longline.eval.types import resolve_fixture

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_FILE = PROJECT_ROOT / "evals" / "compression.jsonl"
FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"

EXPECTED_TOTAL = 20
EXPECTED_FACTS_PER_CASE = 5
MIN_HISTORY_TURNS = 8
MAX_HISTORY_TURNS = 12

# Fixtures the compression suite may reference. One base tree
# (`multiagent_repo`) plus the repo every other suite already shares.
KNOWN_FIXTURES = ("multiagent_repo", "simple_repo")


def _turn_count(case: CompressionCase) -> int:
    """Conversation turns in the scripted history.

    A turn is a user message plus the assistant reply that answers it, so N
    turns is 2N messages. The history is assistant-first (the list is the
    middle of a session, not its opening), which is why this counts user
    messages rather than dividing the length by two.
    """
    return sum(1 for m in case.history if m["role"] == "user")


@pytest.fixture(scope="module")
def cases() -> list[CompressionCase]:
    return load_compression_cases(CASE_FILE)


class TestFileShape:
    def test_case_file_exists_and_loads(self) -> None:
        assert CASE_FILE.is_file(), f"missing case file: {CASE_FILE}"

    def test_total_is_twenty(self, cases: list[CompressionCase]) -> None:
        assert len(cases) == EXPECTED_TOTAL

    def test_ids_are_unique(self, cases: list[CompressionCase]) -> None:
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_every_case_has_a_continuation_task(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            assert c.continuation_task.strip(), f"{c.id}: empty continuation_task"

    def test_history_length_is_in_range(self, cases: list[CompressionCase]) -> None:
        """8-12 turns (plan §4.3).

        Under 8 the transcript is too short for the two-turn compaction window
        to be meaningful; over 12 the case stops being about a *long* context.
        """
        for c in cases:
            n = _turn_count(c)
            assert MIN_HISTORY_TURNS <= n <= MAX_HISTORY_TURNS, f"{c.id}: {n} turns"

    def test_history_is_assistant_first_and_alternating(
        self, cases: list[CompressionCase]
    ) -> None:
        """The transcript is [assistant, user, assistant, user, ...].

        It does not start with a user message: the list is the *middle* of a
        session, where the opening user turn was already consumed by the first
        assistant reply. `normalize_messages_for_api` would insert a synthetic
        "Begin." message to repair a user-first list, which would put an
        unscripted message in front of a scripted stream.
        """
        for c in cases:
            roles = [m["role"] for m in c.history]
            assert roles[0] == "assistant", f"{c.id}: history starts with {roles[0]}"
            assert all(
                roles[i] != roles[i + 1] for i in range(len(roles) - 1)
            ), f"{c.id}: history roles do not alternate: {roles}"

    def test_every_case_has_exactly_five_facts(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            assert len(c.key_facts) == EXPECTED_FACTS_PER_CASE, (
                f"{c.id}: {len(c.key_facts)} key facts"
            )

    def test_every_case_has_checks(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            assert c.checks, f"{c.id}: no checks (would pass vacuously)"

    def test_checks_mode_is_all_everywhere(self, cases: list[CompressionCase]) -> None:
        """Contract §5.1: 默认全部通过才算成功。"""
        for c in cases:
            assert c.checks_mode == "all", f"{c.id} uses checks_mode={c.checks_mode!r}"

    def test_check_fns_are_known(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            for check in c.checks:
                assert check["fn"] in _JUDGES, f"{c.id}: unknown judge {check['fn']!r}"

    def test_command_checks_use_an_argument_list_and_an_allowlist(
        self, cases: list[CompressionCase]
    ) -> None:
        """Contract §8.4: 命令必须是参数列表, 且只允许声明的受控命令."""
        banned = {"sh", "bash", "zsh", "cmd", "powershell", "pwsh"}
        for c in cases:
            for check in c.checks:
                if check["fn"] not in ("command_ok", "command_output_contains", "python_test"):
                    continue
                argv = check["args"]["command"]
                assert isinstance(argv, list), f"{c.id}: command must be a list"
                allowed = check["args"].get("allowed_commands")
                assert allowed, f"{c.id}: command judge with no allowed_commands"
                assert argv[0] in allowed, f"{c.id}: {argv[0]!r} not in {allowed}"
                assert Path(str(argv[0])).name.lower() not in banned, f"{c.id}: shell command"


class TestFactKinds:
    """The five fact kinds the plan names must all be represented."""

    def test_all_five_kinds_appear(self, cases: list[CompressionCase]) -> None:
        seen = {f.kind for c in cases for f in c.key_facts}
        missing = sorted(set(FACT_KINDS) - seen)
        assert not missing, f"no case carries a fact of kind {missing}"

    def test_kind_is_declared_not_inferred(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            for f in c.key_facts:
                assert f.kind in FACT_KINDS, f"{c.id}: fact {f.id} has kind {f.kind!r}"

    def test_the_five_kinds_per_case_are_distinct(self, cases: list[CompressionCase]) -> None:
        """A case's five facts must be five kinds, not five paths.

        The plan lists five kinds because they fail differently under
        compaction -- a path is salient to a summarizer, a *reason* is not. A
        case carrying five file paths would measure one kind of loss five
        times and quietly drop the other four from the dataset.
        """
        for c in cases:
            kinds = [f.kind for f in c.key_facts]
            assert len(set(kinds)) == len(kinds), f"{c.id}: repeated fact kinds {kinds}"

    def test_kinds_are_covered_more_than_once_across_the_set(
        self, cases: list[CompressionCase]
    ) -> None:
        counts = dict.fromkeys(FACT_KINDS, 0)
        for c in cases:
            for f in c.key_facts:
                counts[f.kind] += 1
        assert set(counts.values()) == {EXPECTED_TOTAL}, counts

    def test_at_least_one_retrieval_kind_fact_per_case(
        self, cases: list[CompressionCase]
    ) -> None:
        """Every case carries one fact answerable from the repo, not from history.

        This is the control against a label-leak reading of the metric. If all
        five facts lived only in the transcript, an agent could score 5/5 by
        recovering a summary of the conversation, and the number would say
        nothing about whether the *workspace* was still usable after
        compaction. One fact per case is planted where the agent can only reach
        it by continuing to work.
        """
        retrieval = ("file-path", "symbol-name")
        for c in cases:
            kinds = {f.kind for f in c.key_facts}
            assert kinds & set(retrieval), f"{c.id}: no file-path/symbol-name fact"


class TestJudgingIsByFollowUpNotByKeyword:
    """A fact is retained only if it is USED, and each fact says how.

    The metric contract is explicit (§5.3): 关键信息保留率通过压缩后的追问结果和
    最终产物共同判定; 不能只在 summary 文本里搜索关键词. The mechanical form of
    that rule is the `probe` field -- a question whose answer IS the fact -- and
    a `check` that scores the answer. Keyword-searching the summary cannot
    express "the agent can still use this", so a fact without a probe is
    untestable and is rejected here.
    """

    def test_every_fact_carries_a_probe_question(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            for f in c.key_facts:
                assert f.probe.strip(), f"{c.id}: fact {f.id} has no probe question"

    def test_every_fact_probe_is_a_question(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            for f in c.key_facts:
                # Both ASCII '?' and its fullwidth form, written as escapes so
                # the source itself carries no ambiguous punctuation (RUF001).
                assert f.probe.rstrip().endswith(("?", chr(0xFF1F))), (
                    f"{c.id}: fact {f.id} probe is not phrased as a question: {f.probe!r}"
                )

    def test_every_fact_declares_what_it_is_scored_against(
        self, cases: list[CompressionCase]
    ) -> None:
        """Every fact declares the value its check pins, or the trap that beats it.

        Two shapes, and a fact must be one of them:

        - `answer` set: the case author knows the expected value, so the check
          is a positive assertion on it. `trap` may ALSO be set -- the most
          confusable wrong value -- and is what a mutation test flips to.
        - `answer` empty, `trap` set: a fact whose value is only readable out of
          the repo (a token planted in the fixture), where the author knows the
          wrong answer but not the right one before the run.

        A fact with neither declares nothing, so its check asserts a value
        nobody has stated is the right one; that is unverifiable and rejected.
        """
        for c in cases:
            for f in c.key_facts:
                assert f.answer.strip() or f.trap.strip(), (
                    f"{c.id}: fact {f.id} declares neither an answer nor a trap, "
                    "so its check pins a value nothing names"
                )

    def test_a_trap_never_equals_the_answer(self, cases: list[CompressionCase]) -> None:
        """A trap that IS the answer is not a trap, and its mutation is a no-op."""
        for c in cases:
            for f in c.key_facts:
                if f.trap and f.answer:
                    assert f.trap != f.answer, (
                        f"{c.id}: fact {f.id} has trap == answer ({f.answer!r}); "
                        "mutating to the trap would change nothing"
                    )

    def test_answers_and_traps_are_distinct_across_a_case(
        self, cases: list[CompressionCase]
    ) -> None:
        """No two facts in a case are scored against the same string.

        A collapsed pair would make "which fact was lost" unanswerable: one
        wrong line would fail two facts, or one right line would pass two.
        """
        for c in cases:
            values = [f.answer or f.trap for f in c.key_facts]
            assert len(set(values)) == len(values), f"{c.id}: repeated fact values {values}"

    def test_every_fact_carries_a_check(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            for f in c.key_facts:
                check = f.check
                assert check, f"{c.id}: fact {f.id} has no check"
                assert check.get("fn") in _JUDGES, f"{c.id}: fact {f.id} unknown judge"

    def test_no_fact_check_is_a_summary_keyword_scan(
        self, cases: list[CompressionCase]
    ) -> None:
        """A fact's check must read the ANSWER, never the transcript.

        `file_content` against a file the agent writes is the shape that makes
        this hold: the agent must produce the answer, and the answer file has
        no relation to what the summary said. The banned shape is a check whose
        `path` points at a transcript/summary artifact -- there is no such
        artifact in the sandbox, so a check written that way fails loudly here
        rather than by scanning the wrong thing and reporting a comfortable
        number.
        """
        for c in cases:
            for f in c.key_facts:
                args = f.check.get("args", {})
                path = str(args.get("path", ""))
                assert "summary" not in path and "transcript" not in path, (
                    f"{c.id}: fact {f.id} checks {path!r} -- a summary keyword "
                    "scan is not a retention measurement"
                )


class TestFixtureContainment:
    def test_referenced_fixtures_exist(self, cases: list[CompressionCase]) -> None:
        for c in cases:
            if c.fixture:
                assert (FIXTURES_DIR / c.fixture).is_dir(), f"{c.id}: missing fixture {c.fixture}"

    def test_known_fixture_vocabulary(self, cases: list[CompressionCase]) -> None:
        used = {c.fixture for c in cases if c.fixture}
        assert used <= set(KNOWN_FIXTURES), f"unknown fixtures: {used - set(KNOWN_FIXTURES)}"

    def test_every_case_sets_a_fixture(self, cases: list[CompressionCase]) -> None:
        """Every continuation task asserts a final artifact, so a sandbox is required."""
        for c in cases:
            assert c.fixture, f"{c.id}: no fixture -- the continuation task writes files"

    def test_fixture_names_cannot_escape(self) -> None:
        with pytest.raises(Exception, match="escapes the fixtures root"):
            resolve_fixture(FIXTURES_DIR, "../secrets", case_id="cc-x")


class TestFixturesAreNotPreSatisfied:
    """The artifact checks must NOT already hold before the agent runs."""

    def test_no_case_passes_on_its_untouched_fixture(
        self, cases: list[CompressionCase], tmp_path: Path
    ) -> None:
        offenders: dict[str, list[str]] = {}
        for c in cases:
            sandbox = tmp_path / c.id
            sandbox.mkdir(parents=True)
            shutil.copytree(FIXTURES_DIR / c.fixture, sandbox, dirs_exist_ok=True)
            passed, detail = case_passed(c.checks, sandbox, mode=c.checks_mode)
            if passed:
                offenders[c.id] = [str(d["fn"]) for d in detail]
        assert not offenders, (
            f"these cases already pass before the agent runs: {offenders}"
        )


class TestMultiAgentFixtureShape:
    """The toolkit the continuation tasks are written against.

    These are structural assertions, not a byte freeze. A renamed function or a
    dropped module would turn every artifact check that imports it into a
    permanent zero, and a permanent zero looks like a model failure rather than
    a data bug.
    """

    def test_package_files_exist(self) -> None:
        root = FIXTURES_DIR / "multiagent_repo"
        for rel in (
            "README.md",
            "docs/pipeline_notes.md",
            "multiagent/__init__.py",
            "multiagent/coordinator.py",
            "multiagent/fanout.py",
            "multiagent/prompts.py",
            "tests/test_coordinator.py",
        ):
            assert (root / rel).is_file(), f"multiagent_repo is missing {rel}"

    def test_coordinator_and_fanout_symbols_exist(self) -> None:
        root = FIXTURES_DIR / "multiagent_repo"
        fanout = (root / "multiagent" / "fanout.py").read_text(encoding="utf-8")
        coordinator = (root / "multiagent" / "coordinator.py").read_text(encoding="utf-8")
        assert "def chunk_tasks(" in fanout
        assert "def merge_results(" in fanout
        assert "def run_single(" in fanout
        assert "class Coordinator" in coordinator
        assert "def plan(" in coordinator

    def test_coordinator_test_suite_passes_on_the_fixture(self, tmp_path: Path) -> None:
        """The frozen bug must be real: the fixture's own test has to be red."""
        root = tmp_path / "s"
        shutil.copytree(FIXTURES_DIR / "multiagent_repo", root)
        passed, detail = case_passed(
            [
                {
                    "fn": "command_output_contains",
                    "args": {
                        "command": ["python", "-m", "pytest", "tests", "-q"],
                        "contains": "1 passed",
                        "allowed_commands": ["python"],
                    },
                }
            ],
            root,
        )
        assert passed is False, f"the frozen bug is already fixed: {detail}"
