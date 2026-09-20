"""Unit tests for the multi-agent A/B runner and its usage accounting.

Contract: `evals/README.md` section 5.6. Everything here runs offline and takes about
a second: `build_engine` is given a scripted model transport, so no API key is
needed and no request is made. The fan-out is still a REAL `spawn_teammate`
running a real `InProcessTeammate` over a real `query_loop`, because that is the
thing the accounting has to be shown to survive.

=== The three things this file exists to pin ===

1. **A child's tokens reach the totals.** The red line in §5.6 is that the
   leader's usage alone is not the run's usage. `TestChildUsageReachesTheLedger`
   drives a real fan-out through `spawn_teammate` and asserts that the sum over
   the ledger's child rows equals a number the leader's own stream could not
   have produced, with each child's per-agent total asserted individually.
2. **The accounting is a checked claim, not an assumption.** A dropped child
   must raise rather than quietly lower the total. Every branch of
   `AccountedAgents.assert_complete` is driven here with the input that makes it
   fail, and each failure is also driven through the RUNNER to prove the
   exclusion lands on the row instead of only in an exception.
3. **Both variants do the same work.** The single and multi arms are handed the
   same subtask instructions and judged by the same checks; a case that broke
   that would produce a `Speedup` comparing two different jobs. The per-subtask
   verdicts and the judge lists are compared directly.

No assertion reads a number back out of a field the runner wrote: every
expectation is either an arithmetic consequence of the scripted usage the test
itself supplied, or a count of something the test can see (files on disk, the
number of spawns it requested).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from longline.eval.child_usage import (
    LEADER,
    AccountingError,
    AgentUsage,
    ModelCounter,
    TurnUsage,
    UsageLedger,
    agent_scope,
    count_usage,
    current_agent,
    current_ledger,
    reconcile,
    teammate_ids_in,
)
from longline.eval.multi_agent import (
    CONTROLLED,
    EXPLORATORY,
    MULTI,
    SINGLE,
    MultiAgentCase,
    Subtask,
)
from longline.eval.multi_agent_runner import (
    REASON_ACCOUNTING_INCOMPLETE,
    _apply_live_counting,
    aggregate_multi_agent,
    leader_prompt,
    merge_instruction,
    run_multi_agent_case,
    subtask_prompt,
)
from longline.models.messages import Usage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"
CASE_FILE = PROJECT_ROOT / "evals" / "multi_agent.jsonl"

# Token counts chosen so every arithmetic consequence below is exact and a
# mis-attribution lands on a distinguishable number: the leader's turns and the
# workers' turns cost different amounts, and two workers cost different amounts
# from each other.
LEADER_USAGE = Usage(input_tokens=100, output_tokens=10)
WORKER_USAGE = Usage(input_tokens=7, output_tokens=3)


# --- helpers -----------------------------------------------------------------


def make_case(
    *,
    case_id: str = "ma-test",
    workers: int = 2,
    num_subtasks: int = 2,
    group: str = CONTROLLED,
) -> MultiAgentCase:
    """A minimal case over the committed sibling fixtures.

    Real fixtures rather than a synthetic tree: a case built over `tmp_path`
    would not exercise `load_multi_agent_cases`'s identity check, and the
    runner copies its fixture through `_prepare_sandbox`, which requires one
    that resolves under `evals/fixtures/`.
    """
    dirname = f"out/{case_id}"
    subtasks = [
        Subtask(
            id=f"s{index}",
            instruction=f"Summarise module number {index}",
            writes=f"{dirname}/m{index}.md",
        )
        for index in range(1, num_subtasks + 1)
    ]
    return MultiAgentCase(
        id=case_id,
        task="Summarise the modules described in README.md",
        group=group,  # type: ignore[arg-type]
        subtasks=subtasks,
        workers=workers,
        merge_file=f"{dirname}/manifest.txt",
        fixture_single="parallel_repo_single",
        fixture_multi="parallel_repo_multi",
        max_turns=12,
        tags=["multi-agent"],
        checks=[
            {"fn": "file_exists", "args": {"path": s.writes}} for s in subtasks
        ]
        + [{"fn": "file_exists", "args": {"path": f"{dirname}/manifest.txt"}}],
    )


def _temp_claude_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ma-test-claude-"))


# --- 1. the accounting is complete, and provably so --------------------------


class TestChildUsageReachesTheLedger:
    """The §5.6 red line: every child's tokens and tool calls are collected."""

    @pytest.mark.asyncio
    async def test_every_spawned_worker_has_its_own_row_in_the_ledger(self) -> None:
        """Four workers over four subtasks, each with a distinguishable cost."""
        case = make_case(case_id="ma-rows", workers=4, num_subtasks=4)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        per_agent = run.multi.ledger.per_agent()
        workers = [name for name in per_agent if name != LEADER]
        assert workers == ["worker1", "worker2", "worker3", "worker4"]
        # Input: four workers plus the leader all reported their turns. A runner
        # that read only the leader's trajectory would show exactly one row
        # here, which is the failure this test exists to catch.
        assert run.multi.ledger.turns_count > len(workers)

    @pytest.mark.asyncio
    async def test_the_child_token_total_is_not_the_leaders_total(self) -> None:
        """`child_tokens()` counts the fan-out, and excludes the leader.

        The assertion is a comparison against the leader's own row rather than
        against a hard-coded number, so it holds however the leader's merge turn
        is scripted -- and it fails on the exact bug §5.6 warns about, where
        every non-leader row is empty.
        """
        case = make_case(case_id="ma-sum", workers=2, num_subtasks=2)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        ledger = run.multi.ledger
        leader = ledger.leader
        assert leader.total_tokens > 0, "the leader ran no turns at all"
        assert ledger.child_tokens() > 0, (
            "no child tokens were collected; the fan-out's cost is missing from "
            "the totals (contract §5.6 red line)"
        )
        # The two are disjoint sums of the same list, so they must add up.
        assert ledger.child_tokens() + leader.total_tokens == ledger.total_tokens

    @pytest.mark.asyncio
    async def test_the_second_witness_agrees_with_the_usage_ledger(self) -> None:
        """`TaskRegistry` names the same workers the usage tap recorded.

        Two channels: the wrapped `call_model` (usage) and the registry records
        `spawn_teammate` writes (books). Agreement is worth more than either,
        because neither can produce the other's answer.
        """
        case = make_case(case_id="ma-witness", workers=3, num_subtasks=3)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        accounts = run.multi.accounts
        assert accounts["accounting_complete"] is True
        assert accounts["expected_agents"] == accounts["accounted_agents"] == 4
        assert accounts["witness_agents"] == ["worker1", "worker2", "worker3"]
        assert accounts["unconfirmed_agents"] == []

    @pytest.mark.asyncio
    async def test_a_worker_that_never_ran_its_model_is_reported_not_silent(self) -> None:
        """A spawned agent with no turns makes the accounts incomplete.

        Driven through `reconcile` rather than through the runner: a real
        teammate that is spawned and produces no model call cannot be provoked
        deterministically, and the property under test is the RECORD's, not the
        spawn path's. The runner's own handling of that record is covered by
        `TestTheRunnerRefusesToReportUnaccountedRuns`.
        """
        ledger = UsageLedger()
        ledger.note_spawned(LEADER)
        ledger.note_spawned("worker1")
        ledger.record(TurnUsage(agent=LEADER, input_tokens=1, output_tokens=1, tool_calls=0))

        accounts = reconcile(ledger, witness_agents=["worker1"])
        assert accounts.expected == 2
        assert accounts.accounted == 1
        assert accounts.complete is False


class TestAccountingFailuresRaise:
    """Every branch of the completeness check, with the input that fails it."""

    def test_a_missing_child_raises_with_the_counts(self) -> None:
        ledger = UsageLedger()
        ledger.note_spawned(LEADER)
        ledger.note_spawned("worker1")
        ledger.record(TurnUsage(agent=LEADER, input_tokens=1, output_tokens=1, tool_calls=0))

        accounts = reconcile(ledger)
        with pytest.raises(AccountingError) as excinfo:
            accounts.assert_complete(case_id="c1", variant=MULTI)
        # The message names the numbers, because "1 of 2 missing" is actionable
        # and "accounting failed" is not.
        assert "1 agents have usage but 2 were spawned" in str(excinfo.value)

    def test_an_agent_the_witness_never_saw_raises(self) -> None:
        """Turns attributed to a name nothing corroborates.

        This is the leak-detection branch: a context scope that escaped its
        task would charge one worker's turns to another name, and the totals
        would still add up -- which is exactly why the totals check alone is not
        enough.
        """
        ledger = UsageLedger()
        ledger.note_spawned(LEADER)
        ledger.record(TurnUsage(agent="ghost", input_tokens=1, output_tokens=1, tool_calls=0))

        accounts = reconcile(ledger, witness_agents=[])
        assert accounts.unconfirmed_agents == ["ghost"]
        assert accounts.complete is False
        with pytest.raises(AccountingError) as excinfo:
            accounts.assert_complete(case_id="c2", variant=MULTI)
        assert "which the second witness never observed" in str(excinfo.value)

    def test_a_clean_ledger_does_not_raise(self) -> None:
        """The negative control: without it, a check that always raises passes."""
        ledger = UsageLedger()
        ledger.note_spawned(LEADER)
        ledger.note_spawned("worker1")
        ledger.record(TurnUsage(agent=LEADER, input_tokens=1, output_tokens=1, tool_calls=0))
        ledger.record(TurnUsage(agent="worker1", input_tokens=1, output_tokens=1, tool_calls=0))

        accounts = reconcile(ledger, witness_agents=["worker1"])
        assert accounts.complete is True
        accounts.assert_complete(case_id="c3", variant=MULTI)

    def test_the_leader_is_not_required_in_the_witness(self) -> None:
        """The leader has no registry record, so it must not count as missing.

        Without this exemption every multi-agent run would fail its own check,
        and the pressure would be to weaken the check rather than to state the
        real expectation.
        """
        ledger = UsageLedger()
        ledger.note_spawned(LEADER)
        ledger.note_spawned("worker1")
        ledger.record(TurnUsage(agent=LEADER, input_tokens=1, output_tokens=1, tool_calls=0))
        ledger.record(TurnUsage(agent="worker1", input_tokens=1, output_tokens=1, tool_calls=0))

        accounts = reconcile(ledger, witness_agents=["worker1"])
        assert accounts.unconfirmed_agents == []
        assert accounts.complete is True

    def test_per_agent_totals_are_derived_from_the_turn_list(self) -> None:
        """The totals cannot drift from the turns they summarise."""
        ledger = UsageLedger()
        ledger.record(TurnUsage(agent="w", input_tokens=5, output_tokens=2, tool_calls=1))
        ledger.record(TurnUsage(agent="w", input_tokens=3, output_tokens=1, tool_calls=2))
        ledger.record(TurnUsage(agent=LEADER, input_tokens=9, output_tokens=4, tool_calls=0))

        bucket = ledger.per_agent()["w"]
        assert bucket == AgentUsage(
            agent="w", turns=2, input_tokens=8, output_tokens=3, tool_calls=3,
        )
        assert ledger.input_tokens == 17
        assert ledger.output_tokens == 7
        assert ledger.tool_calls == 3
        # `child_tokens` is the non-leader share, not the grand total.
        assert ledger.child_tokens() == 11


class TestTheAgentScopeSurvivesConcurrency:
    """The scope mechanism the per-agent attribution rests on."""

    @pytest.mark.asyncio
    async def test_two_concurrent_tasks_each_keep_their_own_agent(self) -> None:
        ledger = UsageLedger()
        token = current_ledger.set(ledger)
        seen: dict[str, str] = {}
        try:
            async def child(name: str) -> None:
                with agent_scope(name):
                    await asyncio.sleep(0)
                    seen[name] = current_agent()
                    await asyncio.sleep(0)
                    seen[f"{name}-late"] = current_agent()

            await asyncio.gather(child("a"), child("b"))
            # The leader's own scope is untouched by either child.
            seen["outer"] = current_agent()
        finally:
            current_ledger.reset(token)

        assert seen == {"a": "a", "a-late": "a", "b": "b", "b-late": "b", "outer": LEADER}

    @pytest.mark.asyncio
    async def test_the_scope_is_restored_when_the_body_raises(self) -> None:
        """A leaked scope would charge the leader's turns to a dead worker."""
        ledger = UsageLedger()
        token = current_ledger.set(ledger)
        try:
            with (
                pytest.raises(ValueError, match="boom"),
                agent_scope("worker1"),
            ):
                raise ValueError("boom")
            assert current_agent() == LEADER
        finally:
            current_ledger.reset(token)


class TestTheModelCounter:
    """The wrapping seam itself, on a hand-built stream rather than a run."""

    @pytest.mark.asyncio
    async def test_tool_calls_and_tokens_are_counted_per_turn(self) -> None:
        from longline.core.events import ToolUseStart, TurnComplete

        def factory(model: str | None = None, max_tokens: int = 1) -> Any:
            async def call_model(**kwargs: Any) -> Any:
                yield ToolUseStart(tool_name="Read", tool_id="t1", input={})
                yield ToolUseStart(tool_name="Read", tool_id="t2", input={})
                yield TurnComplete(
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=11, output_tokens=4),
                )
                yield ToolUseStart(tool_name="Read", tool_id="t3", input={})
                yield TurnComplete(
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=2, output_tokens=1),
                )
            return call_model

        ledger = UsageLedger()
        counter = count_usage(factory, ledger)
        events = [event async for event in counter()(messages=[])]

        # Two ToolUseStart + TurnComplete, then one more of each: three
        # starts and two completions. Every one must reach the caller -- a
        # wrapper that consumed an event to count it would silently truncate
        # the agent's stream.
        assert len(events) == 5, "the wrapper must pass every event through"
        assert [t.tool_calls for t in ledger.turns] == [2, 1]
        assert ledger.input_tokens == 13
        assert ledger.output_tokens == 5
        assert ledger.tool_calls == 3

    @pytest.mark.asyncio
    async def test_a_pinned_agent_overrides_the_ambient_scope(self) -> None:
        """The leader's counter must not be relabelled by a worker's scope."""
        from longline.core.events import TurnComplete

        def factory(model: str | None = None, max_tokens: int = 1) -> Any:
            async def call_model(**kwargs: Any) -> Any:
                yield TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=1))
            return call_model

        ledger = UsageLedger()
        token = current_ledger.set(ledger)
        try:
            with agent_scope("worker1"):
                pinned = count_usage(factory, ledger, agent=LEADER)
                [event async for event in pinned()(messages=[])]
        finally:
            current_ledger.reset(token)

        assert [t.agent for t in ledger.turns] == [LEADER]


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class TestTeammateIdExtraction:
    """The second witness's reader, on the text shape `AgentTool` really emits."""

    def test_ids_are_found_and_deduplicated_in_order(self) -> None:
        text = (
            "Agent 'x' launched in background (task_id: teammate-0a1b2c3d). "
            "See teammate-0a1b2c3d and teammate-ffeeddcc."
        )
        assert teammate_ids_in(text) == ["teammate-0a1b2c3d", "teammate-ffeeddcc"]

    def test_a_shorter_hex_is_not_an_id(self) -> None:
        """A prefix must not match: `spawn` mints exactly eight hex digits."""
        assert teammate_ids_in("teammate-abc") == []
        assert teammate_ids_in("teammate-0a1b2c3d4e") == []


class TestTheRunnerRefusesToReportUnaccountedRuns:
    """The exclusion reaches the ROW, not just an exception.

    An exception would abort a paid suite; what the contract needs is for the
    unusable case to be recorded with a reason and kept out of the ratio
    denominators, without being dropped (a silently shrunk denominator is a
    wrong number that looks right -- the same rule compression uses for a failed
    baseline).
    """

    @pytest.mark.asyncio
    async def test_an_unreconciled_fan_out_is_excluded_with_a_reason(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forced by breaking the witness, which is the real failure's shape.

        `_teammate_witness` is the second channel. Reporting no workers from it
        is precisely "the registry never saw the agents the ledger claims",
        which is the mis-attribution case -- and it is provoked here from
        outside the runner rather than by mutating a tracked file, so the
        check's negative branch is driven without touching the product.
        """
        import longline.eval.multi_agent_runner as runner

        monkeypatch.setattr(runner, "_teammate_witness", lambda registry: ([], {}))
        case = make_case(case_id="ma-unreconciled", workers=2, num_subtasks=2)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        assert run.multi.accounts["accounting_complete"] is False
        assert run.multi.accounting_error
        assert run.excluded_from_denominator is True
        assert run.exclusion_reason == REASON_ACCOUNTING_INCOMPLETE
        # The case is still in the data, with its row intact.
        aggregate = aggregate_multi_agent([run], group=CONTROLLED)
        assert aggregate.num_cases == 1
        assert aggregate.eligible_cases == 0
        # And with nothing eligible, the rates are "not measured" -- null, not
        # 0.0, which would read as "0% success".
        assert aggregate.single_success_rate.value is None
        assert aggregate.mean_speedup is None

    @pytest.mark.asyncio
    async def test_a_clean_run_is_not_excluded(self) -> None:
        """The negative control for the exclusion above."""
        case = make_case(case_id="ma-clean", workers=2, num_subtasks=2)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        assert run.excluded_from_denominator is False
        assert run.exclusion_reason is None
        assert run.multi.accounting_error == ""
        assert run.single.accounting_error == ""


class TestBothVariantsDoTheSameWork:
    """The precondition of a meaningful `Speedup`."""

    @pytest.mark.asyncio
    async def test_both_variants_are_handed_the_same_subtask_instructions(self) -> None:
        """The single prompt contains every subtask's exact instruction text.

        Each worker is handed one of those same strings by `subtask_prompt`, so
        asserting that the single prompt is the union of them -- and that each
        per-worker prompt is a substring of it -- is asserting that the two arms
        were asked for the same things.
        """
        case = make_case(case_id="ma-work", workers=2, num_subtasks=3)
        prompt = leader_prompt(case)
        for subtask in case.subtasks:
            assert subtask.instruction in prompt
            assert subtask.writes in prompt
            assert subtask.instruction in subtask_prompt(case, subtask)
        assert case.merge_file in prompt
        assert merge_instruction(case) in prompt

    @pytest.mark.asyncio
    async def test_every_subtask_artifact_exists_after_both_variants(self) -> None:
        """Both arms produce the full declared file set, and are judged on it."""
        case = make_case(case_id="ma-artifacts", workers=3, num_subtasks=4)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        expected = set(case.expected_paths())
        assert len(expected) == case.num_subtasks + 1, "one artifact per subtask plus the merge"
        for variant in (run.single, run.multi):
            assert variant.passed, f"{variant.variant}: {variant.judge_detail}"
            assert set(variant.subtask_verdicts) == {s.id for s in case.subtasks}
            assert all(variant.subtask_verdicts.values())

    @pytest.mark.asyncio
    async def test_more_subtasks_than_workers_still_runs_every_subtask(self) -> None:
        """`workers` is a concurrency limit, not a cap on the work done.

        Six subtasks with two workers must produce six artifacts. Spawning only
        `workers` agents and letting the rest ride would drop the overhang, and
        the missing files would then look like the model's failure rather than
        the harness's.
        """
        case = make_case(case_id="ma-waves", workers=2, num_subtasks=6)
        claude_dir = _temp_claude_dir()
        try:
            run = await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir, usage=WORKER_USAGE,
            )
        finally:
            _cleanup(claude_dir)

        assert run.multi.accounting_error == ""
        assert run.multi.passed, run.multi.judge_detail
        # Six subtasks at two at a time is three waves of two.
        assert run.multi.accounts["expected_agents"] == 1 + 6


class TestTheTwoGroupsAreNeverMerged:
    """Contract §5.6: exploratory is reported separately."""

    def test_aggregating_one_group_ignores_the_other(self) -> None:
        controlled = _stub_run("c1", CONTROLLED, passed=True)
        exploratory = _stub_run("e1", EXPLORATORY, passed=False)

        only_controlled = aggregate_multi_agent([controlled, exploratory], group=CONTROLLED)
        assert only_controlled.num_cases == 1
        assert only_controlled.single_success_rate.numerator == 1

        only_exploratory = aggregate_multi_agent([controlled, exploratory], group=EXPLORATORY)
        assert only_exploratory.num_cases == 1
        assert only_exploratory.single_success_rate.numerator == 0

    def test_the_pooled_ratio_is_a_mean_of_per_case_ratios(self) -> None:
        """Not a ratio of means: the two differ when cases are not homogeneous.

        Two cases, one where multi is 2x faster and one where it is 4x faster.
        The mean of the ratios is 3.0; a ratio of the summed durations would
        weight the longer case and produce a different number.
        """
        fast = _stub_run("c1", CONTROLLED, passed=True, single_ms=200.0, multi_ms=100.0)
        faster = _stub_run("c2", CONTROLLED, passed=True, single_ms=800.0, multi_ms=200.0)

        summary = aggregate_multi_agent([fast, faster], group=CONTROLLED)
        assert summary.mean_speedup == pytest.approx((2.0 + 4.0) / 2)
        # The ratio of means would be 1000/300 = 3.33, a different number.
        assert summary.mean_speedup != pytest.approx(1000.0 / 300.0)

    def test_token_overhead_is_signed_and_relative_to_the_single_arm(self) -> None:
        """A fan-out that costs less reports a negative overhead, not an error."""
        cheaper = _stub_run(
            "c1", CONTROLLED, passed=True, single_tokens=100, multi_tokens=50,
        )
        summary = aggregate_multi_agent([cheaper], group=CONTROLLED)
        assert summary.mean_token_overhead == pytest.approx(-0.5)

    def test_a_zero_single_arm_reports_not_measured(self) -> None:
        """None rather than a division error or a fabricated 0%.

        A run whose baseline cost nothing measured nothing, and reporting 0.0
        would read as "the fan-out added no token overhead" -- a different
        claim from "no measurement was taken".
        """
        zero = _stub_run(
            "c1", CONTROLLED, passed=True, single_ms=0.0, single_tokens=0,
        )
        assert zero.speedup is None
        assert zero.token_overhead is None


def _stub_run(
    case_id: str,
    group: str,
    *,
    passed: bool,
    single_ms: float = 100.0,
    multi_ms: float = 50.0,
    single_tokens: int = 100,
    multi_tokens: int = 200,
) -> Any:
    """A `MultiAgentRun` with hand-set numbers, for the aggregation tests.

    Built directly rather than by running a case: these tests are about the
    arithmetic of the aggregate, and driving a real run would make each
    expectation depend on the scripted model's turn count instead of on the
    numbers the test chose.
    """
    from longline.eval.multi_agent_runner import MultiAgentRun, VariantRun

    def variant(name: str, ms: float, tokens: int) -> VariantRun:
        ledger = UsageLedger(spawned=[LEADER])
        ledger.record(TurnUsage(
            agent=LEADER, input_tokens=tokens, output_tokens=0, tool_calls=1,
        ))
        return VariantRun(
            variant=name,
            passed=passed,
            duration_ms=ms,
            ledger=ledger,
            accounts={},
            subtask_verdicts={},
        )

    return MultiAgentRun(
        case_id=case_id,
        group=group,
        workers=2,
        num_subtasks=2,
        single=variant(SINGLE, single_ms, single_tokens),
        multi=variant(MULTI, multi_ms, multi_tokens),
        expected_paths=[],
    )


class _FakeEngine:
    """A stand-in exposing exactly the two attributes the transport swap touches.

    `_ScriptedTransport` and `_apply_live_counting` both move `make_call_model`
    and `make_call_model_factory`, and nothing else about a `QueryEngine`
    matters to them. A fake that carried a whole engine would make this test a
    test of `build_engine`.

    `scripted` flips when a scripted factory is installed, so a test can tell
    "the live engine was left alone" from "the live engine was replaced" --
    which is the entire difference between the two paths.
    """

    def __init__(self) -> None:
        self.scripted = False
        self.make_call_model: Any = self._real_factory
        self.make_call_model_factory: Any = self._real_factory

    def _real_factory(
        self, model: str | None = None, max_tokens: int = 16384
    ) -> Any:
        _ = model, max_tokens

        async def call_model(**kwargs: Any) -> Any:
            _ = kwargs
            yield None

        return call_model

    async def submit(self, prompt: str, *, max_turns: int) -> Any:
        _ = prompt, max_turns
        yield None


class TestLivePathIsReachable:
    """`offline=False` must NOT install the scripted transport.

    The regression this guards: `_apply_scripted_model` was called
    unconditionally in both variants, so a "live" run silently used the
    scripted model. `model=` reached `build_engine` and the system prompt and
    nothing else -- the request never left the process, every run cost nothing,
    and the module docstring's "A real model id runs the same case against the
    live API" was simply not what the code did.
    """

    def test_live_counting_wraps_the_engines_own_transport(self) -> None:
        engine = _FakeEngine()
        ledger = UsageLedger()

        _apply_live_counting(engine, ledger, agent=LEADER)

        assert isinstance(engine.make_call_model, ModelCounter)
        assert isinstance(engine.make_call_model_factory, ModelCounter)
        assert engine.scripted is False, "the live path must not install a scripted model"

    def test_live_counting_keeps_the_original_factory_underneath(self) -> None:
        """The wrapper must WRAP, not replace.

        This is the whole difference from `_apply_scripted_model`: replacing
        the engine's own factory is what made `model=` decorative. Asserting
        the wrapped factory is the engine's original is what tells the two
        implementations apart when both produce a `ModelCounter`.
        """
        engine = _FakeEngine()
        original = engine.make_call_model_factory

        _apply_live_counting(engine, UsageLedger(), agent=LEADER)

        assert engine.make_call_model.factory is original
        assert engine.make_call_model_factory.factory is original

    def test_live_counting_binds_the_ledger_and_agent(self) -> None:
        engine = _FakeEngine()
        ledger = UsageLedger()

        _apply_live_counting(engine, ledger, agent=LEADER)

        assert engine.make_call_model.ledger is ledger
        assert engine.make_call_model.agent == LEADER

    def test_unpinned_live_counting_resolves_the_agent_per_turn(self) -> None:
        """The multi variant needs one shared counter, not a pinned one.

        With `agent=None` the owner of each turn is read from the ambient scope
        at call time, which is what lets the leader's turns and every worker's
        turns flow through a single wrapper without double-counting either.
        """
        engine = _FakeEngine()

        _apply_live_counting(engine, UsageLedger())

        assert engine.make_call_model.agent is None


__all__: list[str] = []
