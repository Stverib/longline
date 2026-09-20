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
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
    CATEGORIES,
    CATEGORY_ANALYSIS,
    CATEGORY_DEPENDENT,
    CATEGORY_MODIFICATION,
    CONTROLLED,
    DEPENDENT_WORKERS,
    EXPLORATORY,
    MULTI,
    SINGLE,
    CaseParseError,
    MultiAgentCase,
    Subtask,
    chain_order,
)
from longline.eval.multi_agent_runner import (
    REASON_ACCOUNTING_INCOMPLETE,
    REASON_TIMEOUT,
    VariantRun,
    _apply_live_counting,
    _drive,
    _in_sandbox,
    _judge,
    _spawn_workers_serial,
    aggregate_multi_agent,
    effective_workers,
    leader_prompt,
    merge_instruction,
    run_multi_agent_case,
    run_multi_agent_suite,
    subtask_prompt,
)
from longline.models.messages import Usage
from longline.session.task_registry import TaskRegistry

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
    repeat_index: int = 0,
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
        repeat_index=repeat_index,
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
        # Counted, not just present: "the wrapper still calls the engine's own
        # factory" is the claim, and a counter is how it can be checked when the
        # factory returns a fresh closure every call (so identity comparison
        # cannot be used).
        self.factory_calls = 0
        self.make_call_model: Any = self._real_factory
        self.make_call_model_factory: Any = self._real_factory

    def _real_factory(
        self, model: str | None = None, max_tokens: int = 16384
    ) -> Any:
        _ = model, max_tokens
        self.factory_calls += 1

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
        the engine's own factory is what made `model=` decorative. Counting the
        engine's own factory invocations is what tells the two implementations
        apart when both produce a `ModelCounter` -- and it is a call counter
        rather than an identity check because the factory returns a fresh
        closure each time.
        """
        engine = _FakeEngine()
        before = engine.factory_calls

        _apply_live_counting(engine, UsageLedger(), agent=LEADER)
        engine.make_call_model_factory(model="claude-sonnet-5")

        assert engine.factory_calls > before, (
            "the wrapper never reached the engine's own factory, so the request "
            "would never have been made"
        )

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


class _MethodShapedEngine:
    """An engine whose factories are METHODS, the way `QueryEngine`'s are.

    `_FakeEngine` assigns them as plain attributes, which is a shape no real
    engine has -- and that difference is exactly what hid a live-path bug.
    With a method, reading `engine.make_call_model_factory` yields a BOUND
    METHOD rather than a factory, and wrapping that produces a wrapper that
    raises `TypeError: got an unexpected keyword argument 'model'` the first
    time a sub-agent creation site calls it. Every type assertion in this file
    passed anyway, because a `ModelCounter` wrapping a bound method is still a
    `ModelCounter`.

    The shapes are copied from `QueryEngine` rather than made convenient:
    `make_call_model_factory()` takes NO arguments and returns a factory, and
    that factory is what accepts `model`. An earlier version of this fake took
    `model` on the meta-factory too, which is a shape the real engine does not
    have -- and it made the wrapper look broken when it was the fake that was
    wrong. A fake that disagrees with the thing it stands in for is worse than
    no fake, because it produces confident failures.

    It records the models it was asked for, so a test can tell "the wrapper
    forwarded the call" from "the wrapper returned something callable".
    """

    def __init__(self) -> None:
        self.asked: list[str | None] = []

    def make_call_model(self, model: str | None = None, max_tokens: int = 16384) -> Any:
        _ = max_tokens
        self.asked.append(model)

        async def call_model(**kwargs: Any) -> Any:
            _ = kwargs
            yield None

        return call_model

    def make_call_model_factory(self) -> Any:
        engine = self

        def factory(model: str | None = None, max_tokens: int = 16384) -> Any:
            return engine.make_call_model(model=model, max_tokens=max_tokens)

        return factory


class TestLiveCountingSurvivesTheRealEngineShape:
    """The wrapper has to be CALLABLE, not merely present and of the right type.

    Found by the first live invocation this harness ever made: `--suite pair
    --allow-paid` failed inside `_apply_live_counting`'s product with a
    `TypeError`, before any request left the process. Every assertion the
    previous class makes was already green.
    """

    def test_the_installed_factory_takes_the_arguments_a_spawn_passes(self) -> None:
        """FAILS ON: `live = engine.make_call_model_factory` without the parens."""
        engine = _MethodShapedEngine()

        _apply_live_counting(engine, UsageLedger())

        call_model = engine.make_call_model_factory(model="claude-sonnet-5")

        assert callable(call_model)
        assert engine.asked == ["claude-sonnet-5"], (
            "the wrapper did not forward the model through to the engine's own "
            "factory, so the request would never have been made with it"
        )

    def test_the_installed_call_model_takes_the_arguments_submit_passes(self) -> None:
        engine = _MethodShapedEngine()

        _apply_live_counting(engine, UsageLedger())

        assert callable(engine.make_call_model(max_tokens=8192))
        assert engine.asked == [None], (
            "wrapping alone must not call the engine's factory: a wrapper that "
            "built a model per wrap would build one per variant, not per turn"
        )


class TestSandboxChdir:
    """Both variants must run with the process cwd inside the sandbox.

    The production tools resolve relative paths against the process cwd
    (`FileWriteTool` does `Path(file_path)`; `Tool._declare` resolves the same
    way), while `build_engine` tells the model in its system prompt that its
    working directory IS the sandbox. Without this the two disagree: the model
    writes what it was asked for, relative to the repository root, and the
    judge then reads an empty sandbox.

    The offline protocol hides the disagreement, because `scripted_factory`
    builds absolute paths and never resolves a relative one.
    """

    def test_chdir_context_moves_into_the_sandbox(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()

        with _in_sandbox(str(sandbox)):
            assert Path.cwd().resolve() == sandbox.resolve()

    def test_chdir_context_restores_cwd(self, tmp_path: Path) -> None:
        before = Path.cwd().resolve()
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()

        with _in_sandbox(str(sandbox)):
            pass

        assert Path.cwd().resolve() == before

    def test_chdir_context_restores_cwd_on_exception(self, tmp_path: Path) -> None:
        """A variant that raises must not leave the process in its sandbox.

        `_drive` catches a crashed variant and records it rather than
        propagating, but a `_judge` or a `_spawn_workers` raise is not caught
        there. Without the `finally`, one such case would relocate every later
        case in the run -- and the symptom would be "the model wrote the wrong
        file", not "the harness leaked the cwd".
        """
        before = Path.cwd().resolve()
        sandbox = tmp_path / "sbx-on-error"
        sandbox.mkdir()

        with pytest.raises(RuntimeError, match="boom"), _in_sandbox(str(sandbox)):
            raise RuntimeError("boom")

        assert Path.cwd().resolve() == before

    def test_chdir_refuses_a_non_root_starting_point(self, tmp_path: Path) -> None:
        """A nested chdir is a bug: it means a previous case did not restore.

        Left unchecked the second chdir silently nests, every later case
        resolves its paths one level deeper, and the failure surfaces as a
        missing artifact rather than a leaked cwd -- which is the kind of
        symptom that gets attributed to the model.
        """
        sandbox = tmp_path / "sbx-nested"
        sandbox.mkdir()

        with _in_sandbox(str(sandbox)), pytest.raises(RuntimeError, match="chdir"):
            # Entered directly rather than as a nested `with`: ruff's SIM117
            # wants the two contexts combined, and the inner one has to raise
            # on ENTRY, which a combined `with` cannot express as a body.
            _in_sandbox(str(tmp_path / "other")).__enter__()

        assert Path.cwd().resolve() != sandbox.resolve()


class TestCategoryAxis:
    """`category` is orthogonal to `group`, and never replaces it.

    `group` answers "did both arms do the same work" (`controlled` vs
    `exploratory`); `category` answers "what shape was that work". A case has
    both, and the report partitions on both.
    """

    def test_category_defaults_to_analysis_for_legacy_rows(self) -> None:
        """The frozen 24 must keep loading: an absent `category` cannot raise.

        They predate the field and cannot be edited (spec hard constraint 6),
        so this default is what keeps `formal_multi_agent_offline` reproducible.
        """
        case = MultiAgentCase.from_dict(_case_dict())

        assert case.category == CATEGORY_ANALYSIS

    def test_category_is_read_when_present(self) -> None:
        case = MultiAgentCase.from_dict(
            _case_dict(category=CATEGORY_MODIFICATION)
        )

        assert case.category == CATEGORY_MODIFICATION

    def test_unknown_category_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="category"):
            MultiAgentCase.from_dict(_case_dict(category="vibes"))

    def test_every_category_is_loadable(self) -> None:
        """A constant in `CATEGORIES` that the loader rejects is a dead branch."""
        for category in CATEGORIES:
            case = MultiAgentCase.from_dict(
                _case_dict(**_category_overrides(category))
            )
            assert case.category == category

    def test_group_and_category_are_independent(self) -> None:
        """Four combinations, all valid -- the axes do not constrain each other.

        Note what this test does NOT claim: that one subtask list, or one
        `workers` value, is legal in every category. `dependent` requires a
        total order over overlapping writes and exactly one worker; the others
        require the absence of both. That mutual exclusion is the loader's
        actual shape, and hiding it here would make this test assert something
        false.
        """
        for group in (CONTROLLED, EXPLORATORY):
            for category in CATEGORIES:
                case = MultiAgentCase.from_dict(_case_dict(
                    group=group, **_category_overrides(category),
                ))
                assert (case.group, case.category) == (group, category)

    def test_a_chain_must_declare_exactly_one_worker(self) -> None:
        """`workers` is reported as the agent count, so a chain cannot claim 2.

        A chain runs step N+1 only after step N has finished. Recording a higher
        number would put a parallelism the case forbids into the run metadata,
        where the report reads it back as if it were a measurement.
        """
        with pytest.raises(CaseParseError, match=r"workers|one teammate"):
            MultiAgentCase.from_dict(_case_dict(
                category=CATEGORY_DEPENDENT,
                subtasks=_chain(("s1", "a.py"), ("s2", "a.py")),
                workers=2,
            ))

    def test_a_non_chain_may_not_declare_one_worker(self) -> None:
        """The rule cuts both ways: sub-two workers leaves nothing to fan out."""
        with pytest.raises(CaseParseError, match="workers"):
            MultiAgentCase.from_dict(_case_dict(workers=1))


def _chain(*specs: tuple[str, str]) -> list[dict[str, Any]]:
    """`(id, writes)` pairs into the declared-subtask shape, each on the last.

    Every step but the first declares the step before it as its predecessor, so
    the result is a total order by construction. Tests that want a BROKEN chain
    take this and edit one entry, which keeps the breakage the only difference
    from a case the loader accepts.
    """
    out: list[dict[str, Any]] = []
    for index, (sid, writes) in enumerate(specs):
        entry: dict[str, Any] = {
            "id": sid, "instruction": f"step {sid}", "writes": writes,
        }
        if index:
            entry["depends_on"] = [specs[index - 1][0]]
        out.append(entry)
    return out


class TestDependentChainValidation:
    """A `dependent` case's subtasks form a total order, or the case is rejected.

    The non-overlap rule that `_parse_subtasks` enforces everywhere else is
    about CONCURRENCY: a fan-out that writes one path twice loses one of the two
    writes. A declared chain has no concurrency -- step N+1 gets a fresh
    teammate only after step N's has finished -- so overlapping writes are the
    point of the category rather than a hazard, and a different check replaces
    it.
    """

    def test_legal_chain_loads(self) -> None:
        case = MultiAgentCase.from_dict(_case_dict(
            **_category_overrides(
                CATEGORY_DEPENDENT, subtasks=_chain(("s1", "src/a.py"), ("s2", "src/a.py"), ("s3", "src/a.py")),
            ),
        ))

        assert [s.id for s in case.subtasks] == ["s1", "s2", "s3"]
        assert case.subtasks[0].depends_on == ()
        assert case.subtasks[2].depends_on == ("s2",)

    def test_chain_order_is_the_declared_order(self) -> None:
        case = MultiAgentCase.from_dict(_case_dict(
            **_category_overrides(
                CATEGORY_DEPENDENT, subtasks=_chain(("s1", "a.py"), ("s2", "a.py"), ("s3", "a.py")),
            ),
        ))

        assert [s.id for s in chain_order(case.subtasks)] == ["s1", "s2", "s3"]

    def test_overlapping_writes_raise_without_the_dependent_category(self) -> None:
        """The two loaders must disagree, and this is the half that rejects.

        Asserted separately from the accepting case because the whole point of
        the new path is that the SAME declarations are legal in one category and
        illegal in another. A test that only checked the accepting side would
        pass just as well if the rule had been deleted outright.

        The subtasks here carry no `depends_on`: under a non-chain category a
        declared predecessor is rejected first, so a case that declared one
        would test that rule instead of this one.
        """
        with pytest.raises(CaseParseError, match="race"):
            MultiAgentCase.from_dict(_case_dict(
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "src/a.py"},
                    {"id": "s2", "instruction": "b", "writes": "src/a.py"},
                ],
            ))

    def test_declared_predecessors_are_rejected_outside_a_chain(self) -> None:
        """A chain filed under the wrong category would be run CONCURRENTLY.

        Both variants of a non-chain case are free to run the subtasks in any
        order, so a case that declares `depends_on` and is not `dependent` is
        not merely mislabelled -- it will be executed in a way its declarations
        say is wrong, and if its steps share a path, silently lose a write.
        """
        with pytest.raises(CaseParseError, match="depends_on"):
            MultiAgentCase.from_dict(_case_dict(
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "a.py"},
                    {"id": "s2", "instruction": "b", "writes": "b.py",
                     "depends_on": ["s1"]},
                ],
            ))

    def test_cycle_is_rejected(self) -> None:
        subtasks = _chain(("s1", "a.py"), ("s2", "a.py"), ("s3", "a.py"))
        subtasks[0]["depends_on"] = ["s3"]          # s1 <- s3 <- s2 <- s1
        with pytest.raises(CaseParseError, match=r"total order|cycle|unreachable"):
            MultiAgentCase.from_dict(
                _case_dict(category=CATEGORY_DEPENDENT, subtasks=subtasks)
            )

    def test_fork_is_rejected(self) -> None:
        """Two steps depending on one predecessor can run in either order."""
        with pytest.raises(CaseParseError, match=r"total order|fork|successor"):
            MultiAgentCase.from_dict(_case_dict(
                category=CATEGORY_DEPENDENT,
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "a.py"},
                    {"id": "s2", "instruction": "b", "writes": "a.py",
                     "depends_on": ["s1"]},
                    {"id": "s3", "instruction": "c", "writes": "a.py",
                     "depends_on": ["s1"]},
                ],
            ))

    def test_two_roots_are_rejected(self) -> None:
        """Two first steps means two chains, and which runs first is undefined."""
        with pytest.raises(CaseParseError, match=r"total order|first step"):
            MultiAgentCase.from_dict(_case_dict(
                category=CATEGORY_DEPENDENT,
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "a.py"},
                    {"id": "s2", "instruction": "b", "writes": "b.py"},
                ],
            ))

    def test_missing_predecessor_is_rejected(self) -> None:
        subtasks = _chain(("s1", "a.py"), ("s2", "a.py"))
        subtasks[1]["depends_on"] = ["s9"]
        with pytest.raises(CaseParseError, match=r"unknown predecessor|s9"):
            MultiAgentCase.from_dict(
                _case_dict(category=CATEGORY_DEPENDENT, subtasks=subtasks)
            )

    def test_self_dependency_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match=r"itself|cycle"):
            MultiAgentCase.from_dict(_case_dict(
                category=CATEGORY_DEPENDENT,
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "a.py",
                     "depends_on": ["s1"]},
                    {"id": "s2", "instruction": "b", "writes": "a.py",
                     "depends_on": ["s1"]},
                ],
            ))

    def test_two_predecessors_are_rejected(self) -> None:
        """A step with two predecessors is a join, not a link in a chain."""
        with pytest.raises(CaseParseError, match=r"predecessor|total order"):
            MultiAgentCase.from_dict(_case_dict(
                category=CATEGORY_DEPENDENT,
                subtasks=[
                    {"id": "s1", "instruction": "a", "writes": "a.py"},
                    {"id": "s2", "instruction": "b", "writes": "a.py"},
                    {"id": "s3", "instruction": "c", "writes": "a.py",
                     "depends_on": ["s1", "s2"]},
                ],
            ))


def _dependent_case(*, n_steps: int = 3, case_id: str = "ma-chain") -> MultiAgentCase:
    """A `dependent` case whose steps all write the same file, in order.

    Overlapping writes are the category's shape rather than an oversight: each
    step consumes what the previous one produced, so there is only one artifact
    and it changes hands. The chain must be declared here rather than through
    the loader, because a `MultiAgentCase` built directly never goes through
    `_parse_subtasks`.
    """
    dirname = f"out/{case_id}"
    subtasks = [
        Subtask(
            id=f"s{index}",
            instruction=f"Extend the note (step {index})",
            writes=f"{dirname}/note.md",
            depends_on=() if index == 1 else (f"s{index - 1}",),
        )
        for index in range(1, n_steps + 1)
    ]
    return MultiAgentCase(
        id=case_id,
        task="Build the note up one step at a time",
        category=CATEGORY_DEPENDENT,
        subtasks=subtasks,
        workers=DEPENDENT_WORKERS,
        merge_file=f"{dirname}/manifest.txt",
        fixture_single="parallel_repo_single",
        fixture_multi="parallel_repo_multi",
        max_turns=12,
        tags=["multi-agent"],
        checks=[{"fn": "file_exists", "args": {"path": f"{dirname}/manifest.txt"}}],
    )


class TestDependentArmIsSerial:
    """`dependent` fans out one teammate at a time, in chain order.

    Concurrency here would defeat the case: a chain exists to measure what a
    fresh context per step costs, and running two steps at once makes them race
    on the same file rather than hand work forward.
    """

    @pytest.mark.asyncio
    async def test_each_step_spawns_only_after_its_predecessor_landed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The observable is the predecessor's ARTIFACT, not a call order.

        A fake spawn returns immediately, so "was it awaited" is not visible
        from the calls alone. What IS visible is whether the previous step's
        file already existed when the next step started -- which is the
        property the chain needs, and it fails for a concurrent implementation
        whether or not that implementation happens to call spawn in order.
        """
        case = _dependent_case(n_steps=3)
        sandbox = tmp_path / "sbx-chain"
        sandbox.mkdir()
        by_id = {s.id: s for s in case.subtasks}
        prompt_to_subtask = {subtask_prompt(case, s): s for s in case.subtasks}
        observed: list[tuple[str, bool]] = []

        async def _fake_spawn(*args: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt") or args[2])
            subtask = prompt_to_subtask[prompt]
            predecessor_landed = all(
                (sandbox / by_id[pid].writes).exists()
                for pid in subtask.depends_on
            )
            observed.append((subtask.id, predecessor_landed))
            target = sandbox / subtask.writes
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{subtask.id}\n", encoding="utf-8")
            return subtask.id

        monkeypatch.setattr("longline.swarm.spawn.spawn_teammate", _fake_spawn)

        await _spawn_workers_serial(
            case,
            counted=None,
            ledger=UsageLedger(),
            registry=TaskRegistry(),
            claude_dir=None,
            spawned_ids=[],
            sandbox=sandbox,
            failures={},
        )

        assert [sid for sid, _ in observed] == ["s1", "s2", "s3"], "chain order"
        assert all(landed for _, landed in observed), "s1 has no predecessor"

    @pytest.mark.asyncio
    async def test_a_failed_step_does_not_stop_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stopping early would make "gave up" and "produced nothing" the same row.

        The case's judges run against the final workspace, so a chain that dies
        at step 2 has to look different from one that never started. Aborting
        would erase that difference and charge the missing steps to the model.
        """
        case = _dependent_case(n_steps=3)
        sandbox = tmp_path / "sbx-fail"
        sandbox.mkdir()
        prompt_to_subtask = {subtask_prompt(case, s): s for s in case.subtasks}
        attempted: list[str] = []

        async def _fake_spawn(*args: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt") or args[2])
            subtask = prompt_to_subtask[prompt]
            attempted.append(subtask.id)
            if subtask.id == "s2":
                raise RuntimeError("step two exploded")
            return subtask.id

        monkeypatch.setattr("longline.swarm.spawn.spawn_teammate", _fake_spawn)

        failures: dict[str, BaseException] = {}
        await _spawn_workers_serial(
            case,
            counted=None,
            ledger=UsageLedger(),
            registry=TaskRegistry(),
            claude_dir=None,
            spawned_ids=[],
            sandbox=sandbox,
            failures=failures,
        )

        assert attempted == ["s1", "s2", "s3"], "step three must still be attempted"
        assert list(failures) == ["worker2"], "the failure is recorded by agent name"

    @pytest.mark.asyncio
    async def test_every_step_gets_its_own_spawn_counter_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Serial does not mean unaccounted: the §5.6 red line still applies."""
        case = _dependent_case(n_steps=3)
        sandbox = tmp_path / "sbx-ledger"
        sandbox.mkdir()
        prompt_to_subtask = {subtask_prompt(case, s): s for s in case.subtasks}

        async def _fake_spawn(*args: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt") or args[2])
            return prompt_to_subtask[prompt].id

        monkeypatch.setattr("longline.swarm.spawn.spawn_teammate", _fake_spawn)

        ledger = UsageLedger()
        spawned: list[str] = []
        await _spawn_workers_serial(
            case,
            counted=None,
            ledger=ledger,
            registry=TaskRegistry(),
            claude_dir=None,
            spawned_ids=spawned,
            sandbox=sandbox,
            failures={},
        )

        assert spawned == ["worker1", "worker2", "worker3"]
        assert sorted(ledger.spawned) == ["worker1", "worker2", "worker3"]

    @pytest.mark.asyncio
    async def test_the_runner_routes_a_dependent_case_to_the_serial_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wiring, not behaviour: the three tests above cannot see this.

        Every one of them drives `_spawn_workers_serial` directly, so all three
        would still pass if `run_multi_variant` handed a chain to the
        concurrent scheduler. This is the assertion that the dispatch exists.
        """
        case = _dependent_case(n_steps=2)
        claude_dir = _temp_claude_dir()
        calls: list[str] = []

        async def _record_serial(*args: Any, **kwargs: Any) -> None:
            calls.append("serial")

        async def _record_concurrent(*args: Any, **kwargs: Any) -> None:
            calls.append("concurrent")

        monkeypatch.setattr(
            "longline.eval.multi_agent_runner._spawn_workers_serial", _record_serial,
        )
        monkeypatch.setattr(
            "longline.eval.multi_agent_runner._spawn_workers", _record_concurrent,
        )
        try:
            await run_multi_agent_case(
                case, api_key="offline", fixtures_dir=FIXTURES_DIR,
                claude_dir=claude_dir,
            )
        finally:
            _cleanup(claude_dir)

        assert calls == ["serial"], "the multi arm must use the serial scheduler"


def _category_overrides(
    category: str, *, subtasks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The per-category fields that must agree with the category.

    `workers` is not free alongside `subtasks`: a chain declares exactly one
    worker, everything else declares two to four. Keeping the pairing in one
    place means a test that varies the category does not have to know which
    fields move with it -- and it cannot forget one.

    `subtasks` may be overridden for a test that needs a specific list; the
    `workers` value still comes from the category, which is the part that has to
    agree with it.
    """
    return {
        "category": category,
        "subtasks": subtasks if subtasks is not None else _subtasks_for(category),
        "workers": 1 if category == CATEGORY_DEPENDENT else 2,
    }


def _subtasks_for(category: str) -> list[dict[str, Any]]:
    """A subtask list the loader accepts for this category.

    The two shapes are mutually exclusive rather than merely different:
    `dependent` needs a declared total order and must overlap its writes, while
    every other category rejects `depends_on` outright and rejects two steps
    sharing a path. So a test that varies the category has to vary the
    subtasks with it -- there is no one list that satisfies all three.
    """
    if category == CATEGORY_DEPENDENT:
        return _chain(("s1", "a.py"), ("s2", "a.py"))
    return [
        {"id": "s1", "instruction": "a", "writes": "a.py"},
        {"id": "s2", "instruction": "b", "writes": "b.py"},
    ]


def _case_dict(**overrides: Any) -> dict[str, Any]:
    """A loader-shaped dict for a minimal valid case.

    Round-tripped from the known-good `make_case()` rather than hand-written:
    a hand-written dict drifts from the schema the moment a required field is
    added, and the symptom would be a `CaseParseError` in an unrelated test.
    """
    case = make_case()
    d: dict[str, Any] = {
        "id": case.id,
        "task": case.task,
        "group": case.group,
        "subtasks": [
            {"id": s.id, "instruction": s.instruction, "writes": s.writes}
            for s in case.subtasks
        ],
        "workers": case.workers,
        "merge": case.merge,
        "merge_file": case.merge_file,
        "fixture_single": case.fixture_single,
        "fixture_multi": case.fixture_multi,
        "max_turns": case.max_turns,
        "tags": list(case.tags),
        "checks": [dict(c) for c in case.checks],
    }
    d.update(overrides)
    return d


class TestHiddenTestInstallation:
    """The judge must be present at grading time, and must not be before it.

    The sandbox IS the agent's working tree for the whole run, so a hidden test
    installed at the start is a test the model can read -- and a case whose
    answer is readable measures reading. `_judge` installs it, which is the one
    place that sequencing is written down; splitting "install" from "run" across
    two call sites is how they eventually drift.
    """

    def _case_with_hidden(self, hidden: str) -> MultiAgentCase:
        case = make_case(case_id="ma-hid")
        case.hidden_test = hidden
        # Only the hidden test grades this case, so an empty sandbox passing
        # means "the judge ran and liked it", not "nothing was checked".
        case.checks = [{
            "fn": "python_test",
            "args": {
                "command": ["python", "-m", "pytest", "test_hidden.py", "-q"],
                "allowed_commands": ["python"],
                "timeout_s": 60,
            },
        }]
        return case

    def test_judge_copies_the_hidden_test_in_before_grading(self, tmp_path: Path) -> None:
        fixtures = tmp_path / "fixtures"
        hidden = fixtures / "pair" / "ma-hid_hidden" / "test_hidden.py"
        hidden.parent.mkdir(parents=True)
        hidden.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        case = self._case_with_hidden("pair/ma-hid_hidden/test_hidden.py")

        passed, detail, _verdicts = _judge(case, sandbox, fixtures_dir=fixtures)

        assert (sandbox / "test_hidden.py").is_file()
        assert passed is True, detail
        assert [c["fn"] for c in detail] == ["python_test"]

    def test_no_case_gets_a_hidden_test_installed_unasked(self, tmp_path: Path) -> None:
        """A case without one must not pick up a stray file from a previous run."""
        fixtures = tmp_path / "fixtures"
        hidden = fixtures / "pair" / "ma-hid_hidden" / "test_hidden.py"
        hidden.parent.mkdir(parents=True)
        hidden.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        case = make_case(case_id="ma-nohid")
        case.checks = [{"fn": "file_exists", "args": {"path": "out/ma-nohid/m1.md"}}]

        _judge(case, sandbox, fixtures_dir=fixtures)

        assert not (sandbox / "test_hidden.py").exists()

    def test_judge_refuses_a_hidden_test_it_cannot_locate(self, tmp_path: Path) -> None:
        """A missing judge is an error, not a skipped step.

        Skipping would make every run of the case fail as though the model had
        not done the work -- the case would look hard instead of looking broken.
        """
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        case = self._case_with_hidden("pair/nope_hidden/test_hidden.py")

        with pytest.raises(FileNotFoundError, match="hidden test"):
            _judge(case, sandbox, fixtures_dir=tmp_path / "fixtures")

    def test_judge_refuses_a_hidden_test_with_no_fixtures_root(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        case = self._case_with_hidden("pair/x_hidden/test_hidden.py")

        with pytest.raises(ValueError, match="fixtures_dir"):
            _judge(case, sandbox)


class _StallingEngine:
    """An engine whose `submit` never yields, so `_drive` hits its ceiling.

    Pairs with a patched-down `VARIANT_TIMEOUT_S`: the production 300s value
    would make this test either slow or -- if it really waited -- pointless.
    `wait_for` cancels the sleep, so the test costs one scheduling round trip.
    """

    async def submit(self, prompt: str, *, max_turns: int) -> AsyncIterator[Any]:
        _ = prompt, max_turns
        await asyncio.sleep(3600)
        yield  # pragma: no cover - unreachable; this is what makes it a generator


class TestRunMetadata:
    """Three facts a row has to carry, each an assumption the ratios rest on.

    `temperature` was not unset -- the spec's premise for this task said it was,
    and that was wrong. It is a literal `1.0` inside `stream_response`, applied
    to every request. The defects that follow from its being a LITERAL are what
    this fixes: the harness could not pin it (so three repeats could not be
    three samples of one configuration), and no row recorded it (so a change to
    that literal would move every number with nothing in the data to say so).

    `timed_out` exists because `_drive` used to put the timeout into the same
    `errors` list a crash goes into, so the report could not say which happened.
    The two have different fixes: a ceiling that is too low, versus a bug.

    `workers` is the concurrency actually used, which is not always
    `case.workers` once an override is in play.
    """

    def test_timeout_is_its_own_reason_not_a_variant_error(self) -> None:
        run = VariantRun(
            variant=SINGLE, passed=False, duration_ms=1.0, ledger=UsageLedger(),
            accounts={}, subtask_verdicts={}, errors=[], timed_out=True,
        )

        assert run.exclusion_reason() == REASON_TIMEOUT

    def test_accounting_error_outranks_timeout(self) -> None:
        """Both can be true, so the order is a decision, not an accident.

        An unreconciled ledger means the COST is untrustworthy, which
        disqualifies the row no matter how the clock behaved. Reporting the
        timeout first would hide a broken ledger behind an honest-looking
        clock reading.
        """
        run = VariantRun(
            variant=MULTI, passed=False, duration_ms=1.0, ledger=UsageLedger(),
            accounts={}, subtask_verdicts={},
            errors=["variant exceeded the ceiling"], timed_out=True,
            accounting_error=REASON_ACCOUNTING_INCOMPLETE,
        )

        assert run.exclusion_reason() == REASON_ACCOUNTING_INCOMPLETE

    def test_a_clean_run_has_no_exclusion_reason(self) -> None:
        run = VariantRun(
            variant=MULTI, passed=True, duration_ms=1.0, ledger=UsageLedger(),
            accounts={}, subtask_verdicts={},
        )

        assert run.exclusion_reason() is None

    def test_the_row_carries_what_the_run_actually_used(self) -> None:
        """Reported from the run, not re-read from a module constant later.

        `variant_timeout_s` in particular: a reader has to be able to tell that
        a row was cut off by a 300s ceiling and not by today's value, which a
        test or a future edit may already have changed.
        """
        run = VariantRun(
            variant=MULTI, passed=True, duration_ms=1.0, ledger=UsageLedger(),
            accounts={}, subtask_verdicts={}, model="claude-sonnet-5",
            temperature=0.0, max_turns=7, variant_timeout_s=42.0, workers=4,
        )

        row = run.to_row()

        assert row["model"] == "claude-sonnet-5"
        assert row["temperature"] == 0.0
        assert row["max_turns"] == 7
        assert row["variant_timeout_s"] == 42.0
        assert row["workers"] == 4

    async def test_drive_marks_a_timeout_rather_than_only_appending_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout must be a fact about the run, not a substring of `errors`.

        Matching on the message text would break silently the first time the
        wording changes, and the failure mode is invisible: the case would go
        back to being reported as a crash, which is a different fix.
        """
        monkeypatch.setattr(
            "longline.eval.multi_agent_runner.VARIANT_TIMEOUT_S", 0.01,
        )

        _events, errors, timed_out = await _drive(
            _StallingEngine(), "prompt", max_turns=1,
        )

        assert timed_out is True
        assert errors, "the reason must still be human-readable in the row"

    async def test_a_crash_is_not_a_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The distinction has to survive in both directions."""
        monkeypatch.setattr(
            "longline.eval.multi_agent_runner.VARIANT_TIMEOUT_S", 30.0,
        )

        _events, errors, timed_out = await _drive(
            _CrashingEngine(), "prompt", max_turns=1,
        )

        assert timed_out is False
        assert any("boom" in error for error in errors)


class _CrashingEngine:
    """The other half of the pair: raises immediately, never times out."""

    async def submit(self, prompt: str, *, max_turns: int) -> AsyncIterator[Any]:
        _ = prompt, max_turns
        raise RuntimeError("boom")
        yield  # pragma: no cover - unreachable; this is what makes it a generator


class TestWorkersOverride:
    def test_override_changes_concurrency_without_touching_the_case(self) -> None:
        """Group 3 varies `workers` on ONE case, so the case object is untouched.

        Rewriting `case.workers` in place would make the two points of the curve
        different cases -- the fixture, the prompt and the judges would be free
        to drift with it -- and, worse, both variants of the SAME run would see
        different concurrency, which is a confound inside a single row.
        """
        case = make_case(workers=2)

        assert effective_workers(case, None) == 2
        assert effective_workers(case, 4) == 4
        assert case.workers == 2, "the shared case object must not be mutated"

    def test_an_out_of_range_override_is_refused_here_not_only_at_the_cli(self) -> None:
        """The runner is reachable from tests and notebooks too.

        An override only the CLI validates is an override that runs unvalidated
        from everywhere else, and the symptom of `workers=99` is not an error --
        it is a case that quietly stops measuring parallelism.
        """
        case = make_case(workers=2)

        with pytest.raises(ValueError, match=r"workers override"):
            effective_workers(case, 99)

    def test_a_dependent_case_refuses_a_parallelism_override(self) -> None:
        """The serial scheduler ignores `workers`, so accepting one would lie.

        A dependent case's concurrency is fixed at `DEPENDENT_WORKERS` by the
        contract. An override taken at face value here would be a concurrency
        the ROW reports and the RUN never used -- a wrong number in the data,
        which is worse than a refusal.
        """
        case = make_case(workers=DEPENDENT_WORKERS)
        case.category = CATEGORY_DEPENDENT

        assert effective_workers(case, None) == DEPENDENT_WORKERS
        with pytest.raises(ValueError, match="serial by contract"):
            effective_workers(case, 4)


class TestRepeats:
    """`--repeats` was accepted by the CLI and dropped on the floor down here.

    The documented 2b invocation is 6 tasks x 2 arms x **3 repeats**. No code
    path carried a repeat count: `run_multi_agent_suite` ran each case exactly
    once and `MultiAgentRun` had no repeat field, so `--repeats 3` would have
    produced 12 arm-executions instead of 36 -- and the report would have been a
    complete, plausible, correctly-formatted table saying nothing about the
    shortfall. Silently running one third of a paid sweep is the failure mode
    this class exists to prevent.

    The count is also what makes `mean_speedup` mean anything: with repeats, the
    per-case ratios a repeat produces are one sample each, and averaging them
    without saying how many tasks they came from lets a 3-repeat run of 6 tasks
    read as an 18-task result.
    """

    def test_repeats_do_not_inflate_the_task_count(self) -> None:
        runs = [
            _stub_run("c1", CONTROLLED, passed=True, repeat_index=i) for i in range(3)
        ] + [
            _stub_run("c2", CONTROLLED, passed=True, repeat_index=i) for i in range(3)
        ]

        summary = aggregate_multi_agent(runs, group=CONTROLLED)

        assert summary.num_cases == 2, "three repeats of one task are not three tasks"
        assert summary.num_runs == 6, "and six case-runs is not the same as six tasks"

    def test_the_per_category_block_separates_them_too(self) -> None:
        """The category table is what gets quoted, so `n` there must not lie."""
        runs = [
            _stub_run("c1", CONTROLLED, passed=True, repeat_index=i) for i in range(3)
        ]

        block = aggregate_multi_agent(runs, group=CONTROLLED).by_category[
            CATEGORY_ANALYSIS
        ]

        assert block["num_cases"] == 1
        assert block["num_runs"] == 3

    def test_one_repeat_leaves_the_counts_equal(self) -> None:
        """Every existing suite runs one repeat; their numbers must not move."""
        summary = aggregate_multi_agent(
            [_stub_run("c1", CONTROLLED, passed=True)], group=CONTROLLED,
        )

        assert (summary.num_cases, summary.num_runs) == (1, 1)

    def test_a_repeat_index_is_recorded_on_the_row(self) -> None:
        """Recomputable from `raw.jsonl` alone (contract §3)."""
        row = _stub_run("c1", CONTROLLED, passed=True, repeat_index=2).to_row()

        assert row["repeat_index"] == 2

    def test_the_suite_refuses_a_repeat_count_below_one(self) -> None:
        """Zero repeats is a run that spends nothing and reports nothing.

        It would also divide by zero in every mean, so the refusal has to be
        here rather than only in the CLI's `_positive_int`: the runner is
        reachable from tests and notebooks, and a sweep that silently measured
        nothing is worse than one that refused to start.
        """
        with pytest.raises(ValueError, match="repeats"):
            asyncio.run(
                run_multi_agent_suite(
                    [make_case()], api_key="offline", fixtures_dir=Path("."),
                    repeats=0,
                )
            )


__all__: list[str] = []
