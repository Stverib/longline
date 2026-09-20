"""Integration tests: the multi-agent suite driven end to end, offline.

These exercise the real wiring rather than one function at a time: the loader
reads the committed `evals/multi_agent.jsonl` with its fixture-identity check
on, each case runs both arms against real `QueryEngine`s, the fan-out goes
through the production `spawn_teammate`, the judges are the real deterministic
ones, and the aggregate produces the contract's metrics with the right
denominators.

The cost note is the same one the recovery integration file carries: this does
NOT call the API. `run_multi_agent_case(model=None)` replaces the model
transport with a scripted one, so the whole file runs in seconds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from longline.eval.multi_agent import (
    CONTROLLED,
    EXPLORATORY,
    MAX_WORKERS,
    MIN_WORKERS,
    group_of,
    load_multi_agent_cases,
)
from longline.eval.multi_agent_runner import (
    aggregate_multi_agent,
    run_multi_agent_case,
    run_multi_agent_suite,
)
from longline.models.messages import Usage

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "evals" / "multi_agent.jsonl"
FIXTURES = REPO / "evals" / "fixtures"

# The same usage for every turn of every agent, so a mis-attributed turn changes
# the totals in a way arithmetic can catch: a run's tokens are then exactly
# `turns * (input + output)` and the per-agent split is exactly the turn split.
TURN_USAGE = Usage(input_tokens=10, output_tokens=2)


def _hash_tree(root: Path) -> str | None:
    """Digest of a directory's contents, or None when it does not exist."""
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _claude_dir(tmp_path: Path) -> Path:
    """A per-test temp claude_dir.

    Never the user's `~/.longline`: `TeammateMailbox` and `add_member` default
    to it, so a dropped argument would write benchmark team files into the
    operator's real state directory.
    """
    path = tmp_path / "claude"
    path.mkdir()
    return path


@pytest.mark.asyncio
async def test_the_controlled_slice_produces_the_contract_metrics(
    tmp_path: Path,
) -> None:
    """A small controlled slice, both arms, aggregated once."""
    cases = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[:3]

    runs = await run_multi_agent_suite(
        cases, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )
    assert len(runs) == 3

    summary = aggregate_multi_agent(runs, group=CONTROLLED)
    assert summary.num_cases == 3
    # Every case is eligible: both arms completed and both accounted for their
    # agents. An ineligible one here means the harness, not the model.
    assert summary.eligible_cases == 3, [
        (r.case_id, r.exclusion_reason, r.single.errors, r.multi.errors,
         r.multi.accounting_error)
        for r in runs
    ]

    for run in runs:
        assert run.single.passed, f"{run.case_id} single: {run.single.judge_detail}"
        assert run.multi.passed, f"{run.case_id} multi: {run.multi.judge_detail}"
        assert run.single.accounting_error == ""
        assert run.multi.accounting_error == ""

    # Contract §5.6 wants both variants' tokens and tool calls reported, each
    # with its own numerator. The fan-out must cost strictly more here: it runs
    # the same work through more agents, and each agent pays for its own turns.
    assert summary.multi_tokens["total_tokens"] > summary.single_tokens["total_tokens"]
    assert summary.multi_tokens["child_tokens"] > 0
    assert summary.single_tokens["child_tokens"] == 0, (
        "the single arm has no children, so it can have no child tokens"
    )


@pytest.mark.asyncio
async def test_child_tokens_are_counted_from_the_turn_arithmetic(
    tmp_path: Path,
) -> None:
    """The totals equal `turns x per-turn cost`, including every child's turns.

    With a constant per-turn usage the run's token total is fully determined by
    how many turns each agent ran. Comparing the ledger against that product --
    computed from the ledger's OWN turn list and the usage the test supplied --
    catches a dropped child, because a dropped child removes its turns from the
    list AND from the total, and the two would then agree with each other while
    both being wrong. So the check that has teeth is the per-agent one: every
    worker the case spawned must appear with at least one turn.
    """
    case = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[0]
    run = await run_multi_agent_case(
        case, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    per_turn = TURN_USAGE.input_tokens + TURN_USAGE.output_tokens
    ledger = run.multi.ledger
    assert ledger.total_tokens == ledger.turns_count * per_turn
    assert ledger.child_tokens() == sum(
        u.total_tokens for name, u in ledger.per_agent().items() if name != "leader"
    )

    workers = [name for name in ledger.per_agent() if name != "leader"]
    assert len(workers) >= MIN_WORKERS
    assert all(ledger.per_agent()[name].turns > 0 for name in workers)
    # The case declares four subtasks, so however many workers ran there must be
    # at least one turn-bearing worker per subtask -- a worker with no turns
    # would have produced no artifact, and the judge would have failed it.
    assert len(workers) >= case.workers


@pytest.mark.asyncio
async def test_the_fan_out_uses_the_real_swarm_path(tmp_path: Path) -> None:
    """The workers are real teammates, not bare coroutines.

    Checked by what the production path leaves behind: each worker's
    `InProcessTeammate` sends its result to the leader's inbox through
    `TeammateMailbox`, which writes a file under the temp `claude_dir`. A
    `gather` over plain coroutines would pass every numeric assertion in this
    file and leave no inbox file, so this is the assertion that distinguishes a
    measurement of the product from a measurement of a reimplementation.

    The inbox (not the team file) is the witness because `spawn_teammate` only
    best-effort registers members -- `add_member` raises when the team itself
    was never created, and the spawn deliberately swallows that warning. The
    mailbox send happens inside the teammate's own `_execute_with_query_loop`,
    after its query loop has run, so a file there is evidence that a real
    `InProcessTeammate` really executed.
    """
    from longline.swarm.identity import TEAM_LEAD_NAME
    from longline.swarm.mailbox import TeammateMailbox

    case = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[0]
    claude_dir = _claude_dir(tmp_path)
    run = await run_multi_agent_case(
        case, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=claude_dir, usage=TURN_USAGE,
    )
    assert run.multi.passed, run.multi.judge_detail

    inbox = claude_dir / "teams" / case.id / "inboxes" / f"{TEAM_LEAD_NAME}.json"
    assert inbox.is_file(), (
        "no teammate reported to the leader's inbox: the fan-out did not run "
        "through InProcessTeammate"
    )
    messages = json.loads(inbox.read_text(encoding="utf-8"))
    senders = {entry["from"] for entry in messages}
    # One teammate per SUBTASK, not per worker slot: `workers` is the
    # concurrency limit, so a four-subtask case with two workers runs four
    # teammates in two waves. Expecting `case.workers` senders here was the
    # first version of this assertion, and it encoded the wrong model of the
    # fan-out -- which is exactly the bug `_spawn_workers` was fixed for.
    expected_senders = {f"worker{i}" for i in range(1, case.num_subtasks + 1)}
    assert senders == expected_senders, (
        f"{senders} reported; the case declares {case.num_subtasks} subtasks, "
        "so every one of them must have been executed"
    )
    # The mailbox is the production channel, so reading it back through the
    # production reader is the same claim without hand-parsing the file.
    assert len(TeammateMailbox(case.id, claude_dir=claude_dir).receive(TEAM_LEAD_NAME)) >= 1


@pytest.mark.asyncio
async def test_the_two_groups_are_reported_separately(tmp_path: Path) -> None:
    """Aggregating one group never counts the other (contract §5.6)."""
    cases = load_multi_agent_cases(DATASET)
    one_controlled = group_of(cases, CONTROLLED)[0]
    one_exploratory = group_of(cases, EXPLORATORY)[0]

    runs = await run_multi_agent_suite(
        [one_controlled, one_exploratory], api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    controlled = aggregate_multi_agent(runs, group=CONTROLLED)
    exploratory = aggregate_multi_agent(runs, group=EXPLORATORY)
    assert controlled.num_cases == 1
    assert exploratory.num_cases == 1
    assert {r["case_id"] for r in controlled.per_case} == {one_controlled.id}
    assert {r["case_id"] for r in exploratory.per_case} == {one_exploratory.id}
    # The pooled summary sees both, and says so in its own group label rather
    # than pretending to be either.
    assert aggregate_multi_agent(runs).num_cases == 2


@pytest.mark.asyncio
async def test_the_sibling_fixtures_are_untouched_by_a_run(tmp_path: Path) -> None:
    """`evals/fixtures/` must be byte-identical after a suite run.

    The runner copies into a temp sandbox, so this is the check that the copy is
    really a copy: a case that resolved its fixture path back to the repo would
    modify the committed tree, and every later run would start from a different
    state.
    """
    watched = [FIXTURES / "parallel_repo_single", FIXTURES / "parallel_repo_multi"]
    before = {p: _hash_tree(p) for p in watched}

    cases = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[:2]
    await run_multi_agent_suite(
        cases, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    assert {p: _hash_tree(p) for p in watched} == before


@pytest.mark.asyncio
async def test_the_run_does_not_touch_the_real_state_directory(tmp_path: Path) -> None:
    """`TeammateMailbox` and `add_member` both default to `~/.longline`.

    One dropped `claude_dir` anywhere in the spawn chain writes benchmark team
    files into the operator's real state, so the digest is compared across a
    full case run.
    """
    real = Path.home() / ".longline"
    before = _hash_tree(real)

    case = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[0]
    await run_multi_agent_case(
        case, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    assert _hash_tree(real) == before, f"the real state directory {real} was modified"


@pytest.mark.asyncio
async def test_every_row_is_json_serialisable_and_carries_both_variants(
    tmp_path: Path,
) -> None:
    """`raw.jsonl` must be able to carry the whole comparison.

    Contract §3: every number in the summary has to be recomputable from the raw
    rows alone. That requires both variants' tokens, durations and tool calls on
    the SAME row, so the per-case row -- not the two `CaseResult`s -- is what is
    checked here.
    """
    case = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[0]
    run = await run_multi_agent_case(
        case, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    row = run.to_row()
    json.dumps(row)  # must not raise
    for field in ("duration_ms", "input_tokens", "output_tokens", "total_tokens",
                  "tool_calls", "agent_count"):
        assert field in row["single"], f"single arm row is missing {field}"
        assert field in row["multi"], f"multi arm row is missing {field}"
    assert row["single"]["agent_count"] == 1
    assert row["multi"]["agent_count"] >= case.workers
    # And the derived ratios are recomputable from that row.
    assert run.speedup == pytest.approx(
        row["single"]["duration_ms"] / row["multi"]["duration_ms"]
    )


@pytest.mark.asyncio
async def test_repeats_rerun_the_case_and_stamp_the_index(tmp_path: Path) -> None:
    """驱真实 runner 的重复, 而不是只测聚合的函数签名.

    `repeats` 一路从 CLI 走到 `run_multi_agent_suite`, 中间任何一段丢掉它,
    症状都不是报错 -- 是一条跑了 1 次却报成 3 次的记录. 所以这里断言的是
    **真的跑了 3 遍**: 三次运行各自的 `repeat_index` 依次为 0/1/2, 且聚合把
    它们算作 1 个任务, 3 次运行.
    """
    cases = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[:1]

    runs = await run_multi_agent_suite(
        cases, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE, repeats=3,
    )

    assert len(runs) == 3, "three repeats of one case is three runs"
    assert [r.repeat_index for r in runs] == [0, 1, 2]
    assert {r.case_id for r in runs} == {cases[0].id}

    summary = aggregate_multi_agent(runs, group=CONTROLLED)
    assert summary.num_cases == 1
    assert summary.num_runs == 3
    assert summary.eligible_runs == 3


def test_the_dataset_is_committed_and_well_formed() -> None:
    """A number whose dataset is missing is not a number (`evals/README.md` §3)."""
    assert DATASET.is_file()
    lines = [ln for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert 15 <= len(lines) <= 30, "contract §5.6 asks for 15-30 cases"

    cases = load_multi_agent_cases(DATASET)
    assert len(cases) == len(lines)
    controlled = group_of(cases, CONTROLLED)
    exploratory = group_of(cases, EXPLORATORY)
    assert len(controlled) + len(exploratory) == len(cases), "every case is in exactly one group"
    assert controlled and exploratory, "both groups must be non-empty"

    for case in cases:
        assert MIN_WORKERS <= case.workers <= MAX_WORKERS
        if case.group == CONTROLLED:
            assert case.num_subtasks >= MIN_WORKERS
            # The declared artifacts are distinct, so a concurrent fan-out has
            # nothing to race on (contract §8.3).
            assert len(set(case.subtask_paths())) == case.num_subtasks
            assert case.merge_file not in case.subtask_paths()


def test_the_loader_refuses_two_fixture_trees_that_differ(tmp_path: Path) -> None:
    """The sibling-identity check fires, driven end to end through the loader."""
    from longline.eval.multi_agent import assert_fixtures_identical
    from longline.eval.types import CaseParseError

    root = tmp_path / "fixtures"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        (root / name / "f.txt").write_text("same\n", encoding="utf-8")
    case = group_of(load_multi_agent_cases(DATASET), CONTROLLED)[0]
    identical = type(case)(
        id=case.id, task=case.task, fixture_single="a", fixture_multi="b",
        subtasks=list(case.subtasks), merge_file=case.merge_file, checks=list(case.checks),
    )
    assert_fixtures_identical(identical, root)

    (root / "b" / "f.txt").write_text("different\n", encoding="utf-8")
    with pytest.raises(CaseParseError, match="not the same tree"):
        assert_fixtures_identical(identical, root)


@pytest.mark.asyncio
async def test_a_run_carries_its_cases_category(tmp_path: Path) -> None:
    """The report splits on this field, and a defaulted value looks plausible.

    `MultiAgentRun.category` defaults to `parallel_analysis`, so a run that
    never copied the case's category produces a report with ONE section
    covering an 18-case three-category corpus -- a page that looks entirely
    reasonable and is wrong in the way the split exists to prevent. Found
    exactly that way: `--suite pair --offline` printed a single
    `parallel_analysis` block with `n=18`.

    Driven over a case whose category is NOT the default, so the assertion
    cannot pass by coincidence.
    """
    from longline.eval.multi_agent import CATEGORY_ANALYSIS, CATEGORY_DEPENDENT

    benefit = REPO / "evals" / "multi_agent_benefit.jsonl"
    cases = load_multi_agent_cases(benefit)
    case = next(c for c in cases if c.category == CATEGORY_DEPENDENT)
    assert case.category != CATEGORY_ANALYSIS, "the fixture must not be the default"

    run = await run_multi_agent_case(
        case, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    assert run.category == case.category
    assert run.to_row()["category"] == case.category


@pytest.mark.asyncio
async def test_a_multi_category_aggregate_refuses_a_pooled_figure(
    tmp_path: Path,
) -> None:
    """The gate has to fire on runs the runner actually produced.

    A synthetic summary can carry `by_category` without anything having put it
    there; this drives the corpus and checks the aggregate of REAL runs.
    """
    benefit = REPO / "evals" / "multi_agent_benefit.jsonl"
    cases = load_multi_agent_cases(benefit)
    # One case from each of the first two categories present, rather than a
    # head slice: the corpus is ordered BY category, so `[:6]` is one category
    # and the test would have asserted the opposite of what it means.
    seen: set[str] = set()
    picked = []
    for case in cases:
        if case.category not in seen:
            seen.add(case.category)
            picked.append(case)
    assert len(picked) > 1, "the corpus is meant to span several categories"

    runs = await run_multi_agent_suite(
        picked, api_key="offline", fixtures_dir=FIXTURES,
        claude_dir=_claude_dir(tmp_path), usage=TURN_USAGE,
    )

    summary = aggregate_multi_agent(runs)

    assert len(summary.by_category) > 1
    assert summary.is_pooled is True
    assert summary.to_dict()["pooled"] is None
