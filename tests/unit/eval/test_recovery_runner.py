"""Unit tests for `longline/eval/recovery_runner.py` and the resume worker.

Offline and deterministic. The central claims:

1. `success` is derived, and a run that injected nothing can never be one.
2. The acceptance control holds: with recovery off, the fault still fires and
   nothing recovers.
3. Process Kill really kills a child, really resumes in a fresh interpreter, and
   never touches the operator's real `~/.claude`.

The last one is asserted by hashing the real directory before and after.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from longline.eval.faults import (
    CONTEXT_OVERFLOW,
    OUTPUT_TRUNCATE,
    PROCESS_KILL,
    RATE_LIMIT,
    TOOL_FAILURE,
)
from longline.eval.recovery import (
    CWD_PLACEHOLDER,
    RecoveryCase,
    cases_by_fault,
    expand_case,
    extract_fixture_path,
    load_recovery_cases,
    resolve_cwd,
)
from longline.eval.recovery_runner import (
    PER_CASE_FIELDS,
    RecoveryRun,
    aggregate_recovery,
    build_kill_spec,
    latency_percentiles,
    recovery_succeeded,
    run_recovery_case,
)
from longline.eval.recovery_worker import (
    CWD_TAG,
    SESSION_ID,
    build_checkpoint_transcript,
    check_transcript_structure,
    derive_answer,
    prepare,
    resume,
    write_fixture,
)
from longline.eval.types import CaseParseError
from longline.models.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from longline.models.messages import AssistantMessage, UserMessage

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "evals" / "fixtures"
DATASET = REPO / "evals" / "recovery.jsonl"


def _case(case_id: str = "rec-429") -> RecoveryCase:
    cases = load_recovery_cases(DATASET)
    return next(c for c in cases if c.id == case_id)


def _run(**overrides: Any) -> RecoveryRun:
    """A finished run, as the runner would produce it.

    `success` is DERIVED through `recovery_succeeded`, exactly as
    `run_recovery_case` derives it. Setting it by hand would let a test assert
    against a field the production path never assigns that way.
    """
    base: dict[str, Any] = {
        "case_id": "c", "fault": RATE_LIMIT, "fault_injected": True,
        "retry_count": 1, "checkpoint_loaded": False, "transcript_repaired": False,
        "duplicate_persisted_tool_calls": 0, "recovery_latency_ms": 1.0, "passed": True,
    }
    base.update(overrides)
    run = RecoveryRun(**base)
    run.success = recovery_succeeded(run)
    return run


# --- the success predicate ---


class TestRecoverySucceeded:
    def test_all_conditions_met_is_a_success(self) -> None:
        assert recovery_succeeded(_run()) is True

    def test_no_injection_is_never_a_success(self) -> None:
        """The single most important rule in this task.

        A run that injected nothing and passed the judge is an ordinary success.
        Counting it as a recovery would inflate the rate by the share of cases
        that happen to be easy.
        """
        assert recovery_succeeded(_run(fault_injected=False)) is False

    def test_a_fault_that_was_injected_but_not_handled_fails(self) -> None:
        assert recovery_succeeded(_run(retry_count=0)) is False

    def test_a_failed_judge_fails_even_when_the_fault_recovered(self) -> None:
        assert recovery_succeeded(_run(passed=False)) is False

    def test_process_kill_also_needs_the_checkpoint(self) -> None:
        assert recovery_succeeded(
            _run(fault=PROCESS_KILL, checkpoint_loaded=False)
        ) is False
        assert recovery_succeeded(
            _run(fault=PROCESS_KILL, checkpoint_loaded=True)
        ) is True

    def test_process_kill_needs_a_structurally_valid_transcript(self) -> None:
        assert recovery_succeeded(_run(
            fault=PROCESS_KILL, checkpoint_loaded=True,
            structural_errors=["tool_use without tool_result: ['tu-1']"],
        )) is False

    def test_runtime_classes_do_not_need_a_checkpoint(self) -> None:
        """`checkpoint_loaded` is a Process-Kill-only requirement."""
        assert recovery_succeeded(_run(fault=OUTPUT_TRUNCATE, checkpoint_loaded=False)) is True


# --- per-case fields ---


class TestPerCaseFields:
    def test_row_carries_every_contract_field(self) -> None:
        row = _run().to_row()
        for field in PER_CASE_FIELDS:
            assert field in row, f"raw.jsonl row is missing {field}"

    def test_the_six_contract_fields_are_exactly_these(self) -> None:
        assert set(PER_CASE_FIELDS) == {
            "fault_injected", "retry_count", "checkpoint_loaded",
            "transcript_repaired", "duplicate_persisted_tool_calls",
            "recovery_latency_ms",
        }


# --- aggregation ---


class TestAggregateRecovery:
    def test_runtime_and_resume_are_separate_denominators(self) -> None:
        runs = [
            _run(fault=RATE_LIMIT), _run(fault=OUTPUT_TRUNCATE),
            _run(fault=PROCESS_KILL, checkpoint_loaded=True),
        ]
        summary = aggregate_recovery(runs)
        assert summary.runtime_recovery_rate.denominator == 2
        assert summary.session_resume_rate.denominator == 1

    def test_an_empty_class_reports_unmeasured_not_zero(self) -> None:
        summary = aggregate_recovery([_run(fault=RATE_LIMIT)])
        assert summary.by_fault[PROCESS_KILL].denominator == 0
        assert summary.by_fault[PROCESS_KILL].value is None

    def test_uninjected_runs_stay_in_the_denominator(self) -> None:
        """The fault never firing is a finding; dropping the case would hide it."""
        runs = [_run(fault=RATE_LIMIT), _run(fault=RATE_LIMIT, fault_injected=False)]
        summary = aggregate_recovery(runs)
        assert summary.runtime_recovery_rate.denominator == 2
        assert summary.runtime_recovery_rate.numerator == 1

    def test_failures_are_listed(self) -> None:
        summary = aggregate_recovery([_run(fault=RATE_LIMIT, fault_injected=False)])
        assert len(summary.failures) == 1
        assert summary.failures[0]["fault"] == RATE_LIMIT

    def test_latency_counts_successful_runs_only(self) -> None:
        """A failed recovery's time-to-give-up is a different quantity."""
        runs = [
            _run(fault=RATE_LIMIT, recovery_latency_ms=10.0),
            _run(fault=RATE_LIMIT, recovery_latency_ms=999.0, passed=False),
        ]
        summary = aggregate_recovery(runs)
        assert summary.recovery_latency_ms[RATE_LIMIT] == 10.0


class TestLatencyPercentiles:
    def test_reports_mean_p50_p95(self) -> None:
        runs = [_run(recovery_latency_ms=v) for v in (1.0, 2.0, 3.0)]
        stats = latency_percentiles(runs)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["p50"] is not None and stats["p95"] is not None

    def test_unmeasured_when_nothing_succeeded(self) -> None:
        stats = latency_percentiles([_run(passed=False)])
        assert stats == {"mean": None, "p50": None, "p95": None}


# --- dataset ---


class TestRecoveryDataset:
    def test_expands_to_the_contract_matrix(self) -> None:
        """Five runtime classes x 10 = 50, plus Process Kill x 10 = 10.

        These are the two denominators the contract fixes, so the dataset must
        produce exactly them. An extra definition for any class would move a
        denominator and silently change what the reported rate is over.
        """
        cases = load_recovery_cases(DATASET)
        grouped = cases_by_fault(cases)

        assert len(cases) == 60
        for fault, group in grouped.items():
            assert len(group) == 10, f"{fault} has {len(group)} runs, expected 10"

        runtime = [c for c in cases if c.fault != PROCESS_KILL]
        assert len(runtime) == 50, "RuntimeRecoveryRate's denominator"
        assert len(grouped[PROCESS_KILL]) == 10, "SessionResumeRate's denominator"
        assert len({c.id.split("-r")[0] for c in cases}) == 6, "one definition per class"

    def test_expansion_gives_unique_ids(self) -> None:
        cases = load_recovery_cases(DATASET)
        ids = [c.id for c in cases]
        assert len(set(ids)) == len(ids)
        assert len(ids) == 60

    def test_every_runtime_case_names_a_reachable_call_index(self) -> None:
        for case in load_recovery_cases(DATASET):
            if case.fault == PROCESS_KILL:
                continue
            assert case.inject_at_call_indices, f"{case.id} has no injection point"
            assert all(1 <= i <= 2 for i in case.inject_at_call_indices)

    def test_tool_failure_case_names_a_tool(self) -> None:
        for case in load_recovery_cases(DATASET):
            if case.fault == TOOL_FAILURE:
                assert case.fault_tool
            else:
                assert case.fault_tool is None

    def test_overflow_case_carries_a_seed_history(self) -> None:
        """Reactive compact needs something to compress."""
        for case in load_recovery_cases(DATASET):
            if case.fault == CONTEXT_OVERFLOW:
                assert case.seed_history

    def test_every_task_names_a_fixture_the_agent_can_open(self) -> None:
        for case in load_recovery_cases(DATASET):
            assert extract_fixture_path(case.task) is not None, case.id

    def test_kill_task_carries_the_cwd_placeholder(self) -> None:
        for case in load_recovery_cases(DATASET):
            if case.fault == PROCESS_KILL:
                assert CWD_PLACEHOLDER in case.task

    def test_an_overflow_case_without_a_seed_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="seed_history"):
            RecoveryCase.from_dict({
                "id": "x", "type": "recovery", "fault": CONTEXT_OVERFLOW,
                "task": "read <cwd>/notes/value.txt", "answer": "a",
                "checks": [{"fn": "file_exists", "args": {"path": "answer.txt"}}],
            })

    def test_a_tool_failure_case_without_a_tool_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="fault_tool"):
            RecoveryCase.from_dict({
                "id": "x", "type": "recovery", "fault": TOOL_FAILURE,
                "task": "read <cwd>/notes/value.txt", "answer": "a",
                "checks": [{"fn": "file_exists", "args": {"path": "answer.txt"}}],
            })

    def test_an_out_of_range_call_index_is_rejected(self) -> None:
        """Index 0 would never fire, so the case could never be counted."""
        with pytest.raises(CaseParseError, match=r"outside 1\.\.2"):
            RecoveryCase.from_dict({
                "id": "x", "type": "recovery", "fault": RATE_LIMIT,
                "task": "read <cwd>/notes/value.txt", "answer": "a",
                "inject_at_call_indices": [0],
                "checks": [{"fn": "file_exists", "args": {"path": "answer.txt"}}],
            })

    def test_a_missing_answer_is_rejected(self) -> None:
        with pytest.raises(CaseParseError, match="answer"):
            RecoveryCase.from_dict({
                "id": "x", "type": "recovery", "fault": RATE_LIMIT,
                "task": "read <cwd>/notes/value.txt",
                "checks": [{"fn": "file_exists", "args": {"path": "answer.txt"}}],
            })

    def test_expand_case_keeps_the_config_and_varies_the_id(self) -> None:
        case = RecoveryCase(
            id="k", task="t <cwd>/a.txt", fault=RATE_LIMIT, repeat=3,
            answer="a", inject_at_call_indices=[1, 2],
            checks=[{"fn": "file_exists", "args": {"path": "answer.txt"}}],
        )
        expanded = expand_case(case)
        assert [c.id for c in expanded] == ["k", "k-r1", "k-r2"]
        assert all(c.inject_at_call_indices == [1, 2] for c in expanded)

    def test_resolve_cwd_substitutes_the_placeholder(self) -> None:
        assert resolve_cwd("see <cwd>/a.txt", "/tmp/x") == "see /tmp/x/a.txt"


# --- the worker: transcript plumbing ---


class TestWorkerTranscript:
    def test_clean_transcript_reports_no_repair(self) -> None:
        """The positive direction of the `report=` out-parameter.

        Ends on a USER message: a transcript ending on an assistant message is
        itself a structural defect (`normalize_messages_for_api` has to invent
        a message to satisfy the API's alternation rule), so it would not be a
        clean fixture to start from.
        """
        msgs: list[Any] = [
            UserMessage(content="hi"),
            AssistantMessage(content=[TextBlock(text="hello")]),
            UserMessage(content="thanks"),
        ]
        valid, errors = check_transcript_structure(msgs)
        assert valid is True
        assert errors == []

    def test_orphaned_tool_use_is_structurally_invalid(self) -> None:
        msgs: list[Any] = [
            UserMessage(content="go"),
            AssistantMessage(content=[ToolUseBlock(id="tu-1", name="Bash", input={})]),
        ]
        valid, errors = check_transcript_structure(msgs)
        assert valid is False
        assert any("tool_use without tool_result" in e for e in errors)

    def test_transcript_ending_on_an_assistant_message_is_invalid(self) -> None:
        msgs: list[Any] = [
            UserMessage(content="hi"),
            AssistantMessage(content=[TextBlock(text="bye")]),
        ]
        valid, errors = check_transcript_structure(msgs)
        assert valid is False
        assert any("ends on an assistant" in e for e in errors)

    def test_orphan_tool_result_is_invalid(self) -> None:
        msgs: list[Any] = [
            UserMessage(content="hi"),
            UserMessage(content=[ToolResultBlock(tool_use_id="ghost", content="x")]),
        ]
        valid, errors = check_transcript_structure(msgs)
        assert valid is False
        assert any("without tool_use" in e for e in errors)

    def test_checkpoint_transcript_does_not_contain_the_answer(self) -> None:
        """Otherwise the judge would pass on a fact the harness planted.

        The transcript carries the RAW marker (`value=...`); the answer is
        `marker=...`, which `derive_answer` produces by re-reading the fixture
        file after the resume. A leg that could only replay its own history
        would therefore produce the wrong string.
        """
        msgs = build_checkpoint_transcript("marker=alpha-7f3c", value="alpha-7f3c", cwd="/tmp/x")
        rendered = json.dumps([m.to_api_dict() for m in msgs])
        assert "marker=alpha-7f3c" not in rendered
        assert "value=alpha-7f3c" in rendered

    def test_checkpoint_transcript_carries_the_cwd(self) -> None:
        msgs = build_checkpoint_transcript("a", value="v", cwd="/tmp/work")
        rendered = json.dumps([m.to_api_dict() for m in msgs])
        assert f"{CWD_TAG}/tmp/work" in rendered

    def test_derive_answer_reads_the_task_not_the_spec(self, tmp_path: Path) -> None:
        """The trap from Task 4: solvability must come from the task text."""
        write_fixture(tmp_path, value="alpha-7f3c")
        msgs = build_checkpoint_transcript("x", value="alpha-7f3c", cwd=str(tmp_path))
        task = f"Read {CWD_PLACEHOLDER}/notes/value.txt and report it."

        assert derive_answer(msgs, task) == "marker=alpha-7f3c"
        # A task naming a different file gets nothing -- proving the answer is
        # derived from the path in the task, not from a value held aside.
        assert derive_answer(msgs, f"Read {CWD_PLACEHOLDER}/notes/other.txt") is None

    def test_derive_answer_is_none_without_the_transcript(self, tmp_path: Path) -> None:
        write_fixture(tmp_path, value="alpha-7f3c")
        assert derive_answer([], f"Read {CWD_PLACEHOLDER}/notes/value.txt") is None


# --- the worker: prepare and resume ---


class TestWorkerPhases:
    def test_prepare_writes_a_checkpoint_that_reloads(self, tmp_path: Path) -> None:
        report = prepare({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "answer": "marker=alpha-7f3c", "value": "alpha-7f3c",
        })
        assert report["checkpoint_saved"] is True
        assert report["session_file_exists"] is True
        assert report["tasks_file_exists"] is True
        assert report["num_reloaded_messages"] == report["num_committed_messages"]
        # The raw form, not the answer: the transcript holds what the TOOL
        # returned, and the answer is formatted from the fixture after resume.
        assert report["stable_fact"] == "value=alpha-7f3c"

    def test_resume_loads_and_reports_a_clean_transcript(self, tmp_path: Path) -> None:
        prepare({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "answer": "marker=alpha-7f3c", "value": "alpha-7f3c",
        })
        report = resume({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "task": f"Read {CWD_PLACEHOLDER}/notes/value.txt",
        })
        assert report["found"] is True
        assert report["checkpoint_loaded"] is True
        assert report["transcript_repaired"] is False
        assert report["repair_kinds"] == []
        assert report["structurally_valid"] is True
        assert report["structural_errors"] == []
        assert report["transcript_unchanged"] is True
        assert report["derived_answer"] == "marker=alpha-7f3c"

    def test_resume_restores_the_task_snapshot_and_kills_non_terminal_tasks(
        self, tmp_path: Path,
    ) -> None:
        """The contract forbids claiming a background task resumes in place."""
        prepare({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "answer": "a", "value": "v",
        })
        report = resume({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "task": f"Read {CWD_PLACEHOLDER}/notes/value.txt",
        })
        assert report["task_snapshot_loaded"] is True
        assert report["task_states"] == {"b-1a2b3c4d": "killed"}

    def test_resume_of_a_missing_session_is_a_clean_failure(self, tmp_path: Path) -> None:
        report = resume({"claude_dir": str(tmp_path), "session_id": "nope"})
        assert report["found"] is False
        assert report["checkpoint_loaded"] is False

    def test_resume_repairs_a_deliberately_truncated_transcript(self, tmp_path: Path) -> None:
        """The repair path, exercised through the production function."""
        import longline.session.storage as storage

        msgs: list[Any] = [
            UserMessage(content="go"),
            AssistantMessage(content=[ToolUseBlock(id="tu-9", name="Bash", input={})]),
        ]
        storage.save_session(SESSION_ID, msgs, claude_dir=tmp_path)
        report = resume({"claude_dir": str(tmp_path), "session_id": SESSION_ID})
        assert report["transcript_repaired"] is True
        assert "truncated_tail" in report["repair_kinds"]
        assert report["orphaned_tool_use_ids"] == ["tu-9"]
        assert report["structurally_valid"] is True
        assert report["transcript_unchanged"] is False


# --- the real ~/.claude is never touched ---


def _hash_tree(root: Path) -> str | None:
    """A digest of a directory's contents, or None when it does not exist."""
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class TestRealClaudeDirIsUntouched:
    """Every fixture writes to a temp claude_dir; prove the real one is untouched.

    `get_sessions_dir(None)` falls back to `Path.home() / ".longline"`, so a
    single dropped argument anywhere in the worker chain would write benchmark
    sessions into the operator's real state directory. The digest is taken
    before and after and compared.
    """

    def test_prepare_and_resume_leave_the_real_dir_byte_identical(self, tmp_path: Path) -> None:
        real = Path.home() / ".longline"
        before = _hash_tree(real)

        prepare({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "answer": "marker=alpha-7f3c", "value": "alpha-7f3c",
        })
        resume({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "task": f"Read {CWD_PLACEHOLDER}/notes/value.txt",
        })

        assert _hash_tree(real) == before, (
            f"the real state directory {real} changed during a worker run"
        )

    def test_sessions_were_written_into_the_temp_dir_instead(self, tmp_path: Path) -> None:
        """The positive half: proving nothing changed is only meaningful if
        something was supposed to be written somewhere."""
        prepare({
            "claude_dir": str(tmp_path), "session_id": SESSION_ID,
            "answer": "a", "value": "v",
        })
        sessions = tmp_path / "sessions"
        assert (sessions / f"{SESSION_ID}.jsonl").is_file()
        assert (sessions / f"{SESSION_ID}.tasks.json").is_file()

    def test_the_kill_spec_never_names_the_home_directory(self, tmp_path: Path) -> None:
        spec = build_kill_spec(
            claude_dir=tmp_path, session_id=SESSION_ID, answer="a", task="t",
        )
        assert Path(spec["claude_dir"]) == tmp_path
        assert ".claude" not in spec["claude_dir"]


# --- end-to-end offline runs through the real engine ---


class TestRuntimeCasesEndToEnd:
    """Real `QueryEngine`, real tools, real `query_loop`, real judge."""

    @pytest.mark.parametrize("case_id", [
        "rec-429", "rec-529", "rec-tool-failure", "rec-truncate", "rec-overflow",
    ])
    async def test_case_recovers_offline(self, case_id: str) -> None:
        run = await run_recovery_case(
            _case(case_id), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert run.fault_injected is True, f"{case_id}: the fault never fired"
        assert run.retry_count > 0, f"{case_id}: the recovery path never ran"
        assert run.passed is True, f"{case_id}: {run.judge_detail}"
        assert run.success is True

    async def test_tool_failure_retries_at_the_tool_level(self) -> None:
        """The agent must actually call the tool, see the error, and try again."""
        run = await run_recovery_case(
            _case("rec-tool-failure"), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert run.model_call_index >= 2
        assert "tool_fault_counter" in run.injection_proof

    async def test_a_run_that_injects_nothing_is_not_a_success(self) -> None:
        """The anti-fabrication rule, end to end.

        A 429 case pointed at a call index the loop never reaches produces a
        clean task success. It must NOT be counted as a recovery.
        """
        case = RecoveryCase(
            id="rec-unreachable",
            task="Read <cwd>/notes/value.txt and report the marker.",
            fault=RATE_LIMIT,
            inject_at_call_indices=[2],
            answer="marker=alpha-7f3c",
            checks=[{
                "fn": "file_content",
                "args": {"path": "answer.txt", "contains": r"marker=alpha\-7f3c"},
            }],
        )
        run = await run_recovery_case(case, api_key="offline", fixtures_dir=FIXTURES)
        assert run.passed is True, "the task itself succeeds: nothing was broken"
        assert run.fault_injected is False
        assert run.success is False, "a clean run must never count as a recovery"
        assert any("never injected" in n for n in run.notes)

    async def test_a_failure_in_the_wrong_place_does_not_pass(self) -> None:
        """Mutation check: break the artifact, the judge must reject it."""
        case = RecoveryCase(
            id="rec-mutated",
            task="Read <cwd>/notes/value.txt and report the marker.",
            fault=RATE_LIMIT,
            inject_at_call_indices=[1],
            answer="marker=alpha-7f3c",
            checks=[{
                "fn": "file_content",
                "args": {"path": "answer.txt", "contains": r"a value that never appears"},
            }],
        )
        run = await run_recovery_case(case, api_key="offline", fixtures_dir=FIXTURES)
        assert run.passed is False
        assert run.success is False


class TestAcceptanceControl:
    """Plan §4.4: with the recovery strategy disabled the test must fail.

    `drive_query_loop(disable_recovery=True)` passes `max_retry=0` and
    `max_max_output_recovery=0`, which are the two knobs the recovery arms read.
    Both directions are asserted from the same fixture, so a green result cannot
    come from the fault simply not firing.
    """

    async def _drive(self, case: RecoveryCase, *, disable: bool) -> Any:
        """Drive the loop with the back-off recorder injected, not monkeypatched.

        `sleep=_no_sleep` is a `query_loop` parameter. Patching `asyncio.sleep`
        would be process-global -- that name lives on the shared `asyncio`
        module -- and would leak into unrelated coroutines in the same process.
        """
        from longline.eval.faults import ModelFaultInjector, drive_query_loop
        from longline.eval.recovery_runner import _scripted_compact_fn, seed_messages

        inj = ModelFaultInjector(
            fault=case.fault, answer=case.answer,
            at_call_indices=tuple(case.inject_at_call_indices),
        )
        return inj, await drive_query_loop(
            inj, messages=seed_messages(case),
            auto_compact_fn=_scripted_compact_fn(),
            disable_recovery=disable, sleep=_no_sleep,
        )

    @pytest.mark.parametrize("case_id", ["rec-429", "rec-truncate", "rec-overflow"])
    async def test_disabled_recovery_prevents_recovery(self, case_id: str) -> None:
        case = _case(case_id)
        inj_off, out_off = await self._drive(case, disable=True)
        assert inj_off.injected is True, "the fault must still fire with recovery off"
        assert inj_off.recovered_at_call_index is None
        assert case.answer not in out_off.text

    @pytest.mark.parametrize("case_id", ["rec-429", "rec-truncate", "rec-overflow"])
    async def test_enabled_recovery_recovers(self, case_id: str) -> None:
        case = _case(case_id)
        inj_on, out_on = await self._drive(case, disable=False)
        assert inj_on.injected is True
        assert inj_on.recovered_at_call_index is not None
        assert case.answer in out_on.text


# --- Process Kill, end to end ---


class TestProcessKillEndToEnd:
    """Subprocess kill + fresh-interpreter resume; a few seconds per test."""

    async def test_kill_then_resume_recovers(self) -> None:
        run = await run_recovery_case(
            _case("rec-kill"), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert run.fault_injected is True
        assert run.checkpoint_loaded is True
        assert run.transcript_repaired is False
        assert run.structural_errors == []
        assert run.passed is True, run.judge_detail
        assert run.success is True

    async def test_the_task_snapshot_was_restored(self) -> None:
        run = await run_recovery_case(
            _case("rec-kill"), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert run.task_states == {"b-1a2b3c4d": "killed"}

    async def test_no_duplicate_persisted_tool_call(self) -> None:
        run = await run_recovery_case(
            _case("rec-kill"), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert run.duplicate_persisted_tool_calls == 0
        assert run.side_effect_classification == "clean_resume"

    async def test_exactly_once_is_not_claimed(self) -> None:
        """The report must say plainly what it does not prove."""
        run = await run_recovery_case(
            _case("rec-kill"), api_key="offline", fixtures_dir=FIXTURES,
        )
        assert any("exactly-once is NOT claimed" in n for n in run.notes)

    async def test_the_temp_claude_dir_is_removed(self) -> None:
        import tempfile

        from longline.eval.recovery_runner import CLAUDE_DIR_PREFIX

        before = set(Path(tempfile.gettempdir()).glob(f"{CLAUDE_DIR_PREFIX}*"))
        await run_recovery_case(_case("rec-kill"), api_key="offline", fixtures_dir=FIXTURES)
        after = set(Path(tempfile.gettempdir()).glob(f"{CLAUDE_DIR_PREFIX}*"))
        assert after == before, "the harness leaked a claude_dir"


def test_the_recovery_worker_is_runnable_as_a_module() -> None:
    """The kill path shells out to `-m longline.eval.recovery_worker`."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "longline.eval.recovery_worker", "--help"],
        capture_output=True, text=True, cwd=str(REPO), timeout=60, check=False,
    )
    assert proc.returncode == 0
    assert "prepare" in proc.stdout


async def _no_sleep(_seconds: float) -> None:
    return None


def test_no_sleep_is_awaitable() -> None:
    assert asyncio.iscoroutinefunction(_no_sleep)
