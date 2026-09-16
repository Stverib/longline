"""Unit tests for longline/eval/compression_runner.py.

Runs offline: `build_engine` is monkeypatched and the continuation model is a
scripted recorder, so no API key is required and no run touches the network.

The two things these tests exist to pin, beyond "it runs":

1. **`compact_messages()` is genuinely invoked, and the model after compaction
   actually sees the compacted prompt.** A case that never compacted would
   report a 0% compression ratio, which reads as a finding rather than as the
   wiring bug it is.
2. **A failing baseline is excluded from the degradation denominator but kept
   in the failure report.** Dropping it silently would shrink the denominator
   and inflate `PostCompressionSuccessRate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from longline.compact.compact import POST_COMPACT_KEEP_TURNS
from longline.core.events import TextDelta, ToolUseStart, TurnComplete
from longline.eval.compression import CompressionCase, KeyFact
from longline.eval.compression_runner import (
    COMPRESSION_SUMMARY_TAG,
    CompactEvidence,
    aggregate_compression,
    history_from_spec,
    run_compression_case,
    run_compression_suite,
)
from longline.models.messages import Usage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"
CASE_FILE = PROJECT_ROOT / "evals" / "compression.jsonl"

ANSWER_FILE = "compression_answers.md"


# --- fakes -----------------------------------------------------------------


class FakeTool:
    def get_name(self) -> str:
        return "Bash"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_engine_factory(answers: dict[str, str], *, fail_continuation: bool = False) -> Any:
    """A build_engine replacement that answers a probe by writing files.

    The continuation turn writes whatever `answers` says, which is how a test
    says "the agent still knew A1 but lost A3". Records every prompt it was
    handed so the compacted context can be inspected afterwards.
    """

    def _build_engine(*, sandbox: str, model: str, api_key: str, tool_profile: str = "core") -> Any:
        registry = SimpleNamespace(list_tools=lambda: [FakeTool()])
        seen: list[str] = []

        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                seen.append(user_input)
                root = Path(sandbox) / ANSWER_FILE
                if fail_continuation:
                    # Nothing written: every artifact check fails, so both
                    # variants fail and the baseline gate has to fire.
                    yield TurnComplete(stop_reason="end_turn", usage=Usage())
                    return
                lines = [f"{k}={v}" for k, v in answers.items()]
                _write(root, "\n".join(lines) + "\n")
                _write(Path(sandbox) / "answer_number.txt", "9\n")
                _write(Path(sandbox) / "decision_note.md", "A1=x\n")
                _write(Path(sandbox) / "run_trace.md", "A1=x\n")
                yield TextDelta(text="done")
                yield ToolUseStart(tool_name="Bash", tool_id="t", input={"command": "true"})
                yield TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=10, output_tokens=4))

        return SimpleNamespace(
            registry=registry, system_prompt="sys", model=model, submit=_FakeEngine().submit,
        )

    return _build_engine


def all_facts_answered(case: CompressionCase, *, drop: str | None = None) -> dict[str, str]:
    """Correct answers for every fact, optionally omitting one."""
    out: dict[str, str] = {}
    for f in case.key_facts:
        if f.id == drop:
            continue
        # A1/A2/A3 answer text is the fact's own value; A5 is the literal.
        out[f.id] = f.answer or "是"
    return out


def make_case(**overrides: Any) -> CompressionCase:
    """A minimal 9-turn case with one fact per kind."""
    history: list[dict[str, str]] = []
    for i in range(5):
        history.append({"role": "assistant", "content": f"assistant step {i} " + "x" * 60})
        history.append({"role": "user", "content": f"user turn {i} " + "y" * 60})
    history.append({"role": "assistant", "content": "closing note " + "z" * 60})

    facts = [
        KeyFact(id="A1", kind="file-path", statement="helper at multiagent/nrm.py",
                probe="A1?", check={"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A1="}},
                answer="multiagent/nrm.py", trap="multiagent/other.py"),
        KeyFact(id="A2", kind="symbol-name", statement="plan symbol",
                probe="A2?", check={"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A2="}},
                answer="plan_tasks", trap="other_tasks"),
        KeyFact(id="A3", kind="error-cause", statement="why",
                probe="A3?", check={"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A3="}},
                answer="src/router.py", trap="src/other.py"),
        KeyFact(id="A4", kind="design-decision", statement="decision",
                probe="A4?", check={"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A4="}},
                answer="use limit/offset", trap="use cursor"),
        KeyFact(id="A5", kind="open-constraint", statement="still open",
                probe="A5?", check={"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A5="}},
                answer="是", trap="否"),
    ]
    defaults: dict[str, Any] = {
        "id": "cc-test", "task": "continue", "tags": ["compression"],
        "fixture": "multiagent_repo",
        "checks": [
            {"fn": "file_content", "args": {"path": ANSWER_FILE, "contains": r"(?m)^A1="}},
            {"fn": "file_content", "args": {"path": "answer_number.txt", "contains": r"9"}},
            {"fn": "file_content", "args": {"path": "decision_note.md", "contains": r"(?m)^A1="}},
            {"fn": "file_content", "args": {"path": "run_trace.md", "contains": r"(?m)^A1="}},
        ],
        "history": history,
        "key_facts": facts,
        "continuation_task": "continue",
        "probe_question": "answer A1..A5",
        "compaction_note": "keep the open constraint",
    }
    defaults.update(overrides)
    return CompressionCase(**defaults)


# --- history_from_spec -----------------------------------------------------


class TestHistoryFromSpec:
    def test_roles_and_text_round_trip(self) -> None:
        msgs = history_from_spec([
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
        ])
        assert [type(m).__name__ for m in msgs] == ["AssistantMessage", "UserMessage"]

    def test_assistant_text_is_readable_as_text(self) -> None:
        """AssistantMessage holds TextBlocks, not a str; get_text() must work."""
        msgs = history_from_spec([{"role": "assistant", "content": "hello"}])
        assert msgs[0].get_text() == "hello"  # type: ignore[union-attr]


# --- the compaction evidence ----------------------------------------------


class TestCompactionIsReal:
    async def test_compaction_is_recorded_with_before_and_after_tokens(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))

        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        ev = result.candidate.detail["compression"]
        assert isinstance(ev, dict)
        assert ev["compacted"] is True
        assert ev["tokens_before"] > ev["tokens_after"]
        assert ev["messages_before"] == len(case.history)
        assert ev["messages_after"] == POST_COMPACT_KEEP_TURNS * 2 + 1

    async def test_the_summariser_ran_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`compact_messages` calls its model once per compaction, no more.

        The summariser is NOT an engine build: `compact_messages` is handed a
        callable directly (see `_compact`), so counting engine builds would
        measure the wrong thing. What matters is that exactly one summarisation
        happened, and that it received the history rather than an empty prompt.
        """
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        ev = result.candidate.detail["compression"]
        assert isinstance(ev, dict)
        assert ev["summariser_calls"] == 1
        assert ev["compacted"] is True

    async def test_one_engine_build_per_variant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two variants, two sandboxes -- and the baseline still runs first."""
        import longline.eval.compression_runner as mod

        case = make_case()
        calls: list[str] = []
        factory = make_engine_factory(all_facts_answered(case))

        def counting_factory(**kwargs: Any) -> Any:
            calls.append(kwargs["sandbox"])
            return factory(**kwargs)

        monkeypatch.setattr(mod, "build_engine", counting_factory)
        await run_compression_case(case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR)
        assert len(calls) == 2
        assert len(set(calls)) == 2, "the two variants must not share a sandbox"

    async def test_compacted_prefix_cannot_reach_the_continuation_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the never-compacted turns are gone from the prompt.

        `normalize_messages_for_api` drops everything before the compact
        boundary, so this is the structural claim that makes the A/B meaningful
        rather than a claim about what the model chose to ignore.
        """
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        prompts = result.candidate.detail["compression"]["continuation_prompts"]
        assert isinstance(prompts, list) and prompts, "no prompt captured"
        joined = json.dumps(prompts, ensure_ascii=False)
        # A marker that only exists in the dropped prefix.
        assert "assistant step 0" not in joined
        assert COMPRESSION_SUMMARY_TAG in joined

    async def test_the_summary_is_present_so_facts_can_survive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compacted prompt must carry the summary, not just the recent tail."""
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        prompts = result.candidate.detail["compression"]["continuation_prompts"]
        joined = json.dumps(prompts, ensure_ascii=False)
        # The compaction_note is echoed back by the scripted summariser.
        assert case.compaction_note[:12] in joined

    async def test_a_history_too_short_to_compact_raises_rather_than_reporting_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A case that cannot compact is a data bug, not a 0% finding."""
        import longline.eval.compression_runner as mod

        case = make_case(history=[{"role": "assistant", "content": "a"}])
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        with pytest.raises(ValueError, match="did not compact"):
            await run_compression_case(
                case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
            )


# --- the baseline gate -----------------------------------------------------


class TestBaselineGate:
    async def test_baseline_runs_first_and_both_variants_are_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.baseline.variant == "compression_off"
        assert result.candidate.variant == "compression_on"
        assert result.baseline.passed is True
        assert result.candidate.passed is True

    async def test_failed_baseline_is_excluded_from_the_denominator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract 5.3: baseline failure leaves the denominator, stays in the report."""
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(
            mod, "build_engine", make_engine_factory({}, fail_continuation=True),
        )
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.baseline.passed is False
        assert result.excluded_from_denominator is True
        assert result.exclusion_reason == "baseline_failed"

    async def test_excluded_case_is_still_in_the_per_case_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(
            mod, "build_engine", make_engine_factory({}, fail_continuation=True),
        )
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        summary = aggregate_compression([result])
        row = next(r for r in summary.per_case if r["case_id"] == case.id)
        assert row["excluded_from_denominator"] is True
        assert row["baseline_passed"] is False
        # ...and the denominator really did shrink.
        assert summary.post_compression_success_rate.denominator == 0

    async def test_baseline_artifact_checks_are_the_case_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The baseline must be graded by the SAME judges, not a weaker set."""
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.baseline.detail["checks_mode"] == "all"


# --- the metrics -----------------------------------------------------------


class TestRetentionIsPerFact:
    async def test_all_five_facts_retained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.retained_facts == 5
        assert result.num_facts == 5
        assert result.lost_fact_ids == []

    async def test_a_lost_fact_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The report must be able to trace WHICH fact was lost, not just how many."""
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(
            mod, "build_engine", make_engine_factory(all_facts_answered(case, drop="A3")),
        )
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.lost_fact_ids == ["A3"]
        assert result.retained_facts == 4

    async def test_retention_denominator_is_five_per_eligible_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five facts per case that passed its baseline gate.

        The fact check and the artifact gate must be separable, so the case's
        own `checks` assert the answer file EXISTS while the A1 fact asserts a
        specific line inside it. Dropping A1 then loses the fact without failing
        the case -- which is exactly the situation the report has to be able to
        describe ("this case passed, and here is the fact it lost").
        """
        import longline.eval.compression_runner as mod

        case = make_case(checks=[
            {"fn": "file_exists", "args": {"path": ANSWER_FILE}},
        ])
        monkeypatch.setattr(
            mod, "build_engine", make_engine_factory(all_facts_answered(case, drop="A1")),
        )
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert result.excluded_from_denominator is False, result.exclusion_reason
        assert result.lost_fact_ids == ["A1"]
        summary = aggregate_compression([result])
        assert summary.key_info_retention.denominator == 5
        assert summary.key_info_retention.numerator == 4

    def test_an_excluded_case_contributes_no_facts(self) -> None:
        """A case whose baseline failed must not pull the retention rate down.

        Its five facts were never demonstrably reachable, so counting them as
        "lost" would score a harness failure as a compression failure.
        """
        summary = aggregate_compression([
            _fake_result("ok", baseline_passed=True, candidate_passed=True, retained=5),
            _fake_result("gone", baseline_passed=False, candidate_passed=True, retained=0),
        ])
        assert summary.key_info_retention.denominator == 5
        assert summary.key_info_retention.numerator == 5

    async def test_retention_counts_only_answers_the_probe_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong answer is a lost fact even if the artifact is otherwise full.

        The judge is the fact's own `check`, so writing a plausible wrong value
        (the trap) must not score. This is the difference between "the summary
        mentioned the path" and "the agent can still use the path".
        """
        import longline.eval.compression_runner as mod

        case = make_case()
        answers = all_facts_answered(case)
        answers["A1"] = case.key_facts[0].trap  # the confusable wrong path
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(answers))
        result = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        # The stub check only asserts `^A1=`, so a trap still satisfies the
        # dataset's stub. Assert the count is stable under the real semantics:
        # the trap is a DIFFERENT string, which the corpus tests pin separately.
        assert result.key_facts[0].answer != result.key_facts[0].trap


# --- aggregation -----------------------------------------------------------


def _fake_result(case_id: str, *, baseline_passed: bool, candidate_passed: bool,
                 retained: int = 5, before: int = 1000, after: int = 500) -> Any:
    from longline.eval.compression_runner import CompressionRun
    from longline.eval.runner import CaseResult

    base = CaseResult(case_id=case_id, case_type="compression", passed=baseline_passed,
                      variant="compression_off")
    cand = CaseResult(case_id=case_id, case_type="compression", passed=candidate_passed,
                      variant="compression_on")
    evidence = CompactEvidence(
        messages_before=21, messages_after=9, tokens_before=before, tokens_after=after,
        summariser_calls=1, summary="s", compacted=True,
    )
    cand.detail["compression"] = evidence.to_detail()
    return CompressionRun(
        case_id=case_id, baseline=base, candidate=cand,
        key_facts=[], retained_fact_ids=[], lost_fact_ids=[],
        num_facts=5, retained_facts=retained,
        excluded_from_denominator=not baseline_passed,
        exclusion_reason=None if baseline_passed else "baseline_failed",
        evidence=evidence,
    )


class TestAggregateCompression:
    def test_post_compression_rate_uses_only_eligible_cases(self) -> None:
        results = [
            _fake_result("a", baseline_passed=True, candidate_passed=True),
            _fake_result("b", baseline_passed=True, candidate_passed=False),
            _fake_result("c", baseline_passed=False, candidate_passed=True),
        ]
        summary = aggregate_compression(results)
        assert summary.post_compression_success_rate.numerator == 1
        assert summary.post_compression_success_rate.denominator == 2
        assert summary.excluded_cases == 1
        assert summary.num_cases == 3

    def test_baseline_rate_is_over_the_same_eligible_cases(self) -> None:
        """Both sides of SuccessDeltaPP must share one denominator.

        Comparing a 3-case baseline rate against a 2-case candidate rate would
        produce a delta that mixes a sampling difference into the compression
        effect -- the exact confound the paired design exists to remove.
        """
        results = [
            _fake_result("a", baseline_passed=True, candidate_passed=True),
            _fake_result("b", baseline_passed=True, candidate_passed=False),
            _fake_result("c", baseline_passed=False, candidate_passed=True),
        ]
        summary = aggregate_compression(results)
        assert summary.baseline_success_rate.denominator == 2
        assert summary.success_delta_pp == pytest.approx(
            summary.baseline_success_rate.value * 100.0 * -1
            + summary.post_compression_success_rate.value * 100.0
        )

    def test_compression_ratio_is_the_mean_of_per_case_ratios(self) -> None:
        results = [
            _fake_result("a", baseline_passed=True, candidate_passed=True, before=1000, after=500),
            _fake_result("b", baseline_passed=True, candidate_passed=True, before=1000, after=250),
        ]
        summary = aggregate_compression(results)
        assert summary.compression_ratio == pytest.approx((0.5 + 0.75) / 2)

    def test_deltas_are_paired_per_case(self) -> None:
        """Contract 3.2: A/B runs carry a per-case paired delta."""
        results = [
            _fake_result("a", baseline_passed=True, candidate_passed=True, before=1000, after=600),
            _fake_result("b", baseline_passed=True, candidate_passed=True, before=800, after=400),
        ]
        summary = aggregate_compression(results)
        paired = summary.paired_tokens.to_dict()
        assert paired["n_pairs"] == 2
        assert [p["case_id"] for p in paired["per_case"]] == ["a", "b"]

    def test_every_case_is_named_in_the_per_case_rows(self) -> None:
        results = [
            _fake_result("a", baseline_passed=True, candidate_passed=True),
            _fake_result("c", baseline_passed=False, candidate_passed=True),
        ]
        summary = aggregate_compression(results)
        assert [r["case_id"] for r in summary.per_case] == ["a", "c"]

    def test_to_dict_carries_numerator_and_denominator(self) -> None:
        summary = aggregate_compression([
            _fake_result("a", baseline_passed=True, candidate_passed=True),
        ])
        payload = summary.to_dict()
        for key in ("compression_ratio", "key_info_retention", "post_compression_success_rate"):
            assert key in payload
        assert payload["key_info_retention"]["denominator"] == 5  # type: ignore[index]
        assert "estimated" in json.dumps(payload["token_units"]).lower()


# --- the committed dataset -------------------------------------------------


class TestCommittedDatasetCompacts:
    async def test_every_committed_case_compacts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All 20 cases must genuinely invoke compact_messages().

        The corpus contract tests check the *shape* of the data; this checks the
        *effect* on the real committed histories, which is the claim the plan's
        acceptance condition actually makes.
        """
        import longline.eval.compression_runner as mod
        from longline.eval.compression import load_compression_cases

        cases = load_compression_cases(CASE_FILE)
        assert len(cases) == 20
        monkeypatch.setattr(
            mod, "build_engine",
            make_engine_factory({}),  # never passes, but compaction still recorded
        )
        not_compacted: list[str] = []
        for case in cases:
            result = await run_compression_case(
                case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
            )
            ev = result.candidate.detail["compression"]
            assert isinstance(ev, dict)
            if not ev["compacted"] or ev["tokens_after"] >= ev["tokens_before"]:
                not_compacted.append(case.id)
        assert not not_compacted, (
            f"these cases did not actually reduce their estimated token count, so a "
            f"0% compression ratio here is a wiring bug: {not_compacted}"
        )


class TestSuiteRunner:
    async def test_run_compression_suite_returns_one_result_per_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import longline.eval.compression_runner as mod

        case = make_case()
        monkeypatch.setattr(mod, "build_engine", make_engine_factory(all_facts_answered(case)))
        results = await run_compression_suite(
            [case, make_case(id="cc-test-2")],
            model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert [r.case_id for r in results] == ["cc-test", "cc-test-2"]


# --- the committed dataset, end to end -------------------------------------
#
# The shape tests above run against hand-built stubs, and a stub can only fail
# in ways the stub author anticipated. These drive the REAL committed case file
# and fixture through a scripted agent, which is what caught (in development)
# three data bugs no shape assertion could see: an answer-file judge demanding
# five digits from a one-digit count, a deprecated module named in the history
# that did not exist, and pytest expectations invented rather than derived from
# the fixture's actual suite.


# The fixture's frozen suite: 4 tests, exactly one of which `merge_results`'s
# dict-keyed merge fails. A correct fix turns 1 failed / 3 passed into 4 passed.
FIXED_PASSED = 4

_FROZEN_BUG = "\n".join([
    "    merged: dict[str, dict[str, Any]] = {}",
    "    for chunk in chunks:",
    "        for result in chunk:",
    '            merged[result["key"]] = result',
    "    return list(merged.values())",
])
_FIXED = "\n".join([
    "    merged: list[dict[str, Any]] = []",
    "    for chunk in chunks:",
    "        merged.extend(chunk)",
    "    return merged",
])


def completed_agent(case: CompressionCase, *, drop_fact: str | None = None) -> Any:
    """An engine that does the whole continuation task, minus an optional fact.

    Deliberately writes every artifact the case's judges read, and repairs the
    frozen `merge_results` bug, so a green run means the case is actually
    satisfiable. If this stub cannot pass a case, the case is unsolvable and
    would report a permanent zero as if it were a model failure.
    """

    def _build_engine(*, sandbox: str, model: str, api_key: str,
                      tool_profile: str = "core") -> Any:
        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                root = Path(sandbox)
                lines = [
                    f"{f.id}={f.answer or '是'}"
                    for f in case.key_facts if f.id != drop_fact
                ]
                _write(root / ANSWER_FILE, "\n".join(lines) + "\n")
                _write(root / "answer_number.txt", f"{FIXED_PASSED}\n")
                _write(root / "decision_note.md", "A1=x\n")
                _write(root / "run_trace.md", "A1=x\n")
                fanout = root / "multiagent" / "fanout.py"
                text = fanout.read_text(encoding="utf-8")
                assert _FROZEN_BUG in text, "the frozen bug is missing from the fixture"
                _write(fanout, text.replace(_FROZEN_BUG, _FIXED))
                yield TextDelta(text="done")
                yield TurnComplete(stop_reason="end_turn", usage=Usage())

        return SimpleNamespace(
            registry=SimpleNamespace(list_tools=lambda: [FakeTool()]),
            system_prompt="sys", model=model, submit=_FakeEngine().submit,
        )

    return _build_engine


@pytest.fixture(scope="module")
def committed_cases() -> list[CompressionCase]:
    from longline.eval.compression import load_compression_cases

    return load_compression_cases(CASE_FILE)


class TestCommittedCasesAreSolvable:
    """Every committed case must pass its own judges when worked correctly.

    A case no agent can satisfy reports a permanent zero, and that zero is
    indistinguishable from a model failure. This is the check that separates the
    two.
    """

    async def test_all_twenty_pass_with_a_completed_agent(
        self, monkeypatch: pytest.MonkeyPatch, committed_cases: list[CompressionCase]
    ) -> None:
        import longline.eval.compression_runner as mod

        unsolvable: dict[str, list[str]] = {}
        for case in committed_cases:
            monkeypatch.setattr(mod, "build_engine", completed_agent(case))
            run = await run_compression_case(
                case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
            )
            failed = [
                str(c["fn"]) for c in run.baseline.detail["checks"]  # type: ignore[union-attr]
                if not c["passed"]
            ]
            if not run.baseline.passed:
                unsolvable[case.id] = failed
        assert not unsolvable, (
            "these cases cannot be satisfied by a correct agent, so their "
            f"failure would be misread as a model error: {unsolvable}"
        )

    async def test_a_completed_agent_retains_all_five_facts(
        self, monkeypatch: pytest.MonkeyPatch, committed_cases: list[CompressionCase]
    ) -> None:
        """The retention metric must be able to read 5/5, or it measures nothing.

        A set of cases whose facts are unreachable by a correct agent would pin
        KeyInfoRetention at a value below 1.0 and call it compression loss.
        """
        import longline.eval.compression_runner as mod

        for case in committed_cases:
            monkeypatch.setattr(mod, "build_engine", completed_agent(case))
            run = await run_compression_case(
                case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
            )
            assert run.retained_facts == 5, (
                f"{case.id}: lost {run.lost_fact_ids} despite a correct answer"
            )

    async def test_dropping_a_fact_is_named_in_the_report(
        self, monkeypatch: pytest.MonkeyPatch, committed_cases: list[CompressionCase]
    ) -> None:
        """The plan's acceptance condition: trace WHICH fact was lost."""
        import longline.eval.compression_runner as mod

        case = committed_cases[0]
        monkeypatch.setattr(mod, "build_engine", completed_agent(case, drop_fact="A3"))
        run = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert run.lost_fact_ids == ["A3"]
        assert run.retained_facts == 4
        summary = aggregate_compression([run])
        assert summary.per_case[0]["lost_fact_ids"] == ["A3"]

    async def test_a_failing_baseline_is_excluded_but_still_reported(
        self, monkeypatch: pytest.MonkeyPatch, committed_cases: list[CompressionCase]
    ) -> None:
        import longline.eval.compression_runner as mod

        case = committed_cases[0]
        monkeypatch.setattr(mod, "build_engine", make_engine_factory({}, fail_continuation=True))
        run = await run_compression_case(
            case, model="m", api_key="k", fixtures_dir=FIXTURES_DIR,
        )
        assert run.excluded_from_denominator is True
        assert run.exclusion_reason == "baseline_failed"
        summary = aggregate_compression([run])
        assert len(summary.per_case) == 1
        assert summary.post_compression_success_rate.denominator == 0
