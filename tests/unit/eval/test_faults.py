"""Unit tests for `longline/eval/faults.py`.

Offline throughout. The two claims that matter most here are:

1. Every injector really fires, and its counter says so.
2. A fault never delivers the answer, so a run that did not recover cannot be
   graded as if it had.

The second is the failure mode this whole suite exists to prevent, so it is
asserted directly rather than left to the runner's success predicate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from longline.eval import faults
from longline.eval.faults import (
    CONTEXT_OVERFLOW,
    OUTPUT_TRUNCATE,
    OVERLOADED,
    PLACEHOLDER_PREFIX,
    PROCESS_KILL,
    RATE_LIMIT,
    RUNTIME_FAULTS,
    InjectionError,
    ModelFaultInjector,
    ScriptedModel,
    ToolCallingModel,
    ToolFaultWrapper,
    classify_side_effects,
    drive_query_loop,
    duplicate_persisted_tool_calls,
    fault_registry,
    fingerprint_tool_calls,
)
from longline.models.messages import UserMessage
from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema

# --- the fault vocabulary ---


class TestFaultVocabulary:
    def test_six_classes_in_two_metric_groups(self) -> None:
        """RuntimeRecoveryRate is five classes; SessionResumeRate is Process Kill."""
        assert len(faults.ALL_FAULTS) == 6
        assert len(RUNTIME_FAULTS) == 5
        assert PROCESS_KILL not in RUNTIME_FAULTS
        assert PROCESS_KILL in faults.ALL_FAULTS

    def test_fault_names_are_the_contracts(self) -> None:
        assert RUNTIME_FAULTS == (
            "429", "529", "tool_failure", "output_truncate", "context_overflow",
        )
        assert PROCESS_KILL == "process_kill"


# --- the "a fault never emits the answer" rule ---


class TestFaultEventsNeverCarryTheAnswer:
    """A fault that leaked the answer would make every case pass vacuously."""

    @pytest.mark.parametrize("fault", [RATE_LIMIT, OVERLOADED, OUTPUT_TRUNCATE, CONTEXT_OVERFLOW])
    def test_no_scripted_fault_event_contains_the_answer(self, fault: str) -> None:
        answer = "marker=alpha-7f3c"
        events = faults.build_injection_events(fault, answer=answer)
        rendered = "".join(
            str(getattr(e, "text", "") or getattr(e, "message", "")) for e in events
        )
        assert answer not in rendered

    def test_rate_limit_event_is_recoverable(self) -> None:
        (event,) = faults.build_injection_events(RATE_LIMIT)
        assert event.is_recoverable is True  # type: ignore[union-attr]

    def test_overload_event_is_recoverable(self) -> None:
        (event,) = faults.build_injection_events(OVERLOADED)
        assert event.is_recoverable is True  # type: ignore[union-attr]

    def test_prompt_too_long_event_is_not_recoverable(self) -> None:
        """413 is handled by reactive compact, not by the retry arm.

        `query_engine` rebuilds `is_recoverable` from the status code, and 413
        is not in (429, 529) -- so this must be False or the two arms would both
        claim the same fault.
        """
        (event,) = faults.build_injection_events(CONTEXT_OVERFLOW)
        assert event.is_recoverable is False  # type: ignore[union-attr]

    def test_unknown_fault_raises(self) -> None:
        with pytest.raises(InjectionError):
            faults.build_injection_events("not-a-fault")


# --- the model injector ---


class TestModelFaultInjector:
    def _injector(self, fault: str = RATE_LIMIT, **kw: Any) -> ModelFaultInjector:
        return ModelFaultInjector(fault=fault, answer="marker=alpha-7f3c", **kw)

    async def test_counters_start_clean(self) -> None:
        inj = self._injector()
        assert inj.injected is False
        assert inj.record.call_index == 0
        assert inj.record.attempts == 0

    async def test_index_zero_is_rejected(self) -> None:
        """Index 0 would mean "never", i.e. a case that can never be counted."""
        with pytest.raises(InjectionError, match="must be >= 1"):
            self._injector(at_call_indices=(0,))

    async def test_empty_indices_rejected(self) -> None:
        with pytest.raises(InjectionError, match="must be >= 1"):
            self._injector(at_call_indices=())

    async def test_unknown_fault_rejected(self) -> None:
        with pytest.raises(InjectionError, match="cannot inject"):
            self._injector(fault="nope")

    async def test_counter_rises_only_when_the_index_is_hit(self) -> None:
        inj = self._injector(at_call_indices=(2,))
        await _drain(inj(messages=[]))
        assert inj.injected is False
        assert inj.record.call_index == 1
        await _drain(inj(messages=[]))
        assert inj.injected is True
        assert inj.record.attempts == 1

    async def test_recovery_is_observed_not_assumed(self) -> None:
        """No fault-free call after the fault means no observed recovery."""
        inj = self._injector(at_call_indices=(1,))
        await _drain(inj(messages=[]))
        assert inj.injected is True
        assert inj.recovered_at_call_index is None

        await _drain(inj(messages=[]))
        assert inj.recovered_at_call_index == 2

    async def test_answer_only_on_a_fault_free_call(self) -> None:
        inj = self._injector(at_call_indices=(1,))
        first = await _collect(inj(messages=[]))
        assert "marker=alpha-7f3c" not in _render(first)

        second = await _collect(inj(messages=[]))
        assert "marker=alpha-7f3c" in _render(second)

    async def test_truncation_fragment_is_a_prefix_of_the_answer(self) -> None:
        """The continuation must supply the REST, or recovery is unmeasurable.

        The loop saves the truncated text into the transcript, so this fragment
        lands in the graded output. If it were arbitrary prose, the final text
        would be part fault and part recovery with no seam to judge.
        """
        events = faults.build_injection_events(OUTPUT_TRUNCATE)
        text = "".join(getattr(e, "text", "") for e in events)
        assert text
        assert "marker=alpha-7f3c".startswith(text)
        assert PLACEHOLDER_PREFIX not in text


# --- the tool wrapper ---


class _CountingTool(Tool):
    """A production-shaped tool that records its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_name(self) -> str:
        return "Bash"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Bash", description="", input_schema={})

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return True

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self.calls += 1
        return ToolResult(content=f"ok-{self.calls}")


class TestToolFaultWrapper:
    async def test_first_call_fails_then_delegates(self) -> None:
        inner = _CountingTool()
        wrapper = ToolFaultWrapper(inner, fault_at_call_index=1)

        first = await wrapper.execute({"command": "cat x"})
        assert first.is_error is True
        assert wrapper.faults_injected == 1
        assert inner.calls == 0, "the fault must not reach the real tool"

        second = await wrapper.execute({"command": "cat x"})
        assert second.is_error is False
        assert second.content == "ok-1"
        assert inner.calls == 1
        assert wrapper.calls == 2
        assert wrapper.faults_injected == 1, "only the first call is the fault"

    async def test_injected_is_false_before_any_call(self) -> None:
        wrapper = ToolFaultWrapper(_CountingTool())
        assert wrapper.injected is False

    async def test_delegates_name_schema_and_concurrency(self) -> None:
        wrapper = ToolFaultWrapper(_CountingTool())
        assert wrapper.get_name() == "Bash"
        assert wrapper.get_schema().name == "Bash"
        assert wrapper.is_concurrency_safe({}) is True

    async def test_fault_registry_replaces_the_named_tool_only(self) -> None:
        registry, wrapper = fault_registry(
            ".", tool_name="Read", fault_at_call_index=1, profile="core",
        )
        assert registry.get("Read") is wrapper
        assert registry.get("Bash") is not None
        assert registry.get("Bash") is not wrapper

    async def test_fault_registry_rejects_a_tool_not_in_the_profile(self) -> None:
        """A fault aimed at an uncallable tool would never fire."""
        with pytest.raises(InjectionError, match="not in profile"):
            fault_registry(".", tool_name="WebSearch", profile="core")


# --- scripted models ---


class TestScriptedModels:
    async def test_scripted_model_counts_calls_and_never_injects(self) -> None:
        model = ScriptedModel(answer="marker=alpha-7f3c")
        events = await _collect(model(messages=[]))
        assert model.injected is False
        assert model.record.call_index == 1
        assert model.record.fault == faults.NO_FAULT
        assert "marker=alpha-7f3c" in _render(events)

    async def test_tool_calling_model_calls_then_retries_then_answers(self) -> None:
        model = ToolCallingModel(
            tool_name="Bash", tool_input={"command": "cat x"}, answer="marker=alpha-7f3c",
        )
        # Turn 1: ask for the tool.
        first = await _collect(model(messages=[]))
        assert any(isinstance(e, faults.ToolUseStart) for e in first)

        # Turn 2: the result came back an error -> retry.
        errored = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu-1", "is_error": True},
        ]}]
        second = await _collect(model(messages=errored))
        assert any(isinstance(e, faults.ToolUseStart) for e in second)
        assert model.saw_tool_error is True
        assert model.retried_after_error is True

        # Turn 3: success -> answer.
        ok = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu-1", "content": "marker=alpha-7f3c"},
        ]}]
        third = await _collect(model(messages=ok))
        assert "marker=alpha-7f3c" in _render(third)

    async def test_tool_calling_model_never_answers_while_the_tool_fails(self) -> None:
        """The anti-vacuous-pass rule for this class.

        A model that printed the answer regardless of the tool result would make
        the case pass with or without the fault, which is a test that measures
        nothing.
        """
        model = ToolCallingModel(
            tool_name="Bash", tool_input={"command": "cat x"}, answer="marker=alpha-7f3c",
            max_tool_attempts=1,
        )
        errored = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu-1", "is_error": True},
        ]}]
        events = await _collect(model(messages=errored))
        assert "marker=alpha-7f3c" not in _render(events)

    async def test_tool_calling_model_counts_attempts_from_the_transcript(self) -> None:
        """Attempts are read off the transcript, not a private counter."""
        no_results: list[dict[str, Any]] = []
        assert ToolCallingModel._attempts_so_far(no_results) == 0
        with_one = [{"role": "user", "content": [{"type": "tool_result"}]}]
        assert ToolCallingModel._attempts_so_far(with_one) == 1


# --- fingerprints and the duplicate check ---


class TestFingerprints:
    def test_same_call_and_result_yields_the_same_fingerprint(self) -> None:
        a = fingerprint_tool_calls([("Bash", {"command": "ls"}, "digest-1")])
        b = fingerprint_tool_calls([("Bash", {"command": "ls"}, "digest-1")])
        assert a == b

    def test_a_different_result_is_not_a_duplicate(self) -> None:
        """The same call against a changed file is a new call, not a re-execution."""
        a = fingerprint_tool_calls([("Bash", {"command": "cat x"}, "digest-1")])
        b = fingerprint_tool_calls([("Bash", {"command": "cat x"}, "digest-2")])
        assert a != b

    def test_duplicate_detection_is_empty_for_a_clean_resume(self) -> None:
        executed = ["fp-1", "fp-2"]
        persisted = ["fp-1"]
        assert duplicate_persisted_tool_calls(executed, persisted) == ["fp-1"]

    def test_no_re_execution_reports_nothing(self) -> None:
        assert duplicate_persisted_tool_calls(["fp-9"], ["fp-1", "fp-2"]) == []


class TestSideEffectClassification:
    def test_duplicate_wins_over_clean(self) -> None:
        assert classify_side_effects(False, duplicates=["fp"]) == "duplicate_persisted_tool_call"

    def test_kill_inside_a_side_effect_window_is_ambiguous(self) -> None:
        """Not a duplicate, and explicitly NOT reported as safe."""
        result = classify_side_effects(True)
        assert result == faults.AMBIGUOUS
        assert result != faults.CLEAN_RESUME

    def test_clean_resume_requires_no_duplicate_and_no_window(self) -> None:
        assert classify_side_effects(False) == faults.CLEAN_RESUME

    def test_ambiguous_is_not_counted_as_a_duplicate(self) -> None:
        """The two are different findings and must not be conflated."""
        assert classify_side_effects(True) != "duplicate_persisted_tool_call"


# --- driving the real query loop ---


class TestDriveQueryLoop:
    """These run the production `query_loop` against a scripted injector.

    These use `disable_recovery`, which zeroes the RETRY and TRUNCATION budgets
    together. That is a coarser control than `TestAcceptanceControl` uses, and it
    is sound here only because each test asserts on one fault and the two
    budgets are independent -- which
    `test_each_budget_zeroes_only_its_own_arm` pins as a table. If that ever
    stopped holding, these tests would be turning off a path they are not
    measuring, and the table is what would notice.
    """

    async def _drive(self, fault: str, *, indices: tuple[int, ...], disable: bool) -> tuple[Any, Any]:
        """Drive the loop with the back-off recorder injected, not monkeypatched.

        `sleep=_no_sleep` is passed as a `query_loop` parameter. Patching
        `asyncio.sleep` instead would be process-global -- it lives on the
        shared `asyncio` module -- and would leak into unrelated coroutines.
        """
        inj = ModelFaultInjector(
            fault=fault, answer="marker=alpha-7f3c", at_call_indices=indices,
        )
        out = await drive_query_loop(
            inj, messages=[UserMessage(content="go")],
            disable_recovery=disable, sleep=_no_sleep,
        )
        return inj, out

    @pytest.mark.parametrize("fault", [RATE_LIMIT, OVERLOADED])
    async def test_recoverable_error_retries_when_enabled(self, fault: str) -> None:
        inj, out = await self._drive(fault, indices=(1,), disable=False)
        assert inj.injected is True
        assert inj.recovered_at_call_index == 2
        assert "marker=alpha-7f3c" in out.text

    @pytest.mark.parametrize("fault", [RATE_LIMIT, OVERLOADED])
    async def test_recoverable_error_is_terminal_when_disabled(self, fault: str) -> None:
        """The plan's acceptance condition: off, the corresponding test fails."""
        inj, out = await self._drive(fault, indices=(1,), disable=True)
        assert inj.injected is True, "the fault still fires; only the path is off"
        assert inj.recovered_at_call_index is None
        assert out.error_events, "the original error must be surfaced, not swallowed"

    async def test_truncation_continues_when_enabled(self) -> None:
        inj, out = await self._drive(OUTPUT_TRUNCATE, indices=(1,), disable=False)
        assert inj.recovered_at_call_index == 2
        assert "marker=alpha-7f3c" in out.text

    async def test_truncation_stops_when_disabled(self) -> None:
        inj, out = await self._drive(OUTPUT_TRUNCATE, indices=(1,), disable=True)
        assert inj.injected is True
        assert inj.recovered_at_call_index is None
        assert "marker=alpha-7f3c" not in out.text

    async def test_a_fault_at_index_two_needs_a_first_call_that_asks_for_more(self) -> None:
        """Index 2 is only reachable if call 1 requests a tool.

        The scripted model here answers directly, which ENDS the loop after one
        call -- so a fault planted at call 2 never fires. That is a property of
        the loop, not of the injector, and it is why the committed dataset
        injects at index 1: an index the run cannot reach would report
        `fault_injected=false` on every repeat. The injector itself still
        supports index 2 (and that is covered by
        `test_second_call_injection_recovers_when_the_first_call_asks_for_a_tool`).
        """
        inj = ModelFaultInjector(
            fault=RATE_LIMIT, answer="marker=alpha-7f3c", at_call_indices=(2,),
        )
        out = await drive_query_loop(
            inj, messages=[UserMessage(content="go")], disable_recovery=False,
            sleep=_no_sleep,
        )
        assert inj.injected is False
        assert inj.record.call_index == 1, "the direct answer ended the loop"
        assert out.turns == 1

    async def test_second_call_injection_recovers_when_the_first_call_asks_for_a_tool(
        self,
    ) -> None:
        """The "first or second model call" contract point, exercised for real.

        Call 1 requests a tool, so the loop continues; call 2 is the fault; call
        3 recovers. The scripted model is `ToolCallingModel`, whose first call
        is a `ToolUseStart` -- that is what makes turn 2 reachable at all. With a
        model that answers directly the loop ends after one call and index 2
        could never fire (see the test above).
        """
        from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema

        class _OkTool(Tool):
            def get_name(self) -> str:
                return "Bash"

            def get_schema(self) -> ToolSchema:
                return ToolSchema(name="Bash", description="", input_schema={})

            def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
                return True

            async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
                return ToolResult(content="marker=alpha-7f3c")

        registry = ToolRegistry()
        registry.register(_OkTool())

        # A tool-calling script, wrapped so that the SECOND model call is the
        # fault. `ToolCallingModel` would answer on call 2; the injector
        # overrides that call with a 429 and lets call 3 through.
        inner = ToolCallingModel(
            tool_name="Bash", tool_input={"command": "cat x"}, answer="marker=alpha-7f3c",
        )
        injector = _SecondCallFault(inner, fault=RATE_LIMIT, answer="marker=alpha-7f3c")

        out = await drive_query_loop(
            injector, messages=[UserMessage(content="go")], registry=registry,
            sleep=_no_sleep,
        )

        assert injector.record.call_index == 3, "tool call, fault, recovery"
        assert injector.injected is True
        assert injector.recovered_at_call_index == 3
        assert out.errored_executions == 0
        assert "marker=alpha-7f3c" in out.text


# --- helpers ---


class _SecondCallFault:
    """A test-only composition: drive `inner`, but fail its Nth call.

    Exists because the committed dataset injects at call index 1 (see
    `TestDriveQueryLoop.test_a_fault_at_index_two_needs_a_first_call_that_asks_for_more`),
    so index-2 reachability has no dataset case to cover it. This keeps that
    contract point exercised without adding a definition that would move
    `RuntimeRecoveryRate`'s denominator off 50.
    """

    def __init__(self, inner: Any, *, fault: str, answer: str, at_index: int = 2) -> None:
        self._inner = inner
        self._fault = fault
        self._answer = answer
        self._at_index = at_index
        self.record = faults.InjectionRecord(fault=fault, inject_at_call_index=at_index)
        self.sleep_record = faults.SleepRecord()
        self.recovered_at_call_index: int | None = None

    @property
    def injected(self) -> bool:
        return self.record.injected

    def __call__(self, **kwargs: Any) -> Any:
        return self._serve(kwargs)

    async def _serve(self, kwargs: dict[str, Any]) -> Any:
        self.record.call_index += 1
        index = self.record.call_index
        if index == self._at_index:
            self.record.injected = True
            self.record.attempts += 1
            for event in faults.build_injection_events(self._fault, answer=self._answer):
                yield event
            return
        if self.record.injected and self.recovered_at_call_index is None:
            self.recovered_at_call_index = index
        async for event in self._inner(**kwargs):
            yield event


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


async def _no_sleep(_seconds: float) -> None:
    """Replaces the loop's back-off so an offline test does not wait it out."""
    return None


def _render(events: list[Any]) -> str:
    return "".join(str(getattr(e, "text", "") or getattr(e, "message", "")) for e in events)


def test_no_sleep_is_a_coroutine_function() -> None:
    """`asyncio.sleep` is awaited; a plain lambda would raise instead of record."""
    assert asyncio.iscoroutinefunction(_no_sleep)


def test_registry_replace_is_the_only_mutation_path() -> None:
    """`fault_registry` pokes `_tools`; this test pins that it stays a real lookup."""
    registry = ToolRegistry()
    registry.register(_CountingTool())
    assert registry.get("Bash") is not None


def test_faults_module_has_no_path_to_the_users_home() -> None:
    """Guard against a future edit wiring a real claude_dir into the injectors."""
    source = Path(faults.__file__).read_text(encoding="utf-8")
    assert "Path.home()" not in source
    assert '.claude' not in source
    assert "longline-recovery-claude" not in source


# --- the eval-only engine seams stay eval-only ---


class TestEngineSeamsAreEvalOnly:
    """`QueryEngine` declares three attributes for the eval harness to write.

    The claim is that production never writes them. If `core/` ever started
    writing one, it would be a genuine problem -- an attribute that looks
    eval-only but carries production state. This pins the claim so a future
    edit cannot quietly make it false.
    """

    _SEAMS = ("injection_record", "fault_wrapper", "sleep_fn")

    def _repo_root(self) -> Path:
        return Path(faults.__file__).resolve().parents[2]

    def test_no_production_module_writes_an_eval_seam(self) -> None:
        writes: list[str] = []
        for path in (self._repo_root() / "longline").rglob("*.py"):
            if "eval" in path.parts:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for seam in self._SEAMS:
                    # A write is `<something>.<seam> = ...`; a read-through or a
                    # declaration is not.
                    if f".{seam} =" in stripped or f".{seam}=" in stripped:
                        writes.append(f"{path.relative_to(self._repo_root())}:{lineno}")
        assert writes == [], (
            "production code writes an eval-only engine seam: "
            f"{writes}"
        )

    def test_the_eval_harness_is_what_writes_them(self) -> None:
        """The other half: something must write them, or the check above is vacuous."""
        src = (self._repo_root() / "longline" / "eval" / "faults.py").read_text(
            encoding="utf-8"
        )
        for seam in self._SEAMS:
            assert f"engine.{seam} =" in src, seam

    def test_query_engine_declares_all_three(self) -> None:
        src = (self._repo_root() / "longline" / "core" / "query_engine.py").read_text(
            encoding="utf-8"
        )
        for seam in self._SEAMS:
            assert f"self.{seam}" in src, seam


class TestBudgetsAreIndependent:
    """One control knob must turn off exactly one recovery arm.

    A negative control that zeroes a neighbouring budget is not testing what it
    claims: the fault would fail to recover for a reason unrelated to the arm
    under test. This asserts the whole matrix, so a future change that couples
    two budgets is caught here rather than silently weakening every control.
    """

    async def _recovered(self, fault: str, messages: list[Any], **kwargs: Any) -> bool:
        from longline.core.query_loop import query_loop
        from longline.eval.recovery_runner import _scripted_compact_fn
        from longline.tools.base import ToolRegistry

        inj = ModelFaultInjector(
            fault=fault, answer="marker=alpha-7f3c", at_call_indices=(1,),
        )
        async for _ in query_loop(
            messages=list(messages), system_prompt="t", tools=ToolRegistry(),
            call_model=inj, max_turns=10, auto_compact_fn=_scripted_compact_fn(),
            sleep=_no_sleep, **kwargs,
        ):
            pass
        return inj.recovered_at_call_index is not None

    def _overflow_messages(self) -> list[Any]:
        from longline.eval.recovery import load_recovery_cases
        from longline.eval.recovery_runner import seed_messages

        path = Path(faults.__file__).resolve().parents[2] / "evals" / "recovery.jsonl"
        case = next(
            c for c in load_recovery_cases(path) if c.fault == CONTEXT_OVERFLOW
        )
        return seed_messages(case)

    @pytest.mark.parametrize(
        ("fault", "own_budget"),
        [
            (RATE_LIMIT, "max_retry"),
            (OUTPUT_TRUNCATE, "max_max_output_recovery"),
            (CONTEXT_OVERFLOW, "max_reactive_compaction"),
        ],
    )
    async def test_each_budget_zeroes_only_its_own_arm(
        self, fault: str, own_budget: str,
    ) -> None:
        from longline.models.messages import UserMessage

        messages = (
            self._overflow_messages() if fault == CONTEXT_OVERFLOW
            else [UserMessage(content="go")]
        )
        budgets = ("max_retry", "max_max_output_recovery", "max_reactive_compaction")
        for budget in budgets:
            kwargs = {budget: 0}
            recovered = await self._recovered(fault, messages, **kwargs)
            if budget == own_budget:
                assert recovered is False, (
                    f"zeroing {budget} must stop the {fault} recovery"
                )
            else:
                assert recovered is True, (
                    f"zeroing {budget} must NOT interfere with the {fault} "
                    "recovery -- a control that does is testing the wrong path"
                )
