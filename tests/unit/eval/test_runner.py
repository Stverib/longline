"""Unit tests for longline/eval/runner.py — case runner + suite orchestrator.

Runs offline: build_engine is monkeypatched to a fake that yields scripted
events, so no API key is required.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from longline.core.events import TextDelta, ToolResultReady, ToolUseStart, TurnComplete
from longline.eval.runner import run_case, run_suite
from longline.eval.types import E2ECase, ToolCallCase
from longline.models.messages import Usage


class FakeTool:
    def get_name(self) -> str:
        return "Bash"


def _fake_engine_factory(events: list[Any], *, raises: BaseException | None = None) -> Any:
    """Bootstrap a fake build_engine replacement via monkeypatch."""

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        registry = SimpleNamespace(list_tools=lambda: [FakeTool()])
        system = "test"

        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                for event in events:
                    yield event
                if raises is not None:
                    raise raises

        return SimpleNamespace(
            registry=registry,
            system_prompt=system,
            model=model,
            submit=_FakeEngine().submit,
        )

    return _build_engine


async def test_run_tool_call_case_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Read", tool_id="t", input={"file_path": "/tmp/a.py"}),
        TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=3)),
    ]))
    case = ToolCallCase(
        id="tc-001", task="read a file", expect_tools=["Read"],
        expect_args={"Read": {"file_path": r"a\.py"}},
    )
    result = await run_case(case, model="m", api_key="k", fixtures_dir=Path("does-not-exist"))
    assert result.passed is True
    assert result.turns == 1
    assert result.input_tokens == 5
    assert result.output_tokens == 3


async def test_run_tool_call_case_fails_on_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Grep", tool_id="t", input={"pattern": "x"}),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-002", task="grep then read", expect_tools=["Read"])
    result = await run_case(case, model="m", api_key="k", fixtures_dir=Path("does-not-exist"))
    assert result.passed is False
    assert result.detail["tool_subsequence_ok"] is False


async def test_run_e2e_case_copies_fixture_and_judges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    fixtures = tmp_path / "fixtures"
    (fixtures / "simple_repo").mkdir(parents=True)
    (fixtures / "simple_repo" / "README.md").write_text("hello fixture\n", encoding="utf-8")

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TextDelta(text="done"),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))

    case = E2ECase(
        id="e2e-001", task="do the thing", fixture="simple_repo",
        judge={"fn": "file_content", "args": {"path": "README.md", "contains": "fixture"}},
    )
    result = await run_case(case, model="m", api_key="k", fixtures_dir=fixtures)
    assert result.passed is True
    assert result.text == "done"


async def test_run_suite_runs_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ToolUseStart(tool_name="Read", tool_id="t", input={}),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    cases = [
        ToolCallCase(id="a", task="t1", expect_tools=["Read"]),
        ToolCallCase(id="b", task="t2", expect_tools=["Read"]),
    ]
    results = await run_suite(cases, model="m", api_key="k", fixtures_dir=Path("x"))
    assert len(results) == 2
    assert results[0].case_id == "a"
    assert results[1].case_id == "b"


# --- Task 1: telemetry, clock injection, sandbox lifecycle ---


def _tool_result_events() -> list[Any]:
    return [
        ToolUseStart(tool_name="Read", tool_id="t1", input={"file_path": "a.py"}),
        ToolUseStart(tool_name="Bash", tool_id="t2", input={"command": "ls"}),
        ToolResultReady(tool_id="t2", content="ls out", is_error=False),
        ToolResultReady(tool_id="t1", content="boom", is_error=True),
        TurnComplete(stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=3)),
    ]


async def test_case_result_carries_tool_execution_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory(_tool_result_events()))
    case = ToolCallCase(id="tc-001", task="read", expect_tools=["Read"])
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.num_tool_calls == 2
    assert r.num_tool_calls_executed == 2
    assert r.num_successful_tool_calls == 1
    assert r.execution_success_rate.numerator == 1
    assert r.execution_success_rate.denominator == 2


async def test_case_result_carries_tags_variant_trial_and_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-tag", task="t", tags=["blind", "read"])
    r = await run_case(
        case, model="m", api_key="k", fixtures_dir=Path("x"),
        variant="baseline", repeat_index=2, trial=7, run_id="run-1",
    )
    assert r.tags == ["blind", "read"]
    assert r.variant == "baseline"
    assert r.repeat_index == 2
    assert r.trial == 7
    assert r.run_id == "run-1"
    assert r.duration_ms is not None and r.duration_ms >= 0.0


async def test_run_case_uses_injected_clock_for_deterministic_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A monotonic clock is injectable so latency assertions stay offline-stable."""
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    ticks = iter([0, 250_000_000])  # 250 ms in ns

    def clock() -> int:
        return next(ticks)

    case = ToolCallCase(id="tc-clock", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"), clock=clock)
    assert r.duration_ms == 250.0


async def test_num_rounds_matches_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="tool_use", usage=Usage()),
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-rounds", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.turns == 2
    assert r.num_rounds == 2


async def test_error_type_is_none_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-ok", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.error_type is None


async def test_error_type_from_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.core.events as ev
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        ev.ErrorEvent(message="Max turns (8) reached", is_recoverable=False),
    ]))
    case = ToolCallCase(id="tc-err", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.error_type == "max_turns"


async def test_error_type_from_exception_and_readable_by_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([], raises=RuntimeError("kaboom")))
    case = ToolCallCase(id="tc-boom", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.passed is False
    assert r.error_type == "runtime_error"
    assert any("kaboom" in e for e in r.errors)


async def test_success_path_cleans_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-clean", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.sandbox is not None
    assert not Path(r.sandbox).exists()
    assert r.sandbox_kept is False


async def test_exception_path_cleans_sandbox_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([], raises=RuntimeError("kaboom")))
    case = ToolCallCase(id="tc-clean-err", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.sandbox is not None
    assert not Path(r.sandbox).exists()
    assert r.sandbox_kept is False


# --- layered failure semantics: infra propagates, case is recorded ---


async def test_build_engine_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infra fault means nothing was measured, so it must abort loudly.

    Recording it as a case failure would push an unmeasured case into the
    denominator and silently depress the success rate.
    """
    import longline.eval.runner as mod

    def _boom(*, sandbox: str, model: str, api_key: str) -> Any:
        raise RuntimeError("no harness")

    monkeypatch.setattr(mod, "build_engine", _boom)
    case = ToolCallCase(id="tc-infra", task="t")
    with pytest.raises(RuntimeError, match="no harness"):
        await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))


async def test_build_engine_failure_leaves_no_sandbox_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The propagating path must still clean up its temp dir."""
    import longline.eval.runner as mod

    created: list[str] = []

    def _boom(*, sandbox: str, model: str, api_key: str) -> Any:
        created.append(sandbox)
        raise RuntimeError("no harness")

    monkeypatch.setattr(mod, "build_engine", _boom)
    case = ToolCallCase(id="tc-infra-clean", task="t")
    with pytest.raises(RuntimeError):
        await run_case(case, model="m", api_key="k", fixtures_dir=tmp_path)
    assert len(created) == 1
    assert not Path(created[0]).exists()


async def test_midrun_failure_is_recorded_not_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the engine exists, a fault is a fact about the case, not the run."""
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([], raises=RuntimeError("kaboom")))
    case = ToolCallCase(id="tc-mid", task="t")
    r = await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))
    assert r.passed is False
    assert r.error_type == "runtime_error"
    assert any("kaboom" in e for e in r.errors)


async def test_suite_continues_after_a_midrun_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failed case stays in the results (and therefore the denominator)."""
    import longline.eval.runner as mod

    calls = {"n": 0}

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        calls["n"] += 1
        fail = calls["n"] == 2

        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                if fail:
                    raise RuntimeError("case 2 blew up")
                yield TurnComplete(stop_reason="end_turn", usage=Usage())

        return SimpleNamespace(
            registry=SimpleNamespace(list_tools=lambda: [FakeTool()]),
            system_prompt="test", model=model, submit=_FakeEngine().submit,
        )

    monkeypatch.setattr(mod, "build_engine", _build_engine)
    cases = [ToolCallCase(id=f"c{i}", task="t", expect_tools=[]) for i in range(3)]
    results = await run_suite(cases, model="m", api_key="k", fixtures_dir=Path("x"))

    assert [r.case_id for r in results] == ["c0", "c1", "c2"]
    assert results[1].passed is False
    assert results[1].error_type == "runtime_error"
    assert results[0].passed is True


async def test_suite_aborts_on_infra_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infra fault aborts the whole suite rather than degrading the rate."""
    import longline.eval.runner as mod

    def _boom(*, sandbox: str, model: str, api_key: str) -> Any:
        raise RuntimeError("no harness")

    monkeypatch.setattr(mod, "build_engine", _boom)
    cases = [ToolCallCase(id="c0", task="t"), ToolCallCase(id="c1", task="t")]
    with pytest.raises(RuntimeError, match="no harness"):
        await run_suite(cases, model="m", api_key="k", fixtures_dir=Path("x"))


# --- cancellation is not a case failure ---


async def test_keyboard_interrupt_propagates_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C must abort, not be laundered into a case failure.

    Only Exception is a statement about the case; KeyboardInterrupt and
    CancelledError are statements about the run, and recording them would
    prevent the operator from actually stopping the suite.
    """
    import longline.eval.runner as mod

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                raise KeyboardInterrupt
                yield  # pragma: no cover

        return SimpleNamespace(
            registry=SimpleNamespace(list_tools=lambda: [FakeTool()]),
            system_prompt="test", model=model, submit=_FakeEngine().submit,
        )

    monkeypatch.setattr(mod, "build_engine", _build_engine)
    case = ToolCallCase(id="tc-kb", task="t")
    with pytest.raises(KeyboardInterrupt):
        await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))


async def test_cancelled_error_propagates_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio cancellation must propagate so the task can actually cancel."""
    import asyncio

    import longline.eval.runner as mod

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                raise asyncio.CancelledError
                yield  # pragma: no cover

        return SimpleNamespace(
            registry=SimpleNamespace(list_tools=lambda: [FakeTool()]),
            system_prompt="test", model=model, submit=_FakeEngine().submit,
        )

    monkeypatch.setattr(mod, "build_engine", _build_engine)
    case = ToolCallCase(id="tc-cancel", task="t")
    with pytest.raises(asyncio.CancelledError):
        await run_case(case, model="m", api_key="k", fixtures_dir=Path("x"))


async def test_cancellation_still_cleans_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Propagating a cancellation must not leak the temp sandbox."""
    import longline.eval.runner as mod

    seen: list[str] = []

    def _build_engine(*, sandbox: str, model: str, api_key: str) -> Any:
        seen.append(sandbox)

        class _FakeEngine:
            async def submit(self, user_input: str, **kwargs: Any) -> Any:
                raise KeyboardInterrupt
                yield  # pragma: no cover

        return SimpleNamespace(
            registry=SimpleNamespace(list_tools=lambda: [FakeTool()]),
            system_prompt="test", model=model, submit=_FakeEngine().submit,
        )

    monkeypatch.setattr(mod, "build_engine", _build_engine)
    case = ToolCallCase(id="tc-kb-clean", task="t")
    with pytest.raises(KeyboardInterrupt):
        await run_case(case, model="m", api_key="k", fixtures_dir=tmp_path)
    assert len(seen) == 1
    assert not Path(seen[0]).exists()


async def test_exception_path_keeps_sandbox_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([], raises=RuntimeError("kaboom")))
    case = ToolCallCase(id="tc-keep", task="t")
    r = await run_case(
        case, model="m", api_key="k", fixtures_dir=Path("x"), keep_sandbox_on_failure=True,
    )
    assert r.sandbox_kept is True
    assert r.sandbox is not None
    assert Path(r.sandbox).is_dir()
    assert any(r.sandbox in e for e in r.errors)
    Path(r.sandbox).rmdir()


async def test_keep_sandbox_flag_does_not_keep_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    case = ToolCallCase(id="tc-ok-keep", task="t")
    r = await run_case(
        case, model="m", api_key="k", fixtures_dir=Path("x"), keep_sandbox_on_failure=True,
    )
    assert r.sandbox_kept is False
    assert r.sandbox is not None
    assert not Path(r.sandbox).exists()


async def test_missing_fixture_raises_before_sandbox_is_left_behind(tmp_path: Path) -> None:
    from longline.eval.runner import _prepare_sandbox

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    with pytest.raises(FileNotFoundError):
        _prepare_sandbox(fixtures, "nope")


async def test_run_suite_records_repeat_index(monkeypatch: pytest.MonkeyPatch) -> None:
    import longline.eval.runner as mod

    monkeypatch.setattr(mod, "build_engine", _fake_engine_factory([
        TurnComplete(stop_reason="end_turn", usage=Usage()),
    ]))
    cases = [ToolCallCase(id="a", task="t1"), ToolCallCase(id="b", task="t2")]
    results = await run_suite(cases, model="m", api_key="k", fixtures_dir=Path("x"), repeat_index=1)
    assert [r.repeat_index for r in results] == [1, 1]
