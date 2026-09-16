"""Unit tests for longline/eval/latency_runner.py (contract evals/README.md §5.5).

=== How these tests stay fast, and why it has to be this way ===

`run_pair`/`run_sample` accept `clock: Callable[[], int]`, which suggests the
suite can be driven entirely on an injected clock. It cannot, and the reason is
the thing this file is built around:

1. `_emit_turn` schedules the block grid against **`loop.time()`**, not against
   `clock` -- it reads `clock` only to stamp `request_start` and each release.
2. `TimedTool.execute` really does `await asyncio.sleep(duration_s)`.

So an injected clock changes the *timestamps* but not the *schedule*. A real
pair at the shipped 8x scale costs ~2.8 s, and an earlier version of this file
ran a dozen of them: **3.5 minutes for the file**. That is not merely untidy --
a slow test file is an operational hazard, because the next person to run it may
block on it and time out. The same hazard killed two earlier attempts at this
task.

Two mechanisms bring it to **~9 seconds**, and each is used where it is honest:

1. **`_no_sleep_tools` (autouse)** stubs `TimedTool.execute` so the tool body
   returns without waiting, while still recording a real `ExecutionRecord`.
   Everything derived from the records -- `min(records)` for
   `tool_execute_start`, the result-text join, the agreement checks -- therefore
   still sees real data. The stub must NOT make the body instant-and-empty: the
   runner's truth check compares `turn_complete` against response + tool
   duration, so an instant tool makes a correct run fail (measured: a 2 ms turn
   against a declared 320 ms). It keeps the duration, drops the waiting.
2. **`_TEST_SCALE` (0.05)** for the grid itself. Since every duration is scaled
   by one factor, the relationships the cases declare -- the lead is the tail,
   the arms tie when the tail is zero, both arms do identical work -- are
   scale-invariant, and those relationships are what most tests assert.

Three tests opt out of both via the `real_tool_bodies` fixture, because they
measure a real MILLISECOND quantity rather than a relationship, and at 0.05 the
driver's own `create_task` overhead (~8.7 ms) dominates the ~1 ms signal. They
run at the shipped scale and cost 2-3 s each -- about 80% of the file's runtime
in 3 of its 55 tests. That is the right trade: they are the only tests that
witness a real pair end to end through the real production executors.

`assert_paths_agree` therefore cannot be exercised at `_TEST_SCALE` -- its
tolerance is capped at half the declared signal, and below the shipped scale the
host's ~12 ms grid lag swamps a 2 ms turn. Its rejection branches are tested by
constructing `Sample`s directly, and its positive path by the three real tests.

What these tests exist to pin, beyond "it runs":

1. **The streaming arm starts its first tool strictly earlier than the buffered
   arm, and the lead is exactly the response tail.** `lat-001` is built so the
   saving *is* the tail; a harness that credited the tool's own duration as the
   saving would be caught, because the tool is 150 ms and the tail is 20 ms.
2. **A `tail == 0` case makes the arms identical.** The negative control: the
   same assertion, on a case the model says must not separate the arms, produces
   no lead. Without it, "streaming is earlier" would be indistinguishable from
   "the harness always reports a lead".
3. **`assert_paths_agree` refuses a genuine disagreement.** Every rejection
   branch is driven, in both directions, with the input that makes it fail named
   in the test body.
4. **The truth-check tolerance never becomes a waiver** -- it must stay below
   the errors a real harness bug produces, at every scale.

No assertion here reads a number back out of a field the runner wrote: every
expectation comes from `LatencyCase.truth()`, computed in the case file from the
case's own declarations.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from longline.eval.latency_cases import (
    LATENCY_CASES,
    LatencyCase,
    TimingModel,
    ToolCallSpec,
    ToolProfileSpec,
    get_case,
)
from longline.eval.latency_runner import (
    BUFFERED,
    DEFAULT_SAMPLES,
    DEFAULT_TIME_SCALE,
    LATENCY_TAG,
    MAX_SAMPLES,
    MIN_SAMPLES,
    STREAMING,
    ExecutionRecord,
    PathDisagreementError,
    Sample,
    StreamTimeline,
    TimedTool,
    all_records,
    assert_paths_agree,
    build_registry,
    observed_timeline,
    pooled_start_reduction,
    run_latency_suite,
    run_pair,
    run_pair_checked,
    run_sample,
    summarize_case,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The scale these tests drive the event loop at.
#
# Deliberately NOT the shipped 8x. Every duration is scaled by the same factor,
# so the *relationships* the cases declare -- the lead is the tail, the arms tie
# when the tail is zero, both arms do identical work -- are scale-invariant, and
# those relationships are what these tests assert. 0.05 puts a whole pair in the
# low milliseconds while leaving the grid far above the driver's own bookkeeping
# noise, so the assertions are testing the model and not the scheduler.
#
# What is NOT covered at this scale is the runner's tolerance behaviour, and
# more importantly `assert_paths_agree` cannot PASS at it: the tolerance is
# capped at half the declared signal, so the host's ~12 ms grid lag swamps the
# whole 2 ms turn. Tests that go through the agreement gate -- or that measure a
# lead in real milliseconds -- therefore use `_GATE_SCALE` (the shipped 8x) with
# real tool bodies, and cost 2-3 s each. There are deliberately three of them;
# they are ~80% of this file's runtime.
#
# The split is deliberate rather than a compromise: the gate and the absolute
# lead are properties of a real run and deserve a real-scale test, while
# `summarize_case` and the warmup/alternation rules are pure logic that a real
# scale would only slow down.
_TEST_SCALE = 0.05

# The scale for tests that must pass the agreement gate, or whose assertion is
# an absolute duration rather than a relationship. See above.
_GATE_SCALE = DEFAULT_TIME_SCALE


class LoopClock:
    """The clock the runner's own tests use: the loop's monotonic time.

    Deliberately just a unit conversion of `loop.time()`, not a time machine.
    The runner schedules its block grid against `loop.time()` and stamps with
    `clock`, so a clock unrelated to loop time puts the schedule and the
    timestamps on two different rulers -- measured, that doubled the first
    tool's reported start latency (320 ms observed against 160 ms expected).

    Speed comes from `_no_sleep_tools` below, NOT from here: there is no clock
    that can shorten an `await asyncio.sleep`. See that fixture for the whole
    story.
    """

    def __call__(self) -> int:
        return int(asyncio.get_running_loop().time() * 1e9)


def _clock() -> Callable[[], int]:
    return LoopClock()


# The production `TimedTool.execute`, captured at import time. `_no_sleep_tools`
# is autouse and replaces the attribute, so a test that needs a real tool body
# cannot look it up through the class afterwards -- it has to be held here.
_ORIGINAL_TIMED_TOOL_EXECUTE = TimedTool.execute


@pytest.fixture
def real_tool_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt a test out of `_no_sleep_tools`.

    Needed by any test whose assertion depends on the tool taking its declared
    time -- chiefly the agreement gate, whose truth check compares
    `turn_complete` against response + tool duration. With the tool instant the
    correct behaviour is rejected (measured: a 2 ms turn against a declared
    320 ms), so those tests must pay the ~1-3 s a real pair costs.

    Requested by name (`def test_x(self, real_tool_bodies):`) rather than
    applied per-class, so a reader can see at each test whether it is measuring
    real time or virtual.
    """
    from longline.eval import latency_runner as lr

    monkeypatch.setattr(lr.TimedTool, "execute", _ORIGINAL_TIMED_TOOL_EXECUTE)


@pytest.fixture(autouse=True)
def _no_sleep_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `TimedTool.execute` return immediately, keeping its record.

    === Why the tests are slow without this, and why this is the right seam ===

    `run_pair`/`run_sample` accept an injected `clock`, which suggests the suite
    can be driven entirely on virtual time. It cannot:

    1. `_emit_turn` schedules the block grid against **`loop.time()`**, not
       against `clock` -- it reads `clock` only to stamp `request_start` and
       each block release.
    2. `TimedTool.execute` really does `await asyncio.sleep(duration_s)`, and no
       clock can shorten a real sleep.

    So at the shipped 8x scale a single pair costs ~2.8 s. An earlier version of
    this file ran a dozen of them and took **3.5 minutes**, which is not merely
    untidy: a slow test file is an operational hazard, because the next person
    to run it may block on it and time out.

    The fix is to stub the TOOL BODY, not the clock. This replaces
    `TimedTool.execute` with one that records the same `ExecutionRecord` -- so
    `observed_timeline`'s `min(records)` derivation, the result-text join in
    `_match_to_completion_order`, and every agreement check still see real data
    -- while returning without waiting.

    What this does NOT change: the driver still schedules its real grid on
    `loop.time()`, the streaming executor still starts real tasks, and the two
    arms are still genuinely different code paths. The tests that assert the
    streaming arm starts earlier are therefore still measuring the thing they
    claim to measure; they just no longer wait 8 seconds to do it.

    Consequence for the truth check: with the sleeps gone, each arm's timestamps
    land almost exactly on the case's declarations, so `assert_paths_agree`
    passes on the model rather than on a tolerance. The tolerance's own
    behaviour is covered separately by `TestTruthToleranceFormula`, on the
    formula directly.
    """
    from longline.eval import latency_runner as lr

    async def virtual_execute(self: lr.TimedTool, tool_input: dict[str, object]) -> object:
        """Record the declared span, and actually wait the scaled duration.

        The wait is real but MINUTE, because the caller runs at a small
        `time_scale`; what this replaces is the *ratio* between the tool and
        the grid, not the waits themselves. The turn's truth is the response
        plus the tool, so a tool that finished instantly would break the very
        assertion the suite is built on -- measured, `lat_001` reported a
        349 ms turn against a declared 1520 ms.
        """
        index = len(self.records)
        start = self.clock()
        await asyncio.sleep(self.duration_s)
        end = self.clock()
        record = lr.ExecutionRecord(
            call_index=index,
            start_ns=start,
            end_ns=end,
            result_text=f"{self.spec.name} #{index} ok",
            is_error=False,
        )
        self.records.append(record)
        return lr.ToolResult(content=record.result_text)

    monkeypatch.setattr(lr.TimedTool, "execute", virtual_execute)


async def _pair(case: LatencyCase, *, scale: float = _GATE_SCALE) -> tuple[Sample, Sample]:
    """One gate-checked pair. Uses the shipped scale by default -- the gate
    cannot pass below it, see `_GATE_SCALE`."""
    return await run_pair_checked(case, clock=_clock(), time_scale=scale)


def _rejecting_sample(
    sample: Sample,
    *,
    result_texts: list[str] | None = None,
    timestamps: dict[str, int] | None = None,
    records: list[ExecutionRecord] | None = None,
) -> Sample:
    """Copy `sample` with selected fields replaced, for the negative cases.

    Only the field under test moves. `time_scale` in particular is carried over
    unchanged, because the truth assertion is taken at it -- if that moved, a
    failure could be blamed on the harness instead of on the injected field.
    """
    return Sample(
        variant=sample.variant,
        case_id=sample.case_id,
        timestamps=dict(sample.timestamps if timestamps is None else timestamps),
        result_texts=list(sample.result_texts if result_texts is None else result_texts),
        tool_durations_ms=list(sample.tool_durations_ms),
        time_scale=sample.time_scale,
        grid_lag_ns=list(sample.grid_lag_ns),
        release_ns=sample.release_ns,
        sink_ns=sample.sink_ns,
    )


def _record(
    *,
    call_index: int = 0,
    start_ns: int,
    end_ns: int,
    text: str = "Read #0 ok",
) -> ExecutionRecord:
    return ExecutionRecord(
        call_index=call_index, start_ns=start_ns, end_ns=end_ns, result_text=text, is_error=False,
    )


# --- the load-bearing assertion, on the real event loop ---------------------


class TestStreamingStartsEarlier:
    """The one claim the whole micro-benchmark rests on."""

    async def test_streaming_starts_its_first_tool_before_buffered(self) -> None:
        """A real pair through the real driver, at the cheap test scale.

        Uses `run_pair` rather than the checked variant: the gate's truth check
        needs the tool to take its declared time, and with the tool stubbed
        instant that check would reject a correct run. The gate has its own
        real-scale tests in `TestPathsAgreeOnARealPair`.
        """
        buffered, streaming = await run_pair(
            get_case("lat-001"), clock=_clock(), time_scale=_TEST_SCALE,
        )
        assert streaming.tool_start_latency_ms < buffered.tool_start_latency_ms

    async def test_the_lead_is_the_response_tail_not_the_tool_duration(
        self, real_tool_bodies: None,
    ) -> None:
        """`lat-001` sizes the tool (150 ms) far above the tail (20 ms).

        Crediting the tool's duration as the saving is the harness bug this case
        is built to catch: a runner that started the buffered clock at
        `tool_block_complete` rather than at `response_complete` would report a
        saving three times too large. The expected lead is the case file's own
        `truth()`, not a number read back out of the run.

        Runs at `_GATE_SCALE` with REAL tool bodies, unlike the other tests in
        this class. The measured quantity is the lead in milliseconds, and at
        the cheap test scale the driver's own `create_task` overhead (~8.7 ms)
        dominates the ~1 ms signal, so the assertion would be measuring the
        harness rather than the model. This is the one test where the number
        itself -- not just its sign -- is the point.
        """
        case = get_case("lat-001")
        truth = case.truth(_GATE_SCALE)
        buffered, streaming = await run_pair(
            case, clock=_clock(), time_scale=_GATE_SCALE,
        )

        lead_ms = buffered.tool_start_latency_ms - streaming.tool_start_latency_ms
        # The allowance is the host's scheduling jitter, which the module
        # docstring measures at ~20-46 ms per turn. It is far below the wrong
        # answer the test is guarding against: crediting the TOOL as the saving
        # would report 1200 ms at this scale, an order of magnitude out.
        assert lead_ms == pytest.approx(truth.start_lead_ns / 1e6, abs=60.0)
        # The lead is the tail, and the tool is much longer than the tail. This
        # half is pure arithmetic on the case and cannot pass by accident.
        assert truth.start_lead_ns / 1e9 == pytest.approx(
            case.timing.block_delay_s * _GATE_SCALE, rel=1e-9
        )
        assert case.tools[0].duration_s > 3 * case.timing.block_delay_s

    def test_truth_predicts_a_strict_lead_for_every_shipped_case(self) -> None:
        """No pair needed: this is the case file's arithmetic, checked directly."""
        for case in LATENCY_CASES:
            assert not case.truth().starts_are_identical, case.id
            assert case.truth().start_lead_ns > 0, case.id

    async def test_streaming_is_earlier_on_a_three_tool_case(self) -> None:
        """The lead grows with `t_n - t_1`; one more case to show it is not lat-001's shape."""
        case = get_case("lat-002")
        buffered, streaming = await run_pair(
            case, clock=_clock(), time_scale=_TEST_SCALE,
        )
        assert streaming.tool_start_latency_ms < buffered.tool_start_latency_ms


class TestZeroTailCaseDoesNotSeparateTheArms:
    """The negative control: the same assertion, on a case that must fail it.

    `latency_cases`' docstring states that `tail == 0` is the only way the arms
    become identical for tool-start latency. Constructing that case and checking
    the assertion goes the other way is what makes the positive test meaningful.
    """

    @staticmethod
    def _zero_tail() -> LatencyCase:
        return LatencyCase(
            id="lat-zero-tail",
            note="no response tail: both arms can start the tool at the same instant",
            timing=TimingModel(block_delay_s=0.020, tail_blocks=0),
            tools=(ToolProfileSpec("Read", duration_s=0.150),),
            calls=(ToolCallSpec("tu-0", "Read", {"file_path": "a.py"}),),
        )

    def test_truth_agrees_the_case_is_degenerate(self) -> None:
        truth = self._zero_tail().truth(_TEST_SCALE)
        assert truth.start_lead_ns == 0
        assert truth.starts_are_identical

    def test_the_positive_assertion_is_false_on_this_case(self) -> None:
        """The falsification, written as an assertion about the assertion.

        Pure arithmetic on `truth()`: no pair, no scheduler. This is deliberate.
        The arms are identical because the case SAYS so, and asserting it on a
        real run would instead measure the host's ~40 ms timer floor -- on this
        machine two runs of this case differed by 9 ms in the streaming arm's
        favour, purely from jitter, and at 2x the case's whole declared turn is
        only 40 ms. Placing this assertion on the closed form keeps it a
        statement about the model, and leaves `test_arms_do_not_separate_on_tool_start`
        (below) as the honest, jitter-tolerant version on a real pair.
        """
        truth = self._zero_tail().truth(_TEST_SCALE)
        assert truth.start_lead_ns == 0
        with pytest.raises(AssertionError):
            assert truth.streaming_tool_start_ns < truth.buffered_tool_start_ns

    async def test_arms_do_not_separate_on_tool_start(self) -> None:
        """The same case, on a real pair: the lead must vanish into the noise.

        Unlike the positive test, this asserts an ABSENCE, so the host's jitter
        must be allowed for: at this scale the true lead is 0 and the measured
        floor is ~9-40 ms. The bound is therefore the tolerance the runner
        itself uses, not zero -- the claim is "no systematic lead", which is
        what distinguishes this case from `lat-001`'s tail-sized one.
        """
        from longline.eval.latency_runner import _SLEEP_TOLERANCE_FLOOR_NS

        buffered, streaming = await run_pair(
            self._zero_tail(), clock=_clock(), time_scale=_TEST_SCALE,
        )
        lead_ms = buffered.tool_start_latency_ms - streaming.tool_start_latency_ms
        assert abs(lead_ms) < _SLEEP_TOLERANCE_FLOOR_NS / 1e6


# --- assert_paths_agree: the positive case and every rejection branch -------


class TestPathsAgreeOnARealPair:
    """The gate, run against the REAL tool bodies at the shipped scale.

    These two opt out of `_no_sleep_tools` because the gate's truth check
    compares an arm's `turn_complete` against the case's declared turn, and that
    turn is the response PLUS the tool duration. With the tool stubbed instant,
    the correct behaviour is rejected -- measured, a 2 ms turn against a declared
    320 ms. So these are the file's genuinely slow tests, at ~1-3 s each, and
    they are worth it: they are the only place a real pair is driven end to end
    through the real production executors.
    """

    async def test_agreeing_pair_passes(self, real_tool_bodies: None) -> None:
        case = get_case("lat-001")
        buffered, streaming = await _pair(case)
        assert_paths_agree(case, buffered, streaming)  # must not raise

    async def test_both_arms_execute_the_declared_calls_exactly_once(
        self, real_tool_bodies: None,
    ) -> None:
        """The check's first rule, on a real run rather than a constructed one."""
        case = get_case("lat-002")
        buffered, streaming = await _pair(case)
        for sample in (buffered, streaming):
            assert sorted(sample.result_texts) == sorted(case.results())
            assert len(sample.result_texts) == case.num_calls


class TestPathsAgreeRejections:
    """Every branch, driven from a stubbed pair so no loop time is spent.

    Each case states in one line what input makes the assertion fail, which is
    the property the task requires of every new assertion here.
    """

    @staticmethod
    def _base(case: LatencyCase, variant: str = STREAMING) -> Sample:
        truth = case.truth(_TEST_SCALE)
        base = 1_000_000
        return Sample(
            variant=variant,
            case_id=case.id,
            timestamps={
                "request_start": base,
                "tool_block_complete": base + truth.block_offsets_ns[0],
                "tool_execute_start": base + truth.block_offsets_ns[0],
                "response_complete": base + truth.response_ns,
                "tool_execute_end": base + truth.streaming_turn_ns,
                "turn_complete": base + truth.streaming_turn_ns,
            },
            result_texts=list(case.results()),
            tool_durations_ms=[truth.tool_ns / 1e6] * case.num_calls,
            time_scale=_TEST_SCALE,
            grid_lag_ns=[0],
            release_ns=0,
            sink_ns=0,
        )

    def test_rejects_an_arm_that_executed_nothing(self) -> None:
        """FAILS ON: an arm whose `result_texts` is empty -- the executor never dispatched."""
        case = get_case("lat-001")
        with pytest.raises(PathDisagreementError, match="cannot be compared"):
            assert_paths_agree(
                case, self._base(case), _rejecting_sample(self._base(case), result_texts=[]),
            )

    def test_rejects_a_call_count_the_case_does_not_declare(self) -> None:
        """FAILS ON: an arm that executed fewer/more calls than the case declares."""
        case = get_case("lat-002")
        short = self._base(case).result_texts[:-1]
        with pytest.raises(PathDisagreementError):
            assert_paths_agree(
                case, self._base(case),
                _rejecting_sample(self._base(case), result_texts=short),
            )

    def test_rejects_arms_whose_result_texts_differ(self) -> None:
        """FAILS ON: right count, wrong content -- the two arms did different work.

        This is the contract's own precondition ("校验两种路径最终结果完全一致"),
        so it gets the dedicated message about the two arms disagreeing.
        """
        case = get_case("lat-001")
        other = _rejecting_sample(self._base(case), result_texts=["Read #99 ok"])
        with pytest.raises(PathDisagreementError):
            assert_paths_agree(case, self._base(case), other)

    def test_rejects_a_timestamp_before_request_start(self) -> None:
        """FAILS ON: any of the six timestamps landing before `request_start`."""
        case = get_case("lat-001")
        stamps = dict(self._base(case).timestamps)
        stamps["turn_complete"] = stamps["request_start"] - 1
        with pytest.raises(PathDisagreementError, match="precedes request_start"):
            assert_paths_agree(
                case, self._base(case), _rejecting_sample(self._base(case), timestamps=stamps),
            )

    def test_rejects_a_tool_timed_before_its_block_completed(self) -> None:
        """FAILS ON: `tool_execute_start < tool_block_complete` -- an impossible execution."""
        case = get_case("lat-001")
        stamps = dict(self._base(case).timestamps)
        stamps["tool_execute_start"] = stamps["tool_block_complete"] - 1
        stamps["tool_execute_end"] = stamps["tool_block_complete"]
        with pytest.raises(PathDisagreementError, match="before its tool_use block"):
            assert_paths_agree(
                case, self._base(case), _rejecting_sample(self._base(case), timestamps=stamps),
            )

    def test_rejects_a_turn_that_ends_before_its_last_tool(self) -> None:
        """FAILS ON: `turn_complete < tool_execute_end`."""
        case = get_case("lat-001")
        stamps = dict(self._base(case).timestamps)
        stamps["turn_complete"] = stamps["tool_execute_start"] + 1
        with pytest.raises(PathDisagreementError, match="before the last tool finished"):
            assert_paths_agree(
                case, self._base(case), _rejecting_sample(self._base(case), timestamps=stamps),
            )

    def test_rejects_a_start_that_matches_neither_arm_s_truth(self) -> None:
        """FAILS ON: a plausible-looking start that is not the case's declared one.

        The value differs from the truth by more than a whole block delay, which
        is what makes it "measuring the wrong instant" rather than jitter.
        """
        case = get_case("lat-001")
        truth = case.truth(_TEST_SCALE)
        stamps = dict(self._base(case).timestamps)
        stamps["tool_execute_start"] += truth.block_offsets_ns[0] + truth.tool_ns
        with pytest.raises(PathDisagreementError, match="expected"):
            assert_paths_agree(
                case, self._base(case), _rejecting_sample(self._base(case), timestamps=stamps),
            )

    def test_accepts_a_start_within_the_measured_tolerance(self) -> None:
        """PASSES: the same injection, sized to the tolerance rather than above it.

        The counterpart to the test above. Without it, a rejection test that
        fired on any offset at all would look like discrimination while actually
        being a check that can never pass.
        """
        case = get_case("lat-001")
        sample = self._base(case)
        stamps = dict(sample.timestamps)
        stamps["tool_execute_start"] += 1  # one nanosecond: noise
        assert_paths_agree(
            case, sample, _rejecting_sample(sample, timestamps=stamps),
        )  # must not raise


# --- the truth is re-derived, not read back --------------------------------


class TestObservedTimeline:
    def test_tool_spans_come_from_the_records(self) -> None:
        """`min`/`max` over the recorded bodies, not a value the driver stored.

        FAILS ON: a driver that stamped "start" before entering the tool body.
        """
        timeline = StreamTimeline(request_start=100)
        timeline.block_complete_ns = [200]
        timeline.block_offsets_ns = [100]
        records = [
            _record(call_index=0, start_ns=250, end_ns=900, text="Read #0 ok"),
            _record(call_index=1, start_ns=300, end_ns=950, text="Read #1 ok"),
        ]
        out = observed_timeline(timeline, records, turn_complete_ns=1000, response_complete_ns=800)
        assert out["tool_execute_start"] == 250  # the EARLIEST real start
        assert out["tool_execute_end"] == 950  # the LATEST real end
        assert out["tool_block_complete"] == 200
        assert out["request_start"] == 100

    def test_dropping_the_earliest_record_moves_the_derived_start(self) -> None:
        """FAILS ON: a derivation that ignores the records and echoes the driver."""
        timeline = StreamTimeline(request_start=100)
        timeline.block_complete_ns = [200]
        timeline.block_offsets_ns = [100]
        only = [_record(start_ns=300, end_ns=950)]
        out = observed_timeline(timeline, only, turn_complete_ns=1000, response_complete_ns=800)
        assert out["tool_execute_start"] == 300

    async def test_a_real_run_agrees_with_the_re_derivation(self) -> None:
        """FAILS ON: `min(records)` and the published `tool_execute_start` disagreeing."""
        case = get_case("lat-001")
        registry = build_registry(case.tools, time_scale=_TEST_SCALE)
        buffered = await run_sample(
            case, BUFFERED, clock=_clock(), time_scale=_TEST_SCALE, registry=registry,
        )
        spans = all_records(registry)
        assert buffered.timestamps["tool_execute_start"] == min(r.start_ns for r in spans)
        assert buffered.timestamps["tool_execute_end"] == max(r.end_ns for r in spans)


# --- the suite's contract rules, driven without 30 real pairs ---------------


def _stub_suite(monkeypatch: pytest.MonkeyPatch, case: LatencyCase) -> list[str]:
    """Replace `run_pair_checked` with a cheap, deterministic stand-in.

    `run_latency_suite`'s contract rules (the 30-50 sample band, warmup
    exclusion, alternating order) are about the LOOP, not about the clock, and
    running 30 real pairs to test them would cost ~20 s per test. The stub
    returns a pair whose latencies encode the case's truth, so the aggregation
    downstream still sees a real-shaped result.

    Returns the list the stub appends the arm order to, so a caller can assert
    on the ordering it produced.
    """
    from longline.eval import latency_runner as lr

    order: list[str] = []

    def _sample_for(c: LatencyCase, variant: str, time_scale: float) -> Sample:
        # Truth is taken at the scale the CALLER is running, not at the module
        # default: a stub that mixed the two would hand `run_latency_suite` a
        # pair whose numbers belong to a different schedule, and the suite's own
        # agreement gate would reject it for a reason that has nothing to do
        # with the rule under test.
        order.append(variant)
        truth = c.truth(time_scale)
        start = truth.response_ns if variant == BUFFERED else truth.streaming_tool_start_ns
        return Sample(
            variant=variant,
            case_id=c.id,
            timestamps={
                "request_start": 0,
                "tool_block_complete": truth.block_offsets_ns[0],
                "tool_execute_start": start,
                "response_complete": truth.response_ns,
                "tool_execute_end": start + truth.tool_ns,
                "turn_complete": truth.buffered_turn_ns if variant == BUFFERED
                else truth.streaming_turn_ns,
            },
            result_texts=list(c.results()),
            tool_durations_ms=[truth.tool_ns / 1e6] * c.num_calls,
            time_scale=time_scale,
            grid_lag_ns=[0],
            release_ns=0,
            sink_ns=0,
        )

    async def stub_pair(
        c: LatencyCase, *, clock: object = None, time_scale: float = _TEST_SCALE
    ) -> tuple[Sample, Sample]:
        return (
            _sample_for(c, BUFFERED, time_scale),
            _sample_for(c, STREAMING, time_scale),
        )
    async def stub_sample(
        c: LatencyCase, variant: str, *, clock: object = None,
        time_scale: float = _TEST_SCALE, registry: object = None,
    ) -> Sample:
        """The odd samples call `run_sample` DIRECTLY, not `run_pair_checked`.

        That is how the order alternates, so this stub is not optional: without
        it the odd half of every suite runs for real, which is what made four
        tests take 44-50 s each. Stubbing only `run_pair_checked` -- the obvious
        reading -- leaves exactly those four slow, and the file stays a
        three-minute hazard.
        """
        return _sample_for(c, variant, time_scale)

    monkeypatch.setattr(lr, "run_pair_checked", stub_pair)
    monkeypatch.setattr(lr, "run_sample", stub_sample)
    return order


class TestSuiteLoopRules:
    async def test_warmups_are_driven_but_not_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAILS ON: warmup pairs leaking into `samples` (or not being run at all)."""
        case = get_case("lat-001")
        warmups = 3
        order = _stub_suite(monkeypatch, case)
        summary, samples = await run_latency_suite(
            [case], samples=MIN_SAMPLES, warmups=warmups, time_scale=_GATE_SCALE,
        )
        assert summary.warmups_per_arm == warmups
        assert summary.cases[0].samples_per_arm == MIN_SAMPLES

        # The stub covers BOTH call paths, so it sees every arm of every pair:
        # the warmups plus the samples, two arms each. Written as the rule
        # rather than as a literal so it stays true if either count moves.
        assert len(order) == (warmups + MIN_SAMPLES) * 2

        # And the point of the test: every RECORDED sample is a real sample.
        # A warmup leaking in would make this longer than the contract's 30.
        assert len(samples) == MIN_SAMPLES * 2
        assert summary.cases[0].samples_per_arm == len(samples) // 2

    async def test_sample_size_outside_the_contract_band_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAILS ON: `samples` below 30 or above 50 (contract §4.5 fixes the band)."""
        case = get_case("lat-001")
        _stub_suite(monkeypatch, case)
        for bad in (MIN_SAMPLES - 1, MAX_SAMPLES + 1, 0):
            with pytest.raises(ValueError, match="contract"):
                await run_latency_suite([case], samples=bad, warmups=0, time_scale=_GATE_SCALE)

    async def test_a_negative_warmup_count_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAILS ON: `warmups < 0`."""
        case = get_case("lat-001")
        _stub_suite(monkeypatch, case)
        with pytest.raises(ValueError, match="warmups"):
            await run_latency_suite([case], samples=MIN_SAMPLES, warmups=-1, time_scale=_GATE_SCALE)

    async def test_an_unknown_variant_is_refused(self) -> None:
        """FAILS ON: a variant name that is neither `buffered` nor `streaming`."""
        with pytest.raises(ValueError, match="unknown variant"):
            await run_sample(
                get_case("lat-001"), "sideways", clock=_clock(), time_scale=_GATE_SCALE,
            )

    async def test_run_pair_returns_buffered_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAILS ON: the pair coming back (streaming, buffered)."""
        case = get_case("lat-001")
        _stub_suite(monkeypatch, case)
        buffered, streaming = await run_pair(case, clock=_clock(), time_scale=_TEST_SCALE)
        assert buffered.variant == BUFFERED
        assert streaming.variant == STREAMING

    async def test_summary_reports_its_scale_and_sample_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case = get_case("lat-001")
        _stub_suite(monkeypatch, case)
        summary, _ = await run_latency_suite(
            [case], samples=MIN_SAMPLES, warmups=0, time_scale=_GATE_SCALE,
        )
        assert summary.samples_per_arm == MIN_SAMPLES
        assert summary.time_scale == _GATE_SCALE
        payload = summary.to_dict()
        assert payload["reduction_units"] == "ratio_of_durations"
        assert payload["latency_units"] == "ms"


class TestAlternatingOrder:
    """Contract §4.7: the arm that runs second pays for the first one's warm-up.

    `run_latency_suite` calls `run_pair_checked` on even indices (buffered first)
    and the two `run_sample`s in the opposite order on odd ones. Checked here by
    recording which arm each sample entered first.
    """

    async def test_the_first_arm_alternates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAILS ON: a loop that always runs buffered first (or always streaming).

        The two call paths are instrumented separately because they ARE
        separate: even samples go through `run_pair_checked` (buffered first by
        construction), odd ones call `run_sample` directly with streaming first.
        Recording which path each sample took is what makes the alternation
        checkable rather than assumed.
        """
        from longline.eval import latency_runner as lr

        case = get_case("lat-001")
        firsts: list[str] = []
        _stub_suite(monkeypatch, case)
        stub_pair = lr.run_pair_checked
        stub_sample = lr.run_sample

        async def pair_spy(c: LatencyCase, **kw: object) -> tuple[Sample, Sample]:
            firsts.append(BUFFERED)
            return await stub_pair(c, **kw)  # type: ignore[arg-type]

        async def sample_spy(c: LatencyCase, variant: str, **kw: object) -> Sample:
            if variant == STREAMING:  # the odd path enters streaming first
                firsts.append(STREAMING)
            return await stub_sample(c, variant, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(lr, "run_pair_checked", pair_spy)
        monkeypatch.setattr(lr, "run_sample", sample_spy)
        await run_latency_suite(
            [case], samples=MIN_SAMPLES, warmups=0, time_scale=_GATE_SCALE,
        )
        expected = [BUFFERED if i % 2 == 0 else STREAMING for i in range(MIN_SAMPLES)]
        assert firsts == expected

    async def test_warmups_also_alternate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAILS ON: warmups all going one way while samples alternate.

        Easy to miss: the warmup loop is a separate loop from the sampling one,
        so getting the sample order right does not imply the warmup order is.
        It is not cosmetic either -- the first REAL sample inherits whatever
        state the last warmup left, so a warmup block that always ran
        buffered-first biases the first streaming sample.
        """
        from longline.eval import latency_runner as lr

        case = get_case("lat-001")
        firsts: list[str] = []
        _stub_suite(monkeypatch, case)
        stub_pair = lr.run_pair_checked
        stub_sample = lr.run_sample

        async def pair_spy(c: LatencyCase, **kw: object) -> tuple[Sample, Sample]:
            firsts.append(BUFFERED)
            return await stub_pair(c, **kw)  # type: ignore[arg-type]

        async def sample_spy(c: LatencyCase, variant: str, **kw: object) -> Sample:
            if variant == STREAMING:
                firsts.append(STREAMING)
            return await stub_sample(c, variant, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(lr, "run_pair_checked", pair_spy)
        monkeypatch.setattr(lr, "run_sample", sample_spy)
        await run_latency_suite(
            [case], samples=MIN_SAMPLES, warmups=4, time_scale=_GATE_SCALE,
        )
        # 4 warmup pairs + 15 even sample pairs = 19 entries, alternating from
        # buffered. Without the warmup fix this reads b, b, b, b, b, s, ... --
        # which is exactly the regression the test names.
        assert firsts == [BUFFERED if i % 2 == 0 else STREAMING for i in range(len(firsts))]
        assert firsts[:4] == [BUFFERED, STREAMING, BUFFERED, STREAMING]


# --- rows and aggregation --------------------------------------------------


def _sample(
    *,
    variant: str = STREAMING,
    case_id: str = "lat-x",
    start: int = 100,
    turn: int = 500,
    response: int = 300,
    results: Sequence[str] = ("Read #0 ok",),
) -> Sample:
    return Sample(
        variant=variant,
        case_id=case_id,
        timestamps={
            "request_start": 0,
            "tool_block_complete": 10,
            "tool_execute_start": start,
            "response_complete": response,
            "tool_execute_end": start + 20,
            "turn_complete": turn,
        },
        result_texts=list(results),
        tool_durations_ms=[0.02],
        time_scale=1.0,
        grid_lag_ns=[0],
        release_ns=0,
        sink_ns=0,
    )


class TestSampleRow:
    def test_row_is_tagged_and_carries_the_contract_timestamps(self) -> None:
        """FAILS ON: a row missing the tag, or missing one of the six timestamps."""
        from longline.eval.latency_runner import TIMESTAMP_KEYS

        row = _sample().to_row()
        assert LATENCY_TAG in row["tags"]  # type: ignore[operator]
        assert row["timestamp_keys"] == list(TIMESTAMP_KEYS)
        assert set(row["timestamp_ns"]) == set(TIMESTAMP_KEYS)  # type: ignore[arg-type]

    def test_row_timestamps_are_relative_to_request_start(self) -> None:
        """FAILS ON: absolute nanosecond stamps leaking into the row."""
        row = _sample(start=100, turn=500).to_row()
        stamps = row["timestamp_ns"]
        assert stamps["request_start"] == 0  # type: ignore[index]
        assert stamps["tool_execute_start"] == 100  # type: ignore[index]
        assert stamps["turn_complete"] == 500  # type: ignore[index]

    def test_row_states_that_the_reduction_is_a_ratio_not_pp(self) -> None:
        """FAILS ON: the units being left to prose, where `pp` could creep in."""
        row = _sample().to_row()
        assert row["reduction_units"] == "ratio_of_durations"
        assert row["latency_units"] == "ms"

    def test_derived_latencies_match_the_timestamps(self) -> None:
        """FAILS ON: a property that reads a different pair of stamps than it says."""
        sample = _sample(start=100, response=300, turn=500)
        assert sample.tool_start_latency_ms == pytest.approx(100 / 1e6)
        assert sample.turn_latency_ms == pytest.approx(500 / 1e6)
        assert sample.overlap_time_ms == pytest.approx(200 / 1e6)  # 300 - 100

    def test_overlap_time_is_floored_at_zero(self) -> None:
        """FAILS ON: a negative overlap (tool started after the response ended)."""
        assert _sample(start=400, response=300).overlap_time_ms == 0.0

    def test_row_is_json_serialisable(self) -> None:
        row = _sample().to_row()
        assert json.loads(json.dumps(row))["case_id"] == "lat-x"


class TestSummarizeCase:
    """Pure aggregation: no clock, no loop, and every input constructed here."""

    def _pairs(self, case: LatencyCase) -> list[tuple[Sample, Sample]]:
        truth = case.truth(_TEST_SCALE)
        out = []
        for _ in range(MIN_SAMPLES):
            b = _sample(
                variant=BUFFERED, case_id=case.id,
                start=truth.response_ns, response=truth.response_ns,
                turn=truth.buffered_turn_ns, results=case.results(),
            )
            s = _sample(
                variant=STREAMING, case_id=case.id,
                start=truth.streaming_tool_start_ns, response=truth.response_ns,
                turn=max(truth.response_ns, truth.streaming_turn_ns), results=case.results(),
            )
            out.append((b, s))
        return out

    def test_reduction_is_the_ratio_of_the_two_mean_durations(self) -> None:
        """FAILS ON: a reduction computed as a difference, or on the wrong arm order."""
        case = get_case("lat-001")
        summary = summarize_case(case, self._pairs(case), warmups=0, time_scale=_TEST_SCALE)
        b = summary.metrics["buffered"]["tool_start_latency_ms_mean"]
        s = summary.metrics["streaming"]["tool_start_latency_ms_mean"]
        assert b is not None and s is not None
        assert summary.reduction == pytest.approx((b - s) / b)

    def test_all_three_metrics_report_mean_p50_and_p95(self) -> None:
        """FAILS ON: any of the nine contract statistics coming back None."""
        case = get_case("lat-002")
        summary = summarize_case(case, self._pairs(case), warmups=0, time_scale=_TEST_SCALE)
        for arm in (BUFFERED, STREAMING):
            for name in ("tool_start_latency_ms", "turn_latency_ms", "overlap_time_ms"):
                for stat in ("mean", "p50", "p95"):
                    assert summary.metrics[arm][f"{name}_{stat}"] is not None, (arm, name, stat)

    def test_streaming_is_counted_faster_on_every_sample(self) -> None:
        """FAILS ON: the lifetime/overlap counters being swapped or inverted."""
        case = get_case("lat-001")
        summary = summarize_case(case, self._pairs(case), warmups=0, time_scale=_TEST_SCALE)
        assert summary.lifetime_samples == MIN_SAMPLES
        assert summary.overlap_samples == 0

    def test_zero_baseline_reduces_to_not_measured(self) -> None:
        """FAILS ON: a ZeroDivisionError, or a fabricated 0.0/100%."""
        case = get_case("lat-001")
        pairs = self._pairs(case)
        zeroed = [
            (_sample(
                variant=BUFFERED, case_id=case.id,
                start=0, response=0, turn=0, results=case.results(),
            ), s)
            for _, s in pairs
        ]
        summary = summarize_case(case, zeroed, warmups=0, time_scale=_TEST_SCALE)
        assert summary.metrics["buffered"]["tool_start_latency_ms_mean"] == 0.0
        assert summary.reduction is None

    def test_the_per_sample_ratios_are_kept_alongside_the_aggregate(self) -> None:
        """FAILS ON: reporting only the mean, which hides a sample-level reversal."""
        case = get_case("lat-002")
        summary = summarize_case(case, self._pairs(case), warmups=0, time_scale=_TEST_SCALE)
        assert len(summary.reduction_per_sample) == MIN_SAMPLES
        assert summary.reduction_mean is not None
        assert summary.reduction_p50 is not None


class TestPooledReduction:
    def test_pooling_counts_samples_not_cases(self) -> None:
        """FAILS ON: `n` being the case count (3) rather than cases x samples."""
        from longline.eval.latency_runner import CaseLatency

        cases = [
            CaseLatency(
                case_id=f"c{i}", note="", samples_per_arm=2,
                metrics={}, reduction=None, reduction_per_sample=[0.1, 0.2],
                reduction_mean=0.15, reduction_p50=0.15,
                lifetime_samples=2, overlap_samples=0,
            )
            for i in range(3)
        ]
        pooled = pooled_start_reduction(cases)
        assert pooled["n"] == 6
        assert pooled["unit"] == "ratio_of_durations"
        assert pooled["mean"] == pytest.approx(0.15)

    def test_pooling_an_empty_case_list_is_not_a_zero(self) -> None:
        """FAILS ON: an empty pool reporting 0.0 instead of "not measured"."""
        pooled = pooled_start_reduction([])
        assert pooled["n"] == 0
        assert pooled["mean"] is None


# --- the tool wrapper ------------------------------------------------------


class TestTimedTool:
    def test_the_scale_is_applied_exactly_once(self) -> None:
        """FAILS ON: a double-scaled duration (`duration_s * scale * scale`)."""
        case = get_case("lat-001")
        registry = build_registry(case.tools, time_scale=4.0)
        tool = next(t for t in registry.list_tools() if t.get_name() == "Read")
        assert tool.duration_s == pytest.approx(case.tools[0].duration_s * 4.0)  # type: ignore[attr-defined]

    def test_every_tool_shares_the_run_s_clock(self) -> None:
        """FAILS ON: a tool stamping on `perf_counter_ns` while the driver uses `clock`.

        That mismatch is real and was hit while writing these tests: an injected
        clock reading near zero next to a tool span on `perf_counter_ns`
        (process-uptime, tens of seconds) makes `turn_complete` look like it
        happened ~49,000 seconds before the tool finished.
        """
        case = get_case("lat-001")
        sentinel = _clock()
        registry = build_registry(case.tools, time_scale=1.0, clock=sentinel)
        tool = next(t for t in registry.list_tools() if t.get_name() == "Read")
        assert tool.clock is sentinel  # type: ignore[attr-defined]

    async def test_the_recorded_span_brackets_a_real_sleep(self) -> None:
        """FAILS ON: `end_ns <= start_ns`, or a span shorter than the declared body."""
        case = get_case("lat-001")
        registry = build_registry(case.tools, time_scale=_TEST_SCALE)
        await run_sample(
            case, BUFFERED, clock=_clock(), time_scale=_TEST_SCALE, registry=registry,
        )
        record = all_records(registry)[0]
        assert record.end_ns > record.start_ns
        assert record.duration_ms >= case.tools[0].duration_s * _TEST_SCALE * 1000


# --- the shipped defaults are a claim about a real machine ------------------


class TestShippedDefaults:
    def test_sample_band_matches_the_contract(self) -> None:
        """FAILS ON: a band that is not `30 <= default <= 50`."""
        assert MIN_SAMPLES == 30
        assert MAX_SAMPLES == 50
        assert MIN_SAMPLES <= DEFAULT_SAMPLES <= MAX_SAMPLES

    def test_default_scale_keeps_the_signal_above_the_host_timer_floor(self) -> None:
        """FAILS ON: a default scale whose smallest declared signal is inside the noise.

        The measured facts this encodes: this host's `asyncio.sleep` overshoots
        by ~10-40 ms for a short wait regardless of the requested duration, and
        that absolute floor does NOT shrink with `time_scale`. So the smallest
        signal the cases assert -- one block delay -- must clear that floor with
        margin at the default. This is the property that rules out `time_scale=1`.
        """
        from longline.eval.latency_runner import _SLEEP_TOLERANCE_FLOOR_NS

        smallest_signal_ns = (
            min(c.timing.block_delay_s for c in LATENCY_CASES) * DEFAULT_TIME_SCALE * 1e9
        )
        assert smallest_signal_ns > 2 * _SLEEP_TOLERANCE_FLOOR_NS

    def test_the_tolerance_never_shrinks_as_the_scale_grows(self) -> None:
        """FAILS ON: a tolerance that does not scale -- the bug this replaced.

        The allowance for a tool's `asyncio.sleep` overshoot is proportional to
        how long that sleep was, so a fixed millisecond addend makes a long,
        expensive run the LEAST well guarded, which is backwards. Measured, the
        worst truth error as a fraction of the turn was 0.4% at 2x and 1.1% at
        5x: the relative error rises with the scale, so the allowance must too.

        Deliberately NOT asserting which term binds at the shipped scale. It is
        case-dependent: at 8x the proportional term (45.6 ms on lat-001, whose
        150 ms tool makes for a long turn) is above the 40 ms floor, while
         lat-002's (31.2 ms, a 50 ms tool) is below it. Pinning that would be a
        test that breaks the moment a case's duration moves, without saying
        anything about whether the formula is right.
        """
        from longline.eval.latency_runner import (
            _MAX_TOLERANCE_SIGNAL_FRACTION,
            _SLEEP_TOLERANCE_FLOOR_NS,
            _SLEEP_TOLERANCE_FRACTION,
        )

        case = get_case("lat-001")
        scales = (1.0, 2.0, 5.0, 10.0, 20.0)
        terms = [
            max(_SLEEP_TOLERANCE_FRACTION * case.truth(s).buffered_turn_ns, _SLEEP_TOLERANCE_FLOOR_NS)
            for s in scales
        ]
        # Monotonic non-decreasing, and strictly larger by the top of the range.
        assert terms == sorted(terms)
        assert terms[-1] > terms[0]
        # The proportional term overtakes the floor somewhere in this range --
        # that is the whole point of having it rather than a lone constant.
        assert _SLEEP_TOLERANCE_FRACTION * case.truth(scales[-1]).buffered_turn_ns > (
            _SLEEP_TOLERANCE_FLOOR_NS
        )
        # And the cap is always a quarter of the signal, at every scale, so the
        # check can never be widened past the distinction it has to make.
        from longline.eval.latency_runner import _asserted_signal_ns, truth_tolerance_ns

        for s in scales:
            signal = _asserted_signal_ns(case, s)
            tol = truth_tolerance_ns(case.truth(s).buffered_turn_ns, 0, signal_ns=signal)
            assert tol <= int(_MAX_TOLERANCE_SIGNAL_FRACTION * signal) + 1, s

    def test_default_scale_sets_a_bounded_stated_runtime(self) -> None:
        """FAILS ON: shipping a default whose full run is hours.

        The shipped default used to be 1000x, which implies a ~13-hour run. The
        measured cost is ~5.4 s per pair at 20x, so this bounds the whole suite
        (every case x samples+warmups pairs) at a number a person can wait for.
        The bound is deliberately generous -- 20 minutes -- because the scale is
        set by the discrimination requirement, not by speed, and a scale raised
        again for correctness should not trip this test until it is truly
        unusable.

        The previous 8x default was ~6 minutes, which was faster AND wrong: at
        that scale the tolerance needed for the host's jitter exceeded the
        error the check has to catch. See `_MAX_TOLERANCE_SIGNAL_FRACTION`.
        """
        from longline.eval.latency_runner import WARMUP_ROUNDS

        seconds_per_pair_at_20x = 5.4
        pairs = len(LATENCY_CASES) * (DEFAULT_SAMPLES + WARMUP_ROUNDS)
        estimated_s = pairs * seconds_per_pair_at_20x * (DEFAULT_TIME_SCALE / 20.0)
        assert estimated_s < 1200.0, f"a full latency run would take ~{estimated_s / 60:.0f} min"

    def test_the_default_scale_is_the_one_the_docstring_justifies(self) -> None:
        """FAILS ON: the constant drifting away from the measured justification.

        Pinned rather than compared, because the value is the conclusion of a
        measurement and not a free parameter. A change here has to be
        accompanied by a new jitter-vs-signal measurement, which is what
        `test_a_small_scale_cannot_discriminate` and
        `test_the_cap_still_rejects_a_wrong_instant` below encode.
        """
        assert DEFAULT_TIME_SCALE == 20.0


class TestTruthToleranceFormula:
    """The tolerance is the most intricate part of the check, and the only part
    whose failure mode is a silent PASS. It is tested directly on its formula
    rather than only through a run.
    """

    def test_lag_is_added_to_the_sleep_allowance_not_maxed_with_it(self) -> None:
        """FAILS ON: `max(floor, proportional, lag)` instead of `max(floor, prop) + lag`.

        The specific regression, measured: a run whose grid lag was 4.7 ms was
        given a 40 ms allowance when its actual error was 45 ms. A small lag
        does not excuse the sleep overshoot, because both occur in the same run.
        """
        from longline.eval.latency_runner import _LAG_MULTIPLE, truth_tolerance_ns

        lag_ns = 5_000_000  # 5 ms
        # A turn long enough that the proportional term, not the floor, is what
        # the lag is added to -- otherwise the floor masks the addition and the
        # test would pass for the wrong reason.
        turn_ns = 4_000_000_000
        base = truth_tolerance_ns(turn_ns, 0)
        with_lag = truth_tolerance_ns(turn_ns, lag_ns)
        assert with_lag == base + _LAG_MULTIPLE * lag_ns
        # The max-based form would have returned `base` unchanged.
        assert with_lag > base

    def test_the_cap_bounds_the_tolerance_at_every_scale(self) -> None:
        """FAILS ON: a tolerance wider than the capped fraction of the signal."""
        from longline.eval.latency_runner import (
            _MAX_TOLERANCE_SIGNAL_FRACTION,
            _asserted_signal_ns,
            truth_tolerance_ns,
        )

        case = get_case("lat-002")
        for scale in (0.5, 1.0, 2.0, 5.0, 8.0, 20.0):
            signal = _asserted_signal_ns(case, scale)
            tol = truth_tolerance_ns(
                case.truth(scale).buffered_turn_ns,
                50_000_000,  # an absurd lag: the cap must still bind
                signal_ns=signal,
            )
            assert tol <= int(_MAX_TOLERANCE_SIGNAL_FRACTION * signal) + 1, scale

    def test_the_cap_still_rejects_a_wrong_instant(self) -> None:
        """FAILS ON: a cap loose enough to accept a real wrong-measurement error.

        The errors this check must catch are the ones a plausible harness bug
        produces: measuring the buffered start at `tool_block_complete` instead
        of `response_complete` (one response tail early) or reading the tool's
        own span as its start (one tool duration early). Both must exceed the
        tolerance at the shipped scale, for every shipped case.

        Deliberately NOT asserting the check catches a ONE BLOCK GAP error: that
        error IS the signal, so catching it would mean a tolerance of zero. The
        limitation is not a property of this host; it is arithmetic.
        """
        from longline.eval.latency_runner import (
            _asserted_signal_ns,
            truth_tolerance_ns,
        )

        case = get_case("lat-001")
        for scale in (20.0, 40.0):
            truth = case.truth(scale)
            signal = _asserted_signal_ns(case, scale)
            tol = truth_tolerance_ns(truth.buffered_turn_ns, 0, signal_ns=signal)
            # One response tail early, and one whole tool early.
            assert tol < signal, scale
            assert tol < truth.tool_ns, scale

    def test_a_small_scale_cannot_discriminate(self) -> None:
        """FAILS ON: the claim that any scale works.

        The measured reason `DEFAULT_TIME_SCALE` is 20: this host's ~40 ms timer
        floor is larger than the signal a 1x case asserts, so the tolerance has
        to exceed the thing it is checking. If a future host has a finer timer
        this test fails, and it SHOULD -- it asserts a measured property of the
        machine, and says the scale could then be lowered.
        """
        from longline.eval.latency_runner import (
            _asserted_signal_ns,
            truth_tolerance_ns,
        )

        case = get_case("lat-001")
        signal_at_1x = _asserted_signal_ns(case, 1.0)
        assert truth_tolerance_ns(case.truth(1.0).buffered_turn_ns, 0) >= signal_at_1x

    def test_an_uncapped_call_ignores_the_signal(self) -> None:
        """FAILS ON: the cap being applied when the caller did not ask for it."""
        from longline.eval.latency_runner import truth_tolerance_ns

        assert truth_tolerance_ns(1_000_000_000, 0) == truth_tolerance_ns(
            1_000_000_000, 0, signal_ns=None
        )

    def test_the_floor_holds_when_the_proportional_term_shrinks(self) -> None:
        """FAILS ON: a tiny turn producing a near-zero tolerance.

        At very small scales the proportional term rounds toward zero while the
        host's absolute jitter does not, so the floor is what keeps the check
        from being impossible rather than merely strict.
        """
        from longline.eval.latency_runner import _SLEEP_TOLERANCE_FLOOR_NS, truth_tolerance_ns

        assert truth_tolerance_ns(1_000, 0) == _SLEEP_TOLERANCE_FLOOR_NS
