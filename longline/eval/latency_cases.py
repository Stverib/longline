"""Micro-benchmark cases for the streaming-vs-buffered latency A/B (contract §5.5).

A case declares three things and nothing else:

1. `tools`  -- the timed stand-ins in the registry, with the duration each real
   `execute()` will take. Same for both arms: "工具耗时完全相同".
2. `calls`  -- the tool calls the scripted model emits, in the order it emits
   them, with the timestamp the model finishes each block.
3. `timing` -- how long the model's response runs after its last tool block.

`truth` is the closed-form expectation for the turn, computed from `calls` and
`timing`. It is written here, from the case's own declared inputs, rather than
read back out of the runner -- a test that derived its expectation from a field
the runner wrote would pass for any runner, including one measuring nothing.

=== The single-tool case is the one that pins the model ===

One tool call makes the shape of the effect decidable with the least machinery.
The block finishes at `t_1` and the response finishes at `T_response = t_1 +
tail`, where the tail is the rest of the response after that block:

```text
                 ToolStartLatency        TurnLatency
buffered         T_response = t_1 + tail t_1 + tail + L
streaming        t_1                     max(t_1 + tail, t_1 + L)
```

So even a single call separates the arms, by exactly the response tail. The
saving is the tail, not the tool: what streaming buys is the *remaining model
time after the decision is known*. `lat-001` asserts both arms agree on results
and pins the difference to the tail precisely, so a harness that reported the
tool's duration as the saving would be caught.

A `tail == 0` case would additionally make the arms identical, and that variant
is asserted separately in the tests by constructing one.

For multiple calls, with `t_j` the j-th block's completion:

```text
                 ToolStartLatency        TurnLatency
buffered         t_n                     t_n + L
streaming        t_1                     max(T_response, max_j(t_j + L))
```

`t_n - t_1` is the extra response time the buffered arm sits on before it can
start anything, which is where a multi-call saving comes from. The streaming
turn is the `max` over EVERY call, not just the first: calls that start later
finish later, so a short tool whose last call lands late can push the turn past
the response. See `Truth.streaming_turn_ns`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- timing model -----------------------------------------------------------


@dataclass(frozen=True)
class TimingModel:
    """The scripted model's response clock.

    `block_delay_s` is the gap between consecutive `tool_use` blocks finishing,
    and also the unit the tail is measured in: the response completes
    `tail_blocks` block-lengths after its last tool block. Both arms are handed
    the identical grid.
    """

    block_delay_s: float
    tail_blocks: int = 1


@dataclass(frozen=True)
class ToolProfileSpec:
    """A timed stand-in tool: its name, how long it runs, and its batching class.

    `concurrency_safe` is not decoration -- it decides how `orchestration`
    partitions the calls into batches and whether `StreamingToolExecutor` starts
    a call immediately or queues it. A case that declared a shape the production
    code would schedule differently would measure a schedule that never happens.
    """

    name: str
    duration_s: float
    concurrency_safe: bool = True


@dataclass(frozen=True)
class ToolCallSpec:
    """One `tool_use` block the scripted model emits."""

    tool_id: str
    name: str
    input: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Truth:
    """The closed-form expectation for one case's turn, in nanoseconds.

    Computed from the case's declarations, never from a run. A runner that
    schedules differently from the case's own claim fails against this.
    """

    # `t_j` for each call, relative to request_start.
    block_offsets_ns: tuple[int, ...]
    # `T_response`, relative to request_start.
    response_ns: int
    # The duration each tool's body takes.
    tool_ns: int
    # Index (0-based) of the call the streaming arm starts first.
    first_call: int = 0

    @property
    def buffered_tool_start_ns(self) -> int:
        """Buffered starts nothing until the response is complete."""
        return self.response_ns

    @property
    def streaming_tool_start_ns(self) -> int:
        """Streaming starts the first call as soon as its block completes."""
        return self.block_offsets_ns[self.first_call]

    @property
    def buffered_turn_ns(self) -> int:
        return self.response_ns + self.tool_ns

    @property
    def streaming_turn_ns(self) -> int:
        """The turn ends when the response is done AND every tool it started is.

        The streaming arm starts each call as its block completes, so call `j`
        starts at `t_j` and finishes at `t_j + tool_ns`. The turn therefore ends
        at the LATEST of those, not at the first one:

            max(response_ns, max_j(t_j + tool_ns))
                              ^^^^^^^^^^^^^^^^^^^^^

        Taking only `t_first + tool_ns` is the tempting simplification and it is
        wrong whenever a later block's tool outlasts the response. With a grid
        `t_j` that grows and a fixed `tool_ns`, later calls start later and
        finish later:

            lat-002 at 2x: t = 40/80/120, tool = 100, resp = 160
              max_j(t_j + tool) = 120 + 100 = 220   <- the turn really ends here
              t_1   + tool     =  40 + 100 = 140    <- what the old form asserted

        The old form passed on any case whose FIRST tool dominates -- which is
        every case in this file with a single call -- so it survived until a
        multi-call case with a short tool was run. `lat-001` is unaffected
        because it has one call.
        """
        return max(
            self.response_ns,
            max(offset + self.tool_ns for offset in self.block_offsets_ns),
        )

    @property
    def start_lead_ns(self) -> int:
        """How much earlier the streaming arm starts its first tool.

        Zero exactly when the response has no tail (`T_response == t_1`), which
        is the only way the arms become identical for tool-start latency.
        """
        return self.buffered_tool_start_ns - self.streaming_tool_start_ns

    @property
    def starts_are_identical(self) -> bool:
        """True when this case predicts NO tool-start saving for either arm."""
        return self.start_lead_ns == 0


@dataclass(frozen=True)
class LatencyCase:
    """One micro-benchmark case.

    `repeat` re-emits the same ordered tool sequence within one turn. It is the
    honest lever for "a long response with many tool calls": repeating a declared
    sequence keeps the modelling content of the case exactly what it says, where
    inventing N distinct tool calls would only pad the case file.
    """

    id: str
    note: str
    timing: TimingModel
    tools: tuple[ToolProfileSpec, ...]
    calls: tuple[ToolCallSpec, ...]
    repeat: int = 1
    tags: tuple[str, ...] = ()

    @property
    def num_calls(self) -> int:
        return len(self.calls) * self.repeat

    def block_offsets_ns(self, time_scale: float = 1.0) -> tuple[int, ...]:
        """When each `tool_use` block finishes, in ns from `request_start`.

        `time_scale` is applied here and nowhere else, so the case's declared
        timing and the run's actual schedule cannot drift apart: the runner asks
        the case for the grid it is about to use rather than rebuilding it.
        """
        delay_ns = int(self.timing.block_delay_s * time_scale * 1e9)
        return tuple(delay_ns * (k + 1) for k in range(self.num_calls))

    def response_ns(self, time_scale: float = 1.0) -> int:
        """`T_response`: when the model's response stream finishes."""
        offsets = self.block_offsets_ns(time_scale)
        delay_ns = int(self.timing.block_delay_s * time_scale * 1e9)
        return offsets[-1] + self.timing.tail_blocks * delay_ns

    def truth(self, time_scale: float = 1.0) -> Truth:
        """The closed-form expectation, derived from this case's declarations.

        `time_scale` must be the same value the run used; `Truth` is otherwise
        the schedule the case describes at 1x.
        """
        offsets = self.block_offsets_ns(time_scale)
        # Every call in a repeat shares a tool profile by construction; a case
        # whose repeated calls had different durations would not have a single
        # `tool_ns`, so it is asserted at construction instead of guessed here.
        durations = {self._duration(name) for _, name in self._expanded()}
        if len(durations) != 1:
            raise ValueError(
                f"{self.id}: repeated calls must share one tool duration, "
                f"got {sorted(durations)}; the closed-form truth needs a single term"
            )
        return Truth(
            block_offsets_ns=offsets,
            response_ns=self.response_ns(time_scale),
            tool_ns=int(durations.pop() * time_scale * 1e9),
        )

    def _expanded(self) -> list[tuple[int, str]]:
        """`(index, tool_name)` for every call in the turn, in emission order."""
        return [
            (k, call.name)
            for k in range(self.num_calls)
            for call in [self.calls[k % len(self.calls)]]
        ]

    def _duration(self, name: str) -> float:
        for spec in self.tools:
            if spec.name == name:
                return spec.duration_s
        raise ValueError(f"{self.id}: no tool profile named {name!r}")

    def results(self) -> list[str]:
        """The exact result texts both arms must produce, in completion order."""
        return [f"{name} #{index} ok" for index, name in self._expanded()]

    def tool_calls(self) -> list[tuple[str, dict[str, object]]]:
        """`(tool_name, input)` for every call, in emission order."""
        return [(name, dict(self.calls[k % len(self.calls)].input)) for k, name in self._expanded()]

    def execution_durations_ns(self) -> list[int]:
        """How long each real `execute()` body must have taken."""
        return [int(self._duration(name) * 1e9) for _, name in self._expanded()]


# --- the suite --------------------------------------------------------------

# A single tool block, with a response tail after it. The smallest case that
# separates the arms: the block finishes at 20 ms, the response at 40 ms, so
# streaming starts 20 ms earlier and the saving is exactly the tail. There is no
# second block to overlap with, so anything the arms save here is the tail alone
# and nothing else -- which is what makes it the case that pins the model.
SINGLE_TOOL = LatencyCase(
    id="lat-001",
    note="one tool; the saving is the response tail, with no second block to hide behind",
    timing=TimingModel(block_delay_s=0.020, tail_blocks=1),
    tools=(ToolProfileSpec("Read", duration_s=0.150),),
    calls=(ToolCallSpec("tu-0", "Read", {"file_path": "a.py"}),),
    tags=("single-tool",),
)

# Three tools, each longer than the gap between blocks. The first starts 20 ms
# into the response instead of 80 ms in, and its 50 ms body buries the rest of
# the response: the turn ends at t1 + L either way.
THREE_TOOLS = LatencyCase(
    id="lat-002",
    note="three tools, each longer than the gap between blocks",
    timing=TimingModel(block_delay_s=0.020, tail_blocks=1),
    tools=(ToolProfileSpec("Read", duration_s=0.050),),
    calls=(
        ToolCallSpec("tu-0", "Read", {"file_path": "a.py"}),
        ToolCallSpec("tu-1", "Read", {"file_path": "b.py"}),
        ToolCallSpec("tu-2", "Read", {"file_path": "c.py"}),
    ),
    tags=("multi-tool",),
)

# A longer response: eight tool calls at 20 ms, so the buffered arm waits a full
# 180 ms before it starts anything at all while the streaming arm started at 20.
MANY_TOOLS = LatencyCase(
    id="lat-003",
    note="eight tool calls; the buffered arm waits out the whole response",
    timing=TimingModel(block_delay_s=0.020, tail_blocks=1),
    tools=(ToolProfileSpec("Read", duration_s=0.040),),
    calls=tuple(
        ToolCallSpec(f"tu-{i}", "Read", {"file_path": f"f{i}.py"}) for i in range(8)
    ),
    tags=("multi-tool", "long-response"),
)

LATENCY_CASES: tuple[LatencyCase, ...] = (SINGLE_TOOL, THREE_TOOLS, MANY_TOOLS)


def get_case(case_id: str, cases: tuple[LatencyCase, ...] = LATENCY_CASES) -> LatencyCase:
    """Look a case up by id, raising rather than returning None."""
    for case in cases:
        if case.id == case_id:
            return case
    raise KeyError(f"no latency case {case_id!r} (known: {[c.id for c in cases]})")


__all__ = [
    "LATENCY_CASES",
    "MANY_TOOLS",
    "SINGLE_TOOL",
    "THREE_TOOLS",
    "LatencyCase",
    "TimingModel",
    "ToolCallSpec",
    "ToolProfileSpec",
    "Truth",
    "get_case",
]
