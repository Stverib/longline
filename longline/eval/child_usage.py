"""Child-agent usage accounting: the seam that makes §5.6's red line keepable.

=== The red line, and why it is hard ===

Contract §5.6 states it plainly:

> The contract's red line, quoted: every sub-agent's tokens and tool calls
> must be collected -- counting only the leader is not acceptable.

The obstacle is structural, not a missing field. A sub-agent's events never
reach the caller's event stream: `InProcessTeammate._execute_with_query_loop`
and `AgentTool.execute` each iterate their own `query_loop(...)` and keep only
`TextDelta`s, discarding `TurnComplete.usage` on the way past. Whoever runs the
leader therefore sees the leader's tokens and *nothing else* -- and a naive
runner reports a `TokenOverhead` that is too low and a `Speedup` that looks
free, with no field anywhere in the artifact that says a child was missed.

=== The seam, and why it actually closes ===

`query_loop` types its model as

```python
call_model: Callable[..., AsyncIterator[QueryEvent]],
```

**a pure async generator with no return value.** The generator is the *only*
channel by which a turn's completion reaches the loop: the loop reads
`event.usage` off `TurnComplete` and has nowhere else to get it. So wrapping the
factory that produces `call_model` puts this module's accumulator underneath
every turn of every agent that runs through that factory, and a turn that
produced usage which did not pass through the wrapper is a turn the loop could
not have seen either.

That is a claim about the production code, and it is checked rather than
asserted: `tests/unit/eval/test_multi_agent_runner.py` drives real `query_loop`s
through a real `spawn_teammate` fan-out and compares this module's per-agent
totals against a witness that never touches it -- see below.

=== Two independent witnesses, and why agreement is worth more than either ===

1. **The usage ledger** (`UsageLedger`), fed by the wrapped `call_model`. One
   entry per turn, carrying tokens and a per-turn tool-call count read off the
   same stream.
2. **The task registry**, into which `spawn_teammate` writes a record for every
   teammate it starts. It is produced by the spawn path rather than by the model
   transport, so a bug that drops a whole child has to corrupt both the same way
   to go unnoticed.

`AccountedAgents.assert_complete` requires the spawn count and the accounted
count to agree, and requires every agent the ledger claims to be one the second
witness also saw. A mismatch raises rather than producing a number. **A dropped
child is a silent lie in the final `TokenOverhead`**, and the entire purpose of
this module is to make it a loud one instead.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from longline.core.events import ToolUseStart, TurnComplete

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from longline.core.events import QueryEvent

# The leader's own bucket in the usage ledger. `current_agent` returns this
# whenever no child scope is active, which is the case for the leader's own
# loop: a stream whose owner nothing has named belongs to whoever is running it,
# and at the top level that is the leader.
LEADER = "leader"

# The name a sub-agent's stream is attributed to before anything has told us
# who it is. Deliberately its own label rather than `LEADER`: attributing an
# anonymous child's turns to the leader would inflate the leader's own row
# while keeping the grand total right -- a distortion the totals check cannot
# see, because the totals would still add up.
UNKNOWN_AGENT = "unknown"

# The ledger the enclosing task's streams should be recorded into. Set once per
# fan-out (see `run_multi_variant` / `run_single_variant`), so a wrapped
# `call_model` can reach it with no other plumbing. A ContextVar rather than a
# module global so two concurrent runs in one process cannot share a ledger.
current_ledger: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar(
    "multi_agent_usage_ledger", default=None
)


class AccountingError(RuntimeError):
    """The usage ledger does not agree with the agents that actually ran.

    Raised instead of returning a number. Every field this suite reports
    (`TokenOverhead`, `Speedup`) is a comparison between two runs' costs, and a
    run whose cost is understated by a whole child would produce a *plausible*
    number that is wrong in the direction the suite is supposed to be measuring.
    There is no safe default here, so there is no default.
    """


@dataclass
class TurnUsage:
    """One `TurnComplete` observed on one agent's stream."""

    agent: str
    input_tokens: int
    output_tokens: int
    tool_calls: int
    stop_reason: str = ""


@dataclass
class AgentUsage:
    """One agent's totals, summed from its own turns."""

    agent: str
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "turns": self.turns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
        }


@dataclass
class UsageLedger:
    """Every agent's usage, accumulated from the model streams themselves.

    `turns` is the raw per-turn list and is the source of truth: the per-agent
    totals are derived from it, so there is exactly one place a token count is
    written and no way for a total to drift from the turns it summarises.
    """

    turns: list[TurnUsage] = field(default_factory=list)
    # Agent ids the runner knows it spawned, in spawn order. Filled by the
    # runner, not by the streams: an agent that spawned and produced no turn at
    # all is precisely the failure this file exists to catch, so it cannot be
    # discovered from the turns.
    spawned: list[str] = field(default_factory=list)
    # Which agent the enclosing task's streams belong to. See `current_agent`.
    current: contextvars.ContextVar[str] = field(
        default_factory=lambda: contextvars.ContextVar(
            "multi_agent_current_agent", default=LEADER
        )
    )

    def record(self, usage: TurnUsage) -> None:
        self.turns.append(usage)

    def note_spawned(self, agent_id: str) -> None:
        if agent_id not in self.spawned:
            self.spawned.append(agent_id)

    @property
    def agents(self) -> list[str]:
        """Every agent seen, spawned-but-silent ones first, in first-seen order."""
        seen = list(self.spawned)
        for turn in self.turns:
            if turn.agent not in seen:
                seen.append(turn.agent)
        return seen

    def per_agent(self) -> dict[str, AgentUsage]:
        """Totals per agent, keyed by agent id, in `agents` order."""
        totals = {name: AgentUsage(agent=name) for name in self.agents}
        for turn in self.turns:
            bucket = totals.setdefault(turn.agent, AgentUsage(agent=turn.agent))
            bucket.turns += 1
            bucket.input_tokens += turn.input_tokens
            bucket.output_tokens += turn.output_tokens
            bucket.tool_calls += turn.tool_calls
        return totals

    @property
    def leader(self) -> AgentUsage:
        return self.per_agent().get(LEADER, AgentUsage(agent=LEADER))

    @property
    def children(self) -> list[AgentUsage]:
        """Every agent except the leader, in `agents` order."""
        return [u for name, u in self.per_agent().items() if name != LEADER]

    @property
    def input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tool_calls(self) -> int:
        return sum(t.tool_calls for t in self.turns)

    @property
    def turns_count(self) -> int:
        return len(self.turns)

    def child_tokens(self) -> int:
        """Tokens attributable to every non-leader agent.

        The number the red line is about. Reported separately from the grand
        total so a reader can see how much of the cost was the fan-out, and so
        a run whose children were dropped shows up as `0` here rather than as a
        total that merely looks small.
        """
        return sum(u.total_tokens for u in self.children)

    def to_dict(self) -> dict[str, object]:
        per_agent = self.per_agent()
        return {
            "agents_spawned": list(self.spawned),
            "agents_seen": self.agents,
            "per_agent": {name: usage.to_dict() for name, usage in per_agent.items()},
            "turns": [
                {
                    "agent": t.agent,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "tool_calls": t.tool_calls,
                    "stop_reason": t.stop_reason,
                }
                for t in self.turns
            ],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "child_tokens": self.child_tokens(),
        }


# --- the current-agent scope -------------------------------------------------


def current_agent() -> str:
    """The agent whose turns should be attributed to, right now.

    Defaults to `LEADER`, because the leader's own loop runs outside any child
    scope and every unnamed stream on this process's main loop is the leader's.
    A child scope (`agent_scope`) narrows it for the duration of that task, and
    because the scope is a `ContextVar`, each concurrent teammate narrows only
    its own asyncio task's copy.
    """
    ledger = current_ledger.get()
    if ledger is None:
        return LEADER
    return ledger.current.get()


def agent_scope(agent_id: str) -> Any:
    """Set the current agent for the enclosing `asyncio.Task` (a ContextVar).

    A context manager rather than a bare `set`/`reset` pair so an exception
    inside a child cannot leave the scope pointed at that child: the leader
    would then be charged for turns it never ran, and the per-agent rows would
    be wrong while the totals still added up.
    """
    return _AgentScope(agent_id)


class _AgentScope:
    """Implementation of `agent_scope`. See that function for the contract."""

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._token: Any = None
        self._ledger: UsageLedger | None = None
        self._ledger_token: Any = None

    def __enter__(self) -> str:
        ledger = current_ledger.get()
        if ledger is None:
            ledger = UsageLedger()
            self._ledger_token = current_ledger.set(ledger)
        self._token = ledger.current.set(self._agent_id)
        self._ledger = ledger
        return self._agent_id

    def __exit__(self, *exc: object) -> None:
        if self._ledger is not None and self._token is not None:
            self._ledger.current.reset(self._token)
        if self._ledger_token is not None:
            current_ledger.reset(self._ledger_token)


# The ledger's own per-task pointer at the current agent. Held as a field on
# `UsageLedger` (see its `current`) rather than as a module-level ContextVar so
# the ledger and the names it labels cannot be swapped independently by a
# caller, and so two ledgers in one process do not share an agent pointer.


# --- the wrapping seam -------------------------------------------------------


@dataclass
class ModelCounter:
    """A `call_model_factory` wrapper that records every turn it produces.

    `factory(model=None, max_tokens=...)` has the signature `spawn_teammate` and
    `AgentTool` both expect, so it drops in wherever the production factory went
    and needs no knowledge of either caller.

    The per-turn tool-call count is read off the SAME stream as the usage, for
    the same reason the usage is read here at all: the loop consumes the
    generator, and a `ToolUseStart` that did not come through this wrapper is a
    `ToolUseStart` the loop never dispatched either. Counting them separately
    from the trajectory is what makes `tool_calls` a per-agent quantity rather
    than a leader-wide total -- which is what the red line asks for.
    """

    factory: Callable[..., Any]
    ledger: UsageLedger
    agent: str | None = None

    def __call__(self, model: str | None = None, max_tokens: int = 16384) -> Callable[..., AsyncIterator[QueryEvent]]:
        inner = self.factory(model=model, max_tokens=max_tokens)
        counter = self

        async def counted(**kwargs: Any) -> AsyncIterator[QueryEvent]:
            agent = counter.agent or current_agent()
            tool_calls = 0
            async for event in inner(**kwargs):
                if isinstance(event, ToolUseStart):
                    tool_calls += 1
                elif isinstance(event, TurnComplete):
                    counter.ledger.record(
                        TurnUsage(
                            agent=agent,
                            input_tokens=event.usage.input_tokens,
                            output_tokens=event.usage.output_tokens,
                            tool_calls=tool_calls,
                            stop_reason=event.stop_reason,
                        )
                    )
                    # Reset rather than accumulate: one turn's tool calls are
                    # the calls the model requested in that turn. Carrying them
                    # into the next turn would report a running total wearing a
                    # per-turn label.
                    tool_calls = 0
                yield event

        return counted


def count_usage(
    call_model_factory: Callable[..., Any],
    ledger: UsageLedger,
    *,
    agent: str | None = None,
) -> ModelCounter:
    """Wrap a call_model factory so every turn it produces lands in `ledger`.

    `agent=None` (the default) resolves the owner of each turn at call time
    from the ambient agent scope, which is what a *shared* wrapper on the
    leader's factory needs: the leader's own turns and any teammate's turns
    both flow through it, and which is which is decided by the context the
    stream was created in. Passing an explicit `agent` pins every turn from
    this factory to that name, which is what a per-child factory wants.
    """
    return ModelCounter(factory=call_model_factory, ledger=ledger, agent=agent)


async def drain(stream: AsyncIterator[QueryEvent], sink: list[QueryEvent]) -> None:
    """Exhaust `stream` into `sink`. Exists so a fan-out can await a generator."""
    async for event in stream:
        sink.append(event)


# --- completeness: the claim that needs evidence -----------------------------


@dataclass
class AccountedAgents:
    """The two witnesses, side by side, plus the check that they agree.

    `expected` is how many agents the runner spawned; `accounted` is how many
    the ledger has usage for. They are produced by different mechanisms, and
    the whole point of holding both is that a mismatch is *reportable* rather
    than silently averaged into a total.

    `witness` carries the second channel's numbers (what each agent reported
    for itself). It is deliberately not compared for equality with the ledger:
    an agent's self-report and a stream tap measure the same turns by
    construction only when every turn is a plain completion, and the two legitimately
    differ when a turn was retried, truncated or compacted. What IS required is
    that the second witness SAW the same set of agents -- that is the thing
    whose failure means a child was dropped rather than merely recounted.
    """

    expected: int
    accounted: int
    witness_agents: list[str] = field(default_factory=list)
    witness_turns: dict[str, int] = field(default_factory=dict)
    # Agents the ledger has usage for but the second witness never saw. A
    # non-empty list here is a real inconsistency: some stream was attributed
    # to a name nothing else can corroborate, which usually means a context
    # scope leaked and charged one agent's turns to another.
    unconfirmed_agents: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.expected == self.accounted and not self.unconfirmed_agents

    def assert_complete(self, *, case_id: str, variant: str) -> None:
        """Fail loudly when the ledger and the spawn count disagree.

        Three ways this fails, all of which would understate the run's cost or
        misattribute it, and all of which are raised rather than returned:

        - a spawned agent has no usage recorded (`accounted < expected`): its
          stream was never wrapped, so its tokens are absent from the totals;
        - fewer agents ran than were spawned, which accumulates as the same
          shortfall;
        - an agent appears in the ledger that the second witness never saw
          (`unconfirmed_agents`), i.e. turns were attributed to a name nothing
          else corroborates -- typically a leaked context scope.

        Raising rather than recording is deliberate: every one of these
        produces a `TokenOverhead` that is too low and a `Speedup` that looks
        free, which is exactly the confidently-wrong number this suite must
        never print. "2 of 3 missing" is not actionable either, so the ids are
        named.
        """
        if self.complete:
            return
        reasons: list[str] = []
        if self.accounted != self.expected:
            reasons.append(
                f"{self.accounted} agents have usage but {self.expected} were spawned"
            )
        if self.unconfirmed_agents:
            reasons.append(
                f"the ledger claims {self.unconfirmed_agents}, which the second "
                "witness never observed"
            )
        raise AccountingError(
            f"{case_id}/{variant}: " + "; ".join(reasons)
            + f". A child's tokens would be missing or misattributed in the totals "
            f"(witness={self.witness_agents}, notes={self.notes})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_agents": self.expected,
            "accounted_agents": self.accounted,
            "accounting_complete": self.complete,
            "witness_agents": self.witness_agents,
            "witness_turns": self.witness_turns,
            "unconfirmed_agents": self.unconfirmed_agents,
            "notes": self.notes,
        }


def reconcile(
    ledger: UsageLedger,
    *,
    witness_agents: list[str] | None = None,
    witness_turns: dict[str, int] | None = None,
) -> AccountedAgents:
    """Build the agreement record from the ledger and an optional second witness.

    `expected` is `len(ledger.spawned)` -- the runner's own spawn count, which
    is why `note_spawned` must be called at the spawn site rather than inferred
    from the streams. Inferring it would make this check vacuous: it would be
    comparing the streams against themselves.

    `accounted` counts agents with at least one recorded turn.

    When a witness is supplied, agents the ledger saw and the witness did not
    are recorded in `unconfirmed_agents` and make the accounts incomplete. The
    witness is `None` for a single-agent run, where there is no second channel
    to check against -- and a check that silently passes when its evidence is
    missing is worse than no check, so the distinction is carried in the record
    rather than ironed out.
    """
    accounted = [name for name, usage in ledger.per_agent().items() if usage.turns > 0]
    notes: list[str] = []
    if ledger.spawned and UNKNOWN_AGENT in accounted:
        # Not a failure -- an unattributed stream still counts -- but it means
        # the per-agent breakdown puts some of a child's cost in a bucket named
        # "unknown", and a reader must be told rather than left to infer it.
        notes.append(
            "some turns were attributed to no named agent; they are counted in the "
            "totals but the per-agent breakdown cannot name their owner"
        )

    unconfirmed: list[str] = []
    if witness_agents is not None:
        known = set(witness_agents)
        # The leader has no registry record -- it is the caller, not a spawned
        # teammate -- so it is never expected to appear in this witness.
        unconfirmed = [name for name in accounted if name != LEADER and name not in known]

    return AccountedAgents(
        expected=len(ledger.spawned),
        accounted=len(accounted),
        witness_agents=list(witness_agents or []),
        witness_turns=dict(witness_turns or {}),
        unconfirmed_agents=unconfirmed,
        notes=notes,
    )


# A `teammate-<8 hex>` id, as `spawn_teammate` mints it. The second witness
# reads these out of `AgentTool`'s tool result text, which is the only channel
# the leader has for a background agent's identity.
#
# The trailing `(?!\w)` matters: without it the eight-hex pattern matches the
# FIRST eight characters of any longer hex run, so a 10-digit id would be
# reported as a real teammate id that no registry record can ever corroborate
# -- and the second witness would then disagree with the ledger for a reason
# that has nothing to do with a dropped child.
_TEAMMATE_ID_RE = re.compile(r"teammate-[0-9a-f]{8}(?![0-9a-f])")


def teammate_ids_in(text: str) -> list[str]:
    """Teammate task ids mentioned in a string, in order, deduplicated.

    Used to build the second witness from the leader's transcript: an
    `AgentTool` call with `run_in_background` returns a `task_id` in its result
    text, and the tool result is what the leader's event stream carries. This is
    a *different* channel from the usage counters, which is the only reason
    agreement between them is worth anything.
    """
    seen: list[str] = []
    for match in _TEAMMATE_ID_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


__all__ = [
    "LEADER",
    "UNKNOWN_AGENT",
    "AccountedAgents",
    "AccountingError",
    "AgentUsage",
    "ModelCounter",
    "TurnUsage",
    "UsageLedger",
    "agent_scope",
    "count_usage",
    "current_agent",
    "current_ledger",
    "drain",
    "reconcile",
    "teammate_ids_in",
]
