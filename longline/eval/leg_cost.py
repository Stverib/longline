"""What one leg of agent work cost: the units the restart-vs-resume comparison needs.

A "leg" is one process running the agent loop. Three of them matter here, and the
first two are the comparison:

- **restart** -- the same task from scratch, uninterrupted. This is what a crashed
  session has to choose between doing and NOT doing.
- **resume** -- continue from the checkpoint. The saving is `restart - resume`.
- **arm** (the killed leg) -- measured and reported, but it is SUNK COST. The
  process died either way, so its work is paid in both branches and subtracting it
  from one of them would flatter the result.

=== Why these counters are where they are ===

`model_calls` and `tool_calls` are counted at the two chokepoints every call
passes through: the model wrapper and the registry's `GatedTool` wrappers. They
are deliberately NOT read off the transcript, because the transcript is written at
step boundaries -- a leg that dies mid-step would report less work than it did,
which is the exact quantity this comparison is about.

`loop_ms` covers the agent loop only, from the first instruction to the last. It
excludes interpreter startup and engine construction, which the parent measures
separately as whole-subprocess wall time. Mixing the two would make a two-second
loop look like a three-second one and hide the difference being measured.

=== What a token count can and cannot say here ===

Under a scripted model the token totals are a deterministic function of the call
count -- the script emits a fixed `Usage` per turn, so `input_tokens` is
`100 * model_calls` and `output_tokens` is `20 * (tool turns) + 25`. The field is
reported because a real model would make it informative, but in THIS suite it
carries no fact that `model_calls` does not already carry, and no claim here
should be read as a token-efficiency result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = ["COST_FIELDS", "LegCost", "mean_cost", "saving"]

# The fields, in one place, so the row, the summary and the table cannot drift.
COST_FIELDS: tuple[str, ...] = (
    "model_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "loop_ms",
)


@dataclass
class LegCost:
    """One leg's work, as counted while it ran."""

    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    loop_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in COST_FIELDS}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Any) -> LegCost:
        """Read a worker report's `cost` block, tolerating a leg that died.

        A killed or failed leg may report nothing at all, and the honest reading of
        "no report" is zeroes rather than an exception: this is a measurement of a
        process that was supposed to die.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            model_calls=_as_int(data.get("model_calls")),
            tool_calls=_as_int(data.get("tool_calls")),
            input_tokens=_as_int(data.get("input_tokens")),
            output_tokens=_as_int(data.get("output_tokens")),
            loop_ms=_as_float(data.get("loop_ms")),
        )


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean_cost(costs: Sequence[LegCost]) -> dict[str, float]:
    """Per-field means, and zeroes for an empty sequence.

    Empty is zero rather than an error because the caller has usually already
    decided whether it has any samples -- and a division that raised here would
    turn "this cell has no restart baselines" into a crash instead of a row of
    zeroes a reader can see and question.
    """
    if not costs:
        return {name: 0.0 for name in COST_FIELDS}
    return {
        name: sum(float(getattr(cost, name)) for cost in costs) / len(costs)
        for name in COST_FIELDS
    }


def saving(restart: float, resume: float) -> float:
    """The fraction of the restart cost the resume saved, as a ratio.

    1.0 means the resume did no work at all, 0.0 means it redid everything. It is
    NOT clamped and NOT defended against a zero restart: a zero there means the
    restart never ran, and a caller that turned that into 0.0 would be reporting a
    measurement it does not have.
    """
    return 1.0 - (resume / restart)
