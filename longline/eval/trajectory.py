"""Extract an evaluation trajectory from a QueryEvent stream.

The agent under test is driven by QueryEngine.submit(); the events it yields
are the single source of truth for what the agent actually did. This module
projects them onto a compact `Trajectory` used by both judge layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from longline.core.events import ErrorEvent, TextDelta, ToolUseStart, TurnComplete

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from longline.core.events import QueryEvent

# tool call = (tool_name, input dict), in call order
ToolCall = tuple[str, dict[str, object]]


@dataclass
class Trajectory:
    """What the agent did during one evaluation run."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    text: str = ""
    errors: list[str] = field(default_factory=list)


async def extract_trajectory(stream: AsyncIterator[QueryEvent]) -> Trajectory:
    """Consume a QueryEvent stream and project it onto a Trajectory."""
    traj = Trajectory()
    async for event in stream:
        if isinstance(event, ToolUseStart):
            traj.tool_calls.append((event.tool_name, event.input))
        elif isinstance(event, TurnComplete):
            traj.turns += 1
            traj.input_tokens += event.usage.input_tokens
            traj.output_tokens += event.usage.output_tokens
        elif isinstance(event, TextDelta):
            traj.text += event.text
        elif isinstance(event, ErrorEvent):
            traj.errors.append(event.message)
    return traj
