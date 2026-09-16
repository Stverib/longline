"""Prompt templates used when fanning work out to sub-agents."""

from __future__ import annotations

# 子 Agent 的指令模板。{count} 是分到的任务数。
SUBAGENT_PROMPT = "You have {count} task(s). Return one result per task, in order."


def render_subagent_prompt(count: int) -> str:
    """Render the sub-agent instruction for `count` tasks."""
    return SUBAGENT_PROMPT.format(count=count)


def render_leader_prompt(num_workers: int) -> str:
    """Render the leader's instruction for `num_workers` sub-agents.

    Returns a fixed English string; its exact text is asserted by
    `tests/test_coordinator.py`.
    """
    return f"Merge the results of {num_workers} sub-agents."
