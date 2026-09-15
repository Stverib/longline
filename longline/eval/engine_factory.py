"""Assemble a QueryEngine under evaluation.

A minimal, deterministic toolset (no networking, no nested-agent, no team or
permission-prompting tools) keeps eval runs cheap and reproducible. The engine
uses a BYPASS permission context so every tool call executes without asking.

The toolset is selected by **profile** (see `longline/eval/eval_tools.py`): a
case that expects the agent to reach for a web or notebook or task tool must
run against a registry that actually offers it, otherwise the expectation is
unsatisfiable and the selection rate measures nothing. The default profile is
the original six-tool core set, so existing runs are unchanged.
"""

from __future__ import annotations

from longline.core.query_engine import QueryEngine

# NOTE: anthropic SDK is imported lazily inside build_engine() so that this
# module stays importable and unit-testable without hitting the network.
from longline.eval.eval_tools import build_eval_registry
from longline.permissions.gate import PermissionContext, PermissionMode
from longline.prompts.builder import build_system_prompt

# Tools the model under evaluation may use during eval runs (default profile).
EVAL_TOOL_NAMES = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")


def build_engine(
    *,
    sandbox: str,
    model: str,
    api_key: str,
    tool_profile: str = "core",
) -> QueryEngine:
    """Build a QueryEngine wired for evaluation.

    - sandbox: absolute path to a temp dir that the Bash tool runs inside.
    - model: model id to evaluate.
    - api_key: key for the client.
    - tool_profile: which tool families to register (see eval_tools). An
      unknown profile raises rather than silently falling back to the core
      set, because a silently smaller toolset still produces a plausible
      accuracy number.
    """
    import anthropic

    system = "\n\n".join(build_system_prompt(cwd=sandbox, model=model))
    permission_ctx = PermissionContext(
        mode=PermissionMode.BYPASS,
        is_interactive=False,
    )
    return QueryEngine(
        client=anthropic.AsyncAnthropic(api_key=api_key),
        model=model,
        registry=build_eval_registry(sandbox, profile=tool_profile),
        system_prompt=system,
        permission_ctx=permission_ctx,
        max_turns=50,
    )
