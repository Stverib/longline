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

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from longline.core.query_engine import QueryEngine

# NOTE: anthropic SDK is imported lazily inside build_engine() so that this
# module stays importable and unit-testable without hitting the network.
from longline.eval.eval_tools import build_eval_registry
from longline.permissions.gate import PermissionContext, PermissionMode
from longline.prompts.builder import build_system_prompt

# Tools the model under evaluation may use during eval runs (default profile).
EVAL_TOOL_NAMES = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")

# The client identity this harness presents to the gateway.
#
# The OpenCode gateway (`opencode.ai/zen/go`) refuses a request that looks like a
# bare SDK call. Its documentation asks a client to "identify itself with its own
# user agent, such as `my-coding-agent/1.0`" rather than an HTTP library's
# default, and to "send a stable session ID in `x-opencode-session` for each
# conversation". A request without both is answered with an HTTP 403 carrying a
# Cloudflare interstitial -- which reads exactly like a network block, and cost
# this project a detour chasing one.
#
# Nothing here is OpenCode-specific in a way that breaks a real Anthropic
# endpoint: an unknown header is ignored, and naming the client is good manners
# against any gateway.
EVAL_USER_AGENT = "longline-eval/1.0"


def client_headers(session_id: str) -> dict[str, str]:
    """Headers that identify this harness and pin one conversation's routing.

    `session_id` is what the gateway routes and caches on, so it must be stable
    WITHIN a conversation and different BETWEEN them. Each eval case is one
    conversation, and its sandbox is unique per case, so the sandbox is the
    natural source -- see `build_engine`.
    """
    return {"x-opencode-session": session_id, "user-agent": EVAL_USER_AGENT}


def build_engine(
    *,
    sandbox: str,
    model: str,
    api_key: str,
    tool_profile: str = "core",
    forbidden: Iterable[str] = (),
    session_id: str | None = None,
) -> QueryEngine:
    """Build a QueryEngine wired for evaluation.

    - sandbox: absolute path to a temp dir that the Bash tool runs inside.
    - model: model id to evaluate.
    - api_key: key for the client.
    - tool_profile: which tool families to register (see eval_tools). An
      unknown profile raises rather than silently falling back to the core
      set, because a silently smaller toolset still produces a plausible
      accuracy number.
    - forbidden: tool names this case may not call (see
      `longline/eval/constraint_enforcer.py`). Stripped from the registry
      the engine runs against, so the harness carries the constraint.
    - session_id: the gateway's routing key for this conversation. Defaults to
      the sandbox's basename, which is already unique per case; callers that
      know something more meaningful (a case id) may pass it instead.
    """
    import anthropic

    system = "\n\n".join(build_system_prompt(cwd=sandbox, model=model))
    permission_ctx = PermissionContext(
        mode=PermissionMode.BYPASS,
        is_interactive=False,
    )
    return QueryEngine(
        client=anthropic.AsyncAnthropic(
            api_key=api_key,
            default_headers=client_headers(session_id or Path(sandbox).name),
        ),
        model=model,
        registry=build_eval_registry(
            sandbox, profile=tool_profile, forbidden=forbidden,
        ),
        system_prompt=system,
        permission_ctx=permission_ctx,
        max_turns=50,
    )
