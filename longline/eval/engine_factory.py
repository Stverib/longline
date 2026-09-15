"""Assemble a QueryEngine under evaluation.

A minimal, deterministic toolset (no networking, no nested-agent, no team or
permission-prompting tools) keeps eval runs cheap and reproducible. The engine
uses a BYPASS permission context so every tool call executes without asking.
"""

from __future__ import annotations

from longline.core.query_engine import QueryEngine

# NOTE: anthropic SDK is imported lazily inside build_engine() so that this
# module stays importable and unit-testable without hitting the network.
from longline.permissions.gate import PermissionContext, PermissionMode
from longline.prompts.builder import build_system_prompt
from longline.tools.base import ToolRegistry
from longline.tools.bash.bash_tool import BashTool
from longline.tools.file_edit.file_edit_tool import FileEditTool
from longline.tools.file_read.file_read_tool import FileReadTool
from longline.tools.file_write.file_write_tool import FileWriteTool
from longline.tools.glob_tool.glob_tool import GlobTool
from longline.tools.grep_tool.grep_tool import GrepTool

# Tools the model under evaluation may use during eval runs.
EVAL_TOOL_NAMES = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")


def _build_registry(sandbox: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(BashTool(cwd=sandbox))
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


def build_engine(*, sandbox: str, model: str, api_key: str) -> QueryEngine:
    """Build a QueryEngine wired for evaluation.

    - sandbox: absolute path to a temp dir that the Bash tool runs inside.
    - model: model id to evaluate.
    - api_key: key for the client.
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
        registry=_build_registry(sandbox),
        system_prompt=system,
        permission_ctx=permission_ctx,
        max_turns=50,
    )
