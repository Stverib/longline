"""Offline tool profiles for the tool-calling benchmark.

The eval registry is a *profile*, not a fixed list: a case that measures "does
the agent reach for the right tool" must run against a registry that actually
contains the plausible alternatives. Showing the model a WebSearch schema it
could never select would make the selection rate meaningless.

Two things are deliberate here:

- **The web tools are offline stand-ins with byte-identical schemas.** Web
  selection is the metric; the network is not. A flaky API would show up as
  model error, and the contract (`evals/README.md` §5.2) forbids that. So the
  stand-ins copy the production `get_schema()` verbatim — name, description and
  input_schema, exactly — and return a fixed response instead of a request.
  `tests/unit/eval/test_eval_tools.py` compares them field by field against the
  production classes, so the two cannot drift apart unnoticed.
- **Notebook and Task tools are the production classes, unmodified.** They have
  no network dependency, so a stand-in would only add a way for the eval
  registry and the production registry to disagree.
"""

from __future__ import annotations

from typing import Any

from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema
from longline.tools.bash.bash_tool import BashTool
from longline.tools.file_edit.file_edit_tool import FileEditTool
from longline.tools.file_read.file_read_tool import FileReadTool
from longline.tools.file_write.file_write_tool import FileWriteTool
from longline.tools.glob_tool.glob_tool import GlobTool
from longline.tools.grep_tool.grep_tool import GrepTool
from longline.tools.notebook.notebook_edit_tool import NotebookEditTool
from longline.tools.task_tools.task_tools import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
    TaskStore,
    TaskUpdateTool,
)
from longline.tools.web_fetch.web_fetch_tool import WEB_FETCH_TOOL_NAME, WebFetchTool
from longline.tools.web_search.web_search_tool import WEB_SEARCH_TOOL_NAME, WebSearchTool

# --- Web family: same schema as production, fixed response ---


class EvalWebSearchTool(Tool):
    """Offline WebSearch stand-in.

    `get_schema()` intentionally delegates to the production implementation
    rather than restating the schema here. A copy would be free to drift the
    moment someone edits the production description, and the drift would be
    invisible: the metric would silently measure a tool the model never sees
    in production.
    """

    def get_name(self) -> str:
        return WEB_SEARCH_TOOL_NAME

    def get_schema(self) -> ToolSchema:
        return WebSearchTool().get_schema()

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return True

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        query = tool_input.get("query", "")
        if not query:
            return ToolResult(content="Error: query is required", is_error=True)
        # 固定响应:内容可复现,且回显 query 让「search 之后是否用上结果」可判.
        return ToolResult(
            content=(
                f"[offline eval stub] Search results for {query!r}:\n"
                "1. Example result — https://example.com/result-1\n"
                "2. Example result — https://example.com/result-2\n"
            )
        )


class EvalWebFetchTool(Tool):
    """Offline WebFetch stand-in. See `EvalWebSearchTool` for the rationale."""

    def get_name(self) -> str:
        return WEB_FETCH_TOOL_NAME

    def get_schema(self) -> ToolSchema:
        return WebFetchTool().get_schema()

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return True

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        url = tool_input.get("url", "")
        if not url:
            return ToolResult(content="Error: url is required", is_error=True)
        return ToolResult(
            content=(
                f"# Offline eval stub\n\nFetched {url}\n\n"
                "This page is a fixed stand-in used by the evaluation suite.\n"
            )
        )


WEB_FAMILY: tuple[str, ...] = (WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME)
NOTEBOOK_FAMILY: tuple[str, ...] = ("NotebookEdit",)
TASK_FAMILY: tuple[str, ...] = (
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop",
)

# Tools whose schema does not depend on the sandbox, so the profile list can be
# stated by name and kept in sync with tests.
STANDIN_FAMILY: tuple[str, ...] = WEB_FAMILY + NOTEBOOK_FAMILY + TASK_FAMILY

CORE_FAMILY: tuple[str, ...] = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")

# Every tool name the tool-calling benchmark may legitimately expect.
ALL_EVAL_TOOL_NAMES: tuple[str, ...] = CORE_FAMILY + STANDIN_FAMILY

# profile name -> extra families layered on top of the core set.
_PROFILES: dict[str, tuple[str, ...]] = {
    "core": (),
    "web": WEB_FAMILY,
    "notebook": NOTEBOOK_FAMILY,
    "task": TASK_FAMILY,
    "all": STANDIN_FAMILY,
}


def build_tool_profile(profile: str) -> tuple[str, ...]:
    """Tool names for a named profile: core tools plus the requested families.

    Raises ValueError on an unknown profile rather than quietly falling back to
    the core set — a typo'd `--tool-profile` would otherwise silently change
    which tools the model can pick, and the resulting number would still look
    like a number.
    """
    extra = _PROFILES.get(profile)
    if extra is None:
        raise ValueError(
            f"unknown tool profile: {profile!r} (known: {sorted(_PROFILES)})"
        )
    return tuple(dict.fromkeys(CORE_FAMILY + extra))


def build_eval_registry(
    sandbox: str,
    *,
    profile: str = "core",
    task_store: TaskStore | None = None,
) -> ToolRegistry:
    """Assemble a ToolRegistry for one eval case.

    `task_store` is injectable so each case gets its own task state: the task
    tools default to a process-wide singleton, and sharing it across cases
    would let one case's tasks leak into the next one's `TaskList` output,
    breaking the contract's "cases do not share mutable state" rule
    (`evals/README.md` §8.3).
    """
    names = build_tool_profile(profile)
    store = task_store if task_store is not None else TaskStore()

    # Factories, not instances: the registry owns exactly one of each, but the
    # sandbox-bound and store-bound ones need their dependency at construction.
    factories: dict[str, Any] = {
        "Bash": lambda: BashTool(cwd=sandbox),
        "Read": FileReadTool,
        "Write": FileWriteTool,
        "Edit": FileEditTool,
        "Glob": GlobTool,
        "Grep": GrepTool,
        "WebSearch": EvalWebSearchTool,
        "WebFetch": EvalWebFetchTool,
        "NotebookEdit": NotebookEditTool,
        "TaskCreate": lambda: TaskCreateTool(store),
        "TaskGet": lambda: TaskGetTool(store),
        "TaskList": lambda: TaskListTool(store),
        "TaskUpdate": lambda: TaskUpdateTool(store),
        "TaskStop": lambda: TaskStopTool(store),
    }

    registry = ToolRegistry()
    for name in names:
        registry.register(factories[name]())
    return registry
