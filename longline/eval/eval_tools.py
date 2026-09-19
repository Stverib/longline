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

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from longline.eval.constraint_enforcer import strip_forbidden
from longline.eval.tool_desc_variants import (
    STEERED_DESCRIPTIONS,
    DescriptionVariantTool,
    steered_tool_names,
)
from longline.tools.base import (
    ReconcileOutcome,
    Tool,
    ToolRegistry,
    ToolResult,
    ToolSchema,
)
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


# --- the sandbox boundary ---------------------------------------------------
#
# Every path-bearing tool is confined to the case's sandbox. Without this the
# suite measures something other than what it claims to.
#
# `Glob`/`Grep` default to the PROCESS working directory, and an eval run's
# process cwd is the repository root, not the sandbox. A model asked about "the
# sandbox's analysis.ipynb" that reaches for `Glob("**/analysis.ipynb")` is
# therefore handed a path inside `evals/fixtures/`, reads it, and edits it IN
# PLACE. Measured rather than hypothesised: one tool-calling run made 12 such
# calls across 4 cases and corrupted the tracked fixture
# `evals/fixtures/notebook_repo/analysis.ipynb`. The case isolation the contract
# requires (`evals/README.md` §8.3) is broken the moment one case writes into
# another case's starting state, and a run like that produces numbers that are
# about the repository rather than about the model.
#
# The rule is the one a shell inside a sandbox would apply, not a blanket
# refusal:
#
# - a RELATIVE path resolves against the sandbox, which IS the working directory
#   the system prompt declares;
# - an ABSOLUTE path inside the sandbox is allowed;
# - an ABSOLUTE path outside it is refused, as a tool error the model sees and
#   can recover from.
#
# Refusing rather than silently rewriting matters: a model that meant to write
# outside must be told it cannot, or the run records a success that a real
# deployment would not have produced.
#
# Known gap: `Bash` is not confined by this. It starts in the sandbox
# (`BashTool(cwd=sandbox)`) but a command can still `cd` out or name an absolute
# path, and no amount of argument inspection confines a shell. The structured
# tools above are where writes happen in practice, and they are the ones this
# closes; a case is free to make a Bash escape a grading criterion, but no case
# should have one happen by accident.

# tool name -> the argument that carries a filesystem path
SANDBOXED_PATH_ARG: dict[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "path",
    "Grep": "path",
    "NotebookEdit": "notebook_path",
}


class SandboxedTool(Tool):
    """Delegates to a production tool, confining its path argument to `sandbox`.

    Subclasses `Tool` rather than duck-typing, so the registry swap is a
    type-level fact -- the same reason `faults.ToolFaultWrapper` and
    `safety_runner.SentinelTool` do.

    `get_name` and `get_schema` are forwarded unchanged: the model must see the
    production tool exactly, or the tool-selection metric is measuring a
    different menu. Only `execute` differs, and only in where the path may point.
    """

    def __init__(self, inner: Tool, sandbox: str, path_arg: str) -> None:
        self._inner = inner
        self._sandbox = Path(sandbox).resolve()
        self._path_arg = path_arg

    def get_name(self) -> str:
        return self._inner.get_name()

    def get_schema(self) -> ToolSchema:
        return self._inner.get_schema()

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return self._inner.is_concurrency_safe(tool_input)

    def _bound(self, tool_input: dict[str, Any]) -> dict[str, Any] | ToolResult:
        """The tool input with its path argument resolved into the sandbox.

        Returns the refusal `ToolResult` when the path points outside, which is
        what `execute` reports. All three of `execute`, `workload` and `reconcile`
        go through here on purpose: `execute` hands the inner tool a REWRITTEN
        argument, so anything reading the raw one would be describing a different
        file from the one the tool actually touches -- an empty read/write set, or
        a digest of a path nothing writes.
        """
        arg = self._path_arg
        raw = tool_input.get(arg)
        if not raw:
            # `Glob`/`Grep` read a missing `path` as "the working directory",
            # which for an eval case is the sandbox -- not wherever the eval
            # process was launched from. The other tools declare the argument
            # required and are left to report that themselves.
            if arg == "path" and raw is None:
                return {**tool_input, arg: str(self._sandbox)}
            return tool_input

        candidate = Path(str(raw))
        resolved = candidate if candidate.is_absolute() else self._sandbox / candidate
        resolved = resolved.resolve()
        if not resolved.is_relative_to(self._sandbox):
            return ToolResult(
                content=(
                    f"Error: {arg}={raw!r} is outside this case's sandbox. "
                    f"The working directory is {self._sandbox}; use a path inside it."
                ),
                is_error=True,
            )
        return {**tool_input, arg: str(resolved)}

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        bound = self._bound(tool_input)
        if isinstance(bound, ToolResult):
            return bound
        return await self._inner.execute(bound)

    def workload(self, tool_input: dict[str, Any]) -> dict[str, str]:
        """The declared paths, resolved exactly as `execute` resolves them.

        Not a pass-through, and not optional: `Tool.workload` defaults to `{}`, so
        a wrapper that neither forwarded nor re-resolved would leave the journal
        with no digests at all -- silently disabling reconciliation and the
        workspace identity, and reporting a clean recovery for a blinded runtime.
        A refused call touches nothing inside the sandbox, so it declares nothing.
        """
        bound = self._bound(tool_input)
        if isinstance(bound, ToolResult):
            return {}
        return self._inner.workload(bound)

    def reconcile(self, tool_input: dict[str, Any]) -> ReconcileOutcome:
        """Forwarded with the same resolution, for the same reason."""
        bound = self._bound(tool_input)
        if isinstance(bound, ToolResult):
            return ReconcileOutcome.UNKNOWN
        return self._inner.reconcile(bound)


def build_eval_registry(
    sandbox: str,
    *,
    profile: str = "core",
    task_store: TaskStore | None = None,
    forbidden: Iterable[str] = (),
    tool_desc_variant: str = "baseline",
) -> ToolRegistry:
    """Assemble a ToolRegistry for one eval case.

    `task_store` is injectable so each case gets its own task state: the task
    tools default to a process-wide singleton, and sharing it across cases
    would let one case's tasks leak into the next one's `TaskList` output,
    breaking the contract's "cases do not share mutable state" rule
    (`evals/README.md` §8.3).

    `tool_desc_variant` replaces the description TEXT of the named tools, for
    an A/B on wording. The registry still offers every tool in the profile:
    this is a wording experiment, never a visibility one, because hiding a tool
    would make the evaluator do half the routing the metric exists to measure.
    `"baseline"` leaves every description exactly as production serves it, so
    the control arm is not perturbed by the scaffold.
    """
    names = build_tool_profile(profile)
    store = task_store if task_store is not None else TaskStore()

    # Factories, not instances: the registry owns exactly one of each, but the
    # sandbox-bound and store-bound ones need their dependency at construction.
    #
    # Every path-bearing tool goes through `SandboxedTool`, which resolves
    # relative paths against the sandbox and refuses absolute ones outside it.
    # `Bash` takes the sandbox as its cwd instead -- see the note above the
    # wrapper for why a shell cannot be confined the same way.
    def _sandboxed(factory: Any, name: str) -> Any:
        return lambda: SandboxedTool(factory(), sandbox, SANDBOXED_PATH_ARG[name])

    factories: dict[str, Any] = {
        "Bash": lambda: BashTool(cwd=sandbox),
        "Read": _sandboxed(FileReadTool, "Read"),
        "Write": _sandboxed(FileWriteTool, "Write"),
        "Edit": _sandboxed(FileEditTool, "Edit"),
        "Glob": _sandboxed(GlobTool, "Glob"),
        "Grep": _sandboxed(GrepTool, "Grep"),
        "WebSearch": EvalWebSearchTool,
        "WebFetch": EvalWebFetchTool,
        "NotebookEdit": _sandboxed(NotebookEditTool, "NotebookEdit"),
        "TaskCreate": lambda: TaskCreateTool(store),
        "TaskGet": lambda: TaskGetTool(store),
        "TaskList": lambda: TaskListTool(store),
        "TaskUpdate": lambda: TaskUpdateTool(store),
        "TaskStop": lambda: TaskStopTool(store),
    }

    registry = ToolRegistry()
    steered = steered_tool_names(tool_desc_variant)
    for name in names:
        tool: Any = factories[name]()
        if name in steered:
            tool = DescriptionVariantTool(tool, STEERED_DESCRIPTIONS[name])
        registry.register(tool)
    strip_forbidden(registry, forbidden)
    return registry
