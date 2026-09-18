"""Tool-description variants for the routing ablation.

A tool description sits closer to the tool-choice decision than any system
prompt paragraph: the model reads it in the same request that it picks a tool.
This round tests whether steering those descriptions changes which tool the
agent reaches for, so the candidate text has to live BESIDE the current text
rather than replace it.

`STEERED_DESCRIPTIONS` is a CANDIDATE, not a decision. When the ablation names a
winner its text is promoted into the production tool class and this module plus
the `--tool-desc-variant` flag are deleted. The switch is scaffolding for one
experiment, not a feature: leaving it in would leave two sources of truth for
what a tool does.

Each entry states the SAME capability as the production text and adds the
routing rule. None of them claims a behaviour the tool does not have -- a
description that promised more than the tool delivers would win the ablation
for the wrong reason.
"""

from __future__ import annotations

from typing import Any

from longline.tools.base import Tool, ToolResult, ToolSchema

BASELINE = "baseline"
STEERED = "steered"
NOTEBOOK = "notebook"

# Steered descriptions, one per tool that has a substitution problem to fix.
# Bash and the three search/read tools address the Bash-for-everything habit;
# Edit and NotebookEdit address the measured Edit-for-NotebookEdit
# substitution behind three of the four notebook failures.
STEERED_DESCRIPTIONS: dict[str, str] = {
    "Bash": (
        "Executes a shell command and returns its output. Use Bash for shell "
        "execution: builds, tests, package managers, process control, git, or "
        "any operation no dedicated tool covers. Do NOT use Bash as a "
        "substitute for a dedicated tool that performs the same operation "
        "directly -- use Read instead of cat/head/tail/sed -n, Grep instead of "
        "grep/rg, Glob instead of find/ls, and Edit instead of sed -i."
    ),
    "Read": (
        "Reads a file from the filesystem. Prefer this over cat, head, tail, "
        "sed or awk for reading repository files; pass offset and limit to "
        "read a region instead of a whole file."
    ),
    "Grep": (
        "Searches file contents with a regex. Prefer this over a shell grep or "
        "rg: it is confined to the working directory, and its result is "
        "structured so the user can review it."
    ),
    "Glob": (
        "Finds files by name pattern. Prefer this over find or ls when "
        "locating files by name rather than by content."
    ),
    "Edit": (
        "Performs exact string replacements in a text file. Not for Jupyter "
        "notebooks: a .ipynb file is JSON, so an exact-string edit corrupts "
        "it. Use NotebookEdit for .ipynb."
    ),
    "NotebookEdit": (
        "Edits a Jupyter notebook (.ipynb) by cell: insert, replace or delete. "
        "Use this for every change to a .ipynb file, including adding or "
        "removing a cell; do not use Edit or Write on notebook JSON."
    ),
}

# The notebook arm steers only the substitution pair, so the cause behind three
# of the four notebook failures is tested on its own.
NOTEBOOK_STEERED: frozenset[str] = frozenset({"Edit", "NotebookEdit"})

_VARIANTS: dict[str, frozenset[str]] = {
    BASELINE: frozenset(),
    STEERED: frozenset(STEERED_DESCRIPTIONS),
    NOTEBOOK: NOTEBOOK_STEERED,
}


def steered_tool_names(variant: str) -> frozenset[str]:
    """Tool names whose description this variant replaces.

    Raises on an unknown variant rather than falling back to the baseline: a
    typo'd flag that silently produced two identical arms would look like a
    clean null result.
    """
    names = _VARIANTS.get(variant)
    if names is None:
        raise ValueError(
            f"unknown tool description variant: {variant!r} "
            f"(known: {sorted(_VARIANTS)})"
        )
    return names


class DescriptionVariantTool(Tool):
    """Delegates to a production tool, replacing only its description text.

    Subclasses `Tool` for the same reason `SandboxedTool` does: the swap is
    then a type-level fact rather than a duck-typing hope. `get_name`,
    `input_schema` and `execute` are forwarded unchanged, so the model sees the
    real tool with different wording -- the wording being the thing under test,
    and nothing else moving.
    """

    def __init__(self, inner: Tool, description: str) -> None:
        self._inner = inner
        self._description = description

    def get_name(self) -> str:
        return self._inner.get_name()

    def get_schema(self) -> ToolSchema:
        schema = self._inner.get_schema()
        return ToolSchema(
            name=schema.name,
            description=self._description,
            input_schema=schema.input_schema,
        )

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        return self._inner.is_concurrency_safe(tool_input)

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        return await self._inner.execute(tool_input)
