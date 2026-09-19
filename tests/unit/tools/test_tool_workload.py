"""`Tool.workload` -- the declaration the journal and the identity both read.

One declaration serves three jobs: what to digest before and after a call, what
belongs in the session's read/write sets, and (for the two write tools) what
"did my effect land" means. None of the three can be answered from the tool's
NAME, because a name is a label and this is a fact about a specific call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from longline.tools.base import Tool, ToolResult, ToolSchema
from longline.tools.file_edit.file_edit_tool import FileEditTool
from longline.tools.file_read.file_read_tool import FileReadTool
from longline.tools.file_write.file_write_tool import FileWriteTool
from longline.tools.notebook.notebook_edit_tool import NotebookEditTool


class _NoWorkload(Tool):
    """A tool that declares nothing -- the default, and the honest answer for Bash."""

    def get_name(self) -> str:
        return "Nothing"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Nothing", description="", input_schema={})

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")


def test_the_default_declares_nothing() -> None:
    """A tool that cannot say what it touches must not guess.

    Bash is the case that matters: `echo x >> NOTES.md` and `git commit` are the
    same shape to this layer, so the only truthful answer is "no declaration",
    and the caller must treat that as "cannot verify" rather than "touches
    nothing".
    """
    assert _NoWorkload().workload({"command": "echo x >> NOTES.md"}) == {}


def test_the_declared_key_is_absolute() -> None:
    """Absolute, because the digester must not resolve it against ITS cwd.

    The tool knows how it resolves a path; the journal does not. Handing the
    journal a relative path would make the digest depend on the process's working
    directory, which differs between the killed leg and the resumed one.
    """
    declared = FileWriteTool().workload({"file_path": "a.py", "content": "x"})
    assert declared == {str(Path("a.py").resolve()): "write"}


@pytest.mark.parametrize(
    ("tool", "arg", "mode"),
    [
        (FileReadTool(), "file_path", "read"),
        (FileWriteTool(), "file_path", "write"),
        (FileEditTool(), "file_path", "write"),
        (NotebookEditTool(), "notebook_path", "write"),
    ],
)
def test_file_tools_declare_their_one_path(tool: Tool, arg: str, mode: str) -> None:
    target = Path("a.py").resolve()
    assert tool.workload({arg: str(target)}) == {str(target): mode}


def test_an_empty_path_declares_nothing_rather_than_the_cwd() -> None:
    """`Path("")` is `Path(".")` -- a whole directory, hashed.

    The tools reject an empty path at execute time, but `workload` is called
    BEFORE execution (that is its whole point), so it must not turn a malformed
    call into a digest of the current directory.
    """
    assert FileWriteTool().workload({"file_path": "", "content": "x"}) == {}
    assert FileReadTool().workload({}) == {}
    assert FileEditTool().workload({"file_path": ""}) == {}
    assert NotebookEditTool().workload({}) == {}


def test_the_declared_path_is_reported_even_though_the_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """A `Write` that CREATES a file declares a path with no pre-image.

    Declaration is not existence: the point of digesting before the call is to
    record that the file was not there.
    """
    target = tmp_path / "brand" / "new" / "file.txt"
    declared = FileWriteTool().workload({"file_path": str(target), "content": "x"})
    assert declared == {str(target): "write"}
    assert not target.exists()
