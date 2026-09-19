"""`Tool.reconcile` -- "did the call that never reported back take effect?"

Answers a question the journal cannot: a `PREPARED` with no `COMMITTED` says a
call STARTED, not whether it landed. Only the tool knows what its own effect
looks like from the outside, so only the tool can answer.

The default is UNKNOWN, and UNKNOWN is a real answer rather than a failure -- it
is what keeps the runtime from replaying a `Bash` command whose effect it cannot
read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from longline.tools.base import ReconcileOutcome, Tool, ToolResult, ToolSchema
from longline.tools.file_edit.file_edit_tool import FileEditTool
from longline.tools.file_write.file_write_tool import FileWriteTool

if TYPE_CHECKING:
    from pathlib import Path


class _Opaque(Tool):
    def get_name(self) -> str:
        return "Opaque"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="Opaque", description="", input_schema={})

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")


def test_the_default_cannot_tell() -> None:
    assert _Opaque().reconcile({"command": "rm -rf build"}) is ReconcileOutcome.UNKNOWN


def test_write_is_applied_when_the_file_holds_exactly_what_it_wrote(tmp_path: Path) -> None:
    """A full overwrite is self-identifying: the file either is the content or is not."""
    target = tmp_path / "a.txt"
    target.write_text("new", encoding="utf-8")
    outcome = FileWriteTool().reconcile({"file_path": str(target), "content": "new"})
    assert outcome is ReconcileOutcome.APPLIED


def test_write_is_not_applied_when_the_file_still_holds_the_old_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    outcome = FileWriteTool().reconcile({"file_path": str(target), "content": "new"})
    assert outcome is ReconcileOutcome.NOT_APPLIED


def test_write_to_a_path_that_does_not_exist_is_not_applied(tmp_path: Path) -> None:
    """The file being absent is proof the write did not land, not a missing input.

    NOT_APPLIED is what authorises the retry, and a create that never happened is
    exactly the case that needs one.
    """
    outcome = FileWriteTool().reconcile(
        {"file_path": str(tmp_path / "nope.txt"), "content": "new"}
    )
    assert outcome is ReconcileOutcome.NOT_APPLIED


def test_write_recognises_content_the_writer_stored_with_crlf(tmp_path: Path) -> None:
    """The regression this test was written to catch, found by running it.

    `FileWriteTool.execute` opens its target in TEXT mode, so on Windows a `\\n`
    in the argument reaches the disk as `\\r\\n`. A byte-for-byte comparison then
    reports a successful write as NOT_APPLIED -- and NOT_APPLIED is an
    authorisation to RETRY, so the bug would have produced exactly the duplicated
    side effect this whole mechanism exists to prevent.
    """
    target = tmp_path / "a.txt"
    target.write_bytes(b"a\r\nb\r\n")
    outcome = FileWriteTool().reconcile({"file_path": str(target), "content": "a\nb\n"})
    assert outcome is ReconcileOutcome.APPLIED


def test_write_is_non_ascii_safe(tmp_path: Path) -> None:
    """The comparison must not depend on the locale's default encoding.

    `read_text()` without an encoding uses one, and the killed leg and the
    resumed leg are not guaranteed to be the same machine.
    """
    target = tmp_path / "a.txt"
    target.write_bytes("修正 bug\n".encode())
    outcome = FileWriteTool().reconcile(
        {"file_path": str(target), "content": "修正 bug\n"}
    )
    assert outcome is ReconcileOutcome.APPLIED


def test_edit_recognises_a_new_string_that_spans_a_crlf_line(tmp_path: Path) -> None:
    """The same newline trap, one module over.

    `Edit` reads and writes BYTES, so it preserves whatever endings the file
    had, and the `new_string` it substitutes in carries the model's own `\\n`
    verbatim. The result is a file with MIXED endings, and a substring test
    without normalisation would call that completed edit not-applied -- which
    authorises a retry of an edit that already landed.
    """
    target = tmp_path / "a.py"
    # Exactly what `Edit` produces: the original CRLF lines, with the
    # replacement's own LF left untouched.
    target.write_bytes(b"def add(a, b):\r\n    return a + b\n# fixed\r\n")
    outcome = FileEditTool().reconcile(
        {
            "file_path": str(target),
            "old_string": "return a - b",
            "new_string": "return a + b\n# fixed",
        }
    )
    assert outcome is ReconcileOutcome.APPLIED


def test_edit_is_applied_when_old_is_gone_and_new_is_there(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("return a + b\n", encoding="utf-8")
    outcome = FileEditTool().reconcile(
        {
            "file_path": str(target),
            "old_string": "return a - b",
            "new_string": "return a + b",
        }
    )
    assert outcome is ReconcileOutcome.APPLIED


def test_edit_is_not_applied_when_old_is_still_there(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("return a - b\n", encoding="utf-8")
    outcome = FileEditTool().reconcile(
        {
            "file_path": str(target),
            "old_string": "return a - b",
            "new_string": "return a + b",
        }
    )
    assert outcome is ReconcileOutcome.NOT_APPLIED


def test_edit_is_unknown_when_both_strings_are_present(tmp_path: Path) -> None:
    """Both present means the file is not evidence either way.

    `old_string` appearing somewhere ELSE in the file makes "old is still here"
    compatible with "the edit landed", so the honest answer is that the file
    cannot decide it. Reporting APPLIED here would be the dangerous direction: it
    suppresses a retry that was needed.
    """
    target = tmp_path / "a.py"
    target.write_text("return a + b\n# was: return a - b\n", encoding="utf-8")
    outcome = FileEditTool().reconcile(
        {
            "file_path": str(target),
            "old_string": "return a - b",
            "new_string": "return a + b",
        }
    )
    assert outcome is ReconcileOutcome.UNKNOWN


def test_edit_is_unknown_when_neither_string_is_present(tmp_path: Path) -> None:
    """Somebody else's edit is not proof that ours failed."""
    target = tmp_path / "a.py"
    target.write_text("something else entirely\n", encoding="utf-8")
    outcome = FileEditTool().reconcile(
        {
            "file_path": str(target),
            "old_string": "return a - b",
            "new_string": "return a + b",
        }
    )
    assert outcome is ReconcileOutcome.UNKNOWN


def test_edit_on_a_missing_file_is_unknown_not_not_applied(tmp_path: Path) -> None:
    """A file deleted since the crash is not the same fact as an edit that failed.

    `NOT_APPLIED` would authorise a retry, and retrying an edit against a file
    that no longer exists is a different mistake from the one being repaired.
    """
    outcome = FileEditTool().reconcile(
        {"file_path": str(tmp_path / "gone.py"), "old_string": "a", "new_string": "b"}
    )
    assert outcome is ReconcileOutcome.UNKNOWN


def test_edit_with_an_empty_new_string_is_not_read_as_applied(tmp_path: Path) -> None:
    """`new_string=""` is a deletion, and `"" in content` is trivially true.

    Reading the empty string as present would report every deletion as already
    applied, including the ones that never ran.
    """
    target = tmp_path / "a.py"
    target.write_text("return a - b\n", encoding="utf-8")
    outcome = FileEditTool().reconcile(
        {"file_path": str(target), "old_string": "return a - b", "new_string": ""}
    )
    assert outcome is ReconcileOutcome.NOT_APPLIED
