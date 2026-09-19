"""Workspace identity: did the world move under the checkpoint, and does it matter?

Two different failures hide behind "the workspace changed":

- a file the session READ or WROTE has a different digest than the one the
  session recorded -> the checkpoint describes a world that no longer exists, and
  continuing would act on stale premises. **Reject.**
- a file the session never touched changed -> somebody else's business, and
  refusing to resume would make the runtime unusable in any shared checkout.
  **Warn and continue.**

Reporting one number for both is why the suite currently reports a flat zero and
can say nothing more useful than that: production has no identity at all, so both
cases look alike. They do not: one is a safety property and the other is noise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from longline.session.tool_journal import ToolJournal
from longline.session.workspace_identity import (
    DriftVerdict,
    changed_paths,
    classify_drift,
    current_git_head,
)
from longline.tools.base import ReconcileOutcome
from longline.utils.hashing import sha256_file


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=e@x", "-c", "user.name=n", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit repo in a SUBDIRECTORY of `tmp_path`.

    The subdirectory matters: the journal is written to `tmp_path`, and putting it
    inside the repo would make it an untracked file -- so every case would report
    unrelated drift, for a reason that has nothing to do with the case. Production
    keeps the journal outside the workspace for the same reason, and this fixture
    would otherwise hide that requirement rather than exercise it.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    (root / "src").mkdir()
    (root / "src" / "calc.py").write_text("return a - b\n", encoding="utf-8")
    (root / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "init")
    return root


def _identity(
    journal_dir: Path, root: Path, *, touched: dict[str, str]
) -> ToolJournal:
    """A journal whose header pins this repo's HEAD and whose one op touched `touched`."""
    journal = ToolJournal(journal_dir, "s1")
    journal.write_session_header(
        workspace_root=str(root), git_head=current_git_head(root)
    )
    op = journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Edit", tool_input={}, workload=touched
    )
    journal.commit(
        op, outcome="ok", post_state={p: sha256_file(Path(p)) for p in touched}
    )
    return journal


def _classify(journal: ToolJournal, root: Path):
    return classify_drift(
        root=root, header=journal.session_header(), records=journal.records()
    )


def test_an_untouched_workspace_is_clean(tmp_path: Path, repo: Path) -> None:
    journal = _identity(tmp_path, repo, touched={str(repo / "src" / "calc.py"): "write"})
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.CLEAN
    assert report.git_available is True
    assert report.rejected is False


def test_a_file_the_session_wrote_and_someone_else_changed_is_relevant(
    tmp_path: Path, repo: Path
) -> None:
    """The checkpoint's premise about that file is now false."""
    notes = str(repo / "NOTES.md")
    journal = _identity(tmp_path, repo, touched={notes: "write"})
    (repo / "NOTES.md").write_text("# Notes\ndrifted\n", encoding="utf-8")
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert notes in report.relevant
    assert report.rejected is True


def test_a_file_the_session_only_read_is_also_relevant(tmp_path: Path, repo: Path) -> None:
    """A read is a dependency. Acting on it after it changed is acting on stale input."""
    calc = str(repo / "src" / "calc.py")
    journal = _identity(tmp_path, repo, touched={calc: "read"})
    (repo / "src" / "calc.py").write_text("return a + b\n", encoding="utf-8")
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert calc in report.relevant


def test_a_file_the_session_never_touched_is_only_unrelated(tmp_path: Path, repo: Path) -> None:
    """Refusing to resume because a colleague edited an unrelated file would make the
    runtime useless in any shared checkout."""
    journal = _identity(tmp_path, repo, touched={str(repo / "NOTES.md"): "write"})
    (repo / "src" / "other.py").write_text("x = 1\n", encoding="utf-8")
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.UNRELATED
    assert str(repo / "src" / "other.py") in report.unrelated
    assert report.relevant == []
    assert report.rejected is False


def test_a_deleted_dependent_file_is_relevant(tmp_path: Path, repo: Path) -> None:
    """A file that vanished is at least as stale a premise as one that changed."""
    notes = repo / "NOTES.md"
    journal = _identity(tmp_path, repo, touched={str(notes): "write"})
    notes.unlink()
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT


def test_a_moved_head_with_no_dependent_change_is_relevant(tmp_path: Path, repo: Path) -> None:
    """A new commit is not automatically harmless.

    It is relevant: the revision the checkpoint was taken against is gone. What it
    is not is *specific*, which is why it is reported as HEAD movement rather than
    as a list of files.
    """
    journal = _identity(tmp_path, repo, touched={str(repo / "NOTES.md"): "write"})
    (repo / "src" / "other.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "someone else")
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert report.git_head_changed is True
    assert report.relevant == []


def test_a_directory_that_is_not_a_repo_reports_git_unavailable(tmp_path: Path) -> None:
    """Unrelated drift needs git to enumerate what changed.

    Without a repo the check can still see a DEPENDENT change -- it compares
    hashes it recorded -- but nothing else, and the report must say so rather than
    claiming a clean workspace it cannot see.
    """
    root = tmp_path / "plain"
    root.mkdir()
    (root / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    journal = _identity(tmp_path, root, touched={str(root / "NOTES.md"): "write"})
    report = _classify(journal, root)
    assert report.git_available is False
    assert report.verdict is DriftVerdict.CLEAN


def test_a_dependent_change_is_seen_without_git(tmp_path: Path) -> None:
    """The dependent half of the check does not need a repository at all.

    It compares a hash it recorded against the file now, which works anywhere --
    so a non-repo workspace is weaker, not blind, and the report has to say which.
    """
    root = tmp_path / "plain"
    root.mkdir()
    (root / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    journal = _identity(tmp_path, root, touched={str(root / "NOTES.md"): "write"})
    (root / "NOTES.md").write_text("# Notes\ndrifted\n", encoding="utf-8")
    report = _classify(journal, root)
    assert report.verdict is DriftVerdict.RELEVANT
    assert report.git_available is False


def test_the_journal_inside_the_workspace_would_read_as_drift(tmp_path: Path, repo: Path) -> None:
    """The counterexample that forced the fixture above to use a subdirectory.

    It is recorded here rather than only in a comment because it is a real
    constraint on where the journal may live, and the day someone "simplifies"
    the path into the workspace this test is what explains why not.
    """
    from longline.session.tool_journal import journal_path

    assert not str(journal_path(tmp_path, "s1")).startswith(str(repo))


def test_no_header_means_no_identity_to_check(tmp_path: Path) -> None:
    """A session recorded before this existed. It resumes, with no claim made."""
    report = classify_drift(root=tmp_path, header=None, records=[])
    assert report.verdict is DriftVerdict.CLEAN
    assert report.git_available is False


def test_a_session_that_touched_nothing_is_clean_in_a_dirty_repo(
    tmp_path: Path, repo: Path
) -> None:
    """The before_model case: the kill landed before the first tool call.

    Nothing was touched, so nothing is dependent -- but the repo is not "clean"
    in git's sense, it is simply not ours to judge. Reporting RELEVANT here would
    reject a resume over a file the session never looked at.
    """
    journal = ToolJournal(tmp_path, "s1")
    journal.write_session_header(
        workspace_root=str(repo), git_head=current_git_head(repo)
    )
    (repo / "src" / "other.py").write_text("x = 1\n", encoding="utf-8")
    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.UNRELATED
    assert report.rejected is False


def test_changed_paths_is_none_outside_a_repo(tmp_path: Path) -> None:
    assert changed_paths(tmp_path) is None
    assert current_git_head(tmp_path) is None


def test_changed_paths_lists_an_untracked_file(tmp_path: Path) -> None:
    """A file another writer just CREATED is a change to the workspace.

    `git status --porcelain` includes untracked entries, which is why this uses it
    rather than `git diff`.
    """
    _git(tmp_path, "init")
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")
    assert changed_paths(tmp_path) == {str((tmp_path / "new.txt").resolve())}


# --- an interrupted write is the session's own change, not drift ---
#
# Found by the task-diversity suite, whose `after_tool` arms stop inside an Edit.
# Every other after-write arm in the old suite stopped inside a Bash call, and
# Bash declares no workload -- so its file could never be in the read set and the
# same situation was classified `unrelated`. The tool choice hid it.

NL = chr(10)


_OUTCOME = {
    "RECONCILED": ReconcileOutcome.APPLIED,
    "ABORTED": ReconcileOutcome.NOT_APPLIED,
    "INDETERMINATE": ReconcileOutcome.UNKNOWN,
}


def _interrupted_write(journal_dir: Path, repo: Path, path: Path) -> ToolJournal:
    """A journal whose committed op READ `path`, then prepared an Edit of it."""
    journal = _identity(journal_dir, repo, touched={str(path): "read"})
    journal.prepare(
        turn_id=1,
        tool_call_id="tu-2",
        tool_name="Edit",
        tool_input={},
        workload={str(path): "write"},
    )
    return journal


def _resolve_last(journal: ToolJournal, status: str) -> None:
    journal.resolve(
        journal.records()[-1].operation_id, status=status, outcome=_OUTCOME[status]
    )


def test_a_verified_applied_write_is_not_drift(tmp_path: Path, repo: Path) -> None:
    """The session read the file, edited it, and died inside the edit.

    The edit landed but never COMMITTED, so the write set has no record of it and
    the read set still carries the digest from before. The old rule called that a
    moved dependency and REFUSED the resume -- so a crash inside an edit made the
    session permanently unresumable, refused for having done the thing it was
    asked to do.

    What earns the resume here is that reconciliation ASKED: on the way back the
    runtime ran `Edit.reconcile`, which found the new text and not the old one.
    That verdict is the evidence.
    """
    path = repo / "src" / "calc.py"
    journal = _interrupted_write(tmp_path, repo, path)
    path.write_text("return a + b" + NL, encoding="utf-8")   # the edit landed
    _resolve_last(journal, "RECONCILED")

    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.CLEAN
    assert not report.rejected
    assert [Path(p).name for p in report.verified_applied] == ["calc.py"]


def test_an_unverified_interrupted_write_settles_nothing(
    tmp_path: Path, repo: Path
) -> None:
    """INDETERMINATE is what a tool that cannot read its own effect answers, and
    it must not authorise the resume. This is the Bash case: the file changed,
    but nobody can say who changed it."""
    path = repo / "src" / "calc.py"
    journal = _interrupted_write(tmp_path, repo, path)
    path.write_text("return a + b" + NL, encoding="utf-8")
    _resolve_last(journal, "INDETERMINATE")

    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert report.verified_applied == []


def test_an_aborted_write_settles_nothing(tmp_path: Path, repo: Path) -> None:
    """ABORTED is the arm that keeps the drift benchmark honest.

    `Edit.reconcile` answers NOT_APPLIED by finding the OLD text still intact --
    which is exactly what the dependent-drift arm looks like, because its Edit is
    interrupted BEFORE it runs and the parent then appends to the same file. An
    earlier version of this rule settled any interrupted write whose file had
    changed, and it passed every recovery test while silently masking that arm.
    """
    path = repo / "src" / "calc.py"
    journal = _interrupted_write(tmp_path, repo, path)
    path.write_text(
        "return a - b" + NL + "# drifted-by-another-writer" + NL, encoding="utf-8"
    )
    _resolve_last(journal, "ABORTED")

    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT, "an injected drift was masked"
    assert report.rejected
    assert report.verified_applied == []


def test_somebody_elses_edit_is_still_refused(tmp_path: Path, repo: Path) -> None:
    """No write operation in flight at all: a file the session READ, changed by
    anyone, is a stale premise. If this ever passes, the check has stopped."""
    path = repo / "src" / "calc.py"
    journal = _identity(tmp_path, repo, touched={str(path): "read"})
    path.write_text("return a * b" + NL, encoding="utf-8")

    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert report.rejected
    assert report.verified_applied == []


def test_a_verified_write_does_not_excuse_another_file(
    tmp_path: Path, repo: Path
) -> None:
    """Settling the session's own write must not settle anything else: it
    explains one path, and only that path."""
    written = repo / "src" / "calc.py"
    read = repo / "NOTES.md"
    journal = _identity(tmp_path, repo, touched={str(read): "read"})
    journal.prepare(
        turn_id=1,
        tool_call_id="tu-2",
        tool_name="Edit",
        tool_input={},
        workload={str(written): "write"},
    )
    written.write_text("return a + b" + NL, encoding="utf-8")
    read.write_text("# Notes" + NL + "somebody else" + NL, encoding="utf-8")
    _resolve_last(journal, "RECONCILED")

    report = _classify(journal, repo)
    assert report.verdict is DriftVerdict.RELEVANT
    assert [Path(p).name for p in report.relevant] == ["NOTES.md"]


def test_an_untracked_directory_is_reported_as_its_files(
    tmp_path: Path, repo: Path
) -> None:
    """`git status` collapses a wholly-untracked directory to the directory.

    The set it is compared against holds FILE paths, so a file the session just
    created inside such a directory matched nothing and was reported as somebody
    else's unrelated change. Listing every file makes the two sides comparable.
    """
    (repo / "generated").mkdir()
    (repo / "generated" / "new.py").write_text("X = 1" + NL, encoding="utf-8")
    changed = changed_paths(repo)
    assert changed is not None
    names = sorted(Path(p).name for p in changed)
    assert "new.py" in names, names
    assert "generated" not in names, "the directory was reported instead of its file"
