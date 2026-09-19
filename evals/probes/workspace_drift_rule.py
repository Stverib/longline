"""A/B the workspace-drift rule across two revisions of the tree.

The question this answers, and the only one: **does the fixed rule flip the
verdict for a crash inside a write, and does it leave every other verdict
alone?** A recovery rate cannot answer that -- a rate moves when several things
move, and it says nothing about which case flipped.

It builds the journal the diversity suite builds -- a COMMITTED read of a file,
then an Edit of that same file that was interrupted and later reconciled -- and
prints `classify_drift`'s answer for each reconciliation verdict.

Usage, from a tree that contains the revision you want to ask:

    python evals/probes/workspace_drift_rule.py

To compare revisions, run it in each. The isolated way, which does not disturb
the working tree and is how the numbers in the README were taken:

    git worktree add .tmp/prefix-wt <pre-fix-revision>
    mkdir -p .tmp/prefix-wt/evals/probes
    cp evals/probes/workspace_drift_rule.py .tmp/prefix-wt/evals/probes/
    cd .tmp/prefix-wt && PYTHONPATH="$PWD" python evals/probes/workspace_drift_rule.py
    cd - && git worktree remove --force .tmp/prefix-wt

**The readable line is `workspace_identity from:`.** Two runs that load the same
module measure nothing, and the way that happens is an installed/editable
`longline` shadowing the tree you think you are testing -- which is why the
script prints the path it imported instead of trusting the cwd.

The Read MUST be committed. `workspace_from_records` only digests COMMITTED
operations, so a merely-PREPARED read leaves the read set empty, the edited file
lands in no recorded set at all, and EVERY case reports `unrelated` -- a probe
that measures nothing while looking exactly like one that measured something.
That is not hypothetical; it is the first version of this file.
"""

import subprocess
import tempfile
from pathlib import Path

import longline.session.workspace_identity as wi
from longline.session.tool_journal import ToolJournal
from longline.tools.base import ReconcileOutcome
from longline.utils.hashing import sha256_file

print("workspace_identity from:", wi.__file__)

tmp = Path(tempfile.mkdtemp())
repo = tmp / "repo"
repo.mkdir()
subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
(repo / "src").mkdir()
target = repo / "src" / "calc.py"
target.write_text("return a - b\n", encoding="utf-8")
subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
subprocess.run(
    ["git", "-c", "user.email=e@x", "-c", "user.name=n", "commit", "-m", "init"],
    cwd=repo, check=True, capture_output=True,
)


def op_id_of(op: object) -> str:
    """`prepare` returned a bare id before it returned the record itself."""
    return op if isinstance(op, str) else str(op.operation_id)  # type: ignore[attr-defined]


CASES = (
    ("RECONCILED", ReconcileOutcome.APPLIED, "the tool READ BACK its own effect"),
    ("ABORTED", ReconcileOutcome.NOT_APPLIED, "the OLD text was found intact"),
    ("INDETERMINATE", ReconcileOutcome.UNKNOWN, "the tool cannot read its effect"),
    ("(no resolve)", None, "never asked -- the pre-reconcile shape"),
)

for status, outcome, why in CASES:
    # Back to the pre-edit text, so the read below digests THAT revision. The
    # commit is deliberately outside the loop: a second one is a no-op, and a
    # repo whose HEAD moves would add a second reason for the verdict.
    target.write_text("return a - b\n", encoding="utf-8")
    journal = ToolJournal(tmp / status.replace(" ", "_").replace("(", "").replace(")", ""), "s1")
    journal.write_session_header(workspace_root=str(repo), git_head=None)
    read_id = op_id_of(journal.prepare(
        turn_id=1, tool_call_id="tu-1", tool_name="Read",
        tool_input={}, workload={str(target): "read"},
    ))
    journal.commit(read_id, outcome="ok", post_state={str(target): sha256_file(target)})

    edit_id = op_id_of(journal.prepare(
        turn_id=1, tool_call_id="tu-2", tool_name="Edit",
        tool_input={}, workload={str(target): "write"},
    ))
    # The edit landed; the file on disk is now the session's own revision.
    target.write_text("return a + b\n", encoding="utf-8")
    if outcome is not None:
        journal.resolve(edit_id, status=status, outcome=outcome)

    report = wi.classify_drift(
        root=repo, header=journal.session_header(), records=journal.records()
    )
    print(
        f"  {status:14s} -> verdict={report.verdict.value:9s} "
        f"rejected={report.rejected!s:5s}  ({why})"
    )
