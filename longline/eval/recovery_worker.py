"""Subprocess worker for the Process-Kill fault class.

=== Why a separate process at all ===

`SessionResumeRate` is supposed to measure what happens when the runtime dies
between turns. Nothing in-process can produce that honestly: an exception
unwinds a stack, it does not stop a process, and the "restart" would still be
sharing the live interpreter (and every module-level cache) with the run that
was supposed to be interrupted. So the kill is a real `SIGTERM` to a real child
process, and the resume happens in a *fresh* interpreter that re-reads the files
from disk.

=== The production functions this goes through ===

Deliberately the same calls `longline/main.py` makes on `--resume`, in the same
order, against the same on-disk layout:

1. `save_session()` between turns -- the "saved turn checkpoint". It is the
   existing per-turn persistence point (main.py writes one after every turn), so
   the checkpoint is not a new artifact invented for the benchmark.
2. `load_session()` -> `validate_transcript()` on resume.
3. `load_task_snapshot()` -> `TaskRegistry.restore()` for the task state.

Nothing here re-implements a loader. A parallel loader would measure itself.

=== Two phases ===

`python -m longline.eval.recovery_worker prepare <spec.json>` writes the
checkpoint files and prints a JSON report, including the transcript digests that
`--resume` later verifies against. `... resume <spec.json>` loads the session
back, repairs it if needed, restores the task snapshot, re-reads the stable
transcript, and prints a JSON report.

Both write their report to stdout as a single JSON object. Diagnostics go to
stderr, so a caller can parse stdout unconditionally.

The spec carries a `claude_dir`, and the eval runner always points it at a temp
directory. That is a hard rule rather than a convention: this module must never
read or write the operator's real `~/.claude`, and the runner's integration test
asserts the real directory is byte-identical afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.models.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from longline.models.messages import AssistantMessage, Message, UserMessage
from longline.session.storage import get_sessions_dir, load_session, load_task_snapshot, save_session

if TYPE_CHECKING:
    from collections.abc import Sequence

# The session id every kill fixture uses. Fixed rather than random so the
# resumed-leg command in the dataset is a literal string and the file names are
# checkable by a reader.
SESSION_ID = "recovery-resume"

# The fixture the kill case's task text names. The task is written as
# "Inspect the file notes/value.txt under <cwd> ...", so the path a real agent
# would open is exactly `cwd/notes/value.txt`.
FIXTURE_HEADING = "notes"
FIXTURE_FILE = "value.txt"

# Prefix the checkpoint transcript uses to carry the working directory. It rides
# inside the persisted tool result on purpose: the transcript is the only state
# a resumed process has, so recovering the cwd from it is what makes the resume
# a function of the checkpoint rather than of a spec file the harness still
# holds. An ASCII tag avoids any dependence on the surrounding prose.
CWD_TAG = "cwd="

# A task that was still RUNNING when the process died. Restoring it must mark it
# KILLED -- the contract forbids claiming a background task "resumes in place"
# (`evals/README.md` section 1).
RUNNING_TASK_SNAPSHOT: list[dict[str, Any]] = [
    {
        "task_id": "b-1a2b3c4d",
        "task_type": "local_bash",
        "state": "running",
        "created_at": 0.0,
        "updated_at": 0.0,
        "metadata": {"command": "sleep 600"},
    },
]


def derive_answer(messages: Sequence[Message], task: str) -> str | None:
    """The answer a correctly resumed agent would produce, derived from state.

    This is the resumed leg's *agent*, and it is written to read what a real
    agent would read -- the task text, resolved against the sandbox the task
    names -- and never a field this harness set aside for it. A stub that echoed
    `spec["answer"]` would make every resume case pass by construction, which is
    the exact shape of the defect that hid a task/judge mismatch in an earlier
    suite. So there is no score field in the spec at all: the value is
    recovered from the fixture the task points at and then FORMATTED here.

    The formatting step is what keeps the check honest. The transcript holds the
    raw marker (`value=alpha-7f3c`); the answer is `marker=alpha-7f3c`. A leg
    that could only replay its transcript, without re-reading the file the task
    names, would produce the wrong string.

    Returns None when the task does not name a readable file, which the caller
    reports as an unanswerable case rather than quietly passing.
    """
    claude_dir = _claude_dir_from_transcript(messages)
    relative = _relpath_from_task(task)
    if claude_dir is None or relative is None:
        return None
    target = claude_dir / relative
    if not target.is_file():
        return None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith("marker="):
            return line.strip()
    return None


def _relpath_from_task(task: str) -> str | None:
    """The fixture path the task text names, if it names one."""
    from longline.eval.recovery import extract_fixture_path

    return extract_fixture_path(task)


def _claude_dir_from_transcript(messages: Sequence[Message]) -> Path | None:
    """Recover the working directory from the persisted transcript.

    The transcript is the only state a resumed process has. Recording the cwd
    in it -- rather than passing it alongside in the spec file -- is what makes
    the resume genuinely a function of the checkpoint: if the transcript did not
    survive, this returns None and the case fails, instead of the harness
    silently supplying the path from its own spec.
    """
    for msg in messages:
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and isinstance(block.content, str):
                for token in block.content.split():
                    if token.startswith(CWD_TAG):
                        return Path(token[len(CWD_TAG):])
    return None


def build_checkpoint_transcript(answer: str, *, value: str, cwd: str = "") -> list[Message]:
    """The transcript as it stands at the end of the last completed turn.

    Ends on a completed `Read` call whose result is persisted, which is what
    makes the duplicate check meaningful: the resumed leg knows a complete tool
    result for this call index is already on disk, and must not execute it
    again.

    The persisted tool result carries the RAW marker value, not the formatted
    answer. That distinction is the whole point: `derive_answer` reads the
    marker out of the FIXTURE FILE and formats it, so the answer the judge sees
    is something the resumed leg produced. If the transcript already held the
    formatted answer, a resume that merely echoed its own history would pass a
    judge that was supposed to test whether it could still do the work.

    `answer` is accepted so the signature states what the fixture is about, and
    is embedded nowhere.
    """
    _ = answer
    return [
        UserMessage(content="Inspect value.txt and tell me the marker it holds."),
        AssistantMessage(
            content=[
                TextBlock(text="Reading the file before answering."),
                ToolUseBlock(id="tu-1", name="Read", input={"file_path": "value.txt"}),
            ],
            stop_reason="tool_use",
        ),
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="tu-1",
                    # The raw marker, as a tool would return it -- NOT the
                    # formatted answer. `derive_answer` formats it back into the
                    # answer string, which is what makes the judge a test of the
                    # resumed leg's work rather than of its memory.
                    content=f"value={value} {CWD_TAG}{cwd}",
                    is_error=False,
                ),
            ]
        ),
    ]


def write_fixture(claude_dir: Path, *, value: str) -> Path:
    """Create `value.txt` beside the fixture the task will name.

    The file is written at prepare time, i.e. before the kill, so the resumed
    process finds it on disk. That is deliberate: the thing being measured is
    whether the *transcript* survives, and a value.txt that only appeared after
    the resume would let a run pass on a file the checkpoint never covered.
    """
    target = claude_dir / FIXTURE_HEADING / FIXTURE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"marker={value}\n", encoding="utf-8")
    return target


def _read_spec(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"spec must be a JSON object, got {type(data).__name__}")
    return data


def _fingerprint_messages(messages: Sequence[Message]) -> list[str]:
    """Digest of each message's API shape, in order.

    Compared before and after the resume so the resumed leg can prove it did not
    silently rewrite or drop the persisted transcript. Uses `to_api_dict()` --
    the shape the model is sent -- rather than `repr`, because the API view is
    what "a valid transcript" means here.
    """
    import hashlib

    out: list[str] = []
    for msg in messages:
        # Only the API-shaped message types have `to_api_dict`; a system or
        # compact-boundary message is not part of the API transcript, so it is
        # digested by its repr instead of being silently skipped -- dropping it
        # would make a net-zero change (one added, one removed) invisible.
        view = msg.to_api_dict() if hasattr(msg, "to_api_dict") else repr(msg)
        payload = json.dumps(view, sort_keys=True, default=str).encode("utf-8")
        out.append(hashlib.sha256(payload).hexdigest()[:16])
    return out


def prepare(spec: dict[str, Any]) -> dict[str, Any]:
    """Write the checkpoint and report what a killed process left behind.

    Also PARSES the stable transcript back out of the JSONL before returning.
    Parsing is the point: `save_session` writes and this reads, so a value that
    does not survive the round trip is caught here rather than being discovered
    as a mysterious judge failure in the resumed leg.
    """
    claude_dir = Path(spec["claude_dir"])
    session_id = str(spec.get("session_id", SESSION_ID))
    answer = str(spec["answer"])
    value = str(spec["value"])

    write_fixture(claude_dir, value=value)
    messages = build_checkpoint_transcript(answer, value=value, cwd=str(claude_dir))
    save_session(
        session_id,
        messages,
        claude_dir=claude_dir,
        task_snapshot=RUNNING_TASK_SNAPSHOT,
    )
    session_file = get_sessions_dir(claude_dir) / f"{session_id}.jsonl"
    tasks_file = get_sessions_dir(claude_dir) / f"{session_id}.tasks.json"

    reloaded = load_session(session_id, claude_dir=claude_dir)
    stable_fact = _extract_stable_fact(reloaded or [])

    return {
        "phase": "prepare",
        "session_id": session_id,
        "claude_dir": str(claude_dir),
        "session_file": str(session_file),
        "tasks_file": str(tasks_file),
        "session_file_exists": session_file.is_file(),
        "tasks_file_exists": tasks_file.is_file(),
        "num_committed_messages": len(messages),
        "num_reloaded_messages": 0 if reloaded is None else len(reloaded),
        "message_fingerprints": _fingerprint_messages(reloaded or []),
        "stable_fact": stable_fact,
        "checkpoint_saved": session_file.is_file() and bool(reloaded),
    }


def _extract_stable_fact(messages: Sequence[Message]) -> str | None:
    """The raw marker as it was persisted in the transcript's tool result.

    The resumed command's `command_output_contains` judge asserts on `stdout`,
    so this has to be printed. It carries the RAW `value=` form, read from the
    transcript rather than from the fixture file, so the judge's second check
    proves the transcript survived and the first proves the file could be read.
    Either surviving alone would not carry the case.
    """
    for msg in messages:
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and isinstance(block.content, str):
                for token in block.content.split():
                    if token.startswith("value="):
                        return token
    return None


def resume(spec: dict[str, Any]) -> dict[str, Any]:
    """Reload the checkpoint the way `main.py --resume` does, then report.

    The repair call is the production `validate_transcript()` with its
    `report=` out-parameter, so `transcript_repaired` is the function's own
    verdict rather than a re-derivation here.
    """
    from longline.session.recovery import TranscriptRepairReport, validate_transcript

    claude_dir = Path(spec["claude_dir"])
    session_id = str(spec.get("session_id", SESSION_ID))

    loaded = load_session(session_id, claude_dir=claude_dir)
    if loaded is None:
        return {
            "phase": "resume",
            "session_id": session_id,
            "found": False,
            "checkpoint_loaded": False,
            "error": "session not found",
        }

    before = _fingerprint_messages(loaded)
    report = TranscriptRepairReport()
    repaired = validate_transcript(loaded, report=report)
    after = _fingerprint_messages(repaired)

    # The fingerprint check is reported, not asserted here: whether a changed
    # transcript is a defect depends on whether the repair was legitimate, and
    # that judgement belongs to the runner where the fault is known.
    structurally_valid, structural_errors = check_transcript_structure(repaired)

    task_snapshot = load_task_snapshot(session_id, claude_dir=claude_dir)
    restored_states = _restore_task_snapshot(task_snapshot)

    # The answer is DERIVED from the task plus the fixture on disk, never read
    # from the spec (see `derive_answer`).
    task = str(spec.get("task", ""))
    derived = derive_answer(repaired, task)

    return {
        "phase": "resume",
        "session_id": session_id,
        "found": True,
        "checkpoint_loaded": True,
        "num_loaded_messages": len(loaded),
        "num_repaired_messages": len(repaired),
        "transcript_repaired": report.repaired,
        "repair_kinds": list(report.repairs),
        "orphaned_tool_use_ids": list(report.orphaned_tool_use_ids),
        "message_fingerprints_before": before,
        "message_fingerprints_after": after,
        "transcript_unchanged": before == after,
        "structurally_valid": structurally_valid,
        "structural_errors": structural_errors,
        "task_snapshot_loaded": task_snapshot is not None,
        "task_states": restored_states,
        "derived_answer": derived,
        "mastered_stable_fact": _extract_stable_fact(repaired),
    }


def _restore_task_snapshot(snapshot: list[dict[str, Any]] | None) -> dict[str, str]:
    """Run the production `TaskRegistry.restore()` and report the states it set."""
    from longline.session.task_registry import TaskRegistry

    registry = TaskRegistry()
    if snapshot:
        registry.restore(snapshot)
    return {r.task_id: r.state.value for r in registry.list_all()}


def check_transcript_structure(messages: Sequence[Message]) -> tuple[bool, list[str]]:
    """Structural validity of a resumed transcript: API pairing + alternation.

    Three independent conditions, each of which `main.py` relies on when it
    hands the transcript back to the API:

    - every `tool_use` id has a matching `tool_result` (the API rejects a
      request otherwise, which is what `validate_transcript` exists to fix);
    - no `tool_result` refers to an id that was never requested;
    - no two consecutive messages share a role, and the transcript does not end
      on an assistant message, because `normalize_messages_for_api` would have
      to invent a message to satisfy the alternation rule.

    Returns `(valid, errors)` rather than raising, so a broken resume is
    recorded as a failed case with a reason rather than crashing the suite.
    """
    errors: list[str] = []
    tool_use_ids: list[str] = []
    result_ids: list[str] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for assistant_block in msg.content:
                if isinstance(assistant_block, ToolUseBlock):
                    tool_use_ids.append(assistant_block.id)
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for user_block in msg.content:
                if isinstance(user_block, ToolResultBlock):
                    result_ids.append(user_block.tool_use_id)

    unanswered = [i for i in tool_use_ids if i not in set(result_ids)]
    if unanswered:
        errors.append(f"tool_use without tool_result: {unanswered}")
    orphan_results = [i for i in result_ids if i not in set(tool_use_ids)]
    if orphan_results:
        errors.append(f"tool_result without tool_use: {orphan_results}")

    for i in range(1, len(messages)):
        if type(messages[i]) is type(messages[i - 1]):
            errors.append(f"role alternation violated at message {i}")
    if messages and isinstance(messages[-1], AssistantMessage):
        errors.append("transcript ends on an assistant message")

    return (not errors), errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m longline.eval.recovery_worker")
    parser.add_argument("phase", choices=["prepare", "resume"])
    parser.add_argument("spec", help="Path to the JSON spec file.")
    args = parser.parse_args(argv)

    spec = _read_spec(Path(args.spec))
    report = prepare(spec) if args.phase == "prepare" else resume(spec)
    report["phase"] = args.phase

    # One JSON object on stdout and nothing else, so the caller can parse it
    # unconditionally even when the process was killed mid-write on a later run.
    json.dump(report, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
