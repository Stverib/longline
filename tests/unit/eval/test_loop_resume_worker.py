"""The arm phase: a real loop driven once per instruction, stopped by a gate."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from longline.eval.failpoints import (
    AFTER_CHECKPOINT,
    AFTER_TOOL,
    BEFORE_MODEL,
    BEFORE_TOOL,
    read_sentinel,
)
from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases
from longline.eval.loop_resume_worker import (
    Plan,
    ScriptedToolSequence,
    arm,
    settled_steps,
)
from longline.eval.side_effect_journal import KILLED, RESUMED, read_journal

FIXTURE = Path("evals/fixtures/resume_repo")
DATASET = Path("evals/loop_resume.jsonl")
TASK = 'In {cwd}: fix the bug in src/calc.py and append a line containing "fixed-add" to NOTES.md.'


def _canonical_scenario(sandbox: Path) -> dict[str, Any]:
    """The DATASET's scenario for this task, resolved the way the runner does.

    Not a copy written out here. These tests drive the real worker, and the
    scenario is now data the dataset owns -- a second copy in this file would be
    free to drift from the thing production actually runs, and the tests would
    keep passing while the suite ran something else.
    """
    cases = cases_by_failpoint(load_loop_resume_cases(DATASET))
    scenario = cases["after_tool"][0].scenario
    assert scenario is not None
    return scenario.to_spec(sandbox)


def _drain(agen: Any) -> list[Any]:
    async def _run() -> list[Any]:
        return [event async for event in agen]

    return asyncio.run(_run())


def _api_transcript(
    tool_uses: int,
    *,
    offset: int = 0,
    result: str = "ok",
    is_error: bool = False,
) -> list[dict[str, Any]]:
    """A transcript in the API shape `query_loop` hands the model.

    Each `tool_use` is PAIRED with a `tool_result` carrying its own id. The
    pairing is what the progress rule reads, so the ids have to agree -- they did
    not before, and the mismatch was invisible while progress was a bare count of
    `tool_use` blocks. It is not invisible now: an unpaired `tool_use` is a step
    that is not finished with, which is the whole point of the new rule.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for i in range(tool_uses):
        tool_id = f"tu-{offset + i + 1}"
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                        "is_error": is_error,
                    }
                ],
            }
        )
    return messages


def _sandbox(tmp_path: Path, *, seed: int = 0) -> Path:
    """A seeded sandbox, because an unseeded one never occurs in a real run.

    These tests call `arm` directly instead of going through the runner, so
    nothing would substitute the fixture's seed placeholders and the sandbox
    would carry a literal `<seed>` -- a state the worker never sees in a real
    run, and one that made an assertion here compare against a fixture that no
    longer exists (found by the full suite, not by this file on its own).
    """
    from longline.eval.loop_resume_runner import apply_seed

    sandbox = tmp_path / "sandbox"
    shutil.copytree(FIXTURE, sandbox)
    apply_seed(sandbox, seed)
    return sandbox


def _spec(tmp_path: Path, sandbox: Path, failpoint: str, **overrides: Any) -> dict[str, Any]:
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir(exist_ok=True)
    spec: dict[str, Any] = {
        "claude_dir": str(claude_dir),
        "sandbox": str(sandbox),
        "session_id": "loop-resume",
        "failpoint": failpoint,
        "failpoint_tool": "",
        "at_call_index": 1,
        "api_key": "offline",
        "model": "offline-model",
        "task": TASK.format(cwd=sandbox.as_posix()),
        "scenario": _canonical_scenario(sandbox),
        "no_block": True,
    }
    spec.update(overrides)
    return spec


# --- the scripted sequence ---


def test_scripted_sequence_walks_its_steps_then_answers() -> None:
    model = ScriptedToolSequence(steps=[{"tool": "Read", "input": {"file_path": "a.py"}}])
    first = _drain(model(messages=_api_transcript(0)))
    assert [type(e).__name__ for e in first] == ["ToolUseStart", "TurnComplete"]
    second = _drain(model(messages=_api_transcript(1)))
    assert [type(e).__name__ for e in second] == ["TextDelta", "TurnComplete"]
    assert second[1].stop_reason == "end_turn"


def test_scripted_sequence_resumes_from_transcript_progress() -> None:
    """Progress is derived from the transcript, not a private counter.

    That is what makes the resumed leg continue where the killed leg stopped,
    and -- with the turn-0 checkpoint -- what makes the after_tool replay
    observable at all.
    """
    model = ScriptedToolSequence(
        steps=[
            {"tool": "Read", "input": {}},
            {"tool": "Bash", "input": {"command": "echo x >> NOTES.md"}},
        ]
    )
    out = _drain(model(messages=_api_transcript(1)))
    starts = [e for e in out if type(e).__name__ == "ToolUseStart"]
    assert len(starts) == 1
    assert starts[0].tool_name == "Bash"


def test_scripted_sequence_honours_the_offset() -> None:
    """Instruction 2 must not read instruction 1's tool uses as its own progress."""
    model = ScriptedToolSequence(steps=[{"tool": "Write", "input": {}}], offset=3)
    out = _drain(model(messages=_api_transcript(3, offset=0)))
    starts = [e for e in out if type(e).__name__ == "ToolUseStart"]
    assert len(starts) == 1, "the offset was ignored and the step looked complete"
    assert starts[0].tool_name == "Write"
    assert starts[0].tool_id == "tu-4", "tool ids must stay unique across instructions"


def test_the_plan_reads_the_dataset_scenario_and_resolves_its_offsets() -> None:
    """`offset_for(1)` is how many tool_use blocks instruction 1 contributes.

    Instruction 2's sequence needs it: after instruction 1 the transcript already
    carries that many, and without it the model would read instruction 1's tool
    uses as its own progress and answer without doing anything.
    """
    spec = {
        "task": "one",
        "scenario": {
            "followups": ["two"],
            "steps": [[{"tool": "Read", "input": {}}], [{"tool": "Write", "input": {}}]],
            "artifacts": ["NOTES.md"],
        },
    }
    plan = Plan.from_spec(spec)
    assert plan.instructions == ["one", "two"]
    assert plan.offset_for(0) == 0
    assert plan.offset_for(1) == 1


def test_a_spec_without_a_scenario_is_refused() -> None:
    """Falling back to a built-in scenario would run some other task under this
    case's name, and every metric here would still say the run was fine."""
    import pytest

    from longline.eval.failpoints import FailpointError

    with pytest.raises(FailpointError, match="no scenario"):
        Plan.from_spec({"task": "one"})


# --- the arm phase ---


def test_arm_writes_the_turn_zero_checkpoint_before_the_first_model_call(
    tmp_path: Path,
) -> None:
    """Without it, before_model has nothing to resume and would fail for a
    reason unrelated to recovery."""
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_MODEL)
    report = arm(spec)
    assert report["failpoint_reached"] is True
    assert report["sentinel"]["failpoint"] == BEFORE_MODEL
    assert report["checkpoint_messages_before_loop"] == 1
    assert (tmp_path / "claude" / "sessions" / "loop-resume.jsonl").is_file()
    # Nothing ran, so nothing was appended. Asserted as "the header is intact
    # and the append is absent" rather than as byte equality with the fixture:
    # byte equality makes this test a change-detector for the fixture, and it
    # failed for exactly that reason when the fixture gained a seed line.
    notes = (sandbox / "NOTES.md").read_text(encoding="utf-8")
    assert notes.startswith("# Notes\n"), notes
    assert "fixed-add" not in notes
    assert "return a - b" in (sandbox / "src" / "calc.py").read_text(encoding="utf-8")


def test_before_tool_stops_without_executing_the_tool(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_TOOL, failpoint_tool="Bash")
    report = arm(spec)
    assert report["failpoint_reached"] is True
    assert report["sentinel"]["failpoint"] == BEFORE_TOOL
    assert report["sentinel"]["detail"]["tool"] == "Bash"
    assert "fixed-add" not in (sandbox / "NOTES.md").read_text(encoding="utf-8")


def test_after_tool_journals_the_side_effect_before_it_stops(tmp_path: Path) -> None:
    """The killed leg's journal must already carry the append.

    This is the evidence that survives the kill, and the duplicate metric is
    built on it. If the gate stopped before the journal write, the denominator
    would be zero and the arm would report a clean resume for a side effect
    that really happened.
    """
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_TOOL, failpoint_tool="Bash")
    report = arm(spec)
    assert report["failpoint_reached"] is True
    assert report["sentinel"]["failpoint"] == AFTER_TOOL

    # The side effect really happened...
    assert "fixed-add" in (sandbox / "NOTES.md").read_text(encoding="utf-8")
    # ...and the journal knows it, on disk, from the leg that is about to die.
    entries = read_journal(tmp_path / "claude" / "journal.jsonl")
    assert [e.leg for e in entries] == [KILLED, KILLED]
    changed = [e for e in entries if e.changed_state]
    assert len(changed) == 1, "exactly one state-changing execution preceded the gate"
    assert changed[0].tool == "Bash"

    # The fix, asserted on disk. Production used to write one checkpoint per
    # INSTRUCTION, so this file held the instruction and nothing else -- and the
    # resumed leg, seeing no record that the Bash call had been issued, issued it
    # again. Step-level checkpointing means the model-response write landed before
    # the Bash body could run, so the transcript now carries the call.
    session_lines = [
        ln for ln in
        (tmp_path / "claude" / "sessions" / "loop-resume.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(session_lines) > 1, "step-level checkpoints must have landed"

    records = [json.loads(ln) for ln in session_lines]
    tool_uses = [
        block["name"]
        for record in records
        for block in (record.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    tool_results = [
        block
        for record in records
        for block in (record.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]    # The Bash call is on disk, and its result is NOT -- that is the window this
    # arm injects, and it is precisely the state the journal exists to describe.
    assert tool_uses == ["Read", "Bash"], (
        "the Bash call must be on disk BEFORE the tool body runs, or the resumed "
        "leg has no way to know it was ever issued"
    )
    assert len(tool_results) == 1, "only the Read's result came back before the kill"


def test_the_journal_records_the_bash_call_as_started_but_uncommitted(tmp_path: Path) -> None:
    """The other half of the fix, and the reason the checkpoint alone is not enough.

    The checkpoint says the call was ISSUED. It cannot say whether the call took
    effect, because the result never came back. The journal's
    started-without-COMMITTED is what turns that into a decidable question on
    resume.

    The Bash call carries TWO records here, and the second is the one that makes
    the question decidable rather than merely askable. `PREPARED` alone would
    also describe a call that died before the shell was ever spawned -- and that
    call provably did nothing, so a resume that read the two the same way would
    have to give up on both. `EXECUTING` is the Bash tool reporting that it
    reached `create_subprocess_shell`, which is why this arm's honest answer is
    "it may have landed" and the `before_tool` arm's is "it never began".
    """
    from longline.session.tool_journal import COMMITTED, EXECUTING, PREPARED, ToolJournal

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_TOOL, failpoint_tool="Bash")
    arm(spec)

    records = ToolJournal(tmp_path / "claude", "loop-resume").records()
    # Joined on the OPERATION id: the COMMITTED and EXECUTING records carry no
    # tool name, on purpose -- the start is what owns the operation's identity,
    # and repeating its fields into the later records would be a second copy of
    # a fact that could then disagree.
    starts = {r.operation_id: r for r in records if r.status == PREPARED}
    statuses: dict[str, list[str]] = {}
    for record in records:
        start = starts.get(record.operation_id)
        if start is not None:
            statuses.setdefault(start.tool_name, []).append(record.status)

    assert statuses["Bash"] == [PREPARED, EXECUTING], "the shell was spawned and never reported"
    assert statuses["Read"] == [PREPARED, COMMITTED]

    read_start = next(r for r in records if r.status == PREPARED and r.tool_name == "Read")
    assert read_start.access, "the Read declared the file it touched"
    assert read_start.tool_call_id, "and which call it was"


def test_the_before_tool_arm_leaves_a_call_that_provably_never_began(tmp_path: Path) -> None:
    """The other side of the same distinction, and the one that was lost.

    The gate stops before delegating, so the shell is never spawned and
    `mark_irreversible()` is never called -- which leaves ONE record, not two.
    That single record is the proof: nothing about this call can have changed the
    world, so the resumed leg may retry it.

    Read the two arms together. `after_tool` is `[PREPARED, EXECUTING]` and must
    not be retried; `before_tool` is `[PREPARED]` and must be. Before the marker
    existed those were the same record, so the runtime could only give the safe
    answer to both -- and giving the safe answer here meant the task silently
    never finished.
    """
    from longline.session.tool_journal import ABORTED, ToolJournal

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_TOOL, failpoint_tool="Bash")
    arm(spec)

    journal = ToolJournal(tmp_path / "claude", "loop-resume")
    pending = journal.pending()
    assert [p.record.tool_name for p in pending] == ["Bash"]
    assert [p.started for p in pending] == [False], (
        "the gate stopped before the spawn, so no marker should exist"
    )
    operation_id = pending[0].record.operation_id
    assert pending[0].record.tool_call_id, "the start still carries the id the transcript needs"

    from longline.eval.loop_resume_worker import resume

    resume(spec)

    verdicts = [r for r in journal.records() if r.status == ABORTED]
    assert [r.operation_id for r in verdicts] == [operation_id], (
        "a call that provably never began was not declared safe to retry"
    )


def test_after_checkpoint_saves_instruction_one_before_arming_the_gate(tmp_path: Path) -> None:
    """The one arm whose kill leaves COMPLETED work on disk.

    Asserted on the artifacts and the transcript rather than on a counter:
    instruction 1's three steps must have really run and been persisted before
    the gate stopped the process inside instruction 2.
    """
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_CHECKPOINT)
    report = arm(spec)
    assert report["failpoint_reached"] is True
    assert report["instructions_run"] == 1, "the kill must land inside instruction 2"
    assert "fixed-add" in (sandbox / "NOTES.md").read_text(encoding="utf-8")
    assert "return a + b" in (sandbox / "src" / "calc.py").read_text(encoding="utf-8")
    assert not (sandbox / "REPORT.md").exists(), "instruction 2 must not have run"

    session = (tmp_path / "claude" / "sessions" / "loop-resume.jsonl").read_text(encoding="utf-8")
    assert "tool_use" in session, "instruction 1's completed turn must be on disk"


def test_arm_reports_when_its_gate_never_fires(tmp_path: Path) -> None:
    """A failpoint aimed at a tool the scenario never calls is a wiring bug,
    and it must be visible as `failpoint_reached: false` rather than as a run
    that quietly did the whole task."""
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_TOOL, failpoint_tool="Grep")
    report = arm(spec)
    assert report["failpoint_reached"] is False
    assert read_sentinel(tmp_path / "claude") is None
    assert report["instructions_run"] == 1, "the instruction ran to completion instead"


def test_arm_declares_the_sandbox_outside_the_real_state_dir(tmp_path: Path) -> None:
    """The fixture never touches `~/.longline`; `claude_dir` is the only place
    session state goes, and it is always a temp dir."""
    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_MODEL)
    arm(spec)
    assert (tmp_path / "claude" / "sessions").is_dir()
    assert not (Path.home() / ".longline" / "sessions" / "loop-resume.jsonl").exists()


# --- the resume phase ---


def test_check_transcript_structure_accepts_a_paired_transcript() -> None:
    from longline.eval.loop_resume_worker import check_transcript_structure
    from longline.models.content_blocks import ToolResultBlock, ToolUseBlock
    from longline.models.messages import AssistantMessage, UserMessage

    messages = [
        UserMessage(content="go"),
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Read", input={})], stop_reason="tool_use"
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok", is_error=False)]),
    ]
    ok, errors = check_transcript_structure(messages)
    assert ok is True
    assert errors == []


def test_check_transcript_structure_rejects_an_orphan_tool_use() -> None:
    from longline.eval.loop_resume_worker import check_transcript_structure
    from longline.models.content_blocks import ToolUseBlock
    from longline.models.messages import AssistantMessage, UserMessage

    messages = [
        UserMessage(content="go"),
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Read", input={})], stop_reason="tool_use"
        ),
    ]
    ok, errors = check_transcript_structure(messages)
    assert ok is False
    assert any("without tool_result" in e for e in errors)


def test_check_transcript_structure_rejects_role_alternation_violation() -> None:
    from longline.eval.loop_resume_worker import check_transcript_structure
    from longline.models.messages import UserMessage

    ok, errors = check_transcript_structure([UserMessage(content="a"), UserMessage(content="b")])
    assert ok is False
    assert any("alternation" in e for e in errors)


def test_check_transcript_structure_accepts_a_transcript_ending_on_a_text_reply() -> None:
    """The between-instructions state, which is NOT a defect.

    `main.py` writes a checkpoint after every `run_turn()`, so the file
    legitimately ends on the assistant's final text until the next user message
    is appended. An earlier version of this function rejected that state, and
    it cost the after_checkpoint arm all ten of its state-layer verdicts --
    dragging LoopResumeRate down for a reason unrelated to recovery. Caught by
    the 6x10 sweep, not by any single-run test.
    """
    from longline.eval.loop_resume_worker import check_transcript_structure
    from longline.models.content_blocks import TextBlock
    from longline.models.messages import AssistantMessage, UserMessage

    messages = [
        UserMessage(content="go"),
        AssistantMessage(content=[TextBlock(text="done")], stop_reason="end_turn"),
    ]
    ok, errors = check_transcript_structure(messages)
    assert ok is True, errors


def test_check_transcript_structure_still_rejects_a_trailing_unanswered_tool_use() -> None:
    """Removing the "ends on assistant" rule must not lose the dangerous case
    it was there for: the pairing rule has to catch it on its own."""
    from longline.eval.loop_resume_worker import check_transcript_structure
    from longline.models.content_blocks import TextBlock, ToolUseBlock
    from longline.models.messages import AssistantMessage, UserMessage

    messages = [
        UserMessage(content="go"),
        AssistantMessage(
            content=[TextBlock(text="working"), ToolUseBlock(id="t1", name="Read", input={})],
            stop_reason="tool_use",
        ),
    ]
    ok, errors = check_transcript_structure(messages)
    assert ok is False
    assert any("tool_use without tool_result" in e for e in errors)


def test_resume_reports_not_found_when_no_checkpoint_exists(tmp_path: Path) -> None:
    from longline.eval.loop_resume_worker import resume

    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    report = resume(
        {
            "claude_dir": str(claude_dir),
            "sandbox": str(tmp_path / "s"),
            "session_id": "loop-resume",
            "api_key": "offline",
            "model": "offline-model",
        }
    )
    assert report["checkpoint_loaded"] is False
    assert report["layer_state_ok"] is False


def test_resume_finishes_the_task_the_killed_leg_never_finished(tmp_path: Path) -> None:
    """The resumed leg does WORK, not a read-back."""
    from longline.eval.loop_resume_worker import resume

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_MODEL)
    arm(spec)
    report = resume(spec)

    assert report["checkpoint_loaded"] is True
    assert report["structural_errors"] == []
    assert report["layer_state_ok"] is True
    assert report["instructions_run"] == 1
    assert "fixed-add" in (sandbox / "NOTES.md").read_text(encoding="utf-8")
    assert "return a + b" in (sandbox / "src" / "calc.py").read_text(encoding="utf-8")
    assert report["task_states"] == {"b-9f8e7d6c": "killed"}


def test_after_tool_resume_does_not_replay_the_append(tmp_path: Path) -> None:
    """The core measurement, end to end, after the durability fix.

    The killed leg appended to NOTES.md and the result never reached the
    transcript. Before the fix, the resumed leg -- having no record that the call
    was issued -- appended again, and the file carried the line twice.

    Now the step-level checkpoint has the `tool_use` on disk and the journal has
    the operation as PREPARED-without-COMMITTED, so reconciliation reports the
    outcome as unknown and the scripted model does not repeat the call. The file
    must carry the line ONCE, and the journal must show no replay at all -- a
    re-execution that happened to be harmless would still be a re-execution.
    """
    from longline.eval.loop_resume_worker import resume
    from longline.eval.side_effect_journal import compute_side_effect_metrics

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_TOOL, failpoint_tool="Bash")
    arm(spec)
    assert (sandbox / "NOTES.md").read_text(encoding="utf-8").count("fixed-add") == 1

    report = resume(spec)

    notes = (sandbox / "NOTES.md").read_text(encoding="utf-8")
    assert notes.count("fixed-add") == 1, "the append was replayed"
    # The task still finished: refusing to repeat the call must not mean skipping
    # the steps after it.
    assert "return a + b" in (sandbox / "src" / "calc.py").read_text(encoding="utf-8")
    assert report["layer_state_ok"] is True

    entries = read_journal(tmp_path / "claude" / "journal.jsonl")
    assert {e.leg for e in entries} == {KILLED, RESUMED}
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 1, "the append is the only state-changing execution"
    assert metrics.redundant == 0
    assert metrics.duplicated == 0


def test_after_tool_resume_tells_the_model_the_outcome_is_unknown(tmp_path: Path) -> None:
    """And it must not tell it the call FAILED, which is what the default is.

    `validate_transcript`'s placeholder says "internal error", which is true for a
    call that never ran and a lie for a Bash append that already landed. A model
    shown the lie retries -- so the wording is load-bearing, not cosmetic.

    Bash declares no workload, so the truthful verdict is UNKNOWN: nothing can
    read a shell command's effect back off the filesystem.
    """
    from longline.session.tool_journal import (
        INDETERMINATE,
        RECONCILE_UNKNOWN_PREFIX,
        ToolJournal,
    )

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_TOOL, failpoint_tool="Bash")
    arm(spec)
    assert ToolJournal(tmp_path / "claude", "loop-resume").pending(), (
        "the Bash call must be PREPARED-without-COMMITTED for this to mean anything"
    )

    from longline.eval.loop_resume_worker import resume

    resume(spec)

    journal = ToolJournal(tmp_path / "claude", "loop-resume")
    assert journal.pending() == [], "the orphan was left unresolved"
    verdicts = [r for r in journal.records() if r.status == INDETERMINATE]
    assert verdicts, "the Bash call reconciled to something other than unknown"
    assert verdicts[0].outcome == "unknown"

    session_text = (tmp_path / "claude" / "sessions" / "loop-resume.jsonl").read_text(
        encoding="utf-8"
    )
    assert RECONCILE_UNKNOWN_PREFIX in session_text, (
        "the transcript must carry the reconciled verdict, not the generic placeholder"
    )
    assert "internal error" not in session_text


def test_after_checkpoint_resume_does_not_replay_instruction_one(tmp_path: Path) -> None:
    """The positive control for the execution layer.

    Instruction 1 was completed and persisted before the kill, so the resumed
    leg must NOT redo its tools -- and its own work (REPORT.md) must appear.
    Without this arm the redundant count could never be zero, and the metric
    would have no way to show it can recognise a clean resume.
    """
    from longline.eval.loop_resume_worker import resume
    from longline.eval.side_effect_journal import compute_side_effect_metrics

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_CHECKPOINT)
    arm(spec)
    assert (sandbox / "NOTES.md").read_text(encoding="utf-8").count("fixed-add") == 1

    resume(spec)

    assert (sandbox / "REPORT.md").is_file(), "instruction 2 did not run"
    assert (sandbox / "NOTES.md").read_text(encoding="utf-8").count("fixed-add") == 1, (
        "instruction 1's append was replayed even though it was on disk"
    )

    entries = read_journal(tmp_path / "claude" / "journal.jsonl")
    assert {e.leg for e in entries} == {KILLED, RESUMED}
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 2, "instruction 1 had two state-changing executions"
    assert metrics.redundant == 0
    assert metrics.duplicated == 0
    assert metrics.by_tool == {}


def test_a_torn_tail_is_dropped_by_load_session_not_repaired(tmp_path: Path) -> None:
    """Refutes the design's prediction, and records what actually happens.

    The spec predicted that `truncate_tail` would be the one arm reaching
    `validate_transcript`'s orphan repair. It is not: `load_session` skips the
    unparseable line itself (storage.py), so by the time the repair function
    runs there is no orphaned `tool_use` left for it to fix and `repairs` stays
    empty. The transcript the resume gets is the LAST COMPLETE RECORD, which is
    a clean prefix.

    What this arm therefore measures is corruption tolerance -- the run must
    survive a half-written final record -- and that is worth measuring on its
    own. The prediction in the spec is wrong and this test is where it is
    corrected.
    """
    from longline.eval.failpoints import truncate_last_line
    from longline.eval.loop_resume_worker import resume

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, "truncate_tail")
    arm(spec)
    session = tmp_path / "claude" / "sessions" / "loop-resume.jsonl"
    whole = len([ln for ln in session.read_text(encoding="utf-8").splitlines() if ln.strip()])
    assert whole > 1, "one line would leave nothing to survive the cut"

    truncate_last_line(session)
    report = resume(spec)

    assert report["checkpoint_loaded"] is True, "the corrupt tail destroyed the session"
    assert report["num_loaded_messages"] == whole - 1, "the torn record was not the only loss"
    assert report["transcript_repaired"] is False, (
        "the repair path was reached after all -- the spec's claim that load_session "
        "drops the torn line first no longer holds, and both documents need updating"
    )
    assert report["repairs"] == []
    assert report["structural_errors"] == []
    assert report["layer_state_ok"] is True


def test_workspace_drift_goes_undetected(tmp_path: Path) -> None:
    """The honest result of the drift arm, pinned so a future fix is noticed.

    Production has no workspace identity check: the journal, the transcript and
    the tools all record nothing about which revision of a file the checkpoint
    was taken against. So a resumed leg runs happily on top of a workspace that
    changed underneath it, and no tool error mentions it.

    `tool_errors` is the channel such a check would have to use. Today it stays
    empty. WHEN THIS TEST FAILS, a detection mechanism has appeared: the arm
    becomes a real detection rate, and `evals/README.md` must stop saying
    detection is zero.
    """
    from longline.eval.loop_resume_worker import resume

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, "workspace_drift")
    arm(spec)

    # Another writer touches the workspace after the checkpoint was taken.
    notes = sandbox / "NOTES.md"
    notes.write_text(notes.read_text(encoding="utf-8") + "drifted-by-another-writer\n",
                     encoding="utf-8")
    report = resume(spec)
    assert report["tool_errors"] == [], "the runtime noticed the drift; update the README"

    # And the resumed leg happily appends on top of the foreign edit.
    text = notes.read_text(encoding="utf-8")
    assert "drifted-by-another-writer" in text
    assert text.count("fixed-add") == 1


def test_resume_runs_validate_transcript_on_the_loaded_messages(tmp_path: Path) -> None:
    """The production recovery call is made even when it has nothing to do.

    An arm that never calls it would leave `transcript_repaired` permanently
    False for the wrong reason -- the function would simply never have run.
    """
    from longline.eval.loop_resume_worker import resume

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, BEFORE_MODEL)
    arm(spec)
    report = resume(spec)
    assert report["checkpoint_loaded"] is True
    assert "num_loaded_messages" in report
    assert report["repairs"] == [], "nothing to repair in a clean turn-0 checkpoint"


# --- the settled-result progress rule ---------------------------------------


def test_an_unanswered_tool_use_is_not_settled() -> None:
    """A call that was issued and never answered is a step still in flight.

    This is the state a step-level checkpoint leaves behind when the process dies
    inside a tool, and it is the state the OLD progress rule could not see.
    """
    messages = [{"role": "user", "content": [{"type": "tool_use", "id": "tu-1", "name": "Read", "input": {}}]}]
    assert settled_steps(messages) == 0


def test_a_successful_result_settles_its_step() -> None:
    assert settled_steps(_api_transcript(2)) == 2


def test_a_plain_error_does_not_settle_a_step() -> None:
    """A real model retries a call that failed, so the script does too.

    Anything else would silently skip an operation the task needed.
    """
    assert settled_steps(_api_transcript(1, result="Error: nope", is_error=True)) == 0


def test_an_aborted_reconcile_does_not_settle_a_step() -> None:
    """The runtime PROVED nothing happened, which is the only earned retry."""
    from longline.session.tool_journal import RECONCILE_ABORTED_PREFIX

    assert settled_steps(
        _api_transcript(1, result=RECONCILE_ABORTED_PREFIX, is_error=True)
    ) == 0


def test_an_indeterminate_reconcile_does_settle_a_step() -> None:
    """The one that matters, and the whole point of the fix.

    The runtime could not verify whether the operation landed, so re-running it is
    the blind replay this mechanism exists to prevent. A real model would stop and
    ask; the scripted one takes the conservative half of that -- it does not
    repeat the call -- and that shows up here as "the step is done with".
    """
    from longline.session.tool_journal import RECONCILE_UNKNOWN_PREFIX

    assert settled_steps(
        _api_transcript(1, result=RECONCILE_UNKNOWN_PREFIX, is_error=True)
    ) == 1


def test_an_applied_reconcile_settles_its_step() -> None:
    from longline.session.tool_journal import RECONCILE_APPLIED_PREFIX

    assert settled_steps(
        _api_transcript(1, result=RECONCILE_APPLIED_PREFIX, is_error=False)
    ) == 1


def test_the_transcript_validator_placeholder_does_not_settle_a_step() -> None:
    """No journal record at all means the call never even started.

    `validate_transcript` fills the gap with "internal error", which is true here
    -- so it is retried, which is the right move and the opposite of the
    indeterminate case above.
    """
    from longline.session.recovery import SYNTHETIC_TOOL_RESULT_PLACEHOLDER

    assert settled_steps(
        _api_transcript(1, result=SYNTHETIC_TOOL_RESULT_PLACEHOLDER, is_error=True)
    ) == 0


def test_the_script_retries_an_aborted_step_rather_than_skipping_it() -> None:
    """End to end through the model: an aborted call is re-issued, same step.

    The transcript holds one finished step and one the runtime PROVED did not
    happen, so progress stops at 1 and the script issues step 1 again rather than
    advancing to step 2. Skipping it would leave the appended line missing, and
    the task would fail for a reason that has nothing to do with recovery.
    """
    from longline.session.tool_journal import RECONCILE_ABORTED_PREFIX

    model = ScriptedToolSequence(
        steps=[
            {"tool": "Read", "input": {"file_path": "a.py"}},
            {"tool": "Bash", "input": {"command": "echo x"}},
        ]
    )
    transcript = _api_transcript(1) + _api_transcript(
        1, offset=1, result=RECONCILE_ABORTED_PREFIX, is_error=True
    )[1:]
    out = _drain(model(messages=transcript))
    starts = [e for e in out if type(e).__name__ == "ToolUseStart"]
    assert [s.tool_name for s in starts] == ["Bash"]
    existing_ids = {
        block["id"]
        for message in transcript
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    }
    assert starts[0].tool_id not in existing_ids, (
        "a retry must not reuse the id of the attempt it supersedes"
    )


def test_the_script_does_not_retry_an_indeterminate_step() -> None:
    """The fix, expressed in the script: the indeterminate call is not repeated."""
    from longline.session.tool_journal import RECONCILE_UNKNOWN_PREFIX

    model = ScriptedToolSequence(
        steps=[
            {"tool": "Read", "input": {"file_path": "a.py"}},
            {"tool": "Bash", "input": {"command": "echo x"}},
        ]
    )
    out = _drain(
        model(messages=_api_transcript(1, result=RECONCILE_UNKNOWN_PREFIX, is_error=True))
    )
    starts = [e for e in out if type(e).__name__ == "ToolUseStart"]
    assert [s.tool_name for s in starts] == ["Bash"], "step 1, not a repeat of step 1"
