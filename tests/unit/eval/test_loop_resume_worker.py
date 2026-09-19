"""The arm phase: a real loop driven once per instruction, stopped by a gate."""

from __future__ import annotations

import asyncio
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
from longline.eval.loop_resume_worker import (
    ARTIFACT_PATHS,
    SCENARIO_ONE,
    SCENARIO_TWO,
    ScriptedToolSequence,
    arm,
    instruction_offset,
)
from longline.eval.side_effect_journal import KILLED, RESUMED, read_journal

FIXTURE = Path("evals/fixtures/resume_repo")
TASK = 'In {cwd}: fix the bug in src/calc.py and append a line containing "fixed-add" to NOTES.md.'


def _drain(agen: Any) -> list[Any]:
    async def _run() -> list[Any]:
        return [event async for event in agen]

    return asyncio.run(_run())


def _api_transcript(tool_uses: int, *, offset: int = 0) -> list[dict[str, Any]]:
    """A transcript in the API shape `query_loop` hands the model."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for i in range(tool_uses):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"tu-{offset + i + 1}", "name": "Read", "input": {}}
                ],
            }
        )
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x"}]})
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


def test_scenario_one_gates_on_the_step_whose_side_effect_gets_replayed() -> None:
    """The after_tool arm gates on Bash, and the FIRST Bash is the append. So
    the killed leg's only state-changing execution is the append itself --
    exactly the side effect the resumed leg replays. If someone reorders the
    sequence so a read-only command comes first, the denominator drops to zero
    and the metric silently measures nothing."""
    tools = [s["tool"] for s in SCENARIO_ONE]
    assert tools[0] == "Read", "a read must not be the gated step"
    first_bash = tools.index("Bash")
    assert ">>" in SCENARIO_ONE[first_bash]["input"]["command"]
    assert "Edit" in tools


def test_scenario_two_is_a_single_write() -> None:
    assert [s["tool"] for s in SCENARIO_TWO] == ["Write"]


def test_instruction_offset_is_scenario_one_length() -> None:
    assert instruction_offset() == len(SCENARIO_ONE) == 3


def test_artifact_paths_cover_every_mutated_file() -> None:
    assert set(ARTIFACT_PATHS) == {"NOTES.md", "src/calc.py", "REPORT.md"}


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

    # The turn-0 floor: the transcript on disk holds the instruction and
    # nothing else. The task text itself mentions "fixed-add", so the absence
    # to assert is the ABSENCE OF ANY TOOL TRAFFIC, not of the string.
    session_lines = [
        ln for ln in
        (tmp_path / "claude" / "sessions" / "loop-resume.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(session_lines) == 1, "only the instruction should have been persisted"
    assert "tool_use" not in session_lines[0]
    assert "tool_result" not in session_lines[0]


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


def test_after_tool_resume_replays_the_append_the_transcript_never_saw(tmp_path: Path) -> None:
    """The core measurement, end to end.

    The killed leg appended to NOTES.md and the turn-0 checkpoint has no record
    of it, so the resumed leg appends again. Both the redundant re-execution
    and the duplicated side effect must be visible in the journal -- and the
    file must really carry the line twice, or the metric would be describing
    something that did not happen.
    """
    from longline.eval.loop_resume_worker import resume
    from longline.eval.side_effect_journal import compute_side_effect_metrics

    sandbox = _sandbox(tmp_path)
    spec = _spec(tmp_path, sandbox, AFTER_TOOL, failpoint_tool="Bash")
    arm(spec)
    assert (sandbox / "NOTES.md").read_text(encoding="utf-8").count("fixed-add") == 1

    resume(spec)

    notes = (sandbox / "NOTES.md").read_text(encoding="utf-8")
    assert notes.count("fixed-add") == 2, "the append was not replayed, so nothing was measured"

    entries = read_journal(tmp_path / "claude" / "journal.jsonl")
    assert {e.leg for e in entries} == {KILLED, RESUMED}
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 1, "the append is the only state-changing execution"
    assert metrics.redundant == 1
    assert metrics.duplicated == 1


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
