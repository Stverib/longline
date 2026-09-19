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
from longline.eval.side_effect_journal import KILLED, read_journal

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


def _sandbox(tmp_path: Path) -> Path:
    sandbox = tmp_path / "sandbox"
    shutil.copytree(FIXTURE, sandbox)
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
    # Nothing ran, so nothing was mutated.
    assert (sandbox / "NOTES.md").read_text(encoding="utf-8") == "# Notes\n"
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
