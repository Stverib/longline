"""The loop-resume worker: one process that gets killed, one that resumes.

=== Two phases, two interpreters ===

`arm` runs a REAL agent loop -- production `QueryEngine`, production tools --
driven the way `main.py` drives it. A gate stops the process at the case's
failpoint and parks; the parent then kills it.

`resume` runs in a NEW interpreter, loads what the dead one left behind through
the production recovery path, and CONTINUES THE LOOP. Continuing is the whole
difference from `recovery_worker`: a resume that only re-read a value out of the
transcript cannot observe whether the agent re-executes a tool, which is what
this suite measures.

=== One `run_turn()` per instruction, not one per model call ===

`QueryEngine.run_turn()` drives `query_loop`'s own `while` state machine
(`query_loop.py:165`); the agent may make many model calls inside it. That is
what `main.py` calls once per user input, and `save_session()` runs after it
returns (`main.py:806-809`). So production's checkpoint granularity is ONE PER
INSTRUCTION, and this worker must not invent a finer one: splitting an
instruction into several `run_turn()` calls would put checkpoints where
Longline never puts them, and the suite would then be measuring the harness.

The consequence is the point of the whole experiment: a kill anywhere inside an
instruction leaves the transcript at the turn-0 floor, so the resumed leg has no
record of anything the killed leg did.

=== The scripted continuation policy, stated plainly ===

Both legs are driven by `ScriptedToolSequence`, a fixed list of tool calls the
model "decides" to make. When the resumed leg re-issues a tool call the killed
leg already executed, that is the SCRIPT's decision, not a model's. What this
suite measures is the RUNTIME's recovery semantics -- what the runtime does with
a transcript that does not mention a side effect that really happened -- and it
must never be reported as "Longline's agent duplicates side effects".

=== The turn-0 checkpoint, and why it is a deliberate deviation ===

`main.py` saves only after a `run_turn()` returns, so a crash between the user's
instruction and the first model call leaves no session file at all. Copied
faithfully, the `before_model` failpoint would have nothing to resume and would
fail for a reason unrelated to recovery -- a tautology wearing a control group's
name. So `arm` saves once before starting the loop. This deviation is recorded
in `evals/README.md` and must not be quietly dropped: it is what makes the
control arm a control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.core.events import QueryEvent, TextDelta, ToolUseStart, TurnComplete
from longline.eval.failpoints import (
    AFTER_CHECKPOINT,
    BEFORE_MODEL,
    TRUNCATE_TAIL,
    WORKSPACE_DRIFT,
    FailpointError,
    FailpointGate,
    FailpointReached,
    block_forever,
    read_sentinel,
)
from longline.eval.side_effect_journal import KILLED, RESUMED, SideEffectJournal
from longline.models.messages import Usage, UserMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

SESSION_ID = "loop-resume"
JOURNAL_NAME = "journal.jsonl"

# The failpoints whose stop lands inside INSTRUCTION 2, so instruction 1 must
# complete and be persisted first.
#
# `truncate_tail` is here for a reason found by testing rather than by design:
# the turn-0 checkpoint is ONE line, so cutting its last line leaves nothing at
# all and `load_session` returns None -- there is no resume to test. With
# instruction 1 on disk the file has many lines, the torn one is dropped, and
# what the arm actually exercises becomes visible (see `resume`).
#
# `workspace_drift` is deliberately NOT here: its fault is a mutated workspace,
# and it only interacts with the run if the resumed leg is replaying instruction
# 1. It therefore stops at instruction 1's first model call, exactly like
# `before_model`, which is also its control.
STOPS_IN_INSTRUCTION_TWO: tuple[str, ...] = (AFTER_CHECKPOINT, TRUNCATE_TAIL)

# Which GATE mechanism parks the child for each failpoint. The two parent-side
# failpoints reuse a gate because the child still has to stop somewhere for the
# parent to kill it; what makes them their own failpoint is what the parent does
# afterwards -- cut the session file, or mutate the workspace.
_GATE_TRIGGER: dict[str, str] = {
    AFTER_CHECKPOINT: AFTER_CHECKPOINT,
    TRUNCATE_TAIL: AFTER_CHECKPOINT,
    WORKSPACE_DRIFT: BEFORE_MODEL,
}

# The files the journal digests before and after each tool execution. Only the
# ones the task mutates: hashing the whole sandbox would make `post_state`
# differ for reasons unrelated to the tool under observation.
ARTIFACT_PATHS: tuple[str, ...] = ("NOTES.md", "src/calc.py", "REPORT.md")

# Instruction 2 exists only for the `after_checkpoint` failpoint, whose whole
# point is that instruction 1's completed work is on disk at the moment of the
# kill and must NOT be replayed. Instruction 1 is the case's own task text --
# the dataset owns it, not this module.
INSTRUCTION_TWO = "Now write a short REPORT.md summarising what you changed."

# Instruction 1's fixed tool sequence. Read first so the model has seen the bug,
# then the append (the side effect the after_tool arm duplicates), then the
# edit. Only forms both `cmd.exe` and `/bin/sh` accept: `BashTool` runs whatever
# `create_subprocess_shell` picks, and the two shells are not the same language.
SCENARIO_ONE: tuple[dict[str, Any], ...] = (
    {"tool": "Read", "input": {"file_path": "src/calc.py"}},
    {"tool": "Bash", "input": {"command": "echo fixed-add >> NOTES.md"}},
    {
        "tool": "Edit",
        "input": {
            "file_path": "src/calc.py",
            "old_string": "return a - b",
            "new_string": "return a + b",
        },
    },
)

# Instruction 2's sequence. It gives the after_checkpoint arm something to do
# AFTER the resumed leg has loaded instruction 1's completed work.
SCENARIO_TWO: tuple[dict[str, Any], ...] = (
    {
        "tool": "Write",
        "input": {
            "file_path": "REPORT.md",
            "content": "# Report\n\nFixed add() so it sums its arguments.\n",
        },
    },
)

RUNNING_TASK_SNAPSHOT: list[dict[str, Any]] = [
    {
        "task_id": "b-9f8e7d6c",
        "task_type": "local_bash",
        "state": "running",
        "created_at": 0.0,
        "updated_at": 0.0,
        "metadata": {"command": "sleep 600"},
    },
]


def instruction_offset() -> int:
    """How many tool_use blocks instruction 1 contributes.

    Instruction 2's scripted sequence needs it: progress is derived from the
    transcript, and after instruction 1 the transcript already carries this many
    tool_use blocks. Without the offset the model would read instruction 1's
    tool uses as its own progress and answer without doing anything.
    """
    return len(SCENARIO_ONE)


def _count_tool_uses(messages: Sequence[Any]) -> int:
    """How many tool_use blocks the transcript already carries.

    Reads the API shape (`to_api_dict` via `normalize_messages_for_api`),
    because that is what `query_loop` hands the model -- counting the native
    blocks would work in-process and silently break the moment the transcript
    came back off disk.
    """
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        total += sum(
            1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use"
        )
    return total


@dataclass
class ScriptedToolSequence:
    """A scripted model that walks a fixed tool sequence, then answers.

    Progress is derived from the transcript rather than from a counter on this
    object. Two things follow, and both are load-bearing:

    - The resumed leg continues the sequence where the killed leg stopped,
      because the checkpoint is the only progress record that crossed the kill.
    - When the checkpoint has no record of a side effect that happened (the
      `after_tool` arm), the resumed leg re-issues that call -- which is the
      behaviour the duplicate metrics exist to observe.

    `offset` is how many tool_use blocks earlier instructions already put on the
    transcript.
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    offset: int = 0
    answer: str = ""

    def __call__(self, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        return self._serve(list(kwargs.get("messages", [])))

    async def _serve(self, messages: list[Any]) -> AsyncIterator[QueryEvent]:
        done = max(0, _count_tool_uses(messages) - self.offset)
        if done < len(self.steps):
            step = self.steps[done]
            yield ToolUseStart(
                tool_name=str(step["tool"]),
                tool_id=f"tu-{self.offset + done + 1}",
                input=dict(step["input"]),
            )
            yield TurnComplete(
                stop_reason="tool_use",
                usage=Usage(input_tokens=100, output_tokens=20),
            )
            return
        yield TextDelta(text=self.answer)
        yield TurnComplete(
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=25),
        )


def build_scenario_engine(
    sandbox: str,
    *,
    model: str,
    api_key: str,
    gate: FailpointGate,
    journal: SideEffectJournal | None,
    sequence: ScriptedToolSequence,
) -> Any:
    """A real `QueryEngine` whose model is scripted and whose tools are gated.

    Construction mirrors `faults.fault_engine`: the engine, registry, tools and
    permission context all stay real; only the model transport is replaced.
    """
    from longline.eval.engine_factory import build_engine
    from longline.eval.failpoints import GatedModel, GatedTool

    engine = build_engine(sandbox=sandbox, model=model, api_key=api_key, tool_profile="core")

    for name in ("Read", "Edit", "Write", "Bash"):
        inner = engine.registry.get(name)
        if inner is None:
            raise FailpointError(f"tool {name!r} is not in the core profile")
        engine.registry.swap(
            name,
            GatedTool(
                inner=inner,
                gate=gate,
                journal=journal,
                artifact_paths=ARTIFACT_PATHS,
                artifact_root=sandbox,
            ),
        )

    engine.make_call_model = lambda model=None, max_tokens=16384: GatedModel(
        inner=sequence, gate=gate
    )
    return engine


def _save_checkpoint(engine: Any, claude_dir: Path, session_id: str) -> None:
    """The checkpoint write, in `main.py`'s position and with its argument shape."""
    from longline.session.storage import save_session

    save_session(
        session_id,
        engine.messages,
        claude_dir=claude_dir,
        task_snapshot=RUNNING_TASK_SNAPSHOT,
    )


async def _run_instruction(engine: Any) -> list[str]:
    """Run ONE instruction the way production does: a single `run_turn()`.

    NOT split into one call per model call -- see the module docstring. The
    agent may make many model calls inside this one call, and production writes
    no checkpoint until it returns.

    Returns the text of every errored tool result, in order. That is what a
    workspace-identity check would have to produce for a stale checkpoint to be
    *detected*; with no such mechanism in production the list stays empty, and
    the drift arm's job is to show that it does.
    """
    from longline.core.events import ToolResultReady

    names: dict[str, str] = {}
    errors: list[str] = []
    async for event in engine.run_turn():
        if isinstance(event, ToolUseStart):
            names[event.tool_id] = event.tool_name
        elif isinstance(event, ToolResultReady) and event.is_error:
            errors.append(f"{names.get(event.tool_id, '?')}: {event.content}")
    return errors


def _gate_block(spec: dict[str, Any]) -> Callable[[], None]:
    """The gate's parking behaviour.

    Tests pass `no_block` so the phase returns in-process; a real run parks
    until the parent kills it. Injected rather than monkeypatched, so the
    production path and the test path differ in exactly one value.
    """
    if spec.get("no_block"):
        from longline.eval.failpoints import halt

        return halt
    return block_forever


def _build_gate(spec: dict[str, Any], failpoint: str) -> FailpointGate:
    trigger = _GATE_TRIGGER.get(failpoint, failpoint)
    return FailpointGate(
        claude_dir=Path(spec["claude_dir"]),
        # The sentinel names the CASE's failpoint, not the mechanism that
        # stopped the child, so the runner can assert the right one fired.
        failpoint=failpoint,
        trigger=trigger,
        at_call_index=int(spec.get("at_call_index", 1)),
        at_tool_name=str(spec.get("failpoint_tool", "")),
        # Armed from the start for every arm whose stop is inside instruction 1.
        # For the two that stop inside instruction 2 it stays disarmed until
        # instruction 1 is saved, so `GatedModel`'s armed-only counter starts at
        # instruction 2's first call -- no hand-kept call index to go stale.
        armed=trigger != AFTER_CHECKPOINT,
        block=_gate_block(spec),
    )


def arm(spec: dict[str, Any]) -> dict[str, Any]:
    """Phase 1: build the checkpoint, run instruction 1, and stop at the failpoint.

    For `after_checkpoint` the stop is inside INSTRUCTION 2: instruction 1 runs
    to completion and is saved first, and the gate is armed only afterwards.
    That is what puts a *completed* instruction on disk at the moment of the
    kill -- the one arm where the resumed leg has something it must NOT redo.
    """
    from longline.session.storage import get_sessions_dir

    claude_dir = Path(spec["claude_dir"])
    sandbox = Path(spec["sandbox"])
    session_id = str(spec.get("session_id", SESSION_ID))
    failpoint = str(spec["failpoint"])

    gate = _build_gate(spec, failpoint)
    journal = SideEffectJournal(claude_dir / JOURNAL_NAME, KILLED)
    sequence = ScriptedToolSequence(steps=list(SCENARIO_ONE), offset=0)
    engine = build_scenario_engine(
        str(sandbox),
        model=str(spec.get("model", "offline-model")),
        api_key=str(spec.get("api_key", "offline")),
        gate=gate,
        journal=journal,
        sequence=sequence,
    )

    engine.messages.append(UserMessage(content=str(spec["task"])))
    _save_checkpoint(engine, claude_dir, session_id)
    before_loop = len(engine.messages)

    instructions_run = 0
    error = ""
    stopped_at_failpoint = False
    tool_errors: list[str] = []
    try:
        if failpoint in STOPS_IN_INSTRUCTION_TWO:
            asyncio.run(_run_instruction(engine))
            instructions_run += 1
            _save_checkpoint(engine, claude_dir, session_id)
            gate.armed = True
            sequence.steps = list(SCENARIO_TWO)
            sequence.offset = instruction_offset()
            engine.messages.append(UserMessage(content=INSTRUCTION_TWO))
        tool_errors = asyncio.run(_run_instruction(engine))
        instructions_run += 1
    except FailpointReached:
        # The failpoint fired and `no_block` asked it to unwind instead of
        # parking. The sentinel is already on disk, which is the evidence.
        stopped_at_failpoint = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    sentinel = read_sentinel(claude_dir)
    session_file = get_sessions_dir(claude_dir) / f"{session_id}.jsonl"
    return {
        "phase": "arm",
        "session_id": session_id,
        "claude_dir": str(claude_dir),
        "session_file": str(session_file),
        "failpoint_reached": sentinel is not None,
        "sentinel": sentinel,
        "gate_reached": gate.reached,
        "stopped_at_failpoint": stopped_at_failpoint,
        "instructions_run": instructions_run,
        "checkpoint_messages_before_loop": before_loop,
        "journal_entries": len(journal.entries),
        "tool_errors": tool_errors,
        "error": error,
    }


# --- phase 2: resume ---


def check_transcript_structure(messages: Sequence[Any]) -> tuple[bool, list[str]]:
    """Structural validity of a resumed transcript: API pairing + alternation.

    Two independent conditions, each of which the API enforces when the
    transcript goes back over the wire:

    - every `tool_use` id has a matching `tool_result` (the API rejects the
      request otherwise, which is what `validate_transcript` exists to fix);
    - no `tool_result` refers to an id that was never requested;
    - no two consecutive messages share a role.

    **Deliberately absent: "the transcript must not end on an assistant
    message".** `recovery_worker.check_transcript_structure` has that rule and
    this function was copied from it, but it is wrong here. A checkpoint taken
    between two instructions legitimately ends on the assistant's final text --
    that is what `main.py` writes after every `run_turn()`, and the next user
    message is appended before the next one. The rule cost the
    `after_checkpoint` arm all ten of its state-layer verdicts, dragging
    `LoopResumeRate` down for a reason that had nothing to do with recovery.

    The dangerous case it was meant to catch -- a trailing `tool_use` with no
    result -- is already caught by the pairing rule above, so nothing is lost.
    `truncate_tail` never tripped it only because the torn final line is
    dropped, leaving the transcript on a `tool_result` instead.
    """
    from longline.models.content_blocks import ToolResultBlock, ToolUseBlock
    from longline.models.messages import AssistantMessage, UserMessage

    errors: list[str] = []
    tool_use_ids: list[str] = []
    result_ids: list[str] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    tool_use_ids.append(block.id)
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    result_ids.append(block.tool_use_id)

    unanswered = [i for i in tool_use_ids if i not in set(result_ids)]
    if unanswered:
        errors.append(f"tool_use without tool_result: {unanswered}")
    orphan_results = [i for i in result_ids if i not in set(tool_use_ids)]
    if orphan_results:
        errors.append(f"tool_result without tool_use: {orphan_results}")

    for i in range(1, len(messages)):
        if type(messages[i]) is type(messages[i - 1]):
            errors.append(f"role alternation violated at message {i}")

    return (not errors), errors


def _restore_task_snapshot(snapshot: list[dict[str, Any]] | None) -> dict[str, str]:
    """Run the production `TaskRegistry.restore()` and report the states it set."""
    from longline.session.task_registry import TaskRegistry

    registry = TaskRegistry()
    if snapshot:
        registry.restore(snapshot)
    return {r.task_id: r.state.value for r in registry.list_all()}


def _not_found(session_id: str, detail: str) -> dict[str, Any]:
    return {
        "phase": "resume",
        "session_id": session_id,
        "checkpoint_loaded": False,
        "layer_state_ok": False,
        "structural_errors": [detail],
        "transcript_repaired": False,
        "repairs": [],
        "instructions_run": 0,
        "task_states": {},
        "error": detail,
    }


def resume(spec: dict[str, Any]) -> dict[str, Any]:
    """Phase 2: load what the dead process left, then FINISH THE TASK.

    The production order and the production functions, in `main.py`'s sequence:
    `load_session` -> `validate_transcript` -> `load_task_snapshot` ->
    `TaskRegistry.restore`. Then the loop runs again on the recovered
    transcript -- which is where a replayed side effect shows up.
    """
    from longline.session.recovery import TranscriptRepairReport, validate_transcript
    from longline.session.storage import load_session, load_task_snapshot

    claude_dir = Path(spec["claude_dir"])
    sandbox = Path(spec["sandbox"])
    session_id = str(spec.get("session_id", SESSION_ID))
    failpoint = str(spec.get("failpoint", ""))

    loaded = load_session(session_id, claude_dir=claude_dir)
    if loaded is None:
        return _not_found(session_id, "session not found")

    repair_report = TranscriptRepairReport()
    repaired = validate_transcript(loaded, report=repair_report)
    structural_ok, structural_errors = check_transcript_structure(repaired)

    snapshot = load_task_snapshot(session_id, claude_dir=claude_dir)
    task_states = _restore_task_snapshot(snapshot)

    # Disarmed: the gate is a fixture of the killed leg. Armed here it would
    # park this process forever and every case would time out.
    gate = FailpointGate(
        claude_dir=claude_dir,
        failpoint=failpoint,
        at_call_index=int(spec.get("at_call_index", 1)),
        at_tool_name=str(spec.get("failpoint_tool", "")),
        armed=False,
    )
    journal = SideEffectJournal(claude_dir / JOURNAL_NAME, RESUMED)

    if failpoint in STOPS_IN_INSTRUCTION_TWO:
        # Instruction 1 is already on the transcript, so the model picks up at
        # instruction 2's step -- and instruction 2 itself was never persisted
        # (it was typed and the process died before any save). Re-supplying it
        # is what a user does after a crash, not a convenience for the harness.
        sequence = ScriptedToolSequence(steps=list(SCENARIO_TWO), offset=instruction_offset())
    else:
        sequence = ScriptedToolSequence(steps=list(SCENARIO_ONE), offset=0)

    engine = build_scenario_engine(
        str(sandbox),
        model=str(spec.get("model", "offline-model")),
        api_key=str(spec.get("api_key", "offline")),
        gate=gate,
        journal=journal,
        sequence=sequence,
    )
    engine.messages.extend(repaired)
    if failpoint in STOPS_IN_INSTRUCTION_TWO:
        engine.messages.append(UserMessage(content=INSTRUCTION_TWO))

    error = ""
    instructions_run = 0
    tool_errors: list[str] = []
    try:
        tool_errors = asyncio.run(_run_instruction(engine))
        instructions_run = 1
    except BaseException as exc:  # reported, not swallowed
        error = f"{type(exc).__name__}: {exc}"

    return {
        "phase": "resume",
        "session_id": session_id,
        "checkpoint_loaded": True,
        "num_loaded_messages": len(loaded),
        "num_repaired_messages": len(repaired),
        "transcript_repaired": repair_report.repaired,
        "repairs": list(repair_report.repairs),
        "structural_ok": structural_ok,
        "structural_errors": structural_errors,
        "layer_state_ok": structural_ok and not error,
        "task_states": task_states,
        "instructions_run": instructions_run,
        "journal_entries": len(journal.entries),
        "tool_errors": tool_errors,
        "error": error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m longline.eval.loop_resume_worker")
    parser.add_argument("phase", choices=["arm", "resume"])
    parser.add_argument("spec", help="Path to the JSON spec file.")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    report = arm(spec) if args.phase == "arm" else resume(spec)
    # One JSON object on stdout and nothing else, so the parent can parse it
    # even when a later run's process was killed mid-write.
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
