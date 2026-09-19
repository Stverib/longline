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
    FailpointError,
    FailpointGate,
    FailpointReached,
    block_forever,
    read_sentinel,
)
from longline.eval.side_effect_journal import KILLED, SideEffectJournal
from longline.models.messages import Usage, UserMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

SESSION_ID = "loop-resume"
JOURNAL_NAME = "journal.jsonl"

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


async def _run_instruction(engine: Any) -> int:
    """Run ONE instruction the way production does: a single `run_turn()`.

    NOT split into one call per model call -- see the module docstring. The
    agent may make many model calls inside this one call, and production writes
    no checkpoint until it returns.
    """
    async for _event in engine.run_turn():
        pass
    return 1


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
    return FailpointGate(
        claude_dir=Path(spec["claude_dir"]),
        failpoint=failpoint,
        at_call_index=int(spec.get("at_call_index", 1)),
        at_tool_name=str(spec.get("failpoint_tool", "")),
        # Armed from the start for every arm whose stop is inside instruction 1.
        # For after_checkpoint it stays disarmed until instruction 1 is saved,
        # so `GatedModel`'s armed-only counter starts at instruction 2's first
        # call -- no hand-kept call index to go stale.
        armed=failpoint != AFTER_CHECKPOINT,
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
    try:
        if failpoint == AFTER_CHECKPOINT:
            asyncio.run(_run_instruction(engine))
            instructions_run += 1
            _save_checkpoint(engine, claude_dir, session_id)
            gate.armed = True
            sequence.steps = list(SCENARIO_TWO)
            sequence.offset = instruction_offset()
            engine.messages.append(UserMessage(content=INSTRUCTION_TWO))
        asyncio.run(_run_instruction(engine))
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
        "error": error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m longline.eval.loop_resume_worker")
    parser.add_argument("phase", choices=["arm", "resume"])
    parser.add_argument("spec", help="Path to the JSON spec file.")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    report = arm(spec)
    # One JSON object on stdout and nothing else, so the parent can parse it
    # even when a later run's process was killed mid-write.
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
