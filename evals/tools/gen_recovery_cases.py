"""Generate `evals/recovery.jsonl` — the six fault classes, ten runs each.

The file is generated rather than hand-written because the contract is a matrix:
six classes, ten runs apiece, and every case carries the same judge shape. Ten
copies of a line is ten chances for one of them to drift, and a drifted case
still produces a number.

Each case's task names a fixture path under a `<cwd>` placeholder for the
Process-Kill class (the working directory is a fresh temp dir at run time). For
the five runtime classes `submit(sandbox)` runs from the eval's temp cwd, so the
task names the file relative to it.

The `answer` field is what a RECOVERED run must produce. It is never emitted by
an injector: `faults.build_injection_events` returns only fault events, so a run
that did not recover has no answer to be graded against.

Usage:
    uv run python evals/tools/gen_recovery_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "evals" / "recovery.jsonl"

# The fixture every case reads. Written into the sandbox by the case's task.
FIXTURE_DIR = "notes"
FIXTURE_FILE = "value.txt"
MARKER = "marker=alpha-7f3c"

# The working-directory placeholder. Every task names its fixture through it, so
# the path a scripted agent calls is derivable from the task text alone.
CWD = "<cwd>"

REPEAT = 10


def read_task(prefix: str) -> str:
    """A task that names the fixture file it needs, relative to the cwd.

    The path is written as `<cwd>/notes/value.txt` for EVERY class, including
    the in-process ones. They do not have a placeholder to substitute at run
    time, but the agent needs a callable path and `extract_fixture_path` is what
    reads it -- so writing it the same way everywhere keeps one convention
    rather than a per-class dialect that a reader has to learn twice.
    """
    return (
        f"{prefix} The file {CWD}/" + f"{FIXTURE_DIR}/{FIXTURE_FILE} holds a single line of "
        f"the form marker=<value>. Read it and report what it contains."
    )


def kill_task() -> str:
    """The Process-Kill task. Its cwd is only known at run time."""
    return (
        "Resume the interrupted session and finish its last instruction. "
        f"Inspect the file {CWD}/" + f"{FIXTURE_DIR}/{FIXTURE_FILE} "
        "under the working directory the session was started in, then report "
        "the marker value it holds."
    )


def answer_checks() -> list[dict[str, object]]:
    """The deterministic judge, applied to the answer file.

    `file_content` with a `contains` regex, plus `not_contains` for the
    placeholder an injected run would leave behind. Both directions matter: a
    check that only asserted `contains` would pass a run whose output was the
    injector's placeholder concatenated with a real answer.
    """
    return [
        {
            "fn": "file_content",
            "args": {
                "path": "answer.txt",
                "contains": r"marker=alpha\-7f3c",
                "not_contains": r"fault\-injector",
            },
        },
    ]


def runtime_case(case_id: str, fault: str, **extra: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": case_id,
        "type": "recovery",
        "fault": fault,
        "tags": ["recovery", fault],
        "repeat": REPEAT,
        "max_turns": 8,
        "task": read_task("Read a file and report its contents."),
        "answer": MARKER,
        "checks": answer_checks(),
    }
    case.update(extra)
    return case


def kill_case() -> dict[str, object]:
    """The Process-Kill case: the judge reads the resumed leg's two outputs.

    `checks` asserts on BOTH, and they test different survivals:

    - `answer.txt` holds the answer the resumed process DERIVED -- the fixture
      file re-read after the resume, against the working directory recovered
      from the transcript.
    - `stdout.txt` holds the fact read back OUT of the persisted transcript.
      That is the transcript surviving, and it is the raw `value=` form: the
      transcript never carries the formatted answer.

    Either alone would be satisfiable by a leg that skipped half the resume: the
    fixture carries no record of the interrupted turn, and the transcript
    carries no answer to the task.
    """
    return {
        "id": "rec-kill",
        "type": "recovery",
        "fault": "process_kill",
        "tags": ["recovery", "process_kill"],
        "repeat": REPEAT,
        "max_turns": 8,
        "task": kill_task(),
        "answer": MARKER,
        "expects_repair": False,
        "checks": [
            {
                "fn": "file_content",
                "args": {
                    "path": "answer.txt",
                    "contains": r'"derived_answer": "marker=alpha\-7f3c"',
                },
            },
            {
                "fn": "command_output_contains",
                "args": {
                    "command": ["python", "-c", "print(open('stdout.txt').read())"],
                    "allowed_commands": ["python"],
                    "contains": r"value=alpha\-7f3c",
                },
            },
        ],
    }


def overflow_seed() -> list[dict[str, str]]:
    """The history a context-overflow case starts from.

    Reactive compact summarises the transcript, so a 413 arriving on an empty
    one has nothing to compress and the recovery is a no-op -- a run that could
    never succeed. The seed is therefore part of the case, not decoration, and
    the loader refuses an overflow case without it.

    Assistant-first, matching the convention the compression suite documents: a
    user-first seed makes `normalize_messages_for_api` prepend a synthetic
    "Begin." message, which would put a message in the transcript that no case
    declared.
    """
    turns: list[dict[str, str]] = [
        {"role": "assistant", "content": "Session start. We are auditing a small fixture tree."},
    ]
    for i in range(1, 7):
        turns.append({"role": "user", "content": f"Step {i}: what did we establish?"})
        turns.append({
            "role": "assistant",
            "content": (
                f"Step {i} established decision {i}: the fixture lives under "
                f"{FIXTURE_DIR}/, the marker format is marker=<value>, and no "
                "earlier conclusion is superseded. This is repeated context so "
                "the transcript has enough material for a reactive compact."
            ),
        })
    return turns


def build() -> list[dict[str, object]]:
    """The contract's matrix: five runtime classes x 10, plus Process Kill x 10.

    Exactly one case definition per fault class. `RuntimeRecoveryRate`'s
    denominator is fixed at 50 by the contract ("5 classes x 10 runs"), and an
    extra 429 definition faulting the second call instead of the first would
    push the runtime denominator to 70 -- a rate over a set the contract does
    not describe. Exercising the "second call" injection point is a unit-test
    concern (`test_faults.TestDriveQueryLoop`), not a reason to bend the
    dataset.
    """
    cases: list[dict[str, object]] = [
        runtime_case("rec-429", "429", inject_at_call_indices=[1]),
        runtime_case("rec-529", "529", inject_at_call_indices=[1]),
        runtime_case(
            # `Bash`, not `Read`: the eval registry binds Bash to the sandbox
            # (`BashTool(cwd=sandbox)`), so a Bash call can address the fixture
            # by a path the task derives. `Read` takes an absolute `file_path`
            # and resolves relative ones against the PROCESS cwd, which is not
            # the sandbox -- the retry would then fail with "File does not
            # exist" and the case would look like a failed recovery when the
            # fault had simply picked a tool that cannot reach its own fixture.
            "rec-tool-failure", "tool_failure",
            fault_tool="Bash", tool_profile="core",
        ),
        runtime_case("rec-truncate", "output_truncate", inject_at_call_indices=[1]),
        runtime_case(
            "rec-overflow", "context_overflow",
            inject_at_call_indices=[1], seed_history=overflow_seed(),
        ),
        kill_case(),
    ]
    return cases


def main() -> None:
    cases = build()
    lines = [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cases]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(int(c["repeat"]) for c in cases)
    per_fault: dict[str, int] = {}
    for case in cases:
        per_fault[str(case["fault"])] = per_fault.get(str(case["fault"]), 0) + int(case["repeat"])
    print(f"wrote {OUT}: {len(cases)} case definitions, {total} runs")
    for fault, count in sorted(per_fault.items()):
        print(f"  {fault:18s} {count}")


if __name__ == "__main__":
    main()
