"""Case data model and loader for the recovery / resume suite.

A recovery case is not an E2E case with a flag on it. Three things differ, and
each one is a reason for a separate type rather than a nullable field:

1. **The verdict is composite.** "Recovered" means the fault was really
   injected, the recovery path really fired, and the judge passed -- a single
   `checks` list cannot express the first two, and a case that encoded them as
   extra checks would let a run report `passed=false` with no way to tell which
   of the three legs broke.
2. **Two injection dialects.** The five in-process classes are triggered by
   scripted model events (429/529/truncate/overflow) or by a tool wrapper (tool
   failure). Process Kill is triggered by a signal to a child process. One
   record type has to carry both, so the fields are explicit rather than a
   free-form bag.
3. **The dataset is a matrix.** Ten runs per fault class, six classes. The
   loader therefore supports `repeat` -- the case file declares one case per
   class and the loader expands it, so "10 runs of 429" cannot be produced by
   copy-pasting a line nine times and drifting.

The deterministic judges are the SAME `longline.eval.judges` functions every
other suite uses. Reusing them is what keeps a recovery verdict comparable to a
TaskSuccess verdict; a second judge vocabulary would make the two numbers look
alike while measuring different things.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from longline.eval.faults import (
    ALL_FAULTS,
    CONTEXT_OVERFLOW,
    OUTPUT_TRUNCATE,
    OVERLOADED,
    PROCESS_KILL,
    RATE_LIMIT,
    TOOL_FAILURE,
)
from longline.eval.types import CaseParseError, E2ECase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# The fault classes that are injected by a scripted model stream. Their
# injection point is a call index, and the case file must name one.
MODEL_FAULTS: tuple[str, ...] = (
    RATE_LIMIT, OVERLOADED, OUTPUT_TRUNCATE, CONTEXT_OVERFLOW,
)

# Where a fault class may be injected, in the contract's words: "首次或前两次
# model call". Index 0 would mean "never", which is a case that can never be
# counted as injected, so it is rejected at load time rather than at 3am.
MIN_CALL_INDEX = 1
MAX_CALL_INDEX = 2

# The fixture path every Process-Kill case's task names. Kept beside the case
# model because the task text and the tag are two halves of one contract: the
# resumed leg resolves this path from the task against the working directory it
# recovered from the transcript.
FIXTURE_FILE_PATH = "notes/value.txt"

# Sentinel substituted into the Process-Kill task at load time. The real working
# directory is only known at run time (it is a fresh temp dir), so the case file
# carries a placeholder and the loader checks it is present. A task without it
# is unanswerable offline and would fail for a reason unrelated to the fault.
CWD_PLACEHOLDER = "<cwd>"


@dataclass
class RecoveryCase(E2ECase):
    """One fault injection, run `repeat` times.

    Inherits `E2ECase` so the deterministic judge layer (`checks`, `checks_mode`,
    `case_passed`) is literally the same code path as every other suite. The
    inherited `task` is the real instruction -- unlike a compression case, the
    agent here is asked to do something and its output is graded on its own.

    Fields beyond `E2ECase`:
        fault: one of `faults.ALL_FAULTS`.
        repeat: how many independent runs this case contributes. The dataset
            uses 10 per class (`evals/README.md` §5.4), and the expansion
            happens in `load_recovery_cases` so the id/repeat_index pair is the
            only thing separating two runs.
        inject_at_call_indices: for the four model-stream classes, the model call
            indices to fail. At least one of 1 or 2 (the contract's "first or
            second model call").
        fault_tool: for `tool_failure`, the tool that returns `is_error=true` on
            its first call. Required for that class and forbidden for the rest:
            a tool name on a 429 case would be silently ignored.
        answer: the text a correct (recovered) response contains. It is what the
            judge asserts on, never something the injector emits -- a fault that
            emitted the answer would make the case pass while measuring nothing.
        expects_repair: whether the Process-Kill checkpoint is deliberately cut
            mid-turn so the resumed transcript needs `validate_transcript()`.
            Recorded so `transcript_repaired=true` is an expected result rather
            than a surprise, and so a clean checkpoint case is distinguishable.
        tool_profile: which eval registry profile the case runs against.
        seed_history: for `context_overflow`, the scripted conversation the model
            starts from. Reactive compact can only fire when there is history to
            compress; against an empty transcript it has nothing to do, and the
            case would report a failed recovery for a reason that has nothing to
            do with recovery. Empty for every other class, which start clean.
    """

    fault: str = RATE_LIMIT
    repeat: int = 1
    inject_at_call_indices: list[int] = field(default_factory=lambda: [MIN_CALL_INDEX])
    fault_tool: str | None = None
    answer: str = ""
    expects_repair: bool = False
    tool_profile: str = "core"
    seed_history: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_process_kill(self) -> bool:
        return self.fault == PROCESS_KILL

    @property
    def is_runtime_fault(self) -> bool:
        return self.fault != PROCESS_KILL

    @property
    def run_id_prefix(self) -> str:
        """The id stem two runs of this case share, e.g. `rec-429`."""
        return self.id

    def run_id(self, repeat_index: int) -> str:
        """Unique id for one of this case's runs.

        Derived rather than stored: the id and the repeat index are the same
        fact, and storing both invites a row whose `case_id` says run 3 while
        `repeat_index` says 2.
        """
        return self.id if repeat_index == 0 else f"{self.id}-r{repeat_index}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecoveryCase:
        cid = d.get("id")
        if not isinstance(cid, str) or not cid:
            raise CaseParseError(f"recovery case requires a string 'id', got {d!r}")

        fault = d.get("fault")
        if fault not in ALL_FAULTS:
            raise CaseParseError(
                f"{cid}: unknown fault {fault!r} (known: {list(ALL_FAULTS)})"
            )

        answer = d.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            # Required, not optional: without a declared answer the judge has
            # nothing to assert on, and a case whose judge asserts nothing is
            # the vacuous pass this suite exists to avoid.
            raise CaseParseError(f"{cid}: requires a non-empty string 'answer'")

        repeat = d.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise CaseParseError(f"{cid}: 'repeat' must be an int >= 1, got {repeat!r}")

        indices = cls._parse_indices(d.get("inject_at_call_indices"), case_id=cid, fault=str(fault))
        fault_tool = d.get("fault_tool")
        if fault_tool is not None and not isinstance(fault_tool, str):
            raise CaseParseError(f"{cid}: 'fault_tool' must be a str, got {fault_tool!r}")
        if fault == TOOL_FAILURE and not fault_tool:
            raise CaseParseError(
                f"{cid}: a tool_failure case must name 'fault_tool'; the fault has "
                "nothing to attach to otherwise"
            )
        if fault != TOOL_FAILURE and fault_tool is not None:
            raise CaseParseError(
                f"{cid}: 'fault_tool' is only meaningful for {TOOL_FAILURE}, "
                f"got it on a {fault} case"
            )

        expects_repair = d.get("expects_repair", False)
        if not isinstance(expects_repair, bool):
            raise CaseParseError(f"{cid}: 'expects_repair' must be a bool, got {expects_repair!r}")

        tool_profile = d.get("tool_profile", "core")
        if not isinstance(tool_profile, str):
            raise CaseParseError(f"{cid}: 'tool_profile' must be a str, got {tool_profile!r}")

        seed_history = cls._parse_seed(d.get("seed_history"), case_id=cid, fault=str(fault))

        # `E2ECase` reads the top-level `task`/`checks`; reuse it rather than
        # restating the validation, so this loader cannot drift from the E2E one.
        base = E2ECase.from_dict(d)
        if fault == PROCESS_KILL and CWD_PLACEHOLDER not in base.task:
            raise CaseParseError(
                f"{cid}: a process_kill task must contain {CWD_PLACEHOLDER!r}; the "
                "working directory is only known at run time, and a task that "
                "hard-codes one would name a directory that does not exist"
            )
        if fault != PROCESS_KILL and extract_fixture_path(base.task) is None:
            # Every runtime class's scripted agent reads the path its task names,
            # so a task that names none leaves the agent with nothing to call and
            # the case would fail for a reason unrelated to its fault.
            raise CaseParseError(
                f"{cid}: a {fault} task must name its fixture as "
                f"{CWD_PLACEHOLDER}/<relative path>; no callable path is otherwise "
                "derivable from the task text"
            )

        return cls(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            fixture=base.fixture,
            checks=base.checks,
            checks_mode=base.checks_mode,
            judge=base.judge,
            fault=str(fault),
            repeat=repeat,
            inject_at_call_indices=indices,
            fault_tool=fault_tool,
            answer=answer,
            expects_repair=expects_repair,
            tool_profile=tool_profile,
            seed_history=seed_history,
        )

    @staticmethod
    def _parse_seed(raw: object, *, case_id: str, fault: str) -> list[dict[str, str]]:
        """Validate a seed history, and require one only where it is load-bearing.

        `context_overflow` is the only class whose recovery path (reactive
        compact) needs something to compress. A case in that class with no seed
        would report a failed recovery caused by an empty transcript rather than
        by the fault, so the omission is a data bug and is rejected at load time.
        """
        if raw is None:
            if fault == CONTEXT_OVERFLOW:
                raise CaseParseError(
                    f"{case_id}: a {CONTEXT_OVERFLOW} case needs a non-empty "
                    "'seed_history'; reactive compact has nothing to compress "
                    "against an empty transcript"
                )
            return []
        if not isinstance(raw, list) or not raw:
            raise CaseParseError(f"{case_id}: 'seed_history' must be a non-empty list, got {raw!r}")
        parsed: list[dict[str, str]] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise CaseParseError(f"{case_id}: seed_history[{i}] must be an object, got {entry!r}")
            role = entry.get("role")
            content = entry.get("content")
            if role not in ("assistant", "user"):
                raise CaseParseError(
                    f"{case_id}: seed_history[{i}] role {role!r} must be assistant or user"
                )
            if not isinstance(content, str) or not content:
                raise CaseParseError(
                    f"{case_id}: seed_history[{i}] needs non-empty string content"
                )
            parsed.append({"role": str(role), "content": content})
        return parsed

    @staticmethod
    def _parse_indices(raw: object, *, case_id: str, fault: str) -> list[int]:
        """Validate the injection indices for the fault class in question.

        Every rule here is about making a never-firing injection impossible to
        load rather than about tidiness: a case whose fault cannot fire reports
        `fault_injected=false` on all ten runs, which shrinks the numerator while
        leaving the denominator intact -- a wrong number that looks like a
        finding.
        """
        if fault == PROCESS_KILL:
            if raw is not None:
                raise CaseParseError(
                    f"{case_id}: 'inject_at_call_indices' does not apply to {PROCESS_KILL} "
                    "(there is no model call to fail)"
                )
            return []
        if raw is None:
            return [MIN_CALL_INDEX]
        if not isinstance(raw, list) or not raw:
            raise CaseParseError(
                f"{case_id}: 'inject_at_call_indices' must be a non-empty list, got {raw!r}"
            )
        out: list[int] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, int):
                raise CaseParseError(
                    f"{case_id}: injection indices must be ints, got {value!r}"
                )
            if not (MIN_CALL_INDEX <= value <= MAX_CALL_INDEX):
                raise CaseParseError(
                    f"{case_id}: injection index {value} is outside "
                    f"{MIN_CALL_INDEX}..{MAX_CALL_INDEX} (the contract's "
                    "'first or second model call'; 0 would never fire)"
                )
            out.append(value)
        return sorted(set(out))


def expand_case(case: RecoveryCase) -> list[RecoveryCase]:
    """One case object per declared run, with unique ids.

    The copies share their judge and injection config by value; only the id
    differs. `dataclasses.replace` is not used because the sub-case also needs
    its `run_id` to stay derivable -- the id IS the repeat index, so nothing
    else is stored.
    """
    return [
        RecoveryCase(
            id=case.run_id(i),
            task=case.task,
            max_turns=case.max_turns,
            tags=list(case.tags),
            fixture=case.fixture,
            checks=[dict(c) for c in case.checks],
            checks_mode=case.checks_mode,
            judge=dict(case.judge),
            fault=case.fault,
            repeat=1,
            inject_at_call_indices=list(case.inject_at_call_indices),
            fault_tool=case.fault_tool,
            answer=case.answer,
            expects_repair=case.expects_repair,
            tool_profile=case.tool_profile,
            seed_history=[dict(h) for h in case.seed_history],
        )
        for i in range(case.repeat)
    ]


_PATH_RE = re.compile(re.escape(CWD_PLACEHOLDER) + r"[/\\]([\w./\\-]+)")


def extract_fixture_path(task: str) -> str | None:
    """The fixture path a task names, relative to its `<cwd>` placeholder.

    Written as a function rather than left to the resumed leg so the dataset
    contract test can assert, for every case, that the task names a path at all.
    A task that names nothing is unanswerable offline, and the failure would
    surface as a judge mismatch rather than as the data bug it is.
    """
    match = _PATH_RE.search(task)
    return match.group(1).replace("\\", "/") if match else None


def load_recovery_cases(path: Path, *, fixtures_root: Path | None = None) -> list[RecoveryCase]:
    """Load and expand recovery cases from a JSONL file.

    `repeat` is expanded here rather than at run time so the dataset contract
    tests can count the same thing the runner will: "10 runs of 429" must be a
    property of the loaded case list, not of a loop inside the runner that a
    reader has to go and find.
    """
    from longline.eval.types import validate_fixtures

    cases: list[RecoveryCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseParseError(f"{path}:{lineno}: bad JSON: {exc}") from exc
        if not isinstance(d, dict):
            raise CaseParseError(f"{path}:{lineno}: expected JSON object, got {type(d).__name__}")
        ctype: Literal["recovery"] | str | None = d.get("type")
        if ctype != "recovery":
            raise CaseParseError(f"{path}:{lineno}: unknown case type {ctype!r}")
        cases.extend(expand_case(RecoveryCase.from_dict(d)))

    root = path.parent / "fixtures" if fixtures_root is None else fixtures_root
    validate_fixtures([c for c in cases if c.fixture], root)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise CaseParseError(f"{path}: duplicate case ids after expansion: {dupes}")
    return cases


def cases_by_fault(cases: Iterable[RecoveryCase]) -> dict[str, list[RecoveryCase]]:
    """Group loaded cases by fault class, in `ALL_FAULTS` order."""
    grouped: dict[str, list[RecoveryCase]] = {fault: [] for fault in ALL_FAULTS}
    for case in cases:
        grouped[case.fault].append(case)
    return grouped


def resolve_cwd(task: str, cwd: str) -> str:
    """Substitute the real working directory into a Process-Kill task."""
    return task.replace(CWD_PLACEHOLDER, cwd)


__all__ = [
    "CWD_PLACEHOLDER",
    "FIXTURE_FILE_PATH",
    "MAX_CALL_INDEX",
    "MIN_CALL_INDEX",
    "MODEL_FAULTS",
    "RecoveryCase",
    "cases_by_fault",
    "expand_case",
    "extract_fixture_path",
    "load_recovery_cases",
    "resolve_cwd",
]
