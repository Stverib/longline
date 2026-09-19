"""Case data model and loader for the loop-resume suite.

A loop-resume case is not a recovery case with a different flag. Two things
differ, and each is a reason for its own type:

1. **The verdict is four-layered.** State integrity, execution integrity,
   workspace integrity and task success are checked separately, because a run
   that finishes the task while duplicating a side effect is a different defect
   from one that fails to finish -- and a single boolean could not tell them
   apart.
2. **The fault is a PLACE, not a class.** Recovery cases inject a fault into a
   component (429, a tool error); this suite stops the whole process at a named
   point in the loop. The failpoint vocabulary therefore belongs here, not in
   `faults.py`, whose classes name component failures.

Deliberately NOT reusing `RecoveryCase`: its `repeat` expansion, its
`inject_at_call_indices` and its injection-proof vocabulary are all specific to
component faults, and widening it would give one type two unrelated meanings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from longline.eval.failpoints import AFTER_TOOL, ALL_FAILPOINTS, BEFORE_TOOL, GATED_FAILPOINTS
from longline.eval.types import CaseParseError, E2ECase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# The gated-tool failpoints need a tool name; the others must not carry one.
TOOL_NAMED_FAILPOINTS: tuple[str, ...] = (BEFORE_TOOL, AFTER_TOOL)

# Sentinel the task must carry where the real working directory goes. The
# directory is a fresh temp dir known only at run time, so the case file holds
# a placeholder and the loader checks it is present -- a task without it is
# unanswerable offline and would fail for a reason unrelated to the failpoint.
CWD_PLACEHOLDER = "<cwd>"


@dataclass
class LoopResumeCase(E2ECase):
    """One failpoint, run `repeat` times.

    Inherits `E2ECase` so the deterministic judge layer is literally the same
    code path as every other suite -- a second judge vocabulary would make the
    numbers look alike while measuring different things.
    """

    failpoint: str = ""
    failpoint_tool: str = ""
    repeat: int = 1

    def run_id(self, index: int) -> str:
        """The id of repeat `index`. The index IS the seed, so nothing else is stored."""
        return f"{self.id}#{index}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoopResumeCase:
        base = E2ECase.from_dict(d)

        failpoint = d.get("failpoint")
        if not isinstance(failpoint, str) or not failpoint:
            raise CaseParseError(f"{base.id}: loop_resume case requires a string 'failpoint'")
        if failpoint not in ALL_FAILPOINTS:
            raise CaseParseError(
                f"{base.id}: unknown failpoint {failpoint!r} (known: {sorted(ALL_FAILPOINTS)})"
            )

        tool = str(d.get("failpoint_tool", "") or "")
        if failpoint in TOOL_NAMED_FAILPOINTS and not tool:
            raise CaseParseError(
                f"{base.id}: failpoint {failpoint!r} requires 'failpoint_tool'; without a "
                "tool name the gate can never fire and the case would report a failed "
                "recovery for a reason unrelated to recovery"
            )
        if failpoint in GATED_FAILPOINTS and failpoint not in TOOL_NAMED_FAILPOINTS and tool:
            raise CaseParseError(
                f"{base.id}: failpoint {failpoint!r} is not tool-named but carries "
                f"failpoint_tool={tool!r}"
            )

        if CWD_PLACEHOLDER not in base.task:
            raise CaseParseError(
                f"{base.id}: the task must name the working directory as {CWD_PLACEHOLDER}; "
                "the real path is a fresh temp dir known only at run time"
            )

        repeat = d.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise CaseParseError(f"{base.id}: repeat must be an int >= 1, got {repeat!r}")

        return cls(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=list(base.tags),
            fixture=base.fixture,
            checks=[dict(c) for c in base.checks],
            checks_mode=base.checks_mode,
            judge=dict(base.judge),
            failpoint=failpoint,
            failpoint_tool=tool,
            repeat=repeat,
        )


def expand_case(case: LoopResumeCase) -> list[LoopResumeCase]:
    """One case object per declared run, with unique ids.

    The copies share their judge and failpoint config by value; only the id
    differs. `dataclasses.replace` is not used because the sub-case's id must
    stay derivable -- the id IS the repeat index, so nothing else is stored.
    """
    return [
        LoopResumeCase(
            id=case.run_id(i),
            task=case.task,
            max_turns=case.max_turns,
            tags=list(case.tags),
            fixture=case.fixture,
            checks=[dict(c) for c in case.checks],
            checks_mode=case.checks_mode,
            judge=dict(case.judge),
            failpoint=case.failpoint,
            failpoint_tool=case.failpoint_tool,
            repeat=1,
        )
        for i in range(case.repeat)
    ]


def load_loop_resume_cases(
    path: Path,
    *,
    fixtures_root: Path | None = None,
) -> list[LoopResumeCase]:
    """Load and expand every case line.

    Expansion happens here rather than in the runner so the denominator a
    report divides by is fixed by the dataset, not by how many iterations a
    loop happened to run.
    """
    from longline.eval.types import validate_fixtures

    cases: list[LoopResumeCase] = []
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
        if d.get("type") != "loop_resume":
            raise CaseParseError(f"{path}:{lineno}: unknown case type {d.get('type')!r}")
        cases.extend(expand_case(LoopResumeCase.from_dict(d)))

    root = path.parent / "fixtures" if fixtures_root is None else fixtures_root
    validate_fixtures([c for c in cases if c.fixture], root)
    return cases


def cases_by_failpoint(
    cases: Iterable[LoopResumeCase],
) -> dict[str, list[LoopResumeCase]]:
    """Group runs by failpoint, with a key for every class even when empty.

    A missing key reads as "not applicable"; a present key with an empty list
    reads as "not measured". The report needs the second.
    """
    grouped: dict[str, list[LoopResumeCase]] = {fp: [] for fp in ALL_FAILPOINTS}
    for case in cases:
        grouped.setdefault(case.failpoint, []).append(case)
    return grouped


__all__ = [
    "CWD_PLACEHOLDER",
    "TOOL_NAMED_FAILPOINTS",
    "LoopResumeCase",
    "cases_by_failpoint",
    "expand_case",
    "load_loop_resume_cases",
]
