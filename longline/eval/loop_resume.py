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

from longline.eval.failpoints import (
    AFTER_TOOL,
    ALL_FAILPOINTS,
    BEFORE_TOOL,
    GATED_FAILPOINTS,
    STOPS_IN_INSTRUCTION_TWO,
)
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

# Placeholders the FIXTURE must carry for the repeat seed to reach it. Kept
# here rather than in the runner because they are part of the same contract as
# CWD_PLACEHOLDER: a fixture that lost them would make every repeat byte
# identical, and the suite would silently be ten copies of one run.
SEED_PLACEHOLDER = "<seed>"
SEED_A_PLACEHOLDER = "<seed_a>"
SEED_B_PLACEHOLDER = "<seed_b>"
FIXTURE_SEED_PLACEHOLDERS: tuple[str, ...] = (
    SEED_PLACEHOLDER,
    SEED_A_PLACEHOLDER,
    SEED_B_PLACEHOLDER,
)

# What each placeholder becomes, as a function of the repeat index. The offsets
# are what make the two operands different from each other and from zero.
SEED_VALUES: dict[str, Any] = {
    SEED_PLACEHOLDER: lambda seed: str(seed),
    SEED_A_PLACEHOLDER: lambda seed: str(2 + seed),
    SEED_B_PLACEHOLDER: lambda seed: str(3 + seed),
}

# The placeholders that stand for OPERANDS, as opposed to a label. A zero operand
# can make a fixture's bug accidentally correct -- `add(a, b)` with `a - b` passes
# when `b` is 0 -- which would leave the workspace layer vacuous for that seed.
# `<seed>` itself is 0 for the first repeat and that is fine: it is a header.
SEED_OPERANDS: tuple[str, ...] = (SEED_A_PLACEHOLDER, SEED_B_PLACEHOLDER)

# What the model says once the scripted sequence runs out. The empty string is
# what the sequence has always defaulted to, and it is the right default for any
# task whose deliverable is a FILE rather than an answer. A task whose deliverable
# IS the answer -- a read-only analysis, say -- has to set this explicitly, or its
# transcript ends with the model saying nothing.
DEFAULT_ANSWER = ""


@dataclass(frozen=True)
class Scenario:
    """The scripted agent behaviour for one case, as data.

    Item 6 moved this out of `loop_resume_worker`, and the reason is about
    evidence rather than tidiness. While the tool sequence was a module constant,
    every arm of every case ran the same three steps against the same fixture, so
    "the runtime recovered" could only ever be a statement about one script. A
    scenario per case is what makes a failpoint matrix a matrix.

    `steps[0]` is instruction 1's tool sequence and `steps[i]` belongs to
    `followups[i - 1]`. Instruction 1's TEXT is the case's own `task`, so it is
    not repeated here -- there is no way for the two to disagree.

    `artifacts` is what the side-effect journal digests before and after every
    execution, and `workspace_test` is the fixture's own test suite, which is how
    the workspace layer asks "is this repository still working" rather than "did
    the checks pass".
    """

    followups: tuple[str, ...]
    steps: tuple[tuple[dict[str, Any], ...], ...]
    artifacts: tuple[str, ...]
    workspace_test: dict[str, Any]
    seed_files: tuple[tuple[str, tuple[str, ...]], ...]
    answer: str = DEFAULT_ANSWER

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.followups) + 1:
            raise CaseParseError(
                f"a scenario with {len(self.followups)} follow-up instruction(s) needs "
                f"{len(self.followups) + 1} step lists, got {len(self.steps)}"
            )
        # Two is what the kill/resume choreography implements: the arm stops in
        # instruction 2 and the resume re-supplies instruction 2. A third would be
        # silently ignored rather than rejected, so it is rejected here instead.
        if len(self.followups) > 1:
            raise CaseParseError(
                f"a scenario supports one follow-up instruction, got {len(self.followups)}"
            )

    @property
    def instruction_count(self) -> int:
        """How many user instructions this task is, in total."""
        return len(self.steps)

    def to_spec(self, sandbox: object) -> dict[str, Any]:
        """The scenario as the worker receives it, with the sandbox resolved.

        Resolved here and not in the worker for the same reason `spec['task']` is:
        the working directory is a fresh temp dir known only to the parent, and
        two places doing the substitution would eventually disagree about it.
        """
        root = str(sandbox).replace("\\", "/")
        return {
            "followups": [text.replace(CWD_PLACEHOLDER, root) for text in self.followups],
            "steps": [[dict(step) for step in group] for group in self.steps],
            "artifacts": list(self.artifacts),
            "workspace_test": dict(self.workspace_test),
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, case_id: str) -> Scenario:
        raw_followups = d.get("followups", [])
        if not isinstance(raw_followups, list) or not all(
            isinstance(x, str) and x for x in raw_followups
        ):
            raise CaseParseError(f"{case_id}: scenario 'followups' must be a list of strings")

        raw_steps = d.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise CaseParseError(f"{case_id}: scenario 'steps' must be a non-empty list")
        steps: list[tuple[dict[str, Any], ...]] = []
        for index, group in enumerate(raw_steps):
            if not isinstance(group, list) or not group:
                raise CaseParseError(
                    f"{case_id}: scenario steps[{index}] must be a non-empty list"
                )
            parsed: list[dict[str, Any]] = []
            for step in group:
                if not isinstance(step, dict):
                    raise CaseParseError(f"{case_id}: every step must be an object")
                tool = step.get("tool")
                if not isinstance(tool, str) or not tool:
                    raise CaseParseError(f"{case_id}: every step needs a 'tool' name")
                if not isinstance(step.get("input"), dict):
                    raise CaseParseError(f"{case_id}: step {tool!r} needs an 'input' object")
                parsed.append({"tool": tool, "input": dict(step["input"])})
            steps.append(tuple(parsed))

        raw_artifacts = d.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise CaseParseError(
                f"{case_id}: scenario 'artifacts' must be a non-empty list; with none, "
                "every side-effect metric has a denominator of zero and the case "
                "cannot fail on a duplicated execution"
            )

        workspace_test = d.get("workspace_test")
        if not isinstance(workspace_test, dict) or not workspace_test:
            raise CaseParseError(
                f"{case_id}: scenario 'workspace_test' must be a non-empty object; it is "
                "the fixture's own test suite, which is a different question from the "
                "case's checks"
            )

        raw_seed = d.get("seed")
        if not isinstance(raw_seed, list) or not raw_seed:
            raise CaseParseError(
                f"{case_id}: scenario 'seed' must be a non-empty list of "
                "{path, placeholders} entries. Without it every repeat of this case "
                "starts from a byte-identical fixture and the repeat count is theatre"
            )
        seed_files: list[tuple[str, tuple[str, ...]]] = []
        for entry in raw_seed:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise CaseParseError(f"{case_id}: every seed entry needs a 'path'")
            names = entry.get("placeholders")
            if not isinstance(names, list) or not names:
                raise CaseParseError(
                    f"{case_id}: seed entry {entry['path']!r} declares no placeholders"
                )
            for name in names:
                if name not in FIXTURE_SEED_PLACEHOLDERS:
                    raise CaseParseError(
                        f"{case_id}: unknown seed placeholder {name!r} in {entry['path']!r} "
                        f"(known: {list(FIXTURE_SEED_PLACEHOLDERS)}). A typo here would "
                        "never be substituted, and the repeats would silently be identical"
                    )
            seed_files.append((str(entry["path"]), tuple(str(n) for n in names)))

        return cls(
            followups=tuple(str(x) for x in raw_followups),
            steps=tuple(steps),
            artifacts=tuple(str(x) for x in raw_artifacts),
            workspace_test=dict(workspace_test),
            seed_files=tuple(seed_files),
            answer=str(d.get("answer", DEFAULT_ANSWER)),
        )


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
    seed: int = 0
    scenario: Scenario | None = None

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

        raw_scenario = d.get("scenario")
        if not isinstance(raw_scenario, dict):
            raise CaseParseError(
                f"{base.id}: a loop_resume case requires a 'scenario'. The tool sequence "
                "used to be a module constant, which made every case in the suite the "
                "same task with a different place to die -- a matrix with one row"
            )
        scenario = Scenario.from_dict(raw_scenario, case_id=base.id)

        # The gate can only fire on a tool the task actually calls. A case whose
        # failpoint can never fire reports a failed recovery for a reason that has
        # nothing to do with recovery, which is the defect class this suite exists
        # to catch -- and here it would be baked into the dataset.
        instruction_one_tools = {str(step["tool"]) for step in scenario.steps[0]}
        if failpoint in TOOL_NAMED_FAILPOINTS and tool not in instruction_one_tools:
            raise CaseParseError(
                f"{base.id}: failpoint_tool {tool!r} is not among instruction 1's tools "
                f"({sorted(instruction_one_tools)}); the gate could never fire"
            )
        if failpoint in STOPS_IN_INSTRUCTION_TWO and scenario.instruction_count < 2:
            raise CaseParseError(
                f"{base.id}: failpoint {failpoint!r} stops inside instruction 2, but the "
                f"scenario declares {scenario.instruction_count} instruction(s) -- there "
                "is nothing for it to stop in"
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
            scenario=scenario,
        )


def expand_case(case: LoopResumeCase) -> list[LoopResumeCase]:
    """One case object per declared run, with unique ids AND distinct seeds.

    `seed` is the repeat index, and it is the difference between ten runs and
    one run repeated ten times. `recovery.py`'s `expand_case` copies everything
    but the id, which makes its ten repeats byte-identical -- a fact this suite
    cannot afford to inherit, because the whole point of the failpoint matrix is
    that each run stands on its own evidence.

    What a seed does NOT buy is a distribution. The scenario and the judges are
    identical across repeats, so the OUTCOME stays deterministic; the seed only
    varies the INPUT, which is what makes "60 runs, 0 counterexamples" a
    statement about more than one fixture rather than about one fixture 60
    times. Reported as a rate, these numbers would be a lie.
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
            seed=i,
            scenario=case.scenario,
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
    "DEFAULT_ANSWER",
    "FIXTURE_SEED_PLACEHOLDERS",
    "SEED_A_PLACEHOLDER",
    "SEED_B_PLACEHOLDER",
    "SEED_OPERANDS",
    "SEED_PLACEHOLDER",
    "SEED_VALUES",
    "TOOL_NAMED_FAILPOINTS",
    "LoopResumeCase",
    "Scenario",
    "cases_by_failpoint",
    "expand_case",
    "load_loop_resume_cases",
]
