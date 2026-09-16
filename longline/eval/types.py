"""Case data models and JSONL loader for the agent evaluation suite.

Two case kinds, discriminated by the JSON `type` field:

- ToolCallCase (Layer 1): asserts the agent invokes certain tools (as an ordered
  subsequence) with arguments matching regexes. Measures tool selection and
  argument correctness.
- E2ECase (Layer 2): runs the agent inside a sandbox fixture copy, then applies a
  deterministic judge function. Measures end-to-end task success (pass@1).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable


# Tags that name an E2E reporting category rather than an incidental property
# of the case. This tuple is the single source of the category vocabulary: the
# loader validates against it, `report.category_metrics` is keyed by it, and
# the dataset contract tests count against it. A free-form tag would let a typo
# ("fileops") create a silent 7-case category that still sums to 40.
E2E_CATEGORY_TAGS: tuple[str, ...] = (
    "file-ops",
    "code",
    "retrieval",
    "multi-tool",
    "long-chain",
)


class CaseParseError(ValueError):
    """Raised when a case line is malformed or fails schema validation."""


# Case-kind tags. Defined in production code because the judging and reporting
# layers both need them and tests/ is not an importable target for longline/.
BLIND_TAG = "blind"
INSTRUCTION_FOLLOWING_TAG = "instruction-following"
# A case whose correct action is to call NO tool. BFCL gives roughly a quarter
# of its set to this class (240 Irrelevance + 882 Live Irrelevance); a suite
# where every case demands a call rewards an agent that always calls something,
# which biases ToolSelectionCaseAccuracy upward.
ABSTENTION_TAG = "abstention"


def resolve_fixture(
    fixtures_root: Path,
    name: str,
    *,
    case_id: str = "<unknown>",
) -> Path:
    """Resolve a case's `fixture` name to a directory inside `fixtures_root`.

    Returns the **unresolved** ``fixtures_root / name`` path, because that is
    what gets copied into a sandbox: resolving first would dereference a
    fixture that is itself a symlink, and the sandbox would then be a copy of
    an arbitrary directory somewhere else on disk. Containment is checked on
    the resolved paths, so symlinks and ``..`` are both caught.

    Raises CaseParseError rather than returning None: a fixture that escapes
    the root is a data bug that would silently change what a case measures, and
    it must fail at load time, not at the end of a paid run.
    """
    if not name or name in (os.curdir, os.pardir):
        raise CaseParseError(f"{case_id}: empty or dot fixture name {name!r}")

    candidate = fixtures_root / name
    root = fixtures_root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise CaseParseError(
            f"{case_id}: fixture {name!r} escapes the fixtures root "
            f"({resolved} is not under {root})"
        )
    return candidate


def validate_fixtures(
    cases: Iterable[EvalCase],
    fixtures_root: Path,
) -> None:
    """Check every referenced fixture stays inside `fixtures_root`.

    Split out from the loader so it can run against any case file with any
    fixtures root, and called from `load_cases` so the dataset contract holds
    for every loader, not just the CLI's.
    """
    for case in cases:
        if case.fixture is not None:
            resolve_fixture(fixtures_root, case.fixture, case_id=case.id)


@dataclass
class _CaseBase:
    id: str
    task: str
    max_turns: int = 8
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _CaseBase:
        cid = d.get("id")
        task = d.get("task")
        if not isinstance(cid, str) or not isinstance(task, str):
            raise CaseParseError(f"case requires string 'id' and 'task', got {d!r}")
        return cls(
            id=cid,
            task=task,
            max_turns=int(d.get("max_turns", 8)),
            tags=[str(t) for t in d.get("tags", [])],
        )


@dataclass
class ToolCallCase(_CaseBase):
    """Layer-1 case: expected tool usage, as an ordered list of decision steps.

    accepted_tool_steps: the authoritative expectation. Each inner list is the
        set of tools that would be a *correct* choice at one decision point,
        and the steps must be satisfied IN ORDER by the agent's calls (extra
        calls in between are allowed). ``[["Glob", "Grep"], ["Read"]]`` means
        "first locate the file, by either means, then read it".
    max_extra_calls: how many calls that match no step are tolerated before the
        case is considered to have exceeded its budget. Extra calls never make a
        case fail on their own (they are visible only through ToolCallPrecision
        and this budget), because a single-case boolean cannot express "right
        answer, sloppy route" — the contract wants that surfaced as a rate.
    expect_tools: LEGACY alias for the single-candidate case. When given, it is
        expanded to ``[[t] for t in expect_tools]``. Setting both it and
        accepted_tool_steps is an error rather than a silent precedence rule.
        This is a *derived view* of accepted_tool_steps: it is populated only
        when the legacy field was supplied on input, so no reader can mistake a
        multi-candidate step for a single required tool.
    expect_args: tool name -> {argument name -> regex}. At least one call of
        that tool must match every regex (re.search, case-insensitive).
    fixture: name of a subdirectory under evals/fixtures/ to copy into the
        sandbox as the starting state (optional). Read/search cases target it.
    blind_rationale: one line explaining why the task text does not name or
        hint at the expected tool(s). Required reading for the blind set — the
        leakage check in tests is a heuristic, and this is the human-facing half
        of it (evals/README.md §8.1).
    """

    accepted_tool_steps: list[list[str]] = field(default_factory=list)
    max_extra_calls: int = 0
    expect_tools: list[str] = field(default_factory=list)
    expect_args: dict[str, dict[str, str]] = field(default_factory=dict)
    fixture: str | None = None
    blind_rationale: str | None = None

    def __post_init__(self) -> None:
        """Normalise the legacy `expect_tools` into `accepted_tool_steps`.

        `from_dict` already resolves the two fields, but `ToolCallCase(...)` can
        equally be constructed directly (tests and programmatic callers do), and
        a case built with only `expect_tools` would otherwise carry an EMPTY
        step list — which `judge_steps` reads as "no decision steps expected",
        i.e. every run passes. That silent all-pass is worse than a crash, so
        the normalisation lives here where every construction path reaches it.
        """
        if not self.accepted_tool_steps and self.expect_tools:
            self.accepted_tool_steps = [[t] for t in self.expect_tools]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCallCase:
        base = _CaseBase.from_dict(d)
        steps = cls._parse_steps(d)
        expect_tools = d.get("expect_tools")
        max_extra_calls = d.get("max_extra_calls", 0)
        expect_args = d.get("expect_args", {})
        fixture = d.get("fixture")
        blind_rationale = d.get("blind_rationale")

        if not isinstance(expect_args, dict):
            raise CaseParseError(f"expect_args must be dict, got {d!r}")
        if fixture is not None and not isinstance(fixture, str):
            raise CaseParseError(f"fixture must be str or null, got {fixture!r}")
        if isinstance(max_extra_calls, bool) or not isinstance(max_extra_calls, int):
            raise CaseParseError(f"max_extra_calls must be an int, got {max_extra_calls!r}")
        if max_extra_calls < 0:
            raise CaseParseError(f"max_extra_calls must be >= 0, got {max_extra_calls}")
        if blind_rationale is not None and not isinstance(blind_rationale, str):
            raise CaseParseError(f"blind_rationale must be str or null, got {blind_rationale!r}")

        return ToolCallCase(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            accepted_tool_steps=steps,
            max_extra_calls=max_extra_calls,
            expect_tools=[str(t) for t in expect_tools] if isinstance(expect_tools, list) else [],
            expect_args={str(k): {str(a): str(p) for a, p in v.items()} for k, v in expect_args.items()},
            fixture=fixture,
            blind_rationale=blind_rationale,
        )

    @staticmethod
    def _parse_steps(d: dict[str, Any]) -> list[list[str]]:
        """Resolve `accepted_tool_steps` / the legacy `expect_tools` into steps.

        A case must declare exactly one of the two. Both is contradictory (the
        plan's example and the legacy list can disagree about whether a step
        accepts one tool or several), and neither leaves nothing to judge
        against — either way the ambiguity is a data bug that would silently
        change a reported number, so it is raised at load time.
        """
        steps = d.get("accepted_tool_steps")
        expect_tools = d.get("expect_tools")

        if steps is not None and expect_tools is not None:
            raise CaseParseError(
                "tool_call case must set either accepted_tool_steps or expect_tools, not both: "
                f"got {d!r}"
            )
        if steps is None and expect_tools is None:
            raise CaseParseError(
                f"tool_call case requires accepted_tool_steps (or legacy expect_tools), got {d!r}"
            )

        if steps is None:
            if not isinstance(expect_tools, list) or not all(isinstance(t, str) for t in expect_tools):
                raise CaseParseError(f"tool_call case requires list[str] expect_tools, got {d!r}")
            # Legacy path: every tool is a step with exactly one acceptable candidate.
            return [[str(t)] for t in expect_tools]

        if not isinstance(steps, list) or not all(isinstance(s, list) for s in steps):
            raise CaseParseError(f"accepted_tool_steps must be list[list[str]], got {d!r}")
        parsed: list[list[str]] = []
        for step in steps:
            if not step or not all(isinstance(t, str) for t in step):
                raise CaseParseError(
                    f"accepted_tool_steps steps must be non-empty list[str] (an empty step is "
                    f"unsatisfiable), got {step!r}"
                )
            parsed.append([str(t) for t in step])
        return parsed


@dataclass
class E2ECase(_CaseBase):
    """Layer-2 case: sandbox task with one or more deterministic checks.

    fixture: name of a subdirectory under evals/fixtures/ to copy into the
        sandbox as the starting state (optional).
    checks: the authoritative judge list — a list of ``{"fn": ..., "args": ...}``
        entries. **All of them must pass** for the case to pass (contract §5.1).
        A case that checks only its final artifact can be satisfied by a wrong
        route, so multi-step tasks assert the final artifact *and* the key
        intermediate state.
    checks_mode: "all" (default) or "any". "any" exists for the genuinely
        disjunctive case — an artifact that may legitimately be written in one
        of two file formats, say. It is NOT a loosening knob: an "any" case is
        satisfied by whichever single check hits, so the other checks are
        redundant by construction. Build "all" unless the task really has
        mutually exclusive success shapes.
    judge: LEGACY single-check alias for ``checks``. When given on input it is
        expanded to a one-element ``checks`` list, so every downstream reader
        has exactly one shape to handle. This is the mirror image of
        ToolCallCase.expect_tools: a *derived, input-only* view, never
        back-filled from ``checks``.
    """

    fixture: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    checks_mode: str = "all"
    judge: dict[str, Any] = field(default_factory=dict)

    VALID_CHECKS_MODES = ("all", "any")

    def __post_init__(self) -> None:
        """Normalise the legacy single-judge form into a `checks` list.

        Direct construction (tests, programmatic callers) must behave exactly
        like `from_dict`: a case built with only ``judge`` would otherwise carry
        an EMPTY checks list, which a lenient caller could read as "no
        assertions", i.e. pass everything. The same reasoning as
        ``ToolCallCase.__post_init__``.
        """
        if not self.checks and self.judge:
            self.checks = [dict(self.judge)]
        if self.checks_mode not in self.VALID_CHECKS_MODES:
            raise CaseParseError(
                f"checks_mode must be one of {list(self.VALID_CHECKS_MODES)}, "
                f"got {self.checks_mode!r}"
            )

    @property
    def num_checks(self) -> int:
        return len(self.checks)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> E2ECase:
        """Parse a case line, resolving `judge`/`checks` exactly once.

        Setting both is an error rather than a silent precedence rule — the two
        could name different judges, and "which one did the number come from"
        must not depend on argument order in a dict.
        """
        base = _CaseBase.from_dict(d)
        judge = d.get("judge")
        checks = d.get("checks")
        mode = d.get("checks_mode", "all")

        if judge is not None and checks is not None:
            raise CaseParseError(
                f"e2e case must set either 'checks' or the legacy 'judge', not both: got {d!r}"
            )
        if judge is None and checks is None:
            raise CaseParseError(f"e2e case requires 'checks' (or legacy 'judge'), got {d!r}")
        if checks is None:
            if not isinstance(judge, dict):
                raise CaseParseError(f"e2e case requires dict 'judge', got {d!r}")
            checks = [judge]
        if not isinstance(checks, list) or not checks:
            raise CaseParseError(f"e2e case requires a non-empty list 'checks', got {checks!r}")
        if not isinstance(mode, str):
            raise CaseParseError(f"checks_mode must be a str, got {mode!r}")

        parsed: list[dict[str, Any]] = []
        for entry in checks:
            if not isinstance(entry, dict) or not isinstance(entry.get("fn"), str):
                raise CaseParseError(
                    f"each check must be a dict with a string 'fn', got {entry!r}"
                )
            parsed.append(entry)

        fixture = d.get("fixture")
        if fixture is not None and not isinstance(fixture, str):
            raise CaseParseError(f"fixture must be str or null, got {fixture!r}")

        return E2ECase(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            fixture=fixture,
            checks=parsed,
            checks_mode=mode,
            judge={} if judge is None else judge,
        )

    def category_tag(self) -> str | None:
        """The E2E reporting category this case belongs to, if any.

        Returns None for an ad-hoc case built by a test or a CLI `--case-file`
        that carries no category tag; such a case still counts toward
        TaskSuccessRate (the contract excludes nothing) but is not reported
        under a category heading.
        """
        for tag in E2E_CATEGORY_TAGS:
            if tag in self.tags:
                return tag
        return None


EvalCase = ToolCallCase | E2ECase


def load_cases(path: Path, *, fixtures_root: Path | None = None) -> list[EvalCase]:
    """Load case definitions from a JSONL file, one case per line.

    Raises CaseParseError on the first malformed or unknown line. When
    `fixtures_root` is given, every case's fixture is additionally checked to
    resolve inside it (see `validate_fixtures`).

    `fixtures_root` defaults to ``<case file's directory>/fixtures``, which is
    the layout `evals/` actually uses. Defaulting to None (i.e. "skip the
    check") would mean the containment rule only applied to callers that
    remembered to opt in, and the one caller that matters — the CLI load — is
    exactly the one nobody remembers to update.
    """
    cases: list[EvalCase] = []
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
        ctype: Literal["tool_call", "e2e"] | str | None = d.get("type")
        if ctype == "tool_call":
            cases.append(ToolCallCase.from_dict(d))
        elif ctype == "e2e":
            cases.append(E2ECase.from_dict(d))
        else:
            raise CaseParseError(f"{path}:{lineno}: unknown case type {ctype!r}")

    root = Path(path).parent / "fixtures" if fixtures_root is None else fixtures_root
    validate_fixtures(cases, root)
    return cases
