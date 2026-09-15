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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path


class CaseParseError(ValueError):
    """Raised when a case line is malformed or fails schema validation."""


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
    """Layer-1 case: expected tool usage.

    expect_tools: tools that must appear, IN ORDER, as a subsequence of the
        calls the agent made (extra calls in between are allowed).
    expect_args: tool name -> {argument name -> regex}. At least one call of
        that tool must match every regex (re.search, case-insensitive).
    fixture: name of a subdirectory under evals/fixtures/ to copy into the
        sandbox as the starting state (optional). Read/search cases target it.
    """

    expect_tools: list[str] = field(default_factory=list)
    expect_args: dict[str, dict[str, str]] = field(default_factory=dict)
    fixture: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCallCase:
        base = _CaseBase.from_dict(d)
        expect_tools = d.get("expect_tools")
        expect_args = d.get("expect_args", {})
        if not isinstance(expect_tools, list) or not all(isinstance(t, str) for t in expect_tools):
            raise CaseParseError(f"tool_call case requires list[str] expect_tools, got {d!r}")
        if not isinstance(expect_args, dict):
            raise CaseParseError(f"expect_args must be dict, got {d!r}")
        fixture = d.get("fixture")
        if fixture is not None and not isinstance(fixture, str):
            raise CaseParseError(f"fixture must be str or null, got {fixture!r}")
        return ToolCallCase(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            expect_tools=[str(t) for t in expect_tools],
            expect_args={str(k): {str(a): str(p) for a, p in v.items()} for k, v in expect_args.items()},
            fixture=fixture,
        )


@dataclass
class E2ECase(_CaseBase):
    """Layer-2 case: sandbox task with a deterministic judge.

    fixture: name of a subdirectory under evals/fixtures/ to copy into the
        sandbox as the starting state (optional).
    judge: {"fn": <judge name>, "args": {...}}. Judge functions live in
        longline/eval/judges.py and each receives (sandbox: Path, args: dict).
    """

    fixture: str | None = None
    judge: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> E2ECase:
        base = _CaseBase.from_dict(d)
        judge = d.get("judge")
        if not isinstance(judge, dict):
            raise CaseParseError(f"e2e case requires dict 'judge', got {d!r}")
        fixture = d.get("fixture")
        if fixture is not None and not isinstance(fixture, str):
            raise CaseParseError(f"fixture must be str or null, got {fixture!r}")
        return E2ECase(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            fixture=fixture,
            judge=judge,
        )


EvalCase = ToolCallCase | E2ECase


def load_cases(path: Path) -> list[EvalCase]:
    """Load case definitions from a JSONL file, one case per line.

    Raises CaseParseError on the first malformed or unknown line.
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
    return cases
