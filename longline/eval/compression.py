"""Case data models and loader for the compression A/B suite.

Two things live here that `longline/eval/types.py` has no shape for:

1. **A history.** Every other case kind runs against a sandbox and a fresh
   task; a compression case runs against a *conversation*. The history is a
   scripted transcript because the unit tests are offline and deterministic --
   a real model would make "did compaction drop fact 3" a non-reproducible
   question, which is the one thing a retention metric cannot be.
2. **Key facts with a probe apiece.** `KeyInfoRetention` is judged by asking a
   question whose answer IS the fact, never by searching the summary text for
   the fact's wording. See `CompressionCase` for why that distinction is the
   whole metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from longline.eval.types import CaseParseError, E2ECase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# The five fact kinds plan §4.3 names. They are listed rather than left as free
# text because they are not interchangeable: they fail differently under
# compaction, and a case carrying five file paths would measure one kind of loss
# five times while the other four silently drop out of the dataset.
#
#   file-path          a path that must survive verbatim
#   symbol-name        a function / class / module identifier
#   design-decision    a choice that was made, and why
#   error-cause        a diagnosed cause of a bug
#   open-constraint    an unfinished step or a rule that still binds
FACT_KINDS: tuple[str, ...] = (
    "file-path",
    "symbol-name",
    "design-decision",
    "error-cause",
    "open-constraint",
)

FactKind = Literal["file-path", "symbol-name", "design-decision", "error-cause", "open-constraint"]

# Message roles allowed in a scripted history. Deliberately narrow: a
# `system` or `compact_boundary` role in the SEED would mean the case starts
# mid-compaction, and the before/after token counts would no longer bracket the
# compaction the case is supposed to measure.
HISTORY_ROLES: tuple[str, ...] = ("assistant", "user")


@dataclass
class KeyFact:
    """One fact that must survive compaction, plus how its survival is judged.

    A fact is retained only when the agent can still **use** it. The `probe` is
    the operational form of that requirement: a question whose answer IS the
    fact, and which an agent that lost the fact cannot answer.

    This is the needle-in-a-haystack protocol (and its long-context
    descendants, LongBench and infiniteBench): plant a fact, ask a question
    whose answer is that fact, score by match. The construct is standard; what
    is specific here is that the needle is planted in a *conversation* that is
    then compacted, rather than in a document that is then truncated.

    Fields:
        id: stable identifier, unique within a case. The report names the fact
            by this id, so "which fact was lost" is answerable per case.
        kind: one of `FACT_KINDS`.
        statement: the fact itself, as it appears in the history. Prose for a
            human reader; nothing scores against this string.
        probe: the follow-up question asked after compaction. Must be phrased
            as a question -- a question cannot be answered from a summary that
            merely *mentions* the fact.
        check: the deterministic judge for the probe's answer, same shape as
            `E2ECase.checks`. Must read the answer the agent produced, never
            the transcript.
        answer: the expected answer, when the case author knows it. Empty for a
            retrieval fact whose value is only in the repo.
        trap: the most confusable wrong answer -- the value a compacted agent
            reaches for when it half-remembers. Exactly one of `answer`/`trap`
            is set (enforced by the dataset contract tests), so a check cannot
            be written against a value nobody declared.
    """

    id: str
    kind: str
    statement: str
    probe: str
    check: dict[str, Any]
    answer: str = ""
    trap: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, case_id: str) -> KeyFact:
        fact_id = d.get("id")
        statement = d.get("statement")
        probe = d.get("probe")
        check = d.get("check")
        kind = d.get("kind")
        if not isinstance(fact_id, str) or not fact_id:
            raise CaseParseError(f"{case_id}: key_fact requires a string 'id', got {d!r}")
        if not isinstance(statement, str) or not statement:
            raise CaseParseError(f"{case_id}/{fact_id}: requires a string 'statement'")
        # The probe is required rather than optional: a fact with no follow-up
        # question can only be scored by scanning the summary, which is exactly
        # the measurement the contract forbids.
        if not isinstance(probe, str) or not probe.strip():
            raise CaseParseError(
                f"{case_id}/{fact_id}: requires a 'probe' question; without one "
                "the fact cannot be scored by anything but a keyword scan"
            )
        if not isinstance(check, dict) or not isinstance(check.get("fn"), str):
            raise CaseParseError(f"{case_id}/{fact_id}: 'check' must be a dict with a string 'fn'")
        if kind not in FACT_KINDS:
            raise CaseParseError(
                f"{case_id}/{fact_id}: unknown kind {kind!r} (known: {list(FACT_KINDS)})"
            )
        return cls(
            id=fact_id,
            kind=str(kind),
            statement=statement,
            probe=probe,
            check=check,
            answer=str(d.get("answer", "")),
            trap=str(d.get("trap", "")),
        )


@dataclass
class CompressionCase(E2ECase):
    """A long-context case run twice: with and without compaction.

    Subclasses `E2ECase` deliberately. The continuation task is judged by the
    same deterministic `checks` and the same `case_passed` dispatch as every
    other E2E case -- the A/B changes what the agent *sees*, not how its
    artifact is graded. Reusing the judge layer is what keeps
    `PostCompressionSuccessRate` comparable to `TaskSuccessRate` instead of
    being a second, differently-graded quantity.

    Fields beyond `E2ECase`:
        history: the transcript the two variants start from. A list of
            `{"role": "assistant"|"user", "content": str}` dicts. Deliberately
            assistant-first: the list is the middle of a session, and a
            user-first list would make `normalize_messages_for_api` prepend a
            synthetic "Begin." message (see `tests/unit/eval/test_compression_cases.py`).
        key_facts: exactly five, one per kind (contract §5.3, plan §4.3).
        continuation_task: what the agent is asked to do after compaction.
        probe_question: the follow-up asked after the continuation, whose answer
            must draw on the retained facts.
        probe_checks: judges for the probe answer, same shape as `checks`.
        compaction_note: the brief handed to the compaction model.

    The inherited `task` is unused for these cases -- `continuation_task` is
    the real instruction, and keeping `task` as the serialised continuation
    text means an old reader that only looks at `.task` still sees the right
    prompt rather than an empty string.
    """

    history: list[dict[str, str]] = field(default_factory=list)
    key_facts: list[KeyFact] = field(default_factory=list)
    continuation_task: str = ""
    probe_question: str = ""
    probe_checks: list[dict[str, Any]] = field(default_factory=list)
    compaction_note: str = ""

    @property
    def num_turns(self) -> int:
        """Number of messages in the scripted history (plan §4.3 wants 8-12)."""
        return len(self.history)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompressionCase:
        cid = d.get("id")
        if not isinstance(cid, str) or not cid:
            raise CaseParseError(f"compression case requires a string 'id', got {d!r}")

        continuation = d.get("continuation_task")
        if not isinstance(continuation, str) or not continuation.strip():
            raise CaseParseError(f"{cid}: requires a string 'continuation_task'")

        history = cls._parse_history(d.get("history"), case_id=cid)
        facts = cls._parse_facts(d.get("key_facts"), case_id=cid)

        probe_question = d.get("probe_question", "")
        if not isinstance(probe_question, str):
            raise CaseParseError(f"{cid}: 'probe_question' must be a str")
        probe_checks = cls._parse_checks(d.get("probe_checks", []), case_id=cid, field_name="probe_checks")
        compaction_note = d.get("compaction_note", "")
        if not isinstance(compaction_note, str):
            raise CaseParseError(f"{cid}: 'compaction_note' must be a str")

        # `E2ECase` reads top-level `checks`; reuse it rather than restating the
        # validation, so this loader cannot drift from the E2E one.
        base = E2ECase.from_dict({**d, "task": continuation})
        return cls(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            fixture=base.fixture,
            checks=base.checks,
            checks_mode=base.checks_mode,
            judge=base.judge,
            history=history,
            key_facts=facts,
            continuation_task=continuation,
            probe_question=probe_question,
            probe_checks=probe_checks,
            compaction_note=compaction_note,
        )

    @staticmethod
    def _parse_history(raw: object, *, case_id: str) -> list[dict[str, str]]:
        if not isinstance(raw, list) or not raw:
            raise CaseParseError(f"{case_id}: 'history' must be a non-empty list, got {raw!r}")
        parsed: list[dict[str, str]] = []
        for i, msg in enumerate(raw):
            if not isinstance(msg, dict):
                raise CaseParseError(f"{case_id}: history[{i}] must be an object, got {msg!r}")
            role = msg.get("role")
            content = msg.get("content")
            if role not in HISTORY_ROLES:
                raise CaseParseError(
                    f"{case_id}: history[{i}] role {role!r} not in {list(HISTORY_ROLES)}"
                )
            if not isinstance(content, str) or not content:
                raise CaseParseError(f"{case_id}: history[{i}] needs non-empty string content")
            parsed.append({"role": str(role), "content": content})
        return parsed

    @staticmethod
    def _parse_facts(raw: object, *, case_id: str) -> list[KeyFact]:
        if not isinstance(raw, list) or not raw:
            raise CaseParseError(f"{case_id}: 'key_facts' must be a non-empty list, got {raw!r}")
        facts = [KeyFact.from_dict(f, case_id=case_id) for f in raw]
        ids = [f.id for f in facts]
        if len(set(ids)) != len(ids):
            raise CaseParseError(f"{case_id}: duplicate key_fact ids {ids}")
        return facts

    @staticmethod
    def _parse_checks(raw: object, *, case_id: str, field_name: str) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise CaseParseError(f"{case_id}: {field_name} must be a list, got {raw!r}")
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("fn"), str):
                raise CaseParseError(
                    f"{case_id}: each {field_name} entry needs a string 'fn', got {entry!r}"
                )
        return list(raw)


def load_compression_cases(path: Path, *, fixtures_root: Path | None = None) -> list[CompressionCase]:
    """Load compression cases from a JSONL file, one per line.

    `fixtures_root` defaults to ``<case file's directory>/fixtures``, matching
    `load_cases`. The containment check is not optional for the same reason it
    is not optional there: it is the difference between a case reading its own
    fixture and a case reading an arbitrary path off the disk.
    """
    from longline.eval.types import validate_fixtures

    cases: list[CompressionCase] = []
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
        ctype = d.get("type")
        if ctype != "compression":
            raise CaseParseError(f"{path}:{lineno}: unknown case type {ctype!r}")
        cases.append(CompressionCase.from_dict(d))

    root = path.parent / "fixtures" if fixtures_root is None else fixtures_root
    validate_fixtures(cases, root)
    return cases


def resolve_probe_commands(cases: Iterable[CompressionCase]) -> None:
    """Placeholder kept out of the loader; command validation lives in the tests."""
