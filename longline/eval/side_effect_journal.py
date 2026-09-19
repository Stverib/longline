"""A side-effect journal written OUTSIDE the transcript, and the two metrics.

=== The window this closes ===

`evals/README.md` §5.4 states that a kill landing inside a tool's side-effect
window -- the tool ran, its result never reached the transcript -- cannot be
classified from the persisted data, and marks such runs `ambiguous_side_effect`
rather than claiming they are safe. That is correct about the transcript, and
it is the reason this module exists: the transcript is not the only place a
fact can be written down.

The journal is append-only, lives outside the sandbox, and is fsynced before
the process parks for the kill. It therefore survives exactly the event that
erases the tool result, and the window becomes decidable.

=== What this does NOT claim ===

Being able to OBSERVE a duplicated side effect is not the same as having
prevented one. Nothing here gives the runtime exactly-once semantics; a durable
tool journal with idempotency keys would, and that is deliberately out of scope
(see the spec's non-goals). A report built on this module must say "the runtime
duplicated N side effects", never "the runtime has exactly-once".

=== Why the verdict is mechanical rather than a tool taxonomy ===

The tempting design is a table classifying tools as duplicating / idempotent /
fails-on-replay. A table is a knob: whoever wants a better number edits the
table. Instead the journal records a digest of the declared artifact files
before and after each execution, and the duplicate test is:

    the replayed call SUCCEEDED and the artifact state CHANGED AGAIN

That single rule places all three observed tool shapes correctly without
naming any of them:

| tool shape                       | replay result | state    | redundant | duplicated |
|----------------------------------|---------------|----------|-----------|------------|
| `Bash: echo x >> f` (append)     | ok            | changed  | yes       | yes        |
| `Write` identical content        | ok            | same     | yes       | no         |
| `Edit` same old_string           | error         | same     | yes       | no         |
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.eval.metrics import Ratio

# Re-exported so `side_effect_journal.input_fingerprint` keeps working for every
# existing caller. See the wrapper below for why it is not a second copy.
from longline.utils.hashing import input_fingerprint as _input_fingerprint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

KILLED = "killed"
RESUMED = "resumed"

LEGS: tuple[str, ...] = (KILLED, RESUMED)


@dataclass(frozen=True)
class SideEffectEntry:
    """One completed tool execution, as the wrapper observed it."""

    seq: int
    leg: str
    tool: str
    input_fp: str
    outcome: str
    pre_state: dict[str, str]
    post_state: dict[str, str]

    @property
    def changed_state(self) -> bool:
        """Whether this execution changed any declared artifact."""
        return self.pre_state != self.post_state

    def to_row(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "leg": self.leg,
            "tool": self.tool,
            "input_fp": self.input_fp,
            "outcome": self.outcome,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SideEffectEntry:
        return cls(
            seq=int(row["seq"]),
            leg=str(row["leg"]),
            tool=str(row["tool"]),
            input_fp=str(row["input_fp"]),
            outcome=str(row["outcome"]),
            pre_state={str(k): str(v) for k, v in dict(row["pre_state"]).items()},
            post_state={str(k): str(v) for k, v in dict(row["post_state"]).items()},
        )


def input_fingerprint(tool: str, tool_input: Mapping[str, Any]) -> str:
    """Stable fingerprint of a tool request.

    Re-exported, not reimplemented. The runtime records the same fingerprint now
    (`longline/session/tool_journal.py`) and `longline/` may not import from
    `longline/eval/`, so the implementation moved to `longline/utils/hashing.py`.
    A second copy here would be a second answer to "was this the same request",
    compared across a process boundary where a disagreement is invisible.
    """
    return _input_fingerprint(tool, tool_input)


class SideEffectJournal:
    """Append-only JSONL journal, one line per completed tool execution."""

    def __init__(self, path: Path, leg: str) -> None:
        if leg not in LEGS:
            raise ValueError(f"leg must be one of {list(LEGS)}, got {leg!r}")
        self._path = Path(path)
        self._leg = leg
        self._seq = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def leg(self) -> str:
        return self._leg

    @property
    def entries(self) -> list[SideEffectEntry]:
        return read_journal(self._path)

    def record(
        self,
        *,
        tool: str,
        tool_input: Mapping[str, Any],
        outcome: str,
        pre_state: Mapping[str, str],
        post_state: Mapping[str, str],
    ) -> SideEffectEntry:
        """Append one entry and fsync it.

        fsync, not flush: the process parks and is then killed, and neither
        SIGKILL nor TerminateProcess flushes a buffered stream. An entry that
        only reached the buffer would make a duplicated side effect invisible,
        which is the one failure direction this module must not have.
        """
        self._seq += 1
        entry = SideEffectEntry(
            seq=self._seq,
            leg=self._leg,
            tool=tool,
            input_fp=input_fingerprint(tool, tool_input),
            outcome=outcome,
            pre_state=dict(pre_state),
            post_state=dict(post_state),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_row(), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry


def read_journal(path: Path) -> list[SideEffectEntry]:
    """Parse the journal, skipping an unparseable tail.

    Skipping rather than raising on purpose: a truncated last line is a
    possible outcome of a kill mid-append, and refusing to read the journal
    because of it would turn the fault under study into a harness crash. Only
    lines that parse and carry every required key become entries.
    """
    path = Path(path)
    if not path.is_file():
        return []
    out: list[SideEffectEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out.append(SideEffectEntry.from_row(row))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


@dataclass
class SideEffectMetrics:
    """The two side-effect metrics, sharing one denominator.

    `denominator` counts the killed leg's executions that CHANGED an artifact.
    A tool that changed nothing was not a side effect, so it cannot be counted
    as one that got duplicated -- which is why a `Read` never enters the
    denominator.
    """

    denominator: int = 0
    redundant: int = 0
    duplicated: int = 0
    by_tool: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def duplicate_side_effect_rate(self) -> Ratio:
        return Ratio(self.duplicated, self.denominator)

    @property
    def redundant_re_execution_rate(self) -> Ratio:
        return Ratio(self.redundant, self.denominator)

    def to_row(self) -> dict[str, Any]:
        return {
            "denominator": self.denominator,
            "redundant": self.redundant,
            "duplicated": self.duplicated,
            "by_tool": self.by_tool,
        }


def compute_side_effect_metrics(entries: Sequence[SideEffectEntry]) -> SideEffectMetrics:
    """Compare the killed leg's side effects against the resumed leg's replays.

    `redundant` counts replayed requests: a resumed execution whose
    `(tool, input_fp)` the killed leg already ran.

    `duplicated` is the strict subset where the replay also SUCCEEDED and the
    artifact state moved again -- the only case in which a second side effect
    demonstrably happened. `outcome == "ok"` is required in addition to the
    state difference because a replay that errored cannot have applied
    anything, and the state difference is required because a replay that
    rewrote identical content applied nothing new.
    """
    killed = [e for e in entries if e.leg == KILLED]
    resumed = [e for e in entries if e.leg == RESUMED]

    # First occurrence wins: the comparison is against the state as it stood
    # right after the side effect first happened.
    first_by_fp: dict[str, SideEffectEntry] = {}
    for entry in killed:
        if entry.changed_state:
            first_by_fp.setdefault(entry.input_fp, entry)

    denominator = sum(1 for e in killed if e.changed_state)
    redundant = 0
    duplicated = 0
    by_tool: dict[str, dict[str, int]] = {}

    for entry in resumed:
        origin = first_by_fp.get(entry.input_fp)
        if origin is None:
            continue
        redundant += 1
        bucket = by_tool.setdefault(entry.tool, {"redundant": 0, "duplicated": 0})
        bucket["redundant"] += 1
        if entry.outcome == "ok" and entry.post_state != origin.post_state:
            duplicated += 1
            bucket["duplicated"] += 1

    return SideEffectMetrics(
        denominator=denominator,
        redundant=redundant,
        duplicated=duplicated,
        by_tool=by_tool,
    )


__all__ = [
    "KILLED",
    "LEGS",
    "RESUMED",
    "SideEffectEntry",
    "SideEffectJournal",
    "SideEffectMetrics",
    "compute_side_effect_metrics",
    "input_fingerprint",
    "read_journal",
]
