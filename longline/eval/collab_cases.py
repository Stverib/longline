"""Case declarations for the collaboration-reliability suite.

Declared in Python rather than a JSONL file, for the same reason
`latency_cases.py` is: a case here is a *scenario* -- a sender count, a
failpoint, a pair of conflicting writers -- and no `EvalCase` shape can carry
one. A JSONL row would need a schema for each scenario kind and would still
have nowhere to put the ones that come later.

The counts are small on purpose. The question is whether loss can happen AT ALL,
and a lost message is a structural failure rather than a rate that needs a large
sample to detect: one loss out of forty is already a broken invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class CollabCase:
    """One mailbox-integrity scenario: N senders fanning into one inbox."""

    senders: int
    messages_per_sender: int
    claude_dir: Path
    team_name: str = "collab"
    receiver: str = "worker1"

    @property
    def expected(self) -> int:
        return self.senders * self.messages_per_sender


@dataclass(frozen=True)
class DurabilityCase:
    """One inbox-durability scenario: deliver, break the file, read it back.

    `truncate` is the control. A durability test that only ever measures the
    broken case cannot tell "the reader reports loss" from "the reader always
    reports loss", and the second would pass while being useless.
    """

    delivered: int
    claude_dir: Path
    truncate: bool = True
    team_name: str = "collab-durability"
    receiver: str = "worker1"


@dataclass(frozen=True)
class WorktreeCase:
    """One isolation scenario: N sub-agents asked to work in their own worktree."""

    repo: Path
    agents: int
    marker: str


@dataclass(frozen=True)
class OrphanCase:
    """One coordination scenario: N teammates fan out, the leader may read back.

    `drain` is the control, for the same reason `DurabilityCase.truncate` is: a
    suite that only ever drained its inbox would report an orphan rate of 0.0
    forever and could not distinguish "the delivery chain closes" from "the
    metric cannot see a gap". Withholding the drain has to move the number, or
    the number is not measuring the gap.

    `failing` names the teammates whose model factory raises. It exists so the
    write side can be shown to be READ rather than assumed: if `completed` were
    really just `teammates`, a failure injected here would not change it.
    """

    teammates: int
    claude_dir: Path
    drain: bool = True
    failing: tuple[str, ...] = ()
    team_name: str = "collab-orphan"
