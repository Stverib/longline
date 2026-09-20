"""Collaboration-reliability measurements: the infrastructure, not the agent.

=== What this suite is for ===

The paired-benefit suite asks whether a fan-out is worth its cost. This one asks
a different question: whether the machinery under it holds. No model runs here,
no API key is needed, and nothing is spent -- which is why the checks can be
exhaustive where the benefit suite has to be economical.

Six numbers, and the ones that are EXPECTED to look bad are as important as the
ones expected to look good:

```text
MessageLossRate          messages lost / expected        expected 0
DuplicateMessageRate     messages duplicated / expected  expected 0
InboxDurabilityLossRate  lost after corruption / before   expected ALL
OrphanTaskRate           finished but never consumed      measured
CrossWorktreeLeakRate    writes outside their worktree    expected ALL
conflict handling        injected / detected / silent overwrite / integrated
```

The two "expected ALL" figures are the point of the suite. They are the
measurements that turn an architecture claim ("Git worktree isolation") into a
finding ("isolation is not in effect"), and writing the expectation down in
advance is what stops any result from being explained away afterwards.

=== Accounting is by identity, never by count ===

Every mailbox check compares id SETS. A count cannot tell one lost message plus
one duplicated message from nothing at all -- the two cancel -- and those two
are the only failure modes the mailbox test exists to catch.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from longline.swarm.mailbox import InboxCorruptError, TeammateMailbox, TeammateMessage

if TYPE_CHECKING:
    from longline.eval.collab_cases import CollabCase, DurabilityCase


def _message(text: str, *, from_name: str) -> TeammateMessage:
    """A message whose only distinguishing field is its text.

    `TeammateMessage` carries no id, and giving it one to make this suite tidier
    would change the production message format to suit an eval. The text is what
    the accounting keys on instead.
    """
    return TeammateMessage(from_name=from_name, text=text, timestamp=0.0)


@dataclass
class MailboxIntegrityResult:
    """What a fan-out into one inbox delivered, by identity."""

    expected: int
    senders: int = 0
    messages_per_sender: int = 0
    sent_ids: list[str] = field(default_factory=list)
    received_ids: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    peak_concurrent_sends: int = 0

    @property
    def received(self) -> int:
        return len(self.received_ids)

    @property
    def message_loss_rate(self) -> float | None:
        """Lost / expected, or None when nothing was sent.

        None rather than 0.0: no messages means no rate was measured, while 0.0
        claims a rate that was measured and came out clean. The report prints
        the two differently, and only one of them is a fact about the runtime.
        """
        if self.expected == 0:
            return None
        return len(self.lost) / self.expected

    @property
    def duplicate_message_rate(self) -> float | None:
        if self.expected == 0:
            return None
        return len(self.duplicated) / self.expected


def run_mailbox_integrity(case: CollabCase) -> MailboxIntegrityResult:
    """Fan `senders` into one inbox and account for every message by id.

    Senders run as `asyncio` tasks on ONE loop, which is what the production
    fan-out does (`spawn.py` uses `asyncio.create_task`). Running them as threads
    would test a topology no part of the runtime uses and would "find" a race
    that cannot occur.

    Each send yields to the loop, and that is load-bearing rather than
    decorative: `TeammateMailbox.send` never awaits, so without a yield here the
    first coroutine would run to completion before the second started, and "no
    messages lost" would be a statement about a test that never had two writers.
    `peak_concurrent_sends` records that the overlap really happened.
    """
    mailbox = TeammateMailbox(case.team_name, claude_dir=case.claude_dir)
    sent: list[str] = []
    in_flight = 0
    peak = 0

    async def _send(sender_index: int) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            for message_index in range(case.messages_per_sender):
                message_id = f"s{sender_index}-m{message_index}"
                sent.append(message_id)
                mailbox.send(
                    case.receiver,
                    _message(message_id, from_name=f"sender{sender_index}"),
                )
                await asyncio.sleep(0)
        finally:
            in_flight -= 1

    async def _drive() -> None:
        await asyncio.gather(*(_send(i) for i in range(case.senders)))

    asyncio.run(_drive())

    received = [m.text for m in mailbox.receive_all(case.receiver)]
    seen = Counter(received)
    return MailboxIntegrityResult(
        expected=case.expected,
        senders=case.senders,
        messages_per_sender=case.messages_per_sender,
        sent_ids=sent,
        received_ids=received,
        lost=sorted(set(sent) - set(received)),
        duplicated=sorted(mid for mid, count in seen.items() if count > 1),
        peak_concurrent_sends=peak,
    )


@dataclass
class DurabilityResult:
    """What a corrupted inbox cost, and whether the loss was reported."""

    delivered: int
    survived: int
    reported: bool

    @property
    def loss_rate(self) -> float | None:
        if self.delivered == 0:
            return None
        return (self.delivered - self.survived) / self.delivered


def run_inbox_durability(case: DurabilityCase) -> DurabilityResult:
    """Deliver N messages, optionally break the inbox, then read it back.

    `reported` is the load-bearing field. A run that loses messages AND says so
    is a system with a durability limit; a run that loses messages and reports
    an empty inbox cannot tell the difference, and nothing downstream can react
    to it. Before `_read_inbox` was fixed, `reported` was always False and
    `loss_rate` always looked like 0.0 -- the inbox simply read as empty.

    The corruption is CONSTRUCTED rather than produced by killing a process. A
    timing-dependent kill makes an offline suite flaky, and the bytes are the
    same either way; the half-write itself is held to by the property test in
    `tests/unit/swarm/test_mailbox.py`.
    """
    mailbox = TeammateMailbox(case.team_name, claude_dir=case.claude_dir)
    for index in range(case.delivered):
        mailbox.send(case.receiver, _message(f"m{index}", from_name="sender0"))

    if case.truncate:
        # `_inbox_path` is private; reaching for it keeps the corruption
        # exactly where production would produce it, rather than at a
        # re-derived path that could silently stop matching.
        path = mailbox._inbox_path(case.receiver)
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 2], encoding="utf-8")

    reported = False
    survived: list[TeammateMessage] = []
    try:
        survived = mailbox.receive_all(case.receiver)
    except InboxCorruptError:
        reported = True

    return DurabilityResult(
        delivered=case.delivered, survived=len(survived), reported=reported,
    )
