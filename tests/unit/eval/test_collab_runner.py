"""Unit tests for the collaboration-reliability suite.

These measure the collaboration INFRASTRUCTURE -- the mailbox, the worktree
isolation, the task registry -- rather than the agent. No model runs, no API key
is needed and nothing is spent, which is why this suite can afford to be
exhaustive where the paired-benefit suite has to be economical.

=== Why the mailbox test is a NEGATIVE control ===

`TeammateMailbox`'s docstring warns that concurrent writes to one inbox can lose
messages. That warning is unreachable in this runtime, and the first test here
pins the reasoning rather than hunting for the bug it describes:

- `send()` is synchronous from `_read_inbox` to `_write_inbox`, with no `await`
  between them, so a single event loop cannot interleave two sends;
- every teammate runs as an `asyncio` task on ONE loop (`spawn.py`), and the
  repository contains no `Thread`, `multiprocessing` or `run_in_executor`.

So the read-modify-write is atomic by construction. If this test ever starts
losing messages, the in-process assumption has been broken by something else --
most likely a teammate moved to a thread or a process -- and THAT is what to
investigate.
"""

from __future__ import annotations

from pathlib import Path

from longline.eval.collab_cases import CollabCase, DurabilityCase, WorktreeCase
from longline.eval.collab_runner import (
    run_inbox_durability,
    run_mailbox_integrity,
)


def _case(tmp_path: Path, **overrides: object) -> CollabCase:
    base: dict[str, object] = {
        "senders": 4,
        "messages_per_sender": 10,
        "claude_dir": tmp_path,
    }
    base.update(overrides)
    return CollabCase(**base)  # type: ignore[arg-type]


class TestMailboxIntegrityIsANegativeControl:
    def test_concurrent_sends_from_one_loop_lose_nothing(self, tmp_path: Path) -> None:
        result = run_mailbox_integrity(_case(tmp_path))

        assert result.expected == 40
        assert result.received == 40
        assert result.lost == []
        assert result.duplicated == []

    def test_the_rates_are_zero_and_the_denominator_is_the_expected_count(
        self, tmp_path: Path
    ) -> None:
        result = run_mailbox_integrity(_case(tmp_path))

        assert result.message_loss_rate == 0.0
        assert result.duplicate_message_rate == 0.0
        assert result.expected == result.senders * result.messages_per_sender

    def test_messages_are_accounted_by_identity_not_by_count(
        self, tmp_path: Path
    ) -> None:
        """A count cannot tell one lost plus one duplicated from nothing at all.

        Those are the only two failure modes this test exists to catch, and they
        cancel in a total. The accounting therefore compares id sets, which
        means the ids have to be distinct -- asserted here rather than assumed,
        because a generator that reused an id would make every message look
        duplicated and the suite would report a bug that is its own.
        """
        result = run_mailbox_integrity(_case(tmp_path, senders=3, messages_per_sender=4))

        assert len(set(result.sent_ids)) == result.expected
        assert set(result.sent_ids) == set(result.received_ids)

    def test_senders_actually_run_concurrently(self, tmp_path: Path) -> None:
        """Sequential sends would make the negative control vacuous.

        If the senders were awaited one after another, "no messages lost" would
        be a statement about a test that never had two writers, which is a
        different and much weaker claim than the one the result is read as.
        """
        result = run_mailbox_integrity(_case(tmp_path))

        assert result.peak_concurrent_sends > 1, (
            "the senders never overlapped, so nothing about concurrency was tested"
        )


class TestCorruptInboxIsReportedNotEmptied:
    def test_a_truncated_inbox_costs_messages_visibly(self, tmp_path: Path) -> None:
        """The pre-fix reading was `loss_rate=0.0` with `reported=False`.

        Both numbers are kept so the report can show the before and after rather
        than only the good one. `reported` is the load-bearing field: a run that
        loses messages AND says so is a system with a durability limit, while one
        that loses messages and reports an empty inbox cannot tell the
        difference, and nothing downstream can react to it.
        """
        result = run_inbox_durability(
            DurabilityCase(delivered=8, claude_dir=tmp_path)
        )

        assert result.delivered == 8
        assert result.survived == 0, "the file is unreadable, by construction"
        assert result.reported is True
        assert result.loss_rate == 1.0

    def test_an_intact_inbox_survives_completely(self, tmp_path: Path) -> None:
        """The control for the case above: nothing lost when nothing is broken."""
        result = run_inbox_durability(
            DurabilityCase(delivered=8, claude_dir=tmp_path, truncate=False)
        )

        assert result.survived == 8
        assert result.loss_rate == 0.0
        assert result.reported is False


class TestCasesDeclareTheirOwnShape:
    def test_expected_is_derived_from_the_declaration(self) -> None:
        case = CollabCase(senders=5, messages_per_sender=3, claude_dir=Path("."))

        assert case.expected == 15

    def test_a_case_with_no_messages_has_no_rate(self, tmp_path: Path) -> None:
        """A zero denominator must not be reported as a zero rate.

        0.0 would read as "nothing was lost out of everything", which is a
        claim about a measurement that never happened.
        """
        result = run_mailbox_integrity(
            _case(tmp_path, senders=0, messages_per_sender=0)
        )

        assert result.expected == 0
        assert result.message_loss_rate is None
        assert result.duplicate_message_rate is None


class TestWorktreeCaseShape:
    def test_a_worktree_case_needs_a_repository_and_an_agent_count(self) -> None:
        case = WorktreeCase(repo=Path("."), agents=3, marker="marker.txt")

        assert case.agents == 3
        assert case.marker == "marker.txt"
