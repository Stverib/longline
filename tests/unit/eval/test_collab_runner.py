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

from longline.eval.collab_cases import (
    CollabCase,
    ConflictCase,
    DurabilityCase,
    OrphanCase,
    WorktreeCase,
)
from longline.eval.collab_runner import (
    run_conflict,
    run_inbox_durability,
    run_mailbox_integrity,
    run_orphan_task_rate,
    run_worktree_isolation,
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


class TestWorktreeIsolation:
    """Measured expectation: the writes land OUTSIDE the worktree.

    `AgentTool.execute` creates a worktree and deletes it without ever handing
    the path to the child: `query_loop` has no `cwd` parameter and
    `child_registry` holds the PARENT's tool instances, which resolve relative
    paths against the process cwd. So the child writes into the parent's tree
    and the worktree is deleted empty.

    This test asserts the MEASURED behaviour, not the intended one. It fails the
    day someone threads a cwd through -- and that failure is the signal to flip
    the assertion and rewrite the report line, not evidence the test was wrong.

    Two independent observations carry the finding, and neither depends on
    timing:

    - `main_dirty_after` is the user-visible one: the repository the user owns
      went from clean to holding the child's files.
    - `leftover_worktrees` is the structural one. `cleanup_agent_worktree` KEEPS
      any worktree with uncommitted changes, so a child that had really written
      inside its worktree would have left that worktree behind. Nothing is left
      behind, so the children wrote nothing there.
    """

    def test_a_child_with_worktree_isolation_writes_into_the_parents_tree(
        self, tmp_path: Path
    ) -> None:
        result = run_worktree_isolation(
            WorktreeCase(repo=tmp_path / "repo", agents=3, marker="marker.txt")
        )

        assert result.agents == 3
        assert result.writes == 3, "every child's Write has to land SOMEWHERE"
        assert result.leaked == 3, (
            "expected every write to land outside its worktree; if this now "
            "passes with leaked=0, AgentTool gained a real cwd and the report "
            "line about worktree isolation must be rewritten"
        )
        assert result.in_worktree == 0
        assert result.leftover_worktrees == [], (
            "a worktree with uncommitted changes is kept by cleanup, so a "
            "leftover would mean the isolation worked at least once"
        )
        assert result.main_dirty_before == []
        assert result.main_dirty_after == [
            f"?? marker-{index}.txt" for index in range(3)
        ]

    def test_the_spawn_path_did_not_error(self, tmp_path: Path) -> None:
        """A failed `git worktree add` returns early, so a clean run proves one ran.

        Without this the leak could be explained as "no worktree was ever
        created", which is a different -- and much less interesting -- bug than
        "a worktree was created and never entered".
        """
        result = run_worktree_isolation(
            WorktreeCase(repo=tmp_path / "repo", agents=2, marker="marker.txt")
        )

        assert result.spawn_errors == []

    def test_a_run_with_no_agents_has_no_rate(self, tmp_path: Path) -> None:
        """A zero denominator must not read as a clean zero rate."""
        result = run_worktree_isolation(
            WorktreeCase(repo=tmp_path / "repo", agents=0, marker="marker.txt")
        )

        assert result.writes == 0
        assert result.leak_rate is None


class TestOrphanTaskRate:
    """A task that finished but whose result the leader never took up.

    Two witnesses from different mechanisms, which is the only reason the
    number means anything:

    - the WRITE side is `TaskRegistry`: `spawn_teammate` registers each task and
      its done-callback sets `COMPLETED` (`spawn.py`). That is a fact about the
      runtime, produced by the spawn path rather than by this suite.
    - the READ side is the leader: whether the teammate's reply was delivered to
      its inbox and drained from it.

    A teammate can be COMPLETED and still be an orphan, and that gap is the
    metric. Measuring one side alone gives either "everything finished" or "I
    saw what I saw"; neither can see the gap.

    `drain` is the control, and it is not decoration. A suite that only ever
    drained would report 0.0 forever and could not tell "the delivery chain
    closes" from "this metric is incapable of seeing a gap".
    """

    def test_a_drained_reply_is_not_an_orphan(self, tmp_path: Path) -> None:
        result = run_orphan_task_rate(OrphanCase(teammates=3, claude_dir=tmp_path))

        assert result.spawned == 3
        assert result.completed == 3
        assert result.delivered == 3
        assert result.consumed == 3
        assert result.orphan_rate == 0.0

    def test_an_undrained_reply_is_an_orphan(self, tmp_path: Path) -> None:
        """The control: the same run, minus the leader's read.

        Everything that produces the reply still happens -- the teammates run,
        finish, and post to the inbox. Only the drain is withheld, so anything
        this measures is the gap and nothing else.
        """
        result = run_orphan_task_rate(
            OrphanCase(teammates=3, claude_dir=tmp_path, drain=False)
        )

        assert result.completed == 3
        assert result.delivered == 3, "delivery is not what draining controls"
        assert result.consumed == 0
        assert result.orphan_rate == 1.0

    def test_a_failed_teammate_is_not_counted_as_completed(self, tmp_path: Path) -> None:
        """`completed` is read from the registry, not assumed to be `spawned`.

        Without this the write side could be a constant equal to the fan-out
        size and every orphan rate would be computed against a number that
        never moved. The failing teammate fails in its model factory, which
        `InProcessTeammate` calls OUTSIDE its own try -- so the failure reaches
        the task itself and the done-callback marks it FAILED.
        """
        result = run_orphan_task_rate(
            OrphanCase(teammates=3, claude_dir=tmp_path, failing=("worker1",))
        )

        assert result.spawned == 3
        assert result.completed == 2
        assert result.states["worker1"] == "failed"
        assert result.errors, "a failed teammate has to leave a reason behind"
        assert result.orphan_rate == 0.0, (
            "the denominator is COMPLETED, not SPAWNED: the failed teammate "
            "never produced a reply to orphan"
        )


class TestConflictHandling:
    """Four independent numbers, deliberately not one boolean.

    "A conflict happened" and "a conflict was handled" are different facts, and
    the dangerous outcome is neither: it is two writers both reporting success
    while one of the two edits is simply gone. A single pass/fail collapses all
    three into one bit and loses exactly the distinction the suite exists to
    draw.

    `silent_overwrite` is the headline: the final file holds one writer's text,
    the other writer's tool call returned success, and nothing anywhere reported
    a conflict.

    The two shapes are not two ways of doing the same thing:

    - `Edit` carries a precondition (`old_string`), so the second writer to run
      finds the text already changed and FAILS. The conflict is detected, at the
      cost of a failed task.
    - `Write` is a whole-file overwrite with no precondition, so BOTH succeed
      and the first writer's edit is gone. Nothing errors anywhere. This is the
      shape the case exists for.

    The writers run one after the other rather than interleaving, and that is
    not a simplification: neither tool awaits between reading the file and
    writing it, so on one event loop they could not interleave even if started
    together. What is measured is a LOST UPDATE from a stale read, not a data
    race -- and a data race is unreachable here for the same reason the mailbox
    one is.
    """

    def test_edit_detects_the_conflict(self, tmp_path: Path) -> None:
        result = run_conflict(ConflictCase(workspace=tmp_path, shape="edit"))

        assert result.injected == 2
        assert result.detected == 1
        assert result.silent_overwrite == 0, (
            "the second Edit's old_string no longer matches, so it errors "
            "instead of quietly winning"
        )
        assert result.final_integration_success is True

    def test_write_silently_overwrites(self, tmp_path: Path) -> None:
        result = run_conflict(ConflictCase(workspace=tmp_path, shape="write"))

        assert result.injected == 2
        assert result.detected == 0, "Write has no precondition, so nothing can fail"
        assert result.silent_overwrite == 1
        assert result.final_integration_success is True

    def test_every_writer_records_what_it_believed(self, tmp_path: Path) -> None:
        """The belief and the outcome are recorded separately, per writer.

        `silent_overwrite` is the gap between the two columns: a writer whose
        call returned success whose text is not in the file. Collapsing them
        into one field would make the metric uncomputable from the result.
        """
        result = run_conflict(ConflictCase(workspace=tmp_path, shape="write"))

        assert len(result.writers) == 2
        assert [writer.reported_success for writer in result.writers] == [True, True]
        assert sum(1 for writer in result.writers if writer.survived) == 1

    def test_a_lone_writer_cannot_conflict_with_anyone(self, tmp_path: Path) -> None:
        """The control: the same machinery, no second writer.

        Without it, a `silent_overwrite` of 1 could be a property of the tool
        rather than of the collision.
        """
        result = run_conflict(
            ConflictCase(workspace=tmp_path, shape="write", values=("alpha",))
        )

        assert result.injected == 1
        assert result.detected == 0
        assert result.silent_overwrite == 0
        assert result.writers[0].survived is True

    def test_the_edit_shape_names_which_writer_lost(self, tmp_path: Path) -> None:
        """`error` has to carry the tool's own words, not just a boolean.

        "One writer failed" is not actionable; "old_string not found" says the
        precondition is what caught it, which is the difference between the two
        shapes and the reason `Edit` is the safe one.
        """
        result = run_conflict(ConflictCase(workspace=tmp_path, shape="edit"))

        failed = [writer for writer in result.writers if not writer.reported_success]
        assert len(failed) == 1
        assert "old_string not found" in failed[0].error
        assert failed[0].survived is False
