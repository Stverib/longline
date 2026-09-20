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
import contextlib
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.swarm.mailbox import InboxCorruptError, TeammateMailbox, TeammateMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from longline.eval.collab_cases import CollabCase, DurabilityCase, WorktreeCase


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


# --- worktree isolation ------------------------------------------------------


async def _git(*args: str, cwd: Path) -> str:
    """Run one git command and return its stdout.

    `asyncio.create_subprocess_exec` rather than `subprocess.run`, matching
    `longline/tools/agent/worktree.py`: the module under test drives git
    asynchronously, and a synchronous call here would block the very loop the
    sub-agents are meant to share.
    """
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {stderr.decode().strip()}"
        )
    return stdout.decode()


async def _init_scratch_repo(repo: Path) -> str:
    """Make `repo` a real git repository with one commit; return its HEAD.

    A `WorktreeCase` cannot run against an arbitrary directory: `git worktree
    add` refuses outside a repository, and refuses again on a repository with no
    commit to detach from. Both failures would be reported as `spawn_errors`,
    which reads like a finding about isolation rather than about the fixture. So
    the fixture is made correct here, once, explicitly.

    The identity is set LOCALLY. A machine with no global `user.email` cannot
    commit at all, and "the test needs git configured first" is a fixture
    failure that would be indistinguishable from the leak this case measures.
    """
    repo.mkdir(parents=True, exist_ok=True)
    await _git("init", "-q", cwd=repo)
    await _git("config", "user.email", "eval@longline.invalid", cwd=repo)
    await _git("config", "user.name", "longline eval", cwd=repo)
    (repo / "README.md").write_text(
        "scratch repository for the eval suite\n", encoding="utf-8",
    )
    await _git("add", "-A", cwd=repo)
    await _git("commit", "-qm", "init", cwd=repo)
    return (await _git("rev-parse", "HEAD", cwd=repo)).strip()


def _marker_writer(marker: str) -> Any:
    """A `call_model_factory` whose child issues ONE `Write` at a RELATIVE path.

    Relative on purpose, and that is the whole measurement. `AgentTool` never
    tells the child that a worktree exists -- not in the prompt, not in the
    system prompt, not in the tool schemas -- so nothing available to the child
    could make it name the worktree path. A real child asked to write
    `marker-0.txt` writes `marker-0.txt`, and where that lands is decided by the
    tools, which is exactly the question.

    The turn structure is not decorative: `query_loop` yields `TurnComplete`
    BEFORE it runs the tools, and `AgentTool` breaks out of its `async for` on
    an `end_turn`. A scripted model that announced `end_turn` while also
    emitting a tool call would have its tool call dropped, and the case would
    report "the child wrote nothing" -- a fixture bug wearing the costume of a
    result.
    """
    state = {"wrote": False}

    def factory(model: str | None = None, max_tokens: int = 16384) -> Any:
        _ = model, max_tokens

        async def call_model(**_: Any) -> AsyncIterator[Any]:
            from longline.core.events import ToolUseStart, TurnComplete
            from longline.models.content_blocks import ToolUseBlock
            from longline.models.messages import Usage

            if state["wrote"]:
                yield TurnComplete(stop_reason="end_turn", usage=Usage())
                return
            state["wrote"] = True
            block = ToolUseBlock(
                id="marker-write",
                name="Write",
                input={
                    "file_path": marker,
                    "content": f"written by the child: {marker}\n",
                },
            )
            yield ToolUseStart(
                tool_name=block.name, tool_id=block.id, input=dict(block.input),
            )
            yield TurnComplete(stop_reason="tool_use", usage=Usage())

        return call_model

    return factory


@dataclass
class WorktreeIsolationResult:
    """Where `isolation="worktree"` children actually wrote, and what they left.

    `leaked` is the headline and `leftover_worktrees` is its witness. The two
    are not redundant: `leaked` says the files are in the parent's tree, and
    `leftover_worktrees` says the worktrees were still empty when they were torn
    down. Either alone could be explained away -- a cleanup that ran before the
    scan, a scan that looked in the wrong place -- and together they cannot.
    """

    agents: int
    writes: int
    leaked: int
    in_worktree: int
    main_dirty_before: list[str] = field(default_factory=list)
    main_dirty_after: list[str] = field(default_factory=list)
    leftover_worktrees: list[str] = field(default_factory=list)
    spawn_errors: list[str] = field(default_factory=list)
    markers: dict[str, str] = field(default_factory=dict)

    @property
    def leak_rate(self) -> float | None:
        """Leaked / written, or None when nothing was written.

        None rather than 0.0 for the same reason the mailbox rates return None:
        a run where no child wrote anything measured no leak, while 0.0 claims
        one was measured and came out clean.
        """
        if self.writes == 0:
            return None
        return self.leaked / self.writes


async def _drive_worktree_agents(
    case: WorktreeCase, marker_names: list[str], errors: list[str]
) -> None:
    """Run each agent through the REAL `AgentTool.execute`, recording failures.

    Reaching for `create_agent_worktree` directly would be simpler and would
    measure nothing: the finding is about what `AgentTool.execute` does between
    creating a worktree and deleting it, so that method is what has to run.
    """
    from longline.tools.agent.agent_tool import AgentTool
    from longline.tools.base import ToolRegistry
    from longline.tools.file_write.file_write_tool import FileWriteTool

    registry = ToolRegistry()
    registry.register(FileWriteTool())

    for marker in marker_names:
        tool = AgentTool(
            parent_registry=registry,
            call_model_factory=_marker_writer(marker),
            cwd=str(case.repo),
        )
        result = await tool.execute({
            "prompt": f"Create the file {marker} containing one line of text.",
            "description": f"write {marker}",
            "isolation": "worktree",
        })
        if result.is_error:
            errors.append(str(result.content))


@contextlib.contextmanager
def _chdir(path: Path) -> Any:
    """Run with `path` as the process cwd, restoring it however the body exits.

    The cwd IS the mechanism under test: the tools build `Path(file_path)` with
    no base directory, so a relative path resolves here. Running from somewhere
    else would move the leak out of sight rather than remove it.
    """
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_worktree_isolation(case: WorktreeCase) -> WorktreeIsolationResult:
    """Spawn `agents` children with `isolation="worktree"`; find where they wrote.

    Three observations, because any one alone is weak:

    - `main_dirty_before/after`: the repository's own `git status --porcelain`.
      A child that writes into the parent's tree shows up here, and this is the
      observation that matters to a user -- it is their working tree.
    - `leftover_worktrees`: what `cleanup_agent_worktree` chose to KEEP. It
      keeps any worktree with uncommitted changes, so a child that had really
      written inside its worktree would have left it behind. Finding none is
      positive evidence that the worktrees were empty, and it does not depend on
      catching them before they are deleted.
    - `leaked` / `in_worktree`: which of the marker files were found inside a
      worktree.

    The fixture is built here rather than assumed, so a missing `git` or an
    unconfigured identity surfaces as a fixture error instead of as a leak.
    """
    repo = Path(case.repo)

    async def _measure() -> WorktreeIsolationResult:
        await _init_scratch_repo(repo)
        dirty_before = _lines(await _git("status", "--porcelain", cwd=repo))

        stem = Path(case.marker).stem or "marker"
        suffix = Path(case.marker).suffix
        names = [f"{stem}-{index}{suffix}" for index in range(case.agents)]

        errors: list[str] = []
        with _chdir(repo):
            await _drive_worktree_agents(case, names, errors)

        dirty_after = _lines(await _git("status", "--porcelain", cwd=repo))
        resolved_repo = repo.resolve()
        leftovers = [
            path
            for path in _worktree_paths(
                await _git("worktree", "list", "--porcelain", cwd=repo)
            )
            if Path(path).resolve() != resolved_repo
        ]

        # Matched by NAME rather than by "everything untracked", so the README
        # the fixture commits cannot be mistaken for a child's output.
        wanted = set(names)
        markers: dict[str, str] = {}
        in_worktree = 0
        for root in [repo, *(Path(p) for p in leftovers)]:
            if not root.exists():
                continue
            for found in sorted(root.rglob("*")):
                if found.is_file() and found.name in wanted:
                    markers.setdefault(found.name, str(found))
                    if root != repo:
                        in_worktree += 1

        writes = len(markers)
        return WorktreeIsolationResult(
            agents=case.agents,
            writes=writes,
            leaked=writes - in_worktree,
            in_worktree=in_worktree,
            main_dirty_before=dirty_before,
            main_dirty_after=dirty_after,
            leftover_worktrees=leftovers,
            spawn_errors=errors,
            markers=markers,
        )

    return asyncio.run(_measure())


def _lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()]


def _worktree_paths(porcelain: str) -> list[str]:
    """The `worktree <path>` entries of `git worktree list --porcelain`."""
    prefix = "worktree "
    return [
        line[len(prefix) :].strip()
        for line in porcelain.splitlines()
        if line.startswith(prefix)
    ]
