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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.swarm.mailbox import InboxCorruptError, TeammateMailbox, TeammateMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from longline.eval.collab_cases import (
        CollabCase,
        ConflictCase,
        DurabilityCase,
        OrphanCase,
        WorktreeCase,
    )


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


# --- orphan tasks ------------------------------------------------------------


def _teammate_factory(agent_name: str, *, fails: bool) -> Any:
    """A `call_model_factory` for one teammate: report, or fail on the way in.

    The failing variant raises when the FACTORY is called rather than when the
    model is. That placement is deliberate: `InProcessTeammate` calls the
    factory OUTSIDE its own `try`, so the exception reaches the task itself and
    the spawn path's done-callback marks the record FAILED. Raising from inside
    the model would instead be swallowed -- `_execute_with_query_loop` catches
    it, turns it into an "(Error: ...)" reply, and the teammate still reports
    COMPLETED, which is exactly the ambiguity this case has to avoid.
    """

    def factory(model: str | None = None, max_tokens: int = 16384) -> Any:
        _ = model, max_tokens
        if fails:
            raise RuntimeError(f"scripted failure for {agent_name}")

        async def call_model(**_: Any) -> AsyncIterator[Any]:
            from longline.core.events import TextDelta, TurnComplete
            from longline.models.messages import Usage

            yield TextDelta(text=f"{agent_name} finished its share")
            yield TurnComplete(stop_reason="end_turn", usage=Usage())

        return call_model

    return factory


@dataclass
class OrphanResult:
    """How many teammates finished, and how many the leader actually took up.

    `spawned`, `completed` and `delivered` are kept separately because each
    collapse hides a different failure. `spawned - completed` is work that never
    finished; `completed - delivered` is a reply the runtime never posted;
    `delivered - consumed` is the orphan -- a finished teammate whose result is
    sitting unread, which is the one nobody notices, because every component
    reported success.
    """

    spawned: int
    completed: int
    delivered: int
    consumed: int
    drain: bool = True
    states: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def orphan_rate(self) -> float | None:
        """(completed - consumed) / completed, or None when none completed.

        The denominator is COMPLETED rather than SPAWNED. A teammate that never
        produced a reply cannot have an orphaned one, and counting it would make
        a crash look like an orphan -- two different faults with two different
        fixes, added together into one number that names neither.
        """
        if self.completed == 0:
            return None
        return (self.completed - self.consumed) / self.completed


def run_orphan_task_rate(case: OrphanCase) -> OrphanResult:
    """Spawn `teammates`, optionally drain the leader's inbox, and count the gap.

    The teammates are real: `spawn_teammate` builds the identity, registers the
    record, starts `InProcessTeammate` as an `asyncio` task on this loop, and
    the runner's own done-callback moves the record to a terminal state.

    The tool registry handed to `spawn_teammate` is EMPTY, which is a choice
    rather than an oversight. This case measures the coordination channel -- did
    a finished teammate's result reach the leader -- and giving the teammates
    tools would add side effects that the accounting then has to see past. The
    reply is text because the channel carries text.

    The inbox is read with the mailbox's own `receive` + `mark_all_read`, so
    "consumed" means the same thing here as it does in production: the leader
    looked, and the message stopped being unread.
    """
    from longline.session.task_registry import TaskRegistry, TaskState
    from longline.swarm.identity import TEAM_LEAD_NAME, format_agent_id
    from longline.swarm.spawn import spawn_teammate
    from longline.swarm.team_file import TeamFile, save_team_file

    names = [f"worker{index}" for index in range(case.teammates)]

    async def _drive() -> OrphanResult:
        from longline.tools.base import ToolRegistry

        registry = TaskRegistry()
        parent_tools = ToolRegistry()
        mailbox = TeammateMailbox(case.team_name, claude_dir=case.claude_dir)

        # `spawn_teammate`'s `add_member` is best-effort and only warns when the
        # team is missing, so a fan-out against a nonexistent team would run to
        # completion while silently skipping one of its three registration
        # steps. Creating the team costs one call and leaves nothing about the
        # spawn path unexercised -- and no warnings for a reader to explain away.
        save_team_file(
            TeamFile(
                name=case.team_name,
                description="scratch team for the collaboration-reliability suite",
                created_at=0.0,
                lead_agent_id=format_agent_id(TEAM_LEAD_NAME, case.team_name),
            ),
            case.claude_dir,
        )

        tasks: list[asyncio.Task[str]] = []
        for name in names:
            task_id = await spawn_teammate(
                case.team_name,
                name,
                f"Report your share of the work, {name}.",
                _teammate_factory(name, fails=name in case.failing),
                parent_tools,
                claude_dir=case.claude_dir,
                task_registry=registry,
            )
            # Captured now, not later: the done-callback POPS the task out of
            # `_running_tasks`, so looking the task up after it finishes finds
            # nothing and would silently await an empty list.
            tasks.append(_running_task(task_id))

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        # The done-callbacks that write the terminal state are scheduled with
        # `call_soon`, so they run after `gather` returns, not during it. This
        # yields until they have all landed rather than sleeping a fixed time:
        # bounded, and if the bound is ever hit the records are visibly still
        # non-terminal instead of a number quietly reading as zero.
        wanted = set(names)
        records = [
            record
            for record in registry.list_all()
            if record.metadata.get("agent_name") in wanted
        ]
        for _ in range(100):
            if all(record.is_terminal for record in records):
                break
            await asyncio.sleep(0)

        states = {
            str(record.metadata.get("agent_name")): record.state.value
            for record in records
        }
        # The reason comes from the TASK, not from the record: the registry
        # stores a state and nothing else, and "worker1: failed" is a restatement
        # of `states` rather than something a reader can act on.
        errors = [
            f"{name}: {type(outcome).__name__}: {outcome}"
            for name, outcome in zip(names, outcomes, strict=True)
            if isinstance(outcome, BaseException)
        ]

        delivered = [
            message
            for message in mailbox.receive_all(TEAM_LEAD_NAME)
            if message.from_name in wanted
        ]
        consumed = 0
        if case.drain:
            consumed = len(mailbox.receive(TEAM_LEAD_NAME))
            mailbox.mark_all_read(TEAM_LEAD_NAME)

        return OrphanResult(
            spawned=case.teammates,
            completed=sum(1 for s in states.values() if s == TaskState.COMPLETED.value),
            delivered=len(delivered),
            consumed=consumed,
            drain=case.drain,
            states=states,
            errors=errors,
        )

    return asyncio.run(_drive())


def _running_task(task_id: str) -> asyncio.Task[str]:
    from longline.swarm.spawn import get_running_tasks

    task = get_running_tasks().get(task_id)
    if task is None:
        raise RuntimeError(f"spawn returned {task_id} but registered no task for it")
    return task


# --- conflicting writers -----------------------------------------------------

_CONFLICT_HEADER = '"""Shared module two writers both want to change."""\n\n'
_CONFLICT_FOOTER = "\n\ndef read() -> str:\n    return VALUE\n"


def _conflict_body(value: str) -> str:
    """The whole-file content a `Write`-shaped writer installs."""
    return f'{_CONFLICT_HEADER}VALUE = "{value}"{_CONFLICT_FOOTER}'


@dataclass
class WriterOutcome:
    """One writer's belief and one writer's fate, kept apart on purpose.

    `reported_success` is what the tool call returned, which is the whole of
    what the writer learns -- neither tool reads the file back afterwards, and a
    teammate would not either. `survived` is whether that writer's text is the
    text that ended up in the file. `silent_overwrite` is exactly the writers
    where the first is true and the second is not, so collapsing these two into
    one field would make the headline number uncomputable from the result.
    """

    name: str
    value: str
    reported_success: bool
    survived: bool
    error: str = ""


@dataclass
class ConflictResult:
    """Four numbers, because "happened" and "handled" are different facts.

    `injected` is a fact about the CASE, not a measurement: it says how many
    writers were aimed at the same anchor, which is what the other three are
    read against. The three that ARE measurements:

    - `detected`: writers whose call returned an error, so somebody found out.
    - `silent_overwrite`: writers whose call returned SUCCESS whose text is not
      in the file. Nobody found out, and the writer has no way to.
    - `final_integration_success`: the file holds one writer's complete intended
      content, or it holds something neither writer asked for -- a mixture, a
      half-file, a body composed from a stale read.

    `final_integration_success` is TRUE in both shapes this suite runs, and by
    construction rather than by luck: both tools write through `os.replace`, so
    a half-file cannot survive a crash, and every writer installs a complete
    body. It is reported anyway, and the reason is that "somebody detected the
    conflict" and "the workspace is still coherent" are different claims. A
    reader given only `detected` cannot tell which claim the run supports, and a
    shape that DID produce partial writes -- a writer composing its full body
    from a file another writer has since changed -- would separate them. That
    shape is not implemented here, so this field is a guard rather than a
    finding, and the report says so.
    """

    injected: int
    detected: int
    silent_overwrite: int
    final_integration_success: bool
    shape: str = "edit"
    writers: list[WriterOutcome] = field(default_factory=list)
    final_content: str = ""


def run_conflict(case: ConflictCase) -> ConflictResult:
    """Aim `values` writers at one line through the REAL tools, then look.

    The writers run one after the other, in declared order. That is the honest
    realisation of a fan-out here rather than a simplification: neither
    `FileEditTool.execute` nor `FileWriteTool.execute` awaits between reading
    the file and writing it, so on one event loop two of them could not
    interleave even if they were started together. What is measured is a LOST
    UPDATE from a stale read -- the second writer decides what to write from a
    view of the file the first writer already changed -- and that is the
    failure a real fan-out actually produces.

    Concurrency would not change the answer either way: the `Edit` writer's
    precondition fails on the bytes, not on the timing, and the `Write` writer
    has no precondition at all.
    """
    from longline.tools.file_edit.file_edit_tool import FileEditTool
    from longline.tools.file_write.file_write_tool import FileWriteTool

    workspace = Path(case.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / case.target
    path.write_text(_conflict_body("original"), encoding="utf-8")
    anchor = case.anchor

    async def _run() -> list[WriterOutcome]:
        outcomes: list[WriterOutcome] = []
        for index, value in enumerate(case.values):
            name = f"writer{index}"
            if case.shape == "edit":
                result = await FileEditTool().execute({
                    "file_path": str(path),
                    "old_string": anchor,
                    "new_string": f'VALUE = "{value}"',
                })
            else:
                result = await FileWriteTool().execute({
                    "file_path": str(path),
                    "content": _conflict_body(value),
                })
            outcomes.append(
                WriterOutcome(
                    name=name,
                    value=value,
                    reported_success=not result.is_error,
                    survived=False,
                    error="" if not result.is_error else result.text,
                )
            )
        return outcomes

    written = asyncio.run(_run())
    final_content = path.read_text(encoding="utf-8")
    writers = [
        replace(outcome, survived=f'VALUE = "{outcome.value}"' in final_content)
        for outcome in written
    ]
    return ConflictResult(
        injected=len(case.values),
        detected=sum(1 for writer in writers if not writer.reported_success),
        silent_overwrite=sum(
            1 for writer in writers if writer.reported_success and not writer.survived
        ),
        final_integration_success=final_content
        in {_conflict_body(value) for value in case.values},
        shape=case.shape,
        writers=writers,
        final_content=final_content,
    )
