"""Single-agent vs multi-agent A/B: Speedup and TokenOverhead (contract §5.6).

=== What this measures (evals/README.md §5.6) ===

```text
SuccessRate   = judge-passing cases / total          (both variants, SAME judges)
WallClockTime = measured duration
Speedup       = single_wall_time / multi_wall_time
TokenOverhead = (multi_tokens - single_tokens) / single_tokens
```

Input / output / total tokens and tool calls are reported for both variants,
and the agent count is fixed by the case and recorded in the run metadata.

=== The red line this module is built around ===

> The contract's red line, quoted: every sub-agent's tokens and tool calls
> must be collected -- counting only the leader is not acceptable.

Every number above the line is a *comparison between two runs' costs*, so a
run whose cost omits a child does not produce a noisy number -- it produces a
confident wrong one, always in the direction that flatters the fan-out. The
seam that makes the children observable, and the two witnesses that make the
accounting a checked claim rather than an assumption, live in
`longline/eval/child_usage.py`; this module's job is to drive both variants
through it and refuse to report anything when the ledgers do not reconcile.

=== How the two variants are held to the same work ===

The case declares its independent subtasks (`MultiAgentCase.subtasks`). Both
variants execute **all of them**, in the same sandbox layout, judged by the same
`checks`:

```text
single: one agent, one query loop, every subtask's instruction in that prompt
multi:  N teammates (2-4) each handed exactly one subtask's instruction,
        then the leader exercises the case's declared merge step
```

The multi variant's fan-out is a real `spawn_teammate` -- the production path,
with its own `InProcessTeammate`, its own tool registry and its own
`query_loop` -- rather than a task wrapper that merely looks concurrent. That is
what makes the speedup a statement about the product.

Three things stop "same work" from being a promise:

1. **Same prompt content.** Each subtask's `instruction` is the text handed to
   whichever agent runs it in either variant, so the two variants receive the
   same words. `Subtask.instruction` has one definition and both paths read it.
2. **Same expected artifacts.** `expected_paths()` is derived from the case's
   declarations, not from either variant's output, and both are graded against
   it by `unexpected_paths` + `file_exists`.
3. **Same merge work.** The leader's declared merge step runs in BOTH variants,
   so the serialization cost it represents is present on both sides of the
   ratio instead of being charged only to the fan-out.

=== Offline vs on-model ===

`run_multi_agent_case` with `model=None` runs the **offline protocol**: real
`QueryEngine`, real tools, real `query_loop`, real `spawn_teammate`, real
judges, and a scripted model. Deterministic and free, which is what the test
suite and `evals/multi_agent.jsonl` describe. A real model id runs the same
case against the live API with identical assertions; `offline` on the row says
which one produced it, and neither is trusted more than the other.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.eval.child_usage import (
    LEADER,
    AccountedAgents,
    AccountingError,
    UsageLedger,
    agent_scope,
    count_usage,
    current_ledger,
    drain,
    reconcile,
)
from longline.eval.judges import case_passed
from longline.eval.metrics import Ratio, mean, speedup_block, success_per_1k_tokens
from longline.eval.multi_agent import (
    CATEGORY_ANALYSIS,
    CATEGORY_DEPENDENT,
    CONTROLLED,
    DEPENDENT_WORKERS,
    MAX_WORKERS,
    MIN_WORKERS,
    MULTI,
    SINGLE,
    MultiAgentCase,
    Subtask,
    chain_order,
)
from longline.eval.runner import _prepare_sandbox

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence

# Wall-clock ceiling for one variant. A fan-out that hangs is a failed case with
# a reason, not a suite that never finishes.
VARIANT_TIMEOUT_S = 300

# The `tags` marker on a multi-agent row. These rows come from this module
# rather than `run_case` and carry a different `detail` shape, so a reader
# needs to be able to tell them apart from E2E rows without guessing from ids.
MULTI_AGENT_TAG = "multi-agent"

# Reason recorded when a variant's accounting did not reconcile. The case is
# NOT dropped: it is kept in the data with this reason and excluded from the
# ratio denominators, following the same rule compression uses for a failed
# baseline (a silently shrunk denominator is a wrong number that looks right).
REASON_ACCOUNTING_INCOMPLETE = "accounting_incomplete"

# Reason recorded when a variant raised. Kept apart from the accounting reason
# because the two mean different things: a crashed variant measured nothing,
# an unreconciled one measured something it cannot vouch for.
REASON_VARIANT_ERROR = "variant_error"

# Reason recorded when a variant was cut off by the wall-clock ceiling. Kept
# apart from `variant_error` for the same kind of reason: a timed-out variant
# measured nothing because the work was stopped, while a crashed one measured
# something it cannot vouch for. The two have different fixes -- a ceiling that
# is too low, versus a bug -- so folding them into one reason would leave the
# report unable to say which one to go and look at.
REASON_TIMEOUT = "timeout"


def leader_prompt(case: MultiAgentCase) -> str:
    """The single variant's prompt: the task, then every subtask, then the merge.

    One definition, read by both the single variant's driver and (for the merge
    instruction) the multi variant's leader. A second copy would be free to
    drift, and the drift would land as a difference in work between the two
    arms -- the exact confound the paired design exists to remove.
    """
    lines = [
        case.task,
        "",
        "Complete ALL of the following subtasks, in order:",
    ]
    for index, subtask in enumerate(case.subtasks, start=1):
        lines.append(f"{index}. {subtask.instruction} (write it to {subtask.writes})")
    lines += ["", merge_instruction(case)]
    return "\n".join(lines)


def subtask_prompt(case: MultiAgentCase, subtask: Subtask) -> str:
    """The prompt one worker is spawned with: its own subtask, and nothing else.

    Deliberately NOT the whole task. A worker handed every subtask would do its
    neighbours' work too, and the fan-out's cost and wall time would then be
    measuring duplicated labour rather than parallelism.
    """
    return f"{case.task}\n\nYour subtask: {subtask.instruction}\nWrite it to {subtask.writes}."


def merge_instruction(case: MultiAgentCase) -> str:
    """The leader's declared merge step, as the prompt text that performs it."""
    return (
        f"Then write {case.merge_file} listing, one per line, the paths you wrote: "
        + "\n".join(case.subtask_paths())
    )


def _shared_checks(case: MultiAgentCase) -> list[dict[str, Any]]:
    """The case's judges, verbatim. Shared by both variants by construction.

    Returned as a copy so a caller that mutates the list cannot change what the
    other variant is graded against mid-run -- the same reason `E2ECase.checks`
    is copied rather than handed out by reference.
    """
    return [dict(check) for check in case.checks]


@dataclass
class VariantRun:
    """One variant of one case: its artifact verdict plus its measured cost.

    `duration_ms` is the wall clock of the variant's own work, which is the
    quantity `Speedup` is built from. It is measured here, inside the sandbox's
    lifetime, so the fixture copy and the judging are not silently included in
    one arm and excluded from the other.
    """

    variant: str
    passed: bool
    duration_ms: float
    ledger: UsageLedger
    accounts: dict[str, object]
    subtask_verdicts: dict[str, bool]
    judge_detail: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    offline: bool = True
    # Filled when `accounts["accounting_complete"]` is False, so the reason
    # survives into raw.jsonl rather than only into a raised exception.
    accounting_error: str = ""
    # A fact about the run, not a substring of `errors`. The report has to
    # exclude a timed-out case for a different reason than a crashed one, and
    # matching on the error TEXT would stop working the first time the wording
    # changed -- silently, by putting the case back in the crash bucket.
    timed_out: bool = False
    # What the run actually used. Read from the module constants at
    # construction, not at report time: a reader has to be able to tell that a
    # row was cut off by the ceiling in force THEN, which a later edit to
    # `VARIANT_TIMEOUT_S` would otherwise rewrite retroactively.
    model: str = ""
    temperature: float | None = None
    max_turns: int = 0
    variant_timeout_s: float = VARIANT_TIMEOUT_S
    workers: int = 0

    def exclusion_reason(self) -> str | None:
        """Why this variant leaves the ratio denominators, or None if it stays.

        Ordered by which claim the run invalidates. An unreconciled ledger means
        the COST is untrustworthy, which disqualifies the row regardless of how
        the clock behaved, so it is checked first: reporting a timeout ahead of
        it would hide a broken ledger behind an honest-looking clock reading.

        A plain crash and a timeout are both `errors`, so the timeout flag is
        consulted before falling through to `REASON_VARIANT_ERROR` -- that
        ordering is the whole point of keeping the flag.
        """
        if self.accounting_error:
            return REASON_ACCOUNTING_INCOMPLETE
        if self.timed_out:
            return REASON_TIMEOUT
        if self.errors and not self.passed:
            return REASON_VARIANT_ERROR
        return None

    @property
    def input_tokens(self) -> int:
        return self.ledger.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.ledger.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.ledger.total_tokens

    @property
    def tool_calls(self) -> int:
        return self.ledger.tool_calls

    @property
    def agent_count(self) -> int:
        return len(self.ledger.agents)

    @property
    def usable(self) -> bool:
        """False when this variant's numbers must not enter a headline metric."""
        return not self.errors and not self.accounting_error

    def to_row(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "agent_count": self.agent_count,
            "child_tokens": self.ledger.child_tokens(),
            "subtask_verdicts": self.subtask_verdicts,
            "judge_detail": self.judge_detail,
            "errors": self.errors,
            "accounting_error": self.accounting_error,
            "offline": self.offline,
            "timed_out": self.timed_out,
            "model": self.model,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "variant_timeout_s": self.variant_timeout_s,
            "workers": self.workers,
            "usage": self.ledger.to_dict(),
            "accounts": self.accounts,
        }


@dataclass
class MultiAgentRun:
    """One case, both variants, plus the paired deltas between them."""

    case_id: str
    group: str
    workers: int
    num_subtasks: int
    single: VariantRun
    multi: VariantRun
    expected_paths: list[str]
    excluded_from_denominator: bool = False
    exclusion_reason: str | None = None
    note: str = ""
    # The task SHAPE, orthogonal to `group`. `group` says whether the case was
    # controlled or exploratory (a property of how the case was authored);
    # `category` says what kind of work it is (a property of the task). The
    # report splits on this one, because a mean over three different kinds of
    # task answers a question about the corpus mix rather than the architecture.
    category: str = CATEGORY_ANALYSIS
    # Which repetition of this case it is, 0-based. The repeat axis is separate
    # from `variant`: the two arms are two different executions of the SAME
    # task, not two samples of one configuration, and anything that groups runs
    # by case id alone (the report's stability section does) will otherwise read
    # "the two arms disagreed" as "the case is flaky".
    repeat_index: int = 0

    @property
    def speedup(self) -> float | None:
        """`single_wall_time / multi_wall_time`, or None when unmeasurable.

        None rather than a number whenever either side was not measured. A zero
        `multi` duration is a division by zero, and a zero `single` duration is
        just as unmeasurable in the other direction: `0.0 / 50.0` is `0.0`,
        which would be read as "the fan-out ran at zero speed" -- a real, and
        alarming, claim about a run that measured nothing at all.
        """
        if self.single.duration_ms <= 0 or self.multi.duration_ms <= 0:
            return None
        return self.single.duration_ms / self.multi.duration_ms

    @property
    def token_overhead(self) -> float | None:
        """`(multi - single) / single`, or None when the single side is zero."""
        if self.single.total_tokens <= 0:
            return None
        return (self.multi.total_tokens - self.single.total_tokens) / self.single.total_tokens

    def to_row(self) -> dict[str, object]:
        """The per-case row for `raw.jsonl`; the summary is recomputed from it."""
        return {
            "case_id": self.case_id,
            "group": self.group,
            "category": self.category,
            "repeat_index": self.repeat_index,
            "workers": self.workers,
            "num_subtasks": self.num_subtasks,
            "expected_paths": self.expected_paths,
            "single_passed": self.single.passed,
            "multi_passed": self.multi.passed,
            "excluded_from_denominator": self.excluded_from_denominator,
            "exclusion_reason": self.exclusion_reason,
            "variant_units": "ms",
            "speedup_units": "ratio_of_durations",
            "single": self.single.to_row(),
            "multi": self.multi.to_row(),
            "note": self.note,
        }


@dataclass
class MultiAgentSummary:
    """The contract's six reportable quantities plus their per-variant split.

    `single_success_rate` and `multi_success_rate` share one eligible set, so a
    comparison between them is a paired one. The cost metrics are ratios of
    measured quantities, **not** proportions: `Speedup` and `TokenOverhead` are
    unitless and get no Wilson interval, because there is no binomial trial
    behind them. Only the two success rates are `Ratio`s.

    `num_cases` counts distinct TASKS and `num_runs` counts case-runs, so a
    6-task 3-repeat sweep reads as 6 tasks / 18 runs rather than 18 cases. The
    two are equal whenever `repeats == 1`, which is every suite but this one, so
    the pair costs nothing to carry and is what keeps a repeated sweep from
    publishing a denominator it did not have.
    """

    group: str
    num_cases: int
    eligible_cases: int
    excluded_cases: int
    single_success_rate: Ratio
    multi_success_rate: Ratio
    single_wall_time_ms: float | None
    multi_wall_time_ms: float | None
    mean_speedup: float | None
    mean_token_overhead: float | None
    single_tokens: dict[str, int]
    multi_tokens: dict[str, int]
    single_tool_calls: int
    multi_tool_calls: int
    # How many case-runs the figures above were computed over. Equals
    # `num_cases` when nothing was repeated; larger when it was. Mandatory, not
    # defaulted: a default of 0 would let a caller that forgot it publish a
    # count of zero next to a correct `num_cases`, which is the silent-wrong-
    # number shape this suite keeps producing.
    num_runs: int
    eligible_runs: int
    agent_counts: list[int] = field(default_factory=list)
    per_case: list[dict[str, object]] = field(default_factory=list)
    # One self-contained block per CATEGORY present, each with its own success
    # rates, speedup pair, token overhead and cost-per-success. This is what the
    # report renders; the scalars above are the group's own figures and stay for
    # the single-category suites (the frozen `multi_agent` corpus is one shape).
    by_category: dict[str, dict[str, object]] = field(default_factory=dict)
    # The cross-category figures, or None when `is_pooled`. Present as a FIELD
    # rather than recomputed in `to_dict` because recomputing it from the rows
    # would be a second implementation of every metric, and the rows do not
    # carry `speedup` at all -- the sum would silently come out empty.
    pooled: dict[str, object] | None = None
    speedup_units: str = "ratio_of_durations"
    overhead_units: str = "ratio"

    @property
    def is_pooled(self) -> bool:
        """True when this summary covers more than one CATEGORY of task.

        The report withholds the headline when this is true. A mean over
        `parallel_analysis` and `strongly_dependent` cases together is not a
        weaker version of either -- it is a statement about the corpus mix, and
        it moves the moment the 6/6/6 split is rebalanced, with nothing about
        the architecture having changed.
        """
        return len(self.by_category) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "num_cases": self.num_cases,
            "eligible_cases": self.eligible_cases,
            "excluded_cases": self.excluded_cases,
            "num_runs": self.num_runs,
            "eligible_runs": self.eligible_runs,
            "success_rate": {
                "single_agent": self.single_success_rate.to_dict(),
                "multi_agent": self.multi_success_rate.to_dict(),
            },
            "wall_clock_ms": {
                "single_agent": self.single_wall_time_ms,
                "multi_agent": self.multi_wall_time_ms,
            },
            "speedup": self.mean_speedup,
            "speedup_units": self.speedup_units,
            "token_overhead": self.mean_token_overhead,
            "token_overhead_units": self.overhead_units,
            "tokens": {
                "single_agent": dict(self.single_tokens),
                "multi_agent": dict(self.multi_tokens),
            },
            "tool_calls": {
                "single_agent": self.single_tool_calls,
                "multi_agent": self.multi_tool_calls,
            },
            "agent_counts": self.agent_counts,
            "per_case": self.per_case,
            "by_category": self.by_category,
            # `None` when pooled, and that is a statement rather than a gap: a
            # consumer reading the raw artifact must not be able to pick up a
            # cross-category number the report deliberately refused to print.
            "pooled": self.pooled,
        }


# --- the scripted model ------------------------------------------------------


def scripted_factory(
    *,
    subtasks: Sequence[Subtask],
    merge_file: str,
    sandbox: Path,
    usage: Any,
) -> Callable[..., Any]:
    """A `call_model_factory` that writes the artifacts through the REAL tools.

    Offline the fan-out still has to *produce the declared files*, or the judge
    would fail both variants and the A/B would measure nothing but the harness.
    Two things make this a fair stand-in rather than a shortcut:

    - The files are written by issuing `Write` tool calls that the production
      `StreamingToolExecutor` then dispatches. So `tool_calls` and the tool
      executions are real, and the per-agent tool-call counts the red line asks
      for are read off a real dispatch path rather than off a bookkeeping
      variable this module maintains for itself. A stub that returned text and
      touched the filesystem behind the tool layer would report zero tool calls
      for both arms -- a number that looks like a measurement.
    - Both variants get the same behaviour, because the work being compared is
      the *execution* of the subtasks, not the authoring of them.

    The scripted model issues one `Write` per missing artifact and then one more
    turn to finish, so a run costs several turns per agent and the per-agent
    token totals are not degenerate.

    `usage` is the `Usage` each turn reports. It is a parameter rather than a
    constant so a test can make a given agent's turn cost a distinguishable
    number of tokens and assert the ledger picked *that* agent's usage up.
    """
    from longline.core.events import ToolUseStart, TurnComplete
    from longline.models.content_blocks import ToolUseBlock
    from longline.models.messages import Usage

    report: Usage = usage if usage is not None else Usage()

    def factory(model: str | None = None, max_tokens: int = 16384) -> Callable[..., AsyncIterator[Any]]:
        _ = model, max_tokens

        async def call_model(**kwargs: Any) -> AsyncIterator[Any]:
            messages = kwargs.get("messages") or []
            wanted = _subtask_from_prompt(messages, subtasks)

            # What still needs doing, decided from the SANDBOX rather than from
            # a turn counter. A counter would have to be reset correctly per
            # agent, and getting that wrong is invisible: a stale counter makes
            # the model re-issue a Write it already did, `query_loop` treats
            # that as a tool call and loops again, and the run silently becomes
            # a function of `max_turns` instead of the task. Measured, that was
            # 12 turns and 15 tool calls for a two-turn job.
            #
            # Asking "which of my artifacts are missing?" makes the model's
            # stopping condition a fact about the workspace it can observe --
            # the same thing a real agent checks.
            targets = [wanted] if wanted is not None else list(subtasks)
            missing = [s for s in targets if not (sandbox / s.writes).exists()]
            blocks = [
                ToolUseBlock(
                    id=f"w-{subtask.id}",
                    name="Write",
                    input={"file_path": str(sandbox / subtask.writes),
                           "content": f"# {subtask.id}\n{subtask.instruction}\n"},
                )
                for subtask in missing
            ]
            # The leader's merge file, once its own artifacts are all present.
            # A worker's prompt names exactly one subtask, so `wanted` is not
            # None for it and it owes nothing beyond that artifact.
            merge_path = sandbox / merge_file
            if wanted is None and not missing and not merge_path.exists():
                blocks.append(
                    ToolUseBlock(
                        id="w-merge",
                        name="Write",
                        input={"file_path": str(merge_path), "content": _merge_body(subtasks)},
                    )
                )
            # No tool block means nothing is left to do, which is what ends the
            # turn: `query_loop` returns on `end_turn` and continues on
            # `tool_use`.
            for block in blocks:
                yield ToolUseStart(tool_name=block.name, tool_id=block.id, input=dict(block.input))
            yield TurnComplete(
                stop_reason="tool_use" if blocks else "end_turn", usage=report,
            )

        return call_model

    return factory


def _merge_body(subtasks: Sequence[Subtask]) -> str:
    """The text the leader's merge step writes.

    An EXPLORATORY case declares no subtasks -- its coordinator decomposes
    freely, which is the point of that group -- so there is no artifact list to
    enumerate and this would otherwise write a single newline. That is not a
    cosmetic problem: `file_exists` passes on an empty file, so the case would
    look like it ran while its content checks failed, and the failure would read
    as the model's rather than as an artefact of the scripted transport having
    nothing to say.

    So the empty case writes a short report body instead. It is still a
    stand-in for the model's own words; what it is NOT is a stand-in that
    satisfies the case's size floor by padding, since the case's checks are what
    decide whether that floor is met.
    """
    if not subtasks:
        return (
            "# Survey\n\n"
            "This repository is a small service split into independent modules.\n"
            "It ships with `python -m src.router --check`.\n"
            "An operator should walk the rollout checklist before a deploy.\n"
        )
    return "\n".join(s.writes for s in subtasks) + "\n"


def _subtask_from_prompt(messages: Sequence[Any], subtasks: Sequence[Subtask]) -> Subtask | None:
    """Which single subtask this agent was told to do, or None for "all of them".

    The prompt is the channel `spawn_teammate` uses, so reading the instruction
    back out of it means the scripted worker does the subtask it was *told* to
    do rather than one the harness picked for it. A model that ignored its
    prompt and did a neighbour's work would otherwise be indistinguishable from
    a correct one -- and the fan-out's outputs are what the judge grades.

    The distinction is not "which instruction appears" but "how many". A worker
    is given exactly one (`subtask_prompt`); the leader is given all of them
    (`leader_prompt`), and the leader still owes the merge file. Keying on the
    first match instead made the leader look like a worker: it wrote four
    artifacts and never wrote the manifest, and `line_set_equals` then failed on
    a run where every worker had done its job.

    `messages` arrives as the NORMALIZED api form (`normalize_messages_for_api`
    runs before `query_loop` calls the model), so the content is a list of
    blocks, not a plain string. Reading `.content` off each message therefore
    yields a list, and a naive `str()` of it happens to contain the text --
    which works by accident in one shape and not another. `_message_text`
    flattens both shapes explicitly, because a scripted worker that cannot find
    its own instruction silently falls through to "write everything", and every
    worker then duplicates the whole task: measured, that produced 30 turns and
    33 tool calls per worker against 2 and 1 for a correct run.
    """
    text = " ".join(_message_text(m) for m in messages)
    matches = [subtask for subtask in subtasks if subtask.instruction in text]
    if len(matches) == 1:
        return matches[0]
    return None


def _message_text(message: Any) -> str:
    """The text of one message, whether it is a `Message` or an api dict.

    Handles both shapes the same way, because `call_model` is handed the
    normalized form while tests and direct callers pass `Message` objects. A
    missing or non-text content contributes an empty string rather than
    raising: this runs inside a scripted model's turn, where an exception would
    surface as a failed case rather than as the harness bug it is.
    """
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return " ".join(parts)
    return ""


# --- the two variants --------------------------------------------------------


@contextmanager
def _in_sandbox(sandbox: str) -> Iterator[None]:
    """Run a variant with the process cwd set to its sandbox.

    The production tools resolve relative paths against the process cwd
    (`FileWriteTool` does `Path(file_path)`; `Tool._declare` resolves the same
    way), and `build_engine` tells the model in its system prompt that its
    working directory IS the sandbox. Without this the two disagree: the model
    writes what it was asked for, relative to the repository root, and the
    judge then reads an empty sandbox. The offline protocol never showed the
    disagreement because `scripted_factory` builds absolute paths.

    `os.chdir` is process-global, so **cases must run serially**. That is
    already true of this suite, and it is what makes the shared-sandbox design
    coherent: the teammates of one variant are *supposed* to share a working
    tree, so one cwd per variant is the right granularity rather than a
    limitation. A future parallel runner would have to pass a cwd down to the
    tools instead -- see the spec's option B.

    The entry assertion exists because a nested chdir is indistinguishable
    from a correct one at the point of failure: the next case would resolve its
    paths one level deeper and report a missing artifact rather than a leaked
    cwd.
    """
    repo_root = Path(__file__).resolve().parents[2]
    here = Path.cwd().resolve()
    if here != repo_root:
        raise RuntimeError(
            f"_in_sandbox entered from {here}, expected the repository root "
            f"{repo_root}; a previous variant did not restore the cwd"
        )
    os.chdir(sandbox)
    try:
        yield
    finally:
        os.chdir(here)


async def _drive(
    engine: Any,
    prompt: str,
    *,
    max_turns: int,
) -> tuple[list[Any], list[str], bool]:
    """Run one agent to completion, returning its events, errors and timeout flag.

    The timeout is reported as a flag rather than only as an error string. Both
    facts are kept because they answer different questions: the string is what a
    human reads in the row, and the flag is what the report excludes on. Reading
    the exclusion back out of the string would work until somebody reworded it,
    and the failure would be silent -- the case would quietly rejoin the crash
    bucket, which is a different diagnosis with a different fix.
    """
    events: list[Any] = []
    errors: list[str] = []
    timed_out = False
    try:
        await asyncio.wait_for(
            drain(engine.submit(prompt, max_turns=max_turns), events),
            timeout=VARIANT_TIMEOUT_S,
        )
    except TimeoutError:
        errors.append(f"variant exceeded the {VARIANT_TIMEOUT_S}s wall-clock ceiling")
        timed_out = True
    except Exception as exc:  # a crashed variant is a recorded failure, not an abort
        errors.append(f"{type(exc).__name__}: {exc}")
    return events, errors, timed_out


def _install_hidden_test(
    case: MultiAgentCase, sandbox: Path, fixtures_dir: Path
) -> None:
    """Copy a case's hidden judge into the sandbox, immediately before grading.

    Never earlier. The sandbox IS the agent's working tree for the whole run, so
    a judge sitting in it from the start is a judge the model can read, and a
    case whose answer is readable measures reading rather than doing.

    Both variants receive the SAME file from the SAME source. Grading the two
    arms with different tests would make the comparison a comparison of tests,
    which is the one confound this suite exists to remove.

    A missing source is a hard error rather than a skipped step: grading without
    the judge would make every run of the case fail as though the model had not
    done the work, and the case would look hard rather than broken.
    """
    src = fixtures_dir / case.hidden_test
    if not src.is_file():
        raise FileNotFoundError(
            f"{case.id}: hidden test declared at {case.hidden_test!r} is not "
            f"under {fixtures_dir}"
        )
    shutil.copy2(src, sandbox / Path(case.hidden_test).name)


def _judge(
    case: MultiAgentCase,
    sandbox: Path,
    *,
    fixtures_dir: Path | None = None,
) -> tuple[bool, list[dict[str, Any]], dict[str, bool]]:
    """Grade both variants' artifacts against the case's own judges.

    The subtask verdicts are derived from the SAME `file_exists` question the
    case's checks ask, but per subtask, so a failure report can name which one
    went missing instead of reporting a single opaque boolean. They never
    decide pass/fail -- `case_passed` does that, over the case's declared
    checks, exactly as in every other suite.

    A case that declares a `hidden_test` has it installed here, which is the
    only place that sequencing is written down. Splitting "install the judge"
    from "run the judge" across two call sites is how they eventually drift.
    """
    if case.hidden_test:
        if fixtures_dir is None:
            raise ValueError(
                f"{case.id}: declares a hidden_test but no fixtures_dir was "
                "given, so the judge cannot be located"
            )
        _install_hidden_test(case, sandbox, fixtures_dir)

    passed, detail = case_passed(_shared_checks(case), sandbox, mode=case.checks_mode)
    verdicts = {
        subtask.id: (sandbox / subtask.writes).is_file() for subtask in case.subtasks
    }
    return passed, detail, verdicts


async def run_single_variant(
    case: MultiAgentCase,
    *,
    sandbox: str,
    model: str | None,
    api_key: str,
    offline: bool,
    usage: Any,
    fixtures_dir: Path | None = None,
) -> VariantRun:
    """One agent, one query loop, every subtask's instruction in its prompt.

    Its turns go through the same `count_usage` wrapper the fan-out uses, so
    the two arms' token counts come from one mechanism. Reading the single
    arm's cost off the trajectory instead would make the comparison a
    comparison of two different measurements, and any bias in either one
    would land entirely on the ratio.
    """
    from longline.eval.engine_factory import EVAL_TEMPERATURE, build_engine

    ledger = UsageLedger()
    ledger.note_spawned(LEADER)
    token = current_ledger.set(ledger)
    try:
        # `Any`: the offline marker wrapper below is not a `QueryEngine`, and the
        # two are driven through the same duck-typed surface (`submit`).
        engine: Any = build_engine(
            sandbox=sandbox, model=model or "offline-multi-agent", api_key=api_key,
            tool_profile="core",
        )
        if offline:
            scripted = scripted_factory(
                subtasks=case.subtasks, merge_file=case.merge_file, sandbox=Path(sandbox),
                usage=usage,
            )
            # Pinned to LEADER rather than resolved from the ambient scope: the
            # single variant IS the leader, and reading `current_agent()` here
            # would let a leaked scope from a previous case relabel its cost.
            _apply_scripted_model(engine, count_usage(scripted, ledger, agent=LEADER))
            engine = _OfflineEngine(engine)
        else:
            # Live: count the engine's OWN transport. Replacing it here -- which
            # is what the unconditional `_apply_scripted_model` call used to do
            # -- is what made `model=` decorative: the request never left the
            # process, so a "live" run cost nothing and reported zero turns.
            _apply_live_counting(engine, ledger, agent=LEADER)

        with _in_sandbox(sandbox):
            started = time.perf_counter()
            _events, errors, timed_out = await _drive(
                engine, leader_prompt(case), max_turns=case.max_turns,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
    finally:
        current_ledger.reset(token)

    passed, detail, verdicts = _judge(case, Path(sandbox), fixtures_dir=fixtures_dir)
    # Both variants reconcile through the same gate. A single-agent run has no
    # second witness, so `witness_agents` stays None -- the check is weaker
    # here, and that is stated rather than papered over by skipping it.
    accounts = reconcile(ledger)
    accounting_error = _accounting_error(accounts, case_id=case.id, variant=SINGLE)
    return VariantRun(
        variant=SINGLE,
        passed=passed,
        duration_ms=duration_ms,
        ledger=ledger,
        accounts=accounts.to_dict(),
        subtask_verdicts=verdicts,
        judge_detail=detail,
        errors=errors,
        offline=offline,
        accounting_error=accounting_error,
        timed_out=timed_out,
        model=model or "offline-multi-agent",
        temperature=EVAL_TEMPERATURE,
        max_turns=case.max_turns,
        variant_timeout_s=VARIANT_TIMEOUT_S,
        workers=1,
    )


async def run_multi_variant(
    case: MultiAgentCase,
    *,
    sandbox: str,
    model: str | None,
    api_key: str,
    offline: bool,
    usage: Any,
    claude_dir: Path | None = None,
    fixtures_dir: Path | None = None,
    workers_override: int | None = None,
) -> VariantRun:
    """`workers` teammates in parallel, then the leader's declared merge step.

    The fan-out is `spawn_teammate` -- the production path, running a real
    `InProcessTeammate` with its own registry, its own `query_loop` and its own
    mailbox. A `gather` over bare coroutines would be simpler and would measure
    nothing about the product, which is the whole point of the comparison.

    The leader is a second real agent run: it does the merge step, and it is
    where the workers' replies land. Its own turns are attributed to `LEADER`
    so the report can separate "what the fan-out cost" from "what the
    coordination cost".

    Both witnesses are collected here. The usage ledger comes from the wrapped
    `call_model`; the second channel is the set of teammate tasks that
    `spawn_teammate` returned a task id for, awaited one at a time. Agreement
    between them is what `reconcile` records.
    """
    from longline.eval.engine_factory import EVAL_TEMPERATURE, build_engine
    from longline.session.task_registry import TaskRegistry

    ledger = UsageLedger()
    ledger.note_spawned(LEADER)
    token = current_ledger.set(ledger)
    registry = TaskRegistry()

    started = time.perf_counter()
    errors: list[str] = []
    spawned_ids: list[str] = []
    worker_outcomes: dict[str, BaseException] = {}
    try:
        # `Any`: same reason as the single variant -- `_OfflineEngine` is a
        # marker wrapper, not a `QueryEngine`.
        engine: Any = build_engine(
            sandbox=sandbox, model=model or "offline-multi-agent", api_key=api_key,
            tool_profile="core",
        )
        if offline:
            scripted = scripted_factory(
                subtasks=case.subtasks, merge_file=case.merge_file, sandbox=Path(sandbox),
                usage=usage,
            )
            # ONE counter, shared by the leader and every worker. `count_usage`
            # with no pinned agent resolves each turn's owner from the ambient
            # scope at call time, which is exactly what a shared factory needs:
            # the leader's turns arrive outside any worker's scope and land on
            # LEADER, and each worker's arrive inside its own scope and land on
            # that worker. Wrapping this in a second counter would record every
            # turn twice.
            counted = count_usage(scripted, ledger)
            # The leader's own model calls go through the SAME counter, so the
            # merge turn's tokens land in the ledger too. Without this the
            # leader's cost would be missing and every fan-out would look
            # cheaper than it is.
            _apply_scripted_model(engine, count_usage(scripted, ledger, agent=LEADER))
            engine = _OfflineEngine(engine)
        else:
            # Live: ONE unpinned counter over the engine's OWN transport. The
            # workers' factory IS the engine's here -- unlike the offline path,
            # where they get the scripted factory while the leader's engine
            # gets a LEADER-pinned copy of it. Sharing one wrapper is safe
            # because the owner of each turn is resolved from the ambient scope
            # at call time, and a turn cannot pass through two wrappers, so
            # nothing is counted twice.
            _apply_live_counting(engine, ledger)
            counted = engine.make_call_model

        # Every teammate's file I/O happens inside this block. They are
        # `asyncio` tasks sharing one process cwd, so the chdir has to outlive
        # all of them -- and it is fine that they share it, because the
        # teammates of one variant are supposed to work in one tree.
        with _in_sandbox(sandbox):
            spawned_ids.append(LEADER)
            # Which scheduler runs is a property of the CASE, not a parameter:
            # a `dependent` case driven concurrently would race on the one file
            # its steps share, and a concurrent case driven serially would stop
            # measuring parallelism while still reporting a Speedup for it.
            # Two branches rather than one call through a variable, because the
            # serial scheduler takes no concurrency argument: it is serial by
            # construction, and `workers` would only be a number it ignores.
            spawn_kwargs: dict[str, Any] = {
                "counted": counted,
                "ledger": ledger,
                "registry": registry,
                "claude_dir": claude_dir,
                "spawned_ids": spawned_ids,
                "sandbox": Path(sandbox),
                "failures": worker_outcomes,
            }
            if case.category == CATEGORY_DEPENDENT:
                await _spawn_workers_serial(case, **spawn_kwargs)
            else:
                await _spawn_workers(
                    case, workers=effective_workers(case, workers_override),
                    **spawn_kwargs,
                )
            for agent_name, exc in sorted(worker_outcomes.items()):
                errors.append(f"{agent_name}: {type(exc).__name__}: {exc}")

            _events, leader_errors, leader_timed_out = await _drive(
                engine, merge_instruction(case), max_turns=case.max_turns,
            )
            errors.extend(leader_errors)
    finally:
        current_ledger.reset(token)

    duration_ms = (time.perf_counter() - started) * 1000.0

    witness_agents, witness_turns = _teammate_witness(registry)
    accounts = reconcile(ledger, witness_agents=witness_agents, witness_turns=witness_turns)
    accounting_error = _accounting_error(accounts, case_id=case.id, variant=MULTI)

    passed, detail, verdicts = _judge(case, Path(sandbox), fixtures_dir=fixtures_dir)
    return VariantRun(
        variant=MULTI,
        passed=passed,
        duration_ms=duration_ms,
        ledger=ledger,
        accounts=accounts.to_dict(),
        subtask_verdicts=verdicts,
        judge_detail=detail,
        errors=errors,
        offline=offline,
        accounting_error=accounting_error,
        # Only the leader's drive is bounded by the ceiling; a worker is awaited
        # without one, so `_drive` is the sole possible source of a timeout in
        # this variant. Named for where it came from rather than aliased, so a
        # later change that gives workers their own ceiling has to decide what
        # to do with it here instead of inheriting a value that stopped meaning
        # what it said.
        timed_out=leader_timed_out,
        model=model or "offline-multi-agent",
        temperature=EVAL_TEMPERATURE,
        max_turns=case.max_turns,
        variant_timeout_s=VARIANT_TIMEOUT_S,
        workers=effective_workers(case, workers_override),
    )


def _accounting_error(accounts: AccountedAgents, *, case_id: str, variant: str) -> str:
    """The reconcile failure as a string, or `""` when the accounts are clean.

    The check runs on BOTH variants, not only on the fan-out. A single-agent
    run has one agent and a trivial ledger, so a failure there means the
    measurement itself is broken -- the leader's own stream was not wrapped --
    and comparing a sound fan-out against a broken baseline would report a
    speedup that is really a wiring bug. Same check, same failure mode, both
    sides.
    """
    try:
        accounts.assert_complete(case_id=case_id, variant=variant)
    except AccountingError as exc:
        return str(exc)
    return ""


def effective_workers(case: MultiAgentCase, override: int | None) -> int:
    """The concurrency a run uses, resolving an override without touching the case.

    Group 3 varies `workers` across runs of the SAME case, so the override has
    to travel as a parameter rather than as an assignment to `case.workers`.
    Assigning would make the two points of the curve different cases -- the
    fixture, the prompt and the judges would all be free to drift with it --
    and it would also let the two variants of one run disagree about
    concurrency, which is a confound inside a single row rather than between
    two.

    The override is validated here rather than at the CLI edge because the
    runner is reachable from tests and scripts too, and a `workers=99` that only
    the CLI rejects is a `workers=99` that runs from a notebook. The symptom
    would not be an error either: it would be a case that quietly stopped
    measuring parallelism, because every subtask would land in the first wave.

    A `dependent` case is the other half of the same guard. Its concurrency is
    fixed at `DEPENDENT_WORKERS` by the contract, and the serial scheduler
    ignores the number entirely -- so an override accepted here would be a
    concurrency the ROW reports and the RUN never used, which is worse than
    refusing it. One function decides what "the concurrency this run uses"
    means, and both the scheduler and the row read it from here.
    """
    if case.category == CATEGORY_DEPENDENT:
        if override not in (None, DEPENDENT_WORKERS):
            raise ValueError(
                f"a dependent case is serial by contract, so its concurrency is "
                f"fixed at {DEPENDENT_WORKERS}; got override {override}"
            )
        return DEPENDENT_WORKERS
    if override is None:
        return case.workers
    if not MIN_WORKERS <= override <= MAX_WORKERS:
        raise ValueError(
            f"workers override must be in [{MIN_WORKERS}, {MAX_WORKERS}] "
            f"per contract §5.6, got {override}"
        )
    return override


async def _spawn_workers(
    case: MultiAgentCase,
    *,
    counted: Any,
    ledger: UsageLedger,
    registry: Any,
    claude_dir: Path | None,
    spawned_ids: list[str],
    sandbox: Path,
    failures: dict[str, BaseException],
    workers: int | None = None,
) -> None:
    """Spawn one teammate per SUBTASK, up to `workers` at a time.

    Every subtask must be executed, in either variant -- that is what "the same
    work both ways" means and it is the precondition of the whole comparison.
    So the number of workers is the *concurrency limit*, not the number of
    agents that exist: a case with four subtasks and two workers runs two
    waves of two. Spawning `workers` agents and handing the leftovers to
    whichever one happens to be free would silently drop the overhang, and the
    dropped subtask's file would then be missing from the artifact set both
    variants are judged against -- a failure that looks like the model's.

    `workers` defaults to the case's own value and is passed in only when
    Group 3 is varying concurrency across runs of one case. It arrives as a
    PARAMETER rather than as an assignment to `case.workers`, because the case
    is a shared object: rewriting it would make the two points of the scaling
    curve different cases, and would let the two variants of one run disagree
    about concurrency -- a confound inside a single row.

    `failures` is keyed by AGENT NAME, which is what makes a raised exception
    attributable: awaiting a wave with `return_exceptions=True` returns the
    exceptions in wave order, and pairing them positionally against a global
    spawn list would attribute the third worker's failure to the second the
    moment the two lists stopped lining up. The name is written from inside the
    coroutine that raised, so it cannot be mispaired.
    """
    from longline.swarm.spawn import spawn_teammate

    concurrency = effective_workers(case, workers)
    waves = [
        case.subtasks[start : start + concurrency]
        for start in range(0, len(case.subtasks), concurrency)
    ]
    worker_index = 0
    for wave in waves:
        wave_tasks: list[Any] = []
        for subtask in wave:
            worker_index += 1
            agent_name = f"worker{worker_index}"
            ledger.note_spawned(agent_name)
            spawned_ids.append(agent_name)

            # The scope is entered INSIDE the spawned coroutine rather than
            # around `spawn_teammate`, because `asyncio.create_task` copies the
            # context at creation time: a scope set around the spawn call would
            # be captured by every teammate and they would all claim to be
            # `worker1`.
            async def _one(subtask: Subtask = subtask, agent_name: str = agent_name) -> Any:
                with agent_scope(agent_name):
                    try:
                        task_id = await spawn_teammate(
                            team_name=case.id,
                            agent_name=agent_name,
                            prompt=subtask_prompt(case, subtask),
                            call_model_factory=counted,
                            parent_registry=_registry_for_sandbox(sandbox),
                            claude_dir=claude_dir,
                            task_registry=registry,
                        )
                        return await _await_teammate(task_id)
                    except BaseException as exc:  # recorded against its own name
                        failures[agent_name] = exc
                        return None

            wave_tasks.append(asyncio.create_task(_one()))
        # A wave must finish before the next one starts, or the declared
        # concurrency limit is decorative. Exceptions are swallowed here
        # because `_one` already recorded them, by name.
        await asyncio.gather(*wave_tasks, return_exceptions=True)


async def _spawn_workers_serial(
    case: MultiAgentCase,
    *,
    counted: Any,
    ledger: UsageLedger,
    registry: Any,
    claude_dir: Path | None,
    spawned_ids: list[str],
    sandbox: Path,
    failures: dict[str, BaseException],
) -> None:
    """Run a `dependent` case's subtasks one teammate at a time, in chain order.

    Deliberately NOT `_spawn_workers` with `workers=1`. That function derives
    its order from `case.subtasks` and its concurrency limit from
    `case.workers`, so a later edit to `workers` would silently parallelise a
    chain -- and the case would go on reporting a handoff cost it no longer
    measures. Here the order comes from `chain_order` and there is no
    concurrency parameter to get wrong.

    The scope note from `_spawn_workers` still applies and is why `agent_scope`
    is entered inside the coroutine rather than around it: `spawn_teammate`
    calls `asyncio.create_task` for the teammate, which copies the context at
    that moment, so the scope has to be live when the spawn happens.

    A failed step does NOT stop the chain. The case's judges run against the
    final workspace, so a chain that dies at step 2 has to look different from
    one that never started; aborting would make "the model gave up" and "the
    model produced nothing" the same reported result.

    Each step gets a fresh teammate, handed its own subtask's instruction and
    nothing else -- the previous step's artifact is what carries the work
    forward, through the sandbox rather than through a shared context. That
    handoff is the cost this category exists to price, and it is one number on
    purpose: separating "new context" from "what the handoff cost" would need a
    third arm this suite does not have.
    """
    from longline.swarm.spawn import spawn_teammate

    order = chain_order(case.subtasks, case_id=case.id)
    for index, subtask in enumerate(order, start=1):
        agent_name = f"worker{index}"
        ledger.note_spawned(agent_name)
        spawned_ids.append(agent_name)

        async def _one(subtask: Subtask = subtask, agent_name: str = agent_name) -> Any:
            with agent_scope(agent_name):
                try:
                    task_id = await spawn_teammate(
                        team_name=case.id,
                        agent_name=agent_name,
                        prompt=subtask_prompt(case, subtask),
                        call_model_factory=counted,
                        parent_registry=_registry_for_sandbox(sandbox),
                        claude_dir=claude_dir,
                        task_registry=registry,
                    )
                    return await _await_teammate(task_id)
                except BaseException as exc:  # recorded against its own name
                    failures[agent_name] = exc
                    return None

        # Awaited directly rather than gathered: the chain's whole point is that
        # step N+1 consumes step N's artifact, so starting two at once would
        # race on the shared path instead of handing work forward.
        await _one()


async def _await_teammate(task_id: str) -> Any:
    """Wait for one spawned teammate, via `spawn`'s own running-task table.

    A separate channel from the usage counters on purpose: this is the second
    witness. If the usage tap missed a child, this still knows the child ran --
    which is what let the check in `reconcile` be about *agreement* rather than
    about one mechanism confirming itself.
    """
    from longline.swarm.spawn import get_running_tasks

    task = get_running_tasks().get(task_id)
    if task is None:
        return None
    try:
        return await task
    except Exception as exc:  # a failed teammate is a recorded outcome
        return exc


def _teammate_witness(registry: Any) -> tuple[list[str], dict[str, int]]:
    """The second channel: what `TaskRegistry` recorded for the teammates.

    Returns the agent names the registry has records for, and the number of
    records per name. The counts are deliberately NOT compared against the
    ledger's -- the two measure different things (registry records vs model
    turns, and a retried turn is one registry record and several turns) -- so
    only the agent SET is compared, which is the part whose disagreement means
    a child was missed.

    A name can legitimately appear here and NOT in the ledger: a teammate whose
    spawn registered with the registry before it reached its first model call.
    That direction is not an error (the ledger under-counting *turns* is
    visible in the turn list); the reverse -- a ledger agent the registry never
    saw -- is what `reconcile` flags, because it means turns were attributed to
    a name nothing corroborates.
    """
    agents: list[str] = []
    turns: dict[str, int] = {}
    for record in registry.list_all():
        agent_id = str(record.metadata.get("agent_name", ""))
        if not agent_id:
            continue
        if agent_id not in agents:
            agents.append(agent_id)
        turns[agent_id] = turns.get(agent_id, 0) + 1
    return agents, turns


def _registry_for_sandbox(sandbox: Path) -> Any:
    """The tool registry the workers inherit, bound to the case's sandbox.

    A fresh registry per spawn rather than a shared module-level one, because
    `Bash` carries the sandbox as its cwd: sharing one across cases would run a
    case's shell commands in another case's directory (contract §8.3).
    """
    from longline.eval.eval_tools import build_eval_registry

    return build_eval_registry(str(sandbox), profile="core")


class _OfflineEngine:
    """Marker wrapper: the engine's model transport is scripted.

    Exists so a run's `offline` flag is a fact about the engine that ran rather
    than about which branch a caller took, the same reason
    `recovery_runner._OfflineEngine` exists.
    """

    offline = True

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


class _ScriptedTransport:
    """Swaps a real `QueryEngine`'s model transport for a scripted factory.

    Two attributes have to move, and only one of them is the obvious one:

    - `make_call_model` is what `QueryEngine.submit()` calls directly, so it is
      the one that decides whether a request ever reaches the SDK.
    - `make_call_model_factory` is what a *sub-agent* creation site would call.
      Both are replaced so a scripted run cannot leak to the network through
      either path.

    This mirrors `longline/eval/faults.py::apply_model_fault`, which replaces
    `make_call_model` for the same reason. Assigning only the factory looks
    right and is not: the engine still reaches the API, every turn costs real
    time and real money, and the ledger stays empty -- which is how this bug
    announced itself (a 15-second "offline" variant with zero recorded turns).
    `assert_applied` re-checks behaviourally, because a future refactor that
    renames either attribute would otherwise restore the silent live run.
    """

    def __init__(self, factory: Callable[..., Any]) -> None:
        self._factory = factory

    def apply(self, engine: Any) -> Any:
        engine.make_call_model = self._factory
        engine.make_call_model_factory = self._factory
        return engine

    def assert_applied(self, engine: Any) -> None:
        probe = engine.make_call_model()
        expected = self._factory()
        if type(probe).__name__ != type(expected).__name__:
            raise AccountingError(
                "the scripted model transport was not installed: "
                f"engine.make_call_model() returned {type(probe).__name__}, expected "
                f"{type(expected).__name__}. The run would reach the real SDK and "
                "record no usage, which reads as a cheap fan-out."
            )


def _apply_scripted_model(engine: Any, factory: Callable[..., Any]) -> Any:
    """Point a real `QueryEngine` at a scripted model factory, and verify it."""
    transport = _ScriptedTransport(factory)
    engine = transport.apply(engine)
    transport.assert_applied(engine)
    return engine


def _apply_live_counting(
    engine: Any, ledger: UsageLedger, *, agent: str | None = None
) -> Any:
    """Wrap a real engine's own model transport with the usage counter, in place.

    The live sibling of `_apply_scripted_model`, and the difference is what goes
    underneath: here it is the engine's OWN factory, so the request reaches the
    SDK and only the counting wrapper is added. Replacing that factory (which
    the unconditional `_apply_scripted_model` call did) is what made `model=`
    decorative.

    `agent=None` leaves the owner of each turn to be resolved from the ambient
    scope at call time, which is what the multi variant needs: ONE wrapper on
    the leader's engine, with the leader's own turns arriving outside any
    worker's scope and each worker's arriving inside its own. Pinning an agent
    here would attribute every teammate's turns to the leader.

    `make_call_model` is the ONE insertion point, and both attributes are set to
    the same wrapper. `QueryEngine` routes everything through it: `submit()`
    calls it directly, and `make_call_model_factory()` returns a closure whose
    body is `engine.make_call_model(...)`. So wrapping the factory's PRODUCT
    instead -- the obvious reading of "wrap the factory" -- recurses, because
    the closure calls the attribute this function has just replaced, and the
    wrapper calls the closure again. Measured: `RecursionError` from
    `ModelCounter.__call__`. Capturing the bound method BEFORE the assignment is
    what breaks that cycle, and it is the same object `submit` would have got.

    `_assert_live_counting` is the mirror of `assert_applied`. A refactor that
    renamed either attribute would leave the live path UNCOUNTED -- and an
    uncounted live run reports zero tokens while spending real money, which is
    the same "reads as a cheap fan-out" failure in the opposite direction.
    """
    original = engine.make_call_model
    counted = count_usage(original, ledger, agent=agent)
    engine.make_call_model = counted
    engine.make_call_model_factory = counted
    _assert_live_counting(engine)
    return engine


def _assert_live_counting(engine: Any) -> None:
    """Fail loudly if a live engine's transport is not the counting wrapper.

    Behavioural rather than by convention, for the same reason
    `_ScriptedTransport.assert_applied` is: a future rename of either attribute
    would otherwise silently restore an uncounted live run, and the only
    symptom would be a suspiciously cheap number.
    """
    from longline.eval.child_usage import ModelCounter

    if not isinstance(engine.make_call_model, ModelCounter):
        raise AccountingError(
            "the live usage counter was not installed: "
            f"engine.make_call_model is {type(engine.make_call_model).__name__}, "
            "expected ModelCounter. A live run without it records no tokens, "
            "which reads as a free fan-out."
        )
    if not isinstance(engine.make_call_model_factory, ModelCounter):
        raise AccountingError(
            "the live usage counter was not installed on the sub-agent "
            f"creation path: engine.make_call_model_factory is "
            f"{type(engine.make_call_model_factory).__name__}, expected "
            "ModelCounter. Teammates spawned through it would spend tokens the "
            "ledger never sees."
        )


# --- entry points ------------------------------------------------------------


async def run_multi_agent_case(
    case: MultiAgentCase,
    *,
    api_key: str,
    fixtures_dir: Path,
    model: str | None = None,
    claude_dir: Path | None = None,
    usage: Any = None,
    workers_override: int | None = None,
    repeat_index: int = 0,
) -> MultiAgentRun:
    """Run one case's single and multi variants, each in its own sandbox.

    The two variants get separate sandboxes seeded from the case's two sibling
    fixtures. They are proven byte-identical at load time, so the only thing
    that differs between the arms is how the work was executed -- which is the
    claim the whole comparison rests on.

    A variant whose accounting does not reconcile, or which raised, is recorded
    with a reason and excluded from the ratio denominators. It is **not**
    dropped: a silently removed case would shrink the denominator and inflate
    `Speedup` exactly where the fan-out failed to be accountable, which is the
    one place an excluded case must stay visible.
    """
    single = await _run_variant_in_sandbox(
        case, variant=SINGLE, api_key=api_key, fixtures_dir=fixtures_dir,
        fixture=case.fixture_single, model=model, claude_dir=claude_dir, usage=usage,
    )
    multi = await _run_variant_in_sandbox(
        case, variant=MULTI, api_key=api_key, fixtures_dir=fixtures_dir,
        fixture=case.fixture_multi, model=model, claude_dir=claude_dir, usage=usage,
        workers_override=workers_override,
    )

    run = MultiAgentRun(
        case_id=case.id,
        group=case.group,
        category=case.category,
        workers=(
            effective_workers(case, workers_override)
            if case.group == CONTROLLED
            else 0
        ),
        num_subtasks=case.num_subtasks,
        single=single,
        multi=multi,
        expected_paths=case.expected_paths(),
        repeat_index=repeat_index,
    )
    # Both arms are asked, in that order, and the first reason wins. A broken
    # BASELINE is reported ahead of a broken fan-out because the ratio is
    # meaningless either way, but the baseline is the side a reader would not
    # think to suspect -- and a fan-out compared against it would produce a
    # speedup that is really a wiring bug in `_drive`'s own accounting.
    for variant, arm in ((single, "single-agent"), (multi, "multi-agent")):
        reason = variant.exclusion_reason()
        if reason is None:
            continue
        run.excluded_from_denominator = True
        run.exclusion_reason = reason
        run.note = _exclusion_note(reason, arm)
        break
    return run


def _exclusion_note(reason: str, arm: str) -> str:
    """The sentence a reader sees next to an excluded row.

    Keyed on the REASON, not written inline at the exclusion site: the reason is
    the thing that is checked against the contract, and two copies of "what
    `timeout` means" would drift the moment one of them was reworded.
    """
    if reason == REASON_ACCOUNTING_INCOMPLETE:
        return (
            f"the {arm} arm's token accounting did not reconcile, so its cost "
            "cannot be compared (contract §5.6 red line)"
        )
    if reason == REASON_TIMEOUT:
        return (
            f"the {arm} arm was cut off by the {VARIANT_TIMEOUT_S}s ceiling, so "
            "it measured nothing; Speedup is not reported"
        )
    return f"the {arm} arm did not complete; Speedup is not reported"


async def _run_variant_in_sandbox(
    case: MultiAgentCase,
    *,
    variant: str,
    api_key: str,
    fixtures_dir: Path,
    fixture: str,
    model: str | None,
    claude_dir: Path | None,
    usage: Any,
    workers_override: int | None = None,
) -> VariantRun:
    """One variant, in its own freshly copied sandbox, judged before cleanup."""
    offline = model is None
    sandbox = _prepare_sandbox(fixtures_dir, fixture, case_id=f"{case.id}/{variant}")
    try:
        if variant == SINGLE:
            return await run_single_variant(
                case, sandbox=sandbox, model=model, api_key=api_key,
                offline=offline, usage=usage, fixtures_dir=fixtures_dir,
            )
        return await run_multi_variant(
            case, sandbox=sandbox, model=model, api_key=api_key,
            offline=offline, usage=usage, claude_dir=claude_dir,
            fixtures_dir=fixtures_dir, workers_override=workers_override,
        )
    except Exception as exc:  # a crashed variant is recorded, never propagated
        return VariantRun(
            variant=variant,
            passed=False,
            duration_ms=0.0,
            ledger=UsageLedger(spawned=[LEADER]),
            accounts={},
            subtask_verdicts={s.id: False for s in case.subtasks},
            errors=[f"{type(exc).__name__}: {exc}"],
            offline=offline,
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


async def run_multi_agent_suite(
    cases: Iterable[MultiAgentCase],
    *,
    api_key: str,
    fixtures_dir: Path,
    model: str | None = None,
    claude_dir: Path | None = None,
    usage: Any = None,
    workers_override: int | None = None,
    repeats: int = 1,
) -> list[MultiAgentRun]:
    """Run every case serially, `repeats` times each.

    Serial by contract (§4.6): a parallel quality run would contend for the API
    and for the host's cores, and `WallClockTime` is one of the reported
    quantities -- running two cases at once would measure the contention
    instead. The fan-out *inside* one variant deliberately does run its workers
    concurrently; that is the thing being measured.

    The repeats of one case run back to back rather than the corpus being swept
    once per repeat: `repeat_index` is what the report's stability section
    groups on, and interleaving would make a case's samples differ by whatever
    else happened on the host in between.

    `workers_override` applies to every case in the run, which is what the
    agent-count axis wants: two runs of the SAME case file at 2 and 4, compared
    to each other. A `dependent` case refuses a value other than 1, so a corpus
    containing both shapes cannot be swept blindly -- see `effective_workers`.

    `repeats` is validated here and not only at the CLI: a repeat count of zero
    is a sweep that spends nothing and reports nothing, and it divides by zero
    in every mean. The runner is reachable from tests and notebooks too.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats!r}")
    runs: list[MultiAgentRun] = []
    for case in cases:
        for repeat_index in range(repeats):
            runs.append(
                await run_multi_agent_case(
                    case, api_key=api_key, fixtures_dir=fixtures_dir, model=model,
                    claude_dir=claude_dir, usage=usage,
                    workers_override=workers_override,
                    repeat_index=repeat_index,
                )
            )
    return runs


# --- aggregation -------------------------------------------------------------


def _sum_tokens(runs: Sequence[VariantRun]) -> dict[str, int]:
    return {
        "input_tokens": sum(r.input_tokens for r in runs),
        "output_tokens": sum(r.output_tokens for r in runs),
        "total_tokens": sum(r.total_tokens for r in runs),
        "child_tokens": sum(r.ledger.child_tokens() for r in runs),
        "tool_calls": sum(r.tool_calls for r in runs),
    }


def _pair_block(eligible: Sequence[MultiAgentRun]) -> dict[str, object]:
    """Every reportable figure for one set of eligible runs.

    One implementation, used for the group-level block AND for each category's.
    A second copy for the per-category tables would be free to drift, and the
    drift would show up as two headings in the same report computing `Speedup`
    two different ways -- which a reader has no way to notice.

    `success_per_1k_tokens` is per ARM, and both are reported. The corpus-level
    question the user's spec asks is "how much does a success cost", and a
    fan-out that wins on wall clock while costing 3.3x the tokens has a worse
    cost per success -- which is the whole finding, stated as one number.
    """
    speedups = [v for v in (r.speedup for r in eligible) if v is not None]
    overheads = [v for v in (r.token_overhead for r in eligible) if v is not None]
    single_tokens = _sum_tokens([r.single for r in eligible])
    multi_tokens = _sum_tokens([r.multi for r in eligible])

    def _rate(*, passed: Sequence[bool], tokens: dict[str, int]) -> float | None:
        return success_per_1k_tokens(
            successes=sum(1 for ok in passed if ok), total_tokens=tokens["total_tokens"],
        )

    return {
        "num_cases": len({r.case_id for r in eligible}),
        "num_runs": len(eligible),
        "success_rate": {
            "single_agent": Ratio.fraction(r.single.passed for r in eligible).to_dict(),
            "multi_agent": Ratio.fraction(r.multi.passed for r in eligible).to_dict(),
        },
        "speedup": speedup_block(speedups),
        "token_overhead": mean(overheads),
        "tokens": {"single_agent": single_tokens, "multi_agent": multi_tokens},
        "tool_calls": {
            "single_agent": single_tokens["tool_calls"],
            "multi_agent": multi_tokens["tool_calls"],
        },
        "success_per_1k_tokens": {
            "single_agent": _rate(
                passed=[r.single.passed for r in eligible], tokens=single_tokens,
            ),
            "multi_agent": _rate(
                passed=[r.multi.passed for r in eligible], tokens=multi_tokens,
            ),
        },
    }


def aggregate_multi_agent(
    runs: Sequence[MultiAgentRun],
    *,
    group: str | None = None,
) -> MultiAgentSummary:
    """Collapse per-case runs into the contract's metrics, for ONE group.

    `group` filters, and there is no default that merges them. Contract §5.6 is
    explicit that the exploratory group is reported separately: a controlled
    case's subtasks are pre-declared so both arms do the same work, while an
    exploratory case's coordinator decomposes freely, so an average across the
    two would compare different work under one heading. `group=None` at least
    refuses to guess and aggregates whatever it was handed.

    Eligibility is the accounting gate. Both success rates use the SAME
    eligible set -- comparing a rate over all cases against a rate over the
    subset would fold a sampling difference into the comparison, which is the
    confound the paired design exists to remove.

    `Speedup` and `TokenOverhead` are means of per-case ratios, not ratios of
    means. The two differ whenever the cases are not homogeneous in size, and
    the per-case form is the one the contract's formulas describe
    (`single_wall_time / multi_wall_time` of the same task).
    """
    selected = [r for r in runs if group is None or r.group == group]
    eligible = [r for r in selected if not r.excluded_from_denominator]

    speedups = [v for v in (r.speedup for r in eligible) if v is not None]
    overheads = [v for v in (r.token_overhead for r in eligible) if v is not None]

    def _mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    single_tokens = _sum_tokens([r.single for r in eligible])
    multi_tokens = _sum_tokens([r.multi for r in eligible])

    categories = sorted({r.category for r in eligible})
    by_category = {
        category: _pair_block([r for r in eligible if r.category == category])
        for category in categories
    }

    return MultiAgentSummary(
        group=group or "all",
        num_cases=len({r.case_id for r in selected}),
        eligible_cases=len({r.case_id for r in eligible}),
        excluded_cases=(
            len({r.case_id for r in selected}) - len({r.case_id for r in eligible})
        ),
        num_runs=len(selected),
        eligible_runs=len(eligible),
        single_success_rate=Ratio.fraction(r.single.passed for r in eligible),
        multi_success_rate=Ratio.fraction(r.multi.passed for r in eligible),
        single_wall_time_ms=_mean([r.single.duration_ms for r in eligible]),
        multi_wall_time_ms=_mean([r.multi.duration_ms for r in eligible]),
        mean_speedup=_mean(speedups),
        mean_token_overhead=_mean(overheads),
        single_tokens=single_tokens,
        multi_tokens=multi_tokens,
        single_tool_calls=single_tokens["tool_calls"],
        multi_tool_calls=multi_tokens["tool_calls"],
        agent_counts=sorted({r.multi.agent_count for r in eligible}),
        per_case=[r.to_row() for r in selected],
        by_category=by_category,
        pooled=_pair_block(eligible) if len(categories) <= 1 else None,
    )


__all__ = [
    "MULTI_AGENT_TAG",
    "REASON_ACCOUNTING_INCOMPLETE",
    "REASON_VARIANT_ERROR",
    "VARIANT_TIMEOUT_S",
    "MultiAgentRun",
    "MultiAgentSummary",
    "VariantRun",
    "aggregate_multi_agent",
    "leader_prompt",
    "merge_instruction",
    "run_multi_agent_case",
    "run_multi_agent_suite",
    "run_multi_variant",
    "run_single_variant",
    "scripted_factory",
    "subtask_prompt",
]
