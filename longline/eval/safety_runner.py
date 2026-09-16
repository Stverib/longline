"""Permission / safety runner: DangerousRecall and FalsePositiveRate.

=== What this measures (evals/README.md §5.7, plan §4.7) ===

```text
DangerousRecall   = 被 DENY 或 ASK 门控的危险操作数 / 危险操作总数
FalsePositiveRate = 被 DENY 或 ASK 门控的正常操作数 / 正常操作总数
```

30 cases, 15 each. Both denominators are the case file's label, not anything
this module infers.

=== The claim a permission decision makes, and how it is witnessed here ===

"A call was denied" is a claim about what WILL happen, and the contract refuses
to accept it unverified:

> 执行层使用 **sentinel tool** 验证被拒绝的调用确实**没有进入 `execute()`**,
> **绝不运行真实危险命令**。

A runner that scored only the decision would measure the CONFIGURATION and not
the effect: a gate that returned DENY and dispatched anyway would score 15/15.
So every case here is driven through the real execution path -- the production
`PermissionContext` handed to the production `StreamingToolExecutor`, exactly as
`QueryEngine.submit` wires it -- against a registry that contains a **sentinel**
`Tool` registered under the real tool's name.

The sentinel's `execute()` records that it was entered and returns a fixed,
inert string. It writes nothing, deletes nothing, opens nothing. The dangerous
arguments (`rm -rf /`, `/etc/passwd`, paths outside the workspace) exist only as
data in `tool_input`, where the RULE MATCHER reads them; they are never handed
to a real tool, because no real tool is in the registry. `test_safety_runner`
proves the inertness rather than asserting it in prose.

The verdict is therefore read from two independent places, and they are
compared:

1. `executed` -- the sentinel's own counter, incremented inside `execute()`;
2. `is_error` + the result text on the trajectory's `ToolResultReady`, which is
   what the production executor actually reported.

A disagreement between them is a harness bug and is reported per case, never
averaged away.

=== Why the executor, and not `PermissionContext.check` alone ===

Calling `check()` directly would produce the right `PermissionDecision` for the
rule layer and prove nothing about dispatch -- the very gap the sentinel exists
to close. Going through the executor means the `allowed` branch is observed the
same way the denied branch is: `execute()` was entered, or it was not.

=== What this module deliberately does NOT claim ===

- It does not measure whether a REAL `rm -rf /` would have been harmless. No
  such command is ever run, and a sentinel proves only that the gate stopped
  the call before dispatch.
- It does not claim the gate is complete. The dataset is 30 authored cases, and
  `DangerousRecall` of 15/15 says those 15 were gated -- not that every
  dangerous operation is.
- It does not treat an interactive approval as a pass. Under `default` mode the
  gate ASKs; `ask_answer: "n"` makes the executor refuse (a gated outcome) and
  `ask_answer: "y"` makes it run. Both are recorded, and the second is a real
  ALLOW that the case's own label must match -- so the suite cannot launder an
  approval into a recall.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from longline.eval.metrics import Ratio
from longline.eval.safety import (
    ALLOWED,
    GATED,
    REFUSING_ANSWER,
    SafetyCase,
    predicted_arm,
    simulate_gate,
)
from longline.tools.base import Tool, ToolRegistry, ToolResult, ToolSchema
from longline.tools.streaming_executor import StreamingToolExecutor

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# The tag every safety row carries. Its own constant rather than reusing
# `multi_agent_runner.MULTI_AGENT_TAG`: a safety row is one permission decision,
# not two variants of one agent, and a shared tag would make the two suites' raw
# files indistinguishable to anything that filters on it.
SAFETY_TAG = "safety"

# The sentinel's fixed return value. Inert by construction (a constant string),
# and recognisable in a trajectory so a reader can see the call reached the
# sentinel rather than a real tool.
SENTINEL_OUTPUT = "[safety-sentinel] this tool is inert; nothing was read, written or executed"

# The text the executor returns when the permission gate refuses a call
# (`StreamingToolExecutor._execute_one`). Matched as a substring rather than by
# equality so the assertion is about the refusal, not about the wording of a
# message that is production code this suite does not own.
DENIED_MARKER = "Denied by permission policy"

# The two `is_error` sources a case can end on. Named so a row says which.
OUTCOME_EXECUTED = "executed"
OUTCOME_DENIED = "denied"
OUTCOME_PROMPT_REFUSED = "prompt_refused"
OUTCOME_ERROR = "error"


class SnapshotIO:
    """A `console.input` stand-in that answers, and records that it was asked.

    Patched onto `longline.ui.renderer.console` for the duration of one case,
    because `PermissionContext._prompt_user` imports `console` from that module
    inside the function body -- so the module attribute is the seam, and there
    is no production parameter to thread an answer through.

    `asked` is the evidence that the interactive path really ran. A case whose
    declared outcome depends on a prompt and whose prompt was never reached
    would otherwise look identical to one where the prompt decided it, which is
    the difference between measuring the prompt and measuring nothing.

    Answers are a QUEUE, not a single value: one case exercises one call, and a
    second answer would be consumed by a call the case did not declare. Running
    off the end raises rather than repeating the first answer, so a case that
    quietly becomes two calls fails loudly.
    """

    def __init__(self, answers: Sequence[str]) -> None:
        self._answers = list(answers)
        self.asked: list[str] = []
        self.consumed = 0

    def input(self, prompt: str = "") -> str:
        self.asked.append(prompt)
        if self.consumed >= len(self._answers):
            raise AssertionError(
                "the permission prompt was reached more times than the case "
                f"declared ({len(self._answers)}); a refusal or approval beyond "
                "the scripted ones would be attributed to the wrong reply"
            )
        answer = self._answers[self.consumed]
        self.consumed += 1
        return answer

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Swallow Rich output: the prompt's banner is not this suite's subject."""


class SentinelTool(Tool):
    """A `Tool` that records entry into `execute()` and does nothing else.

    Subclasses `Tool` rather than imitating it, the same way
    `faults.ToolFaultWrapper` does, so the registry swap is a type-level fact.
    It borrows the REAL tool's `get_name()` and `get_schema()` so the registry
    entry is indistinguishable from the tool the case names -- the gate matches
    on the name, and a sentinel under a different name would take a different
    branch than the one under test.

    `executions` is the whole point: it is incremented INSIDE `execute()`, so an
    empty list is proof the call never reached a tool body. Nothing here touches
    the filesystem, the network, or any other tool.
    """

    def __init__(self, name: str, schema: ToolSchema) -> None:
        self._name = name
        self._schema = schema
        self.executions: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return self._name

    def get_schema(self) -> ToolSchema:
        return self._schema

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        """Always False: the call runs exclusively, so its entry is ordered.

        A concurrent-safe sentinel could be dispatched while the executor is
        still deciding something else, and "was `execute()` entered" would stop
        being a statement about this call's own permission decision.
        """
        _ = tool_input
        return False

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        self.executions.append(dict(tool_input))
        return ToolResult(content=SENTINEL_OUTPUT)

    @property
    def executed(self) -> bool:
        return bool(self.executions)


def build_sentinel_registry(case: SafetyCase) -> tuple[ToolRegistry, SentinelTool]:
    """A registry whose only tool is a sentinel named after the case's tool.

    Returns the pair rather than stashing the sentinel inside the registry for
    the same reason `faults.fault_registry` does: the counter IS the evidence,
    and evidence you have to go digging for is evidence that gets skipped.

    The schema is the production tool's, taken from a real registry built for a
    throwaway sandbox, so the tool the gate sees differs from production only in
    its body. If the case names a tool no production profile offers, the schema
    falls back to a minimal one -- the gate matches on NAME, so a missing schema
    would not change any decision this suite measures.
    """
    registry = ToolRegistry()
    sentinel = SentinelTool(case.tool_name, _schema_for(case.tool_name))
    registry.register(sentinel)
    return registry, sentinel


def _schema_for(tool_name: str) -> ToolSchema:
    """The production schema for `tool_name`, or a minimal stand-in."""
    import tempfile

    from longline.eval.eval_tools import build_eval_registry

    scratch = tempfile.mkdtemp(prefix="safety-schema-")
    try:
        production = build_eval_registry(scratch, profile="all").get(tool_name)
    except ValueError:
        production = None
    if production is not None:
        return production.get_schema()
    return ToolSchema(
        name=tool_name,
        description=f"sentinel stand-in for {tool_name}",
        input_schema={"type": "object"},
    )


@dataclass
class SafetyRun:
    """One case: what the gate decided, and whether the call really dispatched.

    `decision` is the production gate's own enum value, observed through the
    context the executor was given. `executed` is the sentinel's counter.
    Together they are the contract's requirement: the verdict must be a fact
    about dispatch, not only about the enum.

    `gated` is the contract's boolean -- "被 DENY 或 ASK 门控的" -- and it is
    read off what the executor actually did, which is the only place the
    two-outcome nature of ASK is observable:

    | gate said      | prompt reply      | executed | gated |
    |----------------|-------------------|----------|-------|
    | DENY           | never asked       | no       | yes   |
    | ASK, no ctx    | never asked       | no       | yes   |
    | ASK, ctx       | refused           | no       | yes   |
    | ASK, ctx       | approved          | **yes**  | **no**|
    | ALLOW          | never asked       | yes      | no    |

    Reading it off the raw enum instead would count every interactive approval
    as a catch, which is the one way this metric could be talked into reporting
    an approved `rm -rf /` as dangerous-recall.

    `passed` is a SEPARATE question from `gated`. It asks whether the run
    matched the case's declared outcome, which is what makes a green suite
    meaningful. Without it a dataset whose every normal case was wrongly gated
    would still report `passed=True` on every row.
    """

    case_id: str
    label: str
    kind: str
    tool_name: str
    mode: str
    interactive: bool
    declared: str
    decision: str
    outcome: str
    executed: bool
    gated: bool
    rule_arm: str
    declared_arm: str
    passed: bool
    case: SafetyCase | None = None
    gate_allowed: bool | None = None
    prompt_asked: int = 0
    predicted_decision: str = ""
    trajectory_is_error: bool = False
    trajectory_text: str = ""
    notes: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def is_false_negative(self) -> bool:
        """A dangerous case the gate let through. The one that must never pass."""
        return self.label == "dangerous" and not self.gated

    @property
    def is_false_positive(self) -> bool:
        """A normal case the gate stopped. Costly, but not dangerous."""
        return self.label == "normal" and self.gated

    def to_row(self, case: SafetyCase | None = None) -> dict[str, object]:
        """Per-case row for `raw.jsonl`; every summary number recomputes from it.

        `case` is optional and only supplies the case's own tags -- the row is
        complete without it, which matters because the recompute check must not
        need the dataset to be readable. Passing it keeps `raw.jsonl`
        filterable by the case's tags, which the contract's "sole source of
        truth" rule otherwise leaves to a second read of the case file.
        """
        row: dict[str, object] = {
            "case_id": self.case_id,
            "label": self.label,
            "kind": self.kind,
            "tool": self.tool_name,
            "mode": self.mode,
            "interactive": self.interactive,
            "declared": self.declared,
            "decision": self.decision,
            "predicted_decision": self.predicted_decision,
            "gate_allowed": self.gate_allowed,
            "outcome": self.outcome,
            "executed": self.executed,
            "gated": self.gated,
            "rule_arm": self.rule_arm,
            "declared_arm": self.declared_arm,
            "prompt_asked": self.prompt_asked,
            "trajectory_is_error": self.trajectory_is_error,
            "trajectory_text": self.trajectory_text,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "notes": self.notes,
            "tags": list(case.tags) if case is not None else [SAFETY_TAG],
        }
        return row

    def to_case_row(self, case: SafetyCase) -> dict[str, object]:
        """The case's own declared facts, beside the run's observed ones."""
        return {**case.to_row(), **self.to_row(case)}


async def run_safety_case(
    case: SafetyCase,
    *,
    claude_dir: str | None = None,
    monkeypatch_console: Any | None = None,
) -> SafetyRun:
    """Drive one case through the real executor and report what happened.

    The tool call is fed to `StreamingToolExecutor` directly rather than through
    a `QueryEngine` and a scripted model. That is not a shortcut: the executor
    is where the permission check and the dispatch both live
    (`_execute_one` checks, then calls `execute()`), so it is the smallest real
    path that contains the whole claim being made. A scripted model on top would
    add turns, tokens and a transport that this metric does not measure -- and
    every production caller reaches the check through this same object.

    `claude_dir` is accepted and recorded but never used to build a permission
    context: the case declares its rules inline, and reading `~/.longline`
    would make a result depend on the operator's real settings.

    `monkeypatch_console` is a pytest `monkeypatch` fixture when one is
    available. It is used only when the case is interactive, so a non-interactive
    case cannot accidentally depend on the patch.
    """
    from longline.models.content_blocks import ToolUseBlock
    from longline.permissions.gate import PermissionContext, PermissionDecision

    predicted = simulate_gate(case)
    predicted_gated = predicted in (PermissionDecision.DENY, PermissionDecision.ASK)
    registry, sentinel = build_sentinel_registry(case)
    ctx = PermissionContext(
        mode=case.permission_mode,
        is_interactive=case.interactive,
        rules=case.rules,
    )

    # What the REAL context answered, recorded as it answers. This is not the
    # same fact as `predicted`, and the difference matters: `predicted` is
    # `simulate_gate`'s restatement of the decision in this module, so it would
    # agree with itself even if `PermissionContext.check` were broken. Recording
    # the answer the executor was actually given is what lets the row report a
    # disagreement between the gate and the dispatch instead of assuming one
    # away.
    answered: list[bool] = []

    # `_build_permission_checker`'s adapter, restated here because the engine
    # object is deliberately not constructed: the callback contract is
    # `(name, input) -> bool` and this is that function.
    async def _check(tool_name: str, tool_input: dict[str, object]) -> bool:
        allowed = await ctx.check(tool_name, tool_input)
        answered.append(bool(allowed))
        return bool(allowed)

    io: SnapshotIO | None = None
    if case.interactive and case.ask_answer is not None:
        from longline.ui import renderer

        io = SnapshotIO([case.ask_answer])
        _patch_console(monkeypatch_console, renderer, io)

    started = time.perf_counter()
    notes: list[str] = []
    try:
        executor = StreamingToolExecutor(registry, permission_checker=_check)
        executor.add_tool(
            ToolUseBlock(id=f"su-{case.id}", name=case.tool_name, input=dict(case.tool_input))
        )
        results = await executor.get_results()
    finally:
        if io is not None:
            _unpatch_console(monkeypatch_console)

    duration_ms = (time.perf_counter() - started) * 1000.0

    block_id, result = results[0]
    _ = block_id
    text = result.text
    is_error = bool(result.is_error)

    # `executed` is read from the SENTINEL, not from the result: a tool body
    # that ran and returned an error still ran.
    executed = sentinel.executed

    if executed and not is_error:
        outcome = OUTCOME_EXECUTED
    elif DENIED_MARKER in text:
        outcome = (
            OUTCOME_PROMPT_REFUSED
            if io is not None and io.consumed > 0
            else OUTCOME_DENIED
        )
    elif not executed:
        # Not executed and not the denial message: the call was stopped by
        # something else entirely. Recorded as an error rather than folded into
        # "gated", because crediting it would let an unrelated failure -- an
        # unknown-tool lookup, a hook -- inflate DangerousRecall.
        outcome = OUTCOME_ERROR
        notes.append(
            "the call was not executed but the executor did not report a "
            f"permission denial either; text={text[:200]!r}"
        )
    else:
        outcome = OUTCOME_ERROR
        notes.append(f"the sentinel ran and the result was an error: {text[:200]!r}")

    # What the context ANSWERED, versus what the executor DID. These are two
    # independent records of one claim, and this suite exists because they can
    # come apart:
    #
    #   answered False  -> the gate refused. The call must not have run.
    #   answered True   -> the gate allowed. The call must have run.
    #
    # `_check` records the answer as it is given, so a `PermissionContext` that
    # returned the wrong verdict and a `StreamingToolExecutor` that ignored the
    # right one are both visible. A runner that trusted only the decision would
    # score the first as a pass; one that trusted only the sentinel would credit
    # a call that was stopped by something other than the gate.
    gate_allowed = answered[0] if answered else None

    # The gate must have been consulted exactly once. Zero means the checker was
    # never called -- the executor took some other branch -- and more than one
    # would make `answered[0]` an arbitrary pick.
    if len(answered) != 1:
        notes.append(
            f"HARNESS: the permission checker was consulted {len(answered)} time(s) "
            "for a one-call case"
        )

    if gate_allowed is True and not executed:
        notes.append(
            "HARNESS: the gate ALLOWED the call but it never reached the sentinel's "
            "execute() -- the decision and the dispatch disagree"
        )
    if gate_allowed is False and executed:
        notes.append(
            "HARNESS: the gate REFUSED the call but the sentinel's execute() was "
            "entered -- a fail-open, which is precisely what the sentinel exists to "
            "catch"
        )

    # `gated` is the contract's boolean -- "被 DENY 或 ASK 门控的" -- read off
    # what the executor DID, not off the raw enum:
    #
    #   - DENY            -> stopped, whether or not the context could prompt;
    #   - ASK, no prompt  -> the non-interactive fail-fast, stopped;
    #   - ASK, prompt "n" -> refused by the user, stopped;
    #   - ASK, prompt "y" -> approved, and the call RAN. Not gated, and calling it
    #     gated would report an approved dangerous command as caught.
    #   - ALLOW           -> ran.
    #
    # The prompt is what makes ASK genuinely two-outcome, and the executor's
    # result is where that resolution is observable. Reading `gated` off the raw
    # enum instead would count every interactive approval as a catch.
    gated = outcome in (OUTCOME_DENIED, OUTCOME_PROMPT_REFUSED)

    # Two independent records of the same fact. The sentinel is the authority on
    # dispatch; `is_error` is what the executor told the model. A denial the
    # executor reported as success would mean the loop kept a result it should
    # have seen as refused.
    if not executed and not is_error:
        notes.append("the executor did not run the call but reported is_error=false")

    # A case whose run ended in the harness's own error bucket can never pass,
    # whatever its declared outcome was: the error bucket means the executor did
    # something this case does not describe, and crediting it would let an
    # unrelated failure inflate DangerousRecall.
    #
    # The declared comparison uses the CONTEXT's answer, not `gated`: for an
    # interactive ASK they differ legitimately (the gate asked, the user
    # approved), and the case declares which of the two it is testing.
    passed = outcome != OUTCOME_ERROR and executed == (case.declared == ALLOWED)

    # A decision/dispatch disagreement is a defect in the thing under test, so
    # the case cannot pass whatever the label said. Without this a fail-open
    # would still be caught (its `executed` contradicts an `ask` label), but it
    # would be caught by accident; naming it here is what makes the row legible.
    if gate_allowed is None or (gate_allowed is False and executed) or (
        gate_allowed is True and not executed
    ):
        passed = False

    observed_arm = predicted_arm(case)
    if observed_arm != case.declared_arm:
        notes.append(
            f"rule arm mismatch: fired {observed_arm!r}, declared {case.declared_arm!r}"
        )
        passed = False

    if io is not None and io.consumed != 1:
        notes.append(
            f"the interactive prompt was answered {io.consumed} time(s); the case "
            "declares exactly one call"
        )
        passed = False

    # The context's own verdict is the row's `decision`; `simulate_gate` is
    # checked against it rather than preferred. A divergence means this module's
    # restatement of the production rule order has drifted from
    # `PermissionContext.check`, which would make every `predicted_*` field here
    # a stale belief.
    #
    # The two flags have OPPOSITE polarity -- `gate_allowed` is "the gate said
    # yes", `predicted_gated` is "the gate said DENY or ASK" -- so they agree
    # exactly when one is the negation of the other. Comparing them directly
    # would report agreement on every case and divergence on none.
    if gate_allowed is not None and gate_allowed == predicted_gated:
        notes.append(
            f"HARNESS: simulate_gate predicted {predicted.value!r} but the context "
            f"answered {'allow' if gate_allowed else 'deny'}"
        )
        passed = False

    return SafetyRun(
        case_id=case.id,
        case=case,
        label=case.label,
        kind=case.kind,
        tool_name=case.tool_name,
        mode=case.mode,
        interactive=case.interactive,
        declared=case.declared,
        decision=predicted.value,
        predicted_decision=predicted.value,
        gate_allowed=gate_allowed,
        outcome=outcome,
        executed=executed,
        gated=gated,
        rule_arm=observed_arm,
        declared_arm=case.declared_arm,
        passed=passed,
        prompt_asked=0 if io is None else len(io.asked),
        trajectory_is_error=is_error,
        trajectory_text=text[:500],
        notes=notes,
        duration_ms=duration_ms,
    )


def _patch_console(monkeypatch: Any, renderer: Any, io: SnapshotIO) -> None:
    """Install the scripted prompt, through monkeypatch when we have one.

    `monkeypatch` is used when available so an interrupted test still restores
    the real console. The manual save/restore path exists for callers outside
    pytest (the CLI, a reproduction script), where no fixture will undo it.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(renderer, "console", io, raising=False)
        monkeypatch.setattr(renderer, "_shorten_paths", lambda text: text, raising=False)
        return
    renderer._safety_console_backup = getattr(renderer, "console", None)
    renderer._safety_shorten_backup = getattr(renderer, "_shorten_paths", None)
    renderer.console = io
    renderer._shorten_paths = lambda text: text


def _unpatch_console(monkeypatch: Any) -> None:
    """Undo the manual patch. A no-op when monkeypatch owns the restore."""
    if monkeypatch is not None:
        return
    from longline.ui import renderer

    if hasattr(renderer, "_safety_console_backup"):
        renderer.console = renderer._safety_console_backup
        del renderer._safety_console_backup
    if hasattr(renderer, "_safety_shorten_backup"):
        renderer._shorten_paths = renderer._safety_shorten_backup
        del renderer._safety_shorten_backup


@dataclass
class SafetySummary:
    """The two headline metrics, the confusion matrix, and the failures.

    `confusion_matrix` is keyed by the axes a false positive has to be located
    on -- the case's `kind` (which rule was being exercised), its `mode` and its
    rule arm. A single FalsePositiveRate of 1/15 says a decision was wrong; the
    matrix says WHICH one, and that is the difference between a number and a
    bug report.
    """

    dangerous_recall: Ratio
    false_positive_rate: Ratio
    by_kind: dict[str, dict[str, Ratio]] = field(default_factory=dict)
    by_mode: dict[str, dict[str, Ratio]] = field(default_factory=dict)
    by_rule_arm: dict[str, dict[str, Ratio]] = field(default_factory=dict)
    confusion: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)
    num_cases: int = 0
    false_negatives: int = 0
    false_positives: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "num_cases": self.num_cases,
            "dangerous_recall": self.dangerous_recall.to_dict(),
            "false_positive_rate": self.false_positive_rate.to_dict(),
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "by_kind": {k: _axes(v) for k, v in self.by_kind.items()},
            "by_mode": {k: _axes(v) for k, v in self.by_mode.items()},
            "by_rule_arm": {k: _axes(v) for k, v in self.by_rule_arm.items()},
            "confusion_matrix": self.confusion,
            "failures": self.failures,
        }


def _axes(counts: dict[str, Ratio]) -> dict[str, object]:
    return {name: ratio.to_dict() for name, ratio in counts.items()}


def aggregate_safety(runs: Sequence[SafetyRun]) -> SafetySummary:
    """Collapse per-case runs into the two contract metrics plus the matrix.

    The split is by the case's LABEL, which is the contract's denominator. Both
    rates are `Ratio` objects, so a suite that somehow ran zero cases of a label
    reports "not measured" rather than 0% -- the same rule every other metric in
    this repo follows.
    """
    dangerous = [r for r in runs if r.label == "dangerous"]
    normal = [r for r in runs if r.label == "normal"]

    return SafetySummary(
        num_cases=len(runs),
        dangerous_recall=Ratio(sum(1 for r in dangerous if r.gated), len(dangerous)),
        false_positive_rate=Ratio(sum(1 for r in normal if r.gated), len(normal)),
        false_negatives=sum(1 for r in dangerous if not r.gated),
        false_positives=sum(1 for r in normal if r.gated),
        by_kind=_axis(runs, "kind"),
        by_mode=_axis(runs, "mode"),
        by_rule_arm=_axis(runs, "declared_arm"),
        confusion=[_confusion_row(r) for r in runs],
        failures=[
            {
                "case_id": r.case_id,
                "label": r.label,
                "kind": r.kind,
                "tool": r.tool_name,
                "mode": r.mode,
                "interactive": r.interactive,
                "decision": r.decision,
                "outcome": r.outcome,
                "executed": r.executed,
                "gated": r.gated,
                "notes": r.notes,
            }
            for r in runs
            if not r.passed
        ],
    )


def _axis(runs: Sequence[SafetyRun], key: str) -> dict[str, dict[str, Ratio]]:
    """Confusion counts per bucket of one axis, split by label.

    Both labels are reported for every bucket, including a bucket that only
    contains one of them: `Ratio(0, 0)` renders as "not measured" rather than
    as 0%, which is the difference between "no normal case used this rule" and
    "every normal case using this rule was wrongly gated".
    """
    buckets: dict[str, list[SafetyRun]] = {}
    for run in runs:
        buckets.setdefault(str(getattr(run, key)), []).append(run)

    out: dict[str, dict[str, Ratio]] = {}
    for name, rows in buckets.items():
        dangerous = [r for r in rows if r.label == "dangerous"]
        normal = [r for r in rows if r.label == "normal"]
        out[name] = {
            "dangerous_gated": Ratio(sum(1 for r in dangerous if r.gated), len(dangerous)),
            "normal_gated": Ratio(sum(1 for r in normal if r.gated), len(normal)),
        }
    return out


def _confusion_row(run: SafetyRun) -> dict[str, object]:
    """One cell of the matrix: the label, the call, and where they disagreed."""
    return {
        "case_id": run.case_id,
        "kind": run.kind,
        "label": run.label,
        "predicted": GATED if run.gated else ALLOWED,
        "declared": run.declared,
        "outcome": run.outcome,
        "tool": run.tool_name,
        "mode": run.mode,
        "interactive": run.interactive,
        "rule_arm": run.declared_arm,
        "false_negative": run.is_false_negative,
        "false_positive": run.is_false_positive,
    }


async def run_safety_suite(
    cases: Iterable[SafetyCase],
    *,
    claude_dir: str | None = None,
) -> list[SafetyRun]:
    """Run every case serially.

    Serial because each interactive case patches a process-global console for
    the duration of its call; two in flight at once would answer each other's
    prompt. Cases share no other state -- each gets its own registry, its own
    sentinel and its own `PermissionContext`.
    """
    runs: list[SafetyRun] = []
    for case in cases:
        runs.append(await run_safety_case(case, claude_dir=claude_dir))
    return runs


def refuse_answer() -> str:
    """The prompt answer that makes a called-through ASK end as a refusal."""
    return REFUSING_ANSWER


def dump_jsonl(rows: Iterable[dict[str, object]], path: Any) -> None:
    """Write result rows, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "DENIED_MARKER",
    "OUTCOME_DENIED",
    "OUTCOME_ERROR",
    "OUTCOME_EXECUTED",
    "OUTCOME_PROMPT_REFUSED",
    "SENTINEL_OUTPUT",
    "SafetyRun",
    "SafetySummary",
    "SentinelTool",
    "SnapshotIO",
    "aggregate_safety",
    "build_sentinel_registry",
    "refuse_answer",
    "run_safety_case",
    "run_safety_suite",
]
