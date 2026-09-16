"""Case data model and loader for the permission / safety suite.

=== Why this is not an E2E case, nor a recovery case ===

A safety case does not run an agent and does not grade a file. It states a
scenario -- a tool, the arguments a caller would pass it, a permission mode, an
optional settings.json rule set -- and a **labelled outcome**: is this operation
one the gate is supposed to stop, or one it is supposed to let through?

That label is not decoration. Both headline metrics (contract §5.7) are computed
against it:

```text
DangerousRecall   = 被 DENY 或 ASK 门控的危险操作数 / 危险操作总数
FalsePositiveRate = 被 DENY 或 ASK 门控的正常操作数 / 正常操作总数
```

=== The dangerous argument is data, and it is never executed ===

Every dangerous case names a real tool (`Bash`, `Write`, `Edit`, ...) and the
arguments a genuinely dangerous call would carry -- `rm -rf /`, a path outside
the workspace, `/etc/passwd`. Those arguments exist so the RULE LAYER can match
them and so the case reads as the thing it is testing.

They are never handed to the tool those arguments would damage. `safety_runner`
registers a **sentinel** `Tool` under the named tool's name, drives the real
permission path through the production `StreamingToolExecutor`, and asserts on
whether the sentinel's `execute()` was entered. The sentinel writes nothing,
deletes nothing and touches no path: see `longline/eval/safety_runner.py` for
the argument, and `evals/README.md` §8.4 for the rule this implements
("Safety 用例使用 sentinel tool ... 绝不运行真实危险命令").

=== Coverage the loader enforces ===

`evals/README.md` §5.7 says judging must cover rule priority, all three
`PermissionMode`s and the non-interactive path. Those are properties of the
DECISION the case is about, not of the JSON syntax, so they are validated here:
a declared decision whose mode, rules and interactivity contradict it is
rejected at load time. See `_validate_declared`.

The cross-cutting axes -- modes, interactivity, rule priority -- are ORTHOGONAL
to the label, and that is deliberate. Only sampling `DangerousRecall` under
`default` would leave the metric blind to the mode the CLI actually runs in
(ACCEPT_EDITS), which is where a real write to `/etc/passwd` is decided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from longline.eval.types import CaseParseError
from longline.permissions.gate import PermissionDecision, PermissionMode
from longline.permissions.rules import PermissionRules

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# The two labels. A dangerous case is one the gate must gate; a normal case is
# one it must let through. Nothing else is admissible -- a third label would
# silently land in one of the two metric denominators.
DANGEROUS = "dangerous"
NORMAL = "normal"
LABELS: tuple[str, ...] = (DANGEROUS, NORMAL)

# The contract fixes both denominators at 15. They are asserted by the runner
# rather than assumed, so a dataset that drifted to 14 dangerous cases cannot
# keep reporting a "DangerousRecall / 15" that no longer divides 15.
DANGEROUS_CASES = 15
NORMAL_CASES = 15

# The three `PermissionMode`s, by their JSON spelling, so a case file is
# readable and a typo is rejected rather than defaulted.
MODE_BY_NAME: dict[str, PermissionMode] = {
    "default": PermissionMode.DEFAULT,
    "acceptEdits": PermissionMode.ACCEPT_EDITS,
    "bypassPermissions": PermissionMode.BYPASS,
}

# The decision the case declares for its operation, in the contract's
# vocabulary. `gated` is the contract's own term for "DENY or ASK" -- the two
# outcomes `DangerousRecall` counts.
GATED = "gated"
ALLOWED = "allowed"
DECISIONS: tuple[str, ...] = (GATED, ALLOWED)

# Which rule arm produced the declared decision, when a rule produced it. This
# is what makes "rule priority" checkable rather than merely executed: a
# deny-overrides-allow case must show the DENY rule's arm, not the allow one's.
RULE_DENY = "deny"
RULE_ALLOW = "allow"
RULE_NONE = "none"
RULE_ARMS: tuple[str, ...] = (RULE_DENY, RULE_ALLOW, RULE_NONE)


@dataclass
class SafetyCase:
    """One permission decision: a scenario, and its labelled outcome.

    Fields:
        id: stable case id.
        label: `dangerous` or `normal` (the metric the case feeds).
        kind: the sub-class, reported in the confusion matrix so a false
            positive can be located to the rule that caused it.
        tool_name: the tool the operation calls. It is registered in the
            runner's registry as a SENTINEL -- the name is real, the body is
            not, and nothing here is handed to a real tool.
        tool_input: the arguments a genuine call would carry. Dangerous
            arguments live here and are read ONLY by the rule matcher.
        mode: which `PermissionMode` the scenario runs under.
        interactive: whether the context can prompt the user. Non-interactive
            contexts fail-fast on ASK (gate.py), which is a *gated* outcome
            too -- counted the same way, with the two distinguished per case.
        rules: the settings.json rule set, or None for "no rules configured".
        declared: `gated` or `allowed` -- what the gate is supposed to do.
        declared_arm: which rule arm must produce that decision (`deny`,
            `allow` or `none`). Checked, not merely recorded: it is how
            deny-overrides-allow is proven rather than asserted.
        ask_answer: the answer a scripted interactive prompt returns. Required
            exactly when the declared outcome depends on it (see
            `_validate_declared`). An interactive ASK is genuinely
            two-outcome -- the same scenario approved is `allowed` and refused
            is `gated` -- so the two outcomes are two separate cases rather than
            one case with a verdict flag. That is why a scenario's `declared`
            may be the opposite of what `check_permission` alone would say: with
            `ask_answer: "y"` an ASK ends in execution, and the case's label is
            about what the gate DID, not about which enum it returned.
        note: one line of human-readable context. Every case gets one, because
            the confusion matrix names cases and a bare id is not enough for a
            reader to judge whether the label was right.
    """

    id: str
    label: str
    kind: str
    tool_name: str
    tool_input: dict[str, Any]
    mode: str
    interactive: bool
    declared: str
    note: str
    rules: PermissionRules | None = None
    declared_arm: str = RULE_NONE
    ask_answer: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def permission_mode(self) -> PermissionMode:
        """The declared mode as the enum the gate consumes.

        Looked up rather than re-parsed: `from_dict` already rejected anything
        outside `MODE_BY_NAME`, so a KeyError here would be a bug in this
        module and not a bad case file, and it should be loud.
        """
        return MODE_BY_NAME[self.mode]

    @property
    def is_dangerous(self) -> bool:
        return self.label == DANGEROUS

    @property
    def declared_gated(self) -> bool:
        """The label as the boolean both metrics are built from."""
        return self.declared == GATED

    def to_row(self) -> dict[str, object]:
        """The case's own facts, for the report's confusion matrix."""
        return {
            "case_id": self.id,
            "label": self.label,
            "kind": self.kind,
            "tool": self.tool_name,
            "mode": self.mode,
            "interactive": self.interactive,
            "rules": _rules_row(self.rules),
            "declared": self.declared,
            "declared_arm": self.declared_arm,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SafetyCase:
        """Parse and validate one case line.

        Every rejection here is a case-file bug that would otherwise show up as
        a metric. The two that matter most:

        - An unknown `label` would have to be counted as dangerous or normal by
          a default, and either default silently misstates a headline number.
        - A declared outcome the scenario's own mode/rules/interactivity
          contradict would make the case fail on a CORRECT gate, which reads as
          a safety defect in the product and is not one.
        """
        cid = d.get("id")
        if not isinstance(cid, str) or not cid:
            raise CaseParseError(f"safety case requires a string 'id', got {d!r}")

        case = cls(
            id=cid,
            label=_require_choice(d, "label", LABELS, case_id=cid),
            kind=_require_str(d, "kind", case_id=cid),
            tool_name=_require_str(d, "tool_name", case_id=cid),
            tool_input=_require_input(d, case_id=cid),
            mode=_require_choice(d, "mode", tuple(MODE_BY_NAME), case_id=cid),
            interactive=_require_bool(d, "interactive", case_id=cid),
            declared=_require_choice(d, "declared", DECISIONS, case_id=cid),
            declared_arm=_require_choice(
                d, "declared_arm", RULE_ARMS, case_id=cid, default=RULE_NONE,
            ),
            note=_require_str(d, "note", case_id=cid),
            rules=_parse_rules(d.get("rules"), case_id=cid),
            ask_answer=_parse_ask_answer(d.get("ask_answer"), case_id=cid),
            tags=[str(t) for t in d.get("tags", [])],
        )
        _validate_declared(case)
        return case

    def to_json(self) -> dict[str, Any]:
        """The inverse of `from_dict`, for tests that round-trip the dataset."""
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "mode": self.mode,
            "interactive": self.interactive,
            "declared": self.declared,
            "declared_arm": self.declared_arm,
            "note": self.note,
            "tags": list(self.tags),
        }
        if self.rules is not None:
            out["rules"] = {"allow": list(self.rules.allow), "deny": list(self.rules.deny)}
        if self.ask_answer is not None:
            out["ask_answer"] = self.ask_answer
        return out


def simulate_gate(case: SafetyCase) -> PermissionDecision:
    """What the production gate returns for this case, without executing anything.

    Composed from **production** functions in the same order
    `PermissionContext.check` applies them:

    1. rules first (`apply_rules`), because `check()` consults them before the
       mode;
    2. on a rule match, that rule's decision wins outright -- including an
       `allow` rule overriding a mode that would otherwise demand ASK;
    3. otherwise the mode whitelist (`check_permission`);
    4. `_always_allow` is deliberately NOT modelled: it is runtime state the
       scenario does not declare, and a case whose verdict depends on a prior
       "always allow" is not reproducible in a fresh session.

    **The prompt is resolved, not left open.** `check_permission` returns ASK --
    a request -- and what the context does with it depends on the answer, so a
    raw ASK is not this function's output. When an interactive case declares an
    `ask_answer` of "y" or "a", the user approves and `PermissionContext.check`
    returns True; that resolution is applied here so the returned value is the
    verdict the runner will actually observe. Leaving ASK unresolved would make
    an approved `y` case look like a disagreement to the runner's
    predicted-vs-observed check, which it is not.

    Returning ASK therefore means exactly one thing: the request was never
    answered -- a non-interactive context, where the fail-fast turns it into a
    refusal.
    """
    from longline.permissions.rules import apply_rules

    if case.rules is not None:
        ruled = apply_rules(case.rules, case.tool_name, case.tool_input)
        if ruled is not None:
            return ruled

    from longline.permissions.gate import check_permission

    decision = check_permission(case.permission_mode, case.tool_name, case.tool_input)
    if (
        decision == PermissionDecision.ASK
        and case.interactive
        and case.ask_answer in ("y", "a")
    ):
        # The prompt is the last step of `PermissionContext.check`, and it is
        # what turns this ASK into an ALLOW.
        return PermissionDecision.ALLOW
    return decision


def _raw_mode_decision(case: SafetyCase) -> PermissionDecision:
    """The decision before any prompt is consulted.

    `simulate_gate` resolves an approved ASK into an ALLOW, which is what the
    runner will observe -- but the loader needs the pre-prompt value too, to
    tell "the user approved" apart from "no prompt was ever involved". Both
    readings come from production functions; they differ only in whether the
    last step of `PermissionContext.check` has been applied.
    """
    from longline.permissions.gate import check_permission

    return check_permission(case.permission_mode, case.tool_name, case.tool_input)


def predicted_arm(case: SafetyCase) -> str:
    """Which rule arm is expected to produce the decision, as a string."""
    if case.rules is None:
        return RULE_NONE
    from longline.permissions.rules import apply_rules

    ruled = apply_rules(case.rules, case.tool_name, case.tool_input)
    if ruled is None:
        return RULE_NONE
    return RULE_DENY if ruled == PermissionDecision.DENY else RULE_ALLOW


def _validate_declared(case: SafetyCase) -> None:
    """Reject a case whose declared outcome its own scenario contradicts.

    This is the executable form of contract §5.7's "判分同时覆盖规则优先级、
    三种 PermissionMode 和非交互模式". Four things are checked, and each is a
    real way a case file can be wrong without looking wrong:

    1. **The declaration matches the rule layer, when a rule decides.** A rule
       match ends the decision in `PermissionContext.check` -- no prompt is
       consulted and no mode applies -- so the rule's DENY or its ALLOW is the
       outcome, full stop.
    2. **The declared rule arm is the one that fires.** With both `deny` and
       `allow` rules matching, only the deny arm may be credited; a case that
       claimed `allow` there would be a deny-overrides-allow case whose outcome
       came from the allow rule, i.e. a test of nothing.
    3. **A case whose outcome rides on a prompt says so.** Under `default` or
       `acceptEdits` with no rule, the gate's `check_permission` returns ASK --
       a request, not a verdict -- and only the answer turns it into one. Such a
       case must declare `ask_answer`, and the declared outcome must match the
       control flow that answer produces.
    4. **An outcome that no answer can change declares none.** In BYPASS, or on
       a rule match, the prompt is never reached; carrying `ask_answer` there
       would imply the user decided something they never saw.
    """
    decision = simulate_gate(case)
    arm = predicted_arm(case)

    if arm != case.declared_arm:
        raise CaseParseError(
            f"{case.id}: declares rule arm {case.declared_arm!r} but the arm that "
            f"actually fires is {arm!r}. A deny-overrides-allow case whose decision "
            "came from the allow rule proves nothing about priority."
        )

    # The prompt is reached only when no rule matched, the gate asked, and the
    # context can ask. `simulate_gate` has already folded in the answer, so the
    # check has to be made BEFORE its ALLOW resolution -- otherwise an approved
    # case would look as though no prompt was involved at all.
    prompt_decides = (
        case.interactive and arm == RULE_NONE and _raw_mode_decision(case) == PermissionDecision.ASK
    )

    if prompt_decides:
        if case.ask_answer is None:
            raise CaseParseError(
                f"{case.id}: the gate returns ASK under mode={case.mode!r} and no "
                "rule matches, so an interactive prompt is what decides this case; "
                "it must declare 'ask_answer' ('n' refuses, 'y' or 'a' approves) -- "
                "the same scenario is gated or allowed depending on the reply."
            )
        expected = ASK_ANSWERS[case.ask_answer] == "deny"
        if expected != case.declared_gated:
            raise CaseParseError(
                f"{case.id}: declares {case.declared!r} but ask_answer="
                f"{case.ask_answer!r} means the executor "
                f"{'refuses' if expected else 'runs'} the call. The declared "
                "outcome must be the one that answer produces."
            )
        return

    if case.ask_answer is not None:
        raise CaseParseError(
            f"{case.id}: 'ask_answer' is only meaningful when an interactive prompt "
            f"decides the case (got gate={decision.value!r}, "
            f"interactive={case.interactive}, rule_arm={arm!r})"
        )

    actually_gated = decision in (PermissionDecision.DENY, PermissionDecision.ASK)
    if actually_gated != case.declared_gated:
        raise CaseParseError(
            f"{case.id}: declares {case.declared!r} but the production gate returns "
            f"{decision.value!r} for tool={case.tool_name!r} mode={case.mode!r}, "
            f"interactive={case.interactive}. The dataset label would be wrong, not "
            "the gate."
        )


def load_safety_cases(path: Path) -> list[SafetyCase]:
    """Load safety cases from a JSONL file, one per line.

    Rejects a dataset whose label split is not the contract's 15/15. The runner
    could count whatever it was given, but every reported figure names /15 in
    the contract, and a dataset that drifted would move the denominator without
    anything saying so.
    """
    cases: list[SafetyCase] = []
    seen: set[str] = set()
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
        case = SafetyCase.from_dict(d)
        if case.id in seen:
            raise CaseParseError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)

    dangerous = sum(1 for c in cases if c.is_dangerous)
    normal = len(cases) - dangerous
    if dangerous != DANGEROUS_CASES or normal != NORMAL_CASES:
        raise CaseParseError(
            f"{path}: expected {DANGEROUS_CASES} dangerous + {NORMAL_CASES} normal "
            f"cases (contract §5.7), got {dangerous} + {normal}"
        )
    return cases


def group_by(cases: Iterable[SafetyCase], key: str) -> dict[str, list[SafetyCase]]:
    """Bucket cases by a field of `to_row`, preserving file order.

    Used for the per-class confusion matrix: the buckets are the case's `kind`,
    its `mode`, and its rule arm, and a false positive is locatable by reading
    off which bucket it fell into.
    """
    buckets: dict[str, list[SafetyCase]] = {}
    for case in cases:
        value = case.to_row()[key]
        label = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        buckets.setdefault(label, []).append(case)
    return buckets


def _rules_row(rules: PermissionRules | None) -> dict[str, list[str]] | None:
    if rules is None:
        return None
    return {"allow": list(rules.allow), "deny": list(rules.deny)}


def _require_str(d: dict[str, Any], key: str, *, case_id: str) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CaseParseError(f"{case_id}: requires a non-empty string {key!r}, got {value!r}")
    return value


def _require_bool(d: dict[str, Any], key: str, *, case_id: str) -> bool:
    value = d.get(key)
    if not isinstance(value, bool):
        raise CaseParseError(f"{case_id}: requires a bool {key!r}, got {value!r}")
    return value


def _require_choice(
    d: dict[str, Any],
    key: str,
    choices: tuple[str, ...],
    *,
    case_id: str,
    default: str | None = None,
) -> str:
    value = d.get(key, default)
    if value not in choices:
        raise CaseParseError(
            f"{case_id}: {key!r} must be one of {list(choices)}, got {value!r}"
        )
    return str(value)


def _require_input(d: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    value = d.get("tool_input")
    if not isinstance(value, dict):
        raise CaseParseError(f"{case_id}: requires an object 'tool_input', got {value!r}")
    return dict(value)


def _parse_rules(value: Any, *, case_id: str) -> PermissionRules | None:
    """The case's `rules` block, or None for "no settings.json rules".

    None and an empty `PermissionRules()` are distinct inputs: None means
    `PermissionContext(rules=None)`, which skips the rule layer entirely, while
    an empty rule set exercises the fall-through with the layer installed. A
    case testing rule priority must be able to say which it means.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CaseParseError(f"{case_id}: 'rules' must be an object or absent, got {value!r}")
    allow = value.get("allow", [])
    deny = value.get("deny", [])
    if not isinstance(allow, list) or not isinstance(deny, list):
        raise CaseParseError(f"{case_id}: 'rules.allow' and 'rules.deny' must be lists")
    return PermissionRules(allow=[str(r) for r in allow], deny=[str(r) for r in deny])


def _parse_ask_answer(value: Any, *, case_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in ("y", "n", "a"):
        raise CaseParseError(
            f"{case_id}: 'ask_answer' must be one of 'y', 'n', 'a', got {value!r}"
        )
    return value


# The scripted prompt answers and what each means for the executor, spelled out
# because `_prompt_user` treats them as three different things: "a" is not
# "y" -- it also writes the tool into `_always_allow`, and a case that used "a"
# while meaning "y" would silently change the NEXT call in the same context.
# A case may only use "a" on a context it alone owns (see `safety_runner`).
ASK_ANSWERS: dict[str, Literal["allow", "deny", "always"]] = {
    "y": "allow",
    "n": "deny",
    "a": "always",
}

# The answer that makes the executor refuse a call, i.e. the one that produces a
# *gated* outcome at an interactive ASK. Named rather than inlined so the
# validation and the runner cannot disagree about which reply is which.
REFUSING_ANSWER = "n"

__all__ = [
    "ALLOWED",
    "ASK_ANSWERS",
    "DANGEROUS",
    "DANGEROUS_CASES",
    "DECISIONS",
    "GATED",
    "LABELS",
    "MODE_BY_NAME",
    "NORMAL",
    "NORMAL_CASES",
    "REFUSING_ANSWER",
    "RULE_ALLOW",
    "RULE_ARMS",
    "RULE_DENY",
    "RULE_NONE",
    "SafetyCase",
    "group_by",
    "load_safety_cases",
    "predicted_arm",
    "simulate_gate",
]
