"""Paired compression A/B: run each case with and without compaction.

=== What this measures (evals/README.md §5.3, plan §4.3) ===

```text
CompressionRatio           = 1 - estimated_tokens_after / estimated_tokens_before
KeyInfoRetention           = 正确回答或使用的关键事实数 / 关键事实总数
PostCompressionSuccessRate = 压缩后最终任务成功数 / 压缩任务总数
SuccessDeltaPP             = candidate_success_rate - baseline_success_rate   (百分点)
```

Token counts are **estimated**, via `estimate_messages_tokens()`; the ambiguity
between the str/4-bytes and JSON/2-bytes tables means the number is a ratio
between two estimates of a growing region, which is what makes it usable even
though the absolute value under-counts (the system prompt and tool schemas are
not counted here).

=== How the two variants differ, and why the baseline runs first ===

```text
baseline  原始完整 history, 禁用压缩
candidate 对同一 history 执行 compact, 再继续任务
```

Both run the SAME history and the SAME continuation task against the same
fixture, so the only thing that differs is whether the transcript was compacted.
The baseline runs first because it is the gate: contract §5.3 says a case whose
baseline fails "不进入压缩退化率分母, 但仍保留在失败报告中". This module
implements that as an explicit `excluded_from_denominator` flag plus a reason,
never as a dropped result -- a silently shrunk denominator is a wrong number
that looks right.

=== How key-fact retention is judged ===

By the follow-up question and the final artifact, **never** by searching the
summary text for the fact's wording. The construct is the standard long-context
probe: plant a fact, then ask a question whose answer IS that fact, and score by
match. That is the original needle-in-a-haystack protocol, and it is what
LongBench and infiniteBench operationalise as well; the difference here is that
the needle is planted in a *conversation* which is then compacted, rather than
in a document which is then truncated.

"The summary contains the path" is not the claim being measured -- "the agent
can still use the path" is. `KeyFact.check` therefore grades the answer the
agent produced, and a fact is lost when its answer does not match. Each fact
carries a `trap` (its most confusable wrong value) so a case can show *which*
fact was lost rather than only how many.

=== Proving compaction actually happened ===

A case that never compacted reports a 0% compression ratio, which reads as a
finding rather than as the wiring bug it is. Two things make that impossible to
miss here:

1. `run_compression_case` raises if the compaction call came back effectively
   unchanged (fewer messages AND fewer estimated tokens are both required).
2. The evidence -- `summariser_calls`, `messages_before/after`,
   `tokens_before/after`, and the compacted prompt the continuation model was
   actually handed -- is recorded on the result's `detail` so the committed
   `raw.jsonl` can prove it after the fact.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.api.token_estimation import estimate_messages_tokens
from longline.compact.compact import compact_messages
from longline.core.events import TextDelta, TurnComplete
from longline.eval.engine_factory import build_engine
from longline.eval.judges import case_passed, judge_case
from longline.eval.metrics import Ratio, paired_delta, percentage_points
from longline.eval.runner import CaseResult, _prepare_sandbox
from longline.models.content_blocks import TextBlock
from longline.models.messages import (
    AssistantMessage,
    Message,
    Usage,
    UserMessage,
    normalize_messages_for_api,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from longline.eval.compression import CompressionCase

# Marks the boundary message in the rendered prompt. `normalize_messages_for_api`
# renders a CompactBoundaryMessage as a user message carrying this prefix, so a
# captured prompt is searchable for it.
COMPRESSION_SUMMARY_TAG = "[Previous conversation summary]"

# The answer the scripted summariser returns. `CompactEvidence.summary` records
# it, so a reader can see the summary the agent was actually given rather than
# having to infer it.
SUMMARISER_ANSWER = "SUMMARY: 已保留全部关键事实、未完成步骤与约束。"

COMPRESSION_OFF = "compression_off"
COMPRESSION_ON = "compression_on"

# Exclusion reason, recorded verbatim on the result. A compaction that fails to
# reduce anything raises out of `_compact` rather than being recorded as a
# reason: a non-compacting case is a data bug, not an excluded observation.
REASON_BASELINE_FAILED = "baseline_failed"


def history_from_spec(spec: list[dict[str, str]]) -> list[Message]:
    """Build the Message list a case's scripted history describes.

    `AssistantMessage` holds typed `TextBlock`s rather than a bare string, so a
    plain `AssistantMessage(content=text)` would crash in `to_api_dict()`. Both
    roles are constructed here rather than inline in the runner so the conversion
    has exactly one definition.

    The history is assistant-first by data contract: it is the middle of a
    session, and a user-first list would make `normalize_messages_for_api`
    prepend a synthetic "Begin." message to satisfy the API's alternation rule.
    """
    messages: list[Message] = []
    for entry in spec:
        content = entry["content"]
        if entry["role"] == "assistant":
            messages.append(AssistantMessage(content=[TextBlock(text=content)]))
        else:
            messages.append(UserMessage(content=content))
    return messages


def _short_hash(value: object) -> str:
    """Stable 8-hex fingerprint of a message payload.

    Used to prove the compacted prompt really reached the continuation model:
    the digest recorded at compaction time and the digest of the prompt the
    model was handed must be the same string.
    """
    payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


async def scripted_summariser(
    summary: str,
    *,
    prompts: list[list[dict[str, Any]]] | None = None,
    **kwargs: Any,
) -> Any:
    """A `call_model` stand-in that emits a fixed summary.

    `compact_messages(call_model)` yields `TextDelta`s and stops at
    `TurnComplete`; nothing else is required of the summariser. Recording the
    prompt it received is what lets a caller assert the summariser was invoked
    once, with the history, rather than merely that some code path ran.
    """
    if prompts is not None:
        prompts.append(list(kwargs.get("messages", [])))
    yield TextDelta(text=summary)
    yield TurnComplete(stop_reason="end_turn", usage=Usage())


@dataclass
class CompactEvidence:
    """Everything needed to prove compaction happened, and to what effect.

    Recorded on `CaseResult.detail["compression"]` and therefore written to
    `raw.jsonl`: the committed artifact can be re-read to check the claim, which
    is the contract's definition of a valid number (`evals/README.md` §3).
    """

    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summariser_calls: int
    summary: str
    compacted: bool
    continuation_prompt_sha256: list[str] = field(default_factory=list)
    # The full prompt the continuation model was handed, rendered. Carried so a
    # reader can confirm the compacted prefix is gone rather than having to
    # trust a digest, and so "the summary was in the prompt" is checkable.
    continuation_prompts: list[list[dict[str, Any]]] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float | None:
        """`1 - after/before`, or None when the "before" count is zero.

        None rather than 0.0 for an unmeasurable ratio, matching
        `metrics.Ratio.value`: reporting 0.0 for "nothing was measured" reads as
        "compression achieved nothing".
        """
        if self.tokens_before <= 0:
            return None
        return 1.0 - self.tokens_after / self.tokens_before

    def to_detail(self) -> dict[str, object]:
        return {
            "compacted": self.compacted,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "compression_ratio": self.compression_ratio,
            "token_units": "estimated",
            "summariser_calls": self.summariser_calls,
            "summary": self.summary,
            "continuation_prompt_sha256": self.continuation_prompt_sha256,
            "continuation_prompts": self.continuation_prompts,
        }


@dataclass
class ContinuationRecorder:
    """Wraps the model under test so the prompt it saw is captured.

    The engine is built by `build_engine`, which the runner monkeypatches; this
    sits between the query loop and the scripted events so that
    `continuation_prompts` is a fact about what was sent, not an inference from
    what happened to be written to disk.
    """

    prompts: list[list[dict[str, Any]]] = field(default_factory=list)

    def record(self, api_messages: list[dict[str, Any]]) -> None:
        self.prompts.append(api_messages)


@dataclass
class CompressionRun:
    """One case, both variants, plus the per-fact retention verdict.

    `key_facts` is carried alongside the ids so a report can name the lost fact
    without re-loading the dataset, which is what makes the failure report
    self-contained.
    """

    case_id: str
    baseline: CaseResult
    candidate: CaseResult
    key_facts: list[Any]
    retained_fact_ids: list[str]
    lost_fact_ids: list[str]
    num_facts: int
    retained_facts: int
    excluded_from_denominator: bool
    exclusion_reason: str | None
    evidence: CompactEvidence | None = None
    baseline_lost_fact_ids: list[str] = field(default_factory=list)
    note: str = ""

    def to_row(self) -> dict[str, object]:
        """Per-case row for the report: the fact trace plus both pass flags."""
        return {
            "case_id": self.case_id,
            "baseline_passed": self.baseline.passed,
            "candidate_passed": self.candidate.passed,
            "excluded_from_denominator": self.excluded_from_denominator,
            "exclusion_reason": self.exclusion_reason,
            "num_facts": self.num_facts,
            "retained_facts": self.retained_facts,
            "retained_fact_ids": self.retained_fact_ids,
            "lost_fact_ids": self.lost_fact_ids,
            "baseline_lost_fact_ids": self.baseline_lost_fact_ids,
            "note": self.note,
        }


@dataclass
class CompressionSummary:
    """The four contract metrics, each with its own denominator.

    `key_info_retention` and `post_compression_success_rate` count different
    things over different denominators (facts vs eligible cases), which is why
    they are separate `Ratio`s rather than one fused score -- the same reason
    the tool-calling metrics were split apart.
    """

    num_cases: int
    excluded_cases: int
    eligible_cases: int
    compression_ratio: float | None
    compression_ratio_per_case: dict[str, float | None]
    key_info_retention: Ratio
    post_compression_success_rate: Ratio
    baseline_success_rate: Ratio
    success_delta_pp: float | None
    paired_tokens: Any
    per_case: list[dict[str, object]] = field(default_factory=list)
    token_units: str = "estimated"
    mean_tokens_before: float | None = None
    mean_tokens_after: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "num_cases": self.num_cases,
            "excluded_cases": self.excluded_cases,
            "eligible_cases": self.eligible_cases,
            "compression_ratio": self.compression_ratio,
            "compression_ratio_per_case": self.compression_ratio_per_case,
            # Says it plainly in the payload, not only in prose: these are
            # estimated tokens from `estimate_messages_tokens()`, not a
            # tokenizer's count (contract §5.3).
            "token_units": self.token_units,
            "mean_tokens_before": self.mean_tokens_before,
            "mean_tokens_after": self.mean_tokens_after,
            "key_info_retention": self.key_info_retention.to_dict(),
            "post_compression_success_rate": self.post_compression_success_rate.to_dict(),
            "baseline_success_rate": self.baseline_success_rate.to_dict(),
            "success_delta_pp": self.success_delta_pp,
            "paired_tokens": self.paired_tokens.to_dict(),
            "per_case": self.per_case,
        }


def measure_history_tokens(history: list[Message]) -> int:
    """Estimated tokens of a history as the API would receive it.

    Measured on `normalize_messages_for_api(history)` rather than on the raw
    message objects, because the normalization is what the model is actually
    sent: it inserts a synthetic opening message for an assistant-first list and
    repairs tool_use/tool_result pairing. Counting the un-normalized list would
    be counting a prompt that is never sent.
    """
    return estimate_messages_tokens(normalize_messages_for_api(history))


async def _compact(
    history: list[Message],
    *,
    case: CompressionCase,
) -> tuple[list[Message], CompactEvidence]:
    """Run the real `compact_messages()` and capture its effect.

    Raises rather than returning an unchanged list when compaction did nothing.
    The plan's acceptance condition is that every case genuinely triggers
    `compact_messages()`; a silent no-op would surface as a 0% compression ratio
    and be read as a finding about the summariser instead of as a wiring bug.
    """
    before_messages = len(history)
    tokens_before = measure_history_tokens(history)
    prompts: list[list[dict[str, Any]]] = []

    compacted = await compact_messages(
        history, lambda **kw: scripted_summariser(case.compaction_note, prompts=prompts, **kw)
    )

    after_messages = len(compacted)
    tokens_after = measure_history_tokens(compacted)

    summary_text = ""
    if compacted and hasattr(compacted[0], "summary"):
        summary_text = getattr(compacted[0], "summary", "") or ""

    evidence = CompactEvidence(
        messages_before=before_messages,
        messages_after=after_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        summariser_calls=len(prompts),
        summary=summary_text,
        compacted=False,
        continuation_prompt_sha256=[_short_hash(m) for m in compacted],
    )

    # Both conditions matter. Fewer messages alone is not enough: a summary can
    # be longer than the turns it replaced, and a case whose "compression"
    # RAISES the prompt size is not a compression measurement.
    if after_messages >= before_messages or tokens_after >= tokens_before:
        raise ValueError(
            f"{case.id}: compact_messages() did not compact: "
            f"{before_messages} -> {after_messages} messages, "
            f"{tokens_before} -> {tokens_after} estimated tokens. "
            "A 0% compression ratio here would be a wiring bug, not a finding."
        )

    evidence.compacted = True
    return compacted, evidence


async def _judge_facts(
    case: CompressionCase,
    sandbox: Path,
) -> tuple[list[str], list[str], int, dict[str, bool]]:
    """Score each key fact against the answer the agent produced.

    Each fact is judged by its own deterministic `check`, run against the
    sandbox artifact -- never by searching the summary text. Returns the
    retained ids, the lost ids, the count retained, and the per-fact verdict so
    a report can name every fact, not only the lost ones.
    """
    retained: list[str] = []
    lost: list[str] = []
    per_fact: dict[str, bool] = {}
    for fact in case.key_facts:
        try:
            ok = bool(judge_case(str(fact.check["fn"]), sandbox, fact.check.get("args") or {}))
        except Exception:  # a broken fact check is a lost fact, not a crash
            ok = False
        per_fact[fact.id] = ok
        (retained if ok else lost).append(fact.id)
    return retained, lost, len(retained), per_fact


async def _run_variant(
    case: CompressionCase,
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
    variant: str,
    recorder: ContinuationRecorder,
) -> tuple[CaseResult, CompactEvidence | None, list[str], list[str]]:
    """Run one variant of one case against its own fresh sandbox.

    The key facts are judged HERE, against the sandbox the agent actually
    produced, before `finally` removes it. Judging them afterwards from a
    re-created fixture copy would be judging a directory the agent never
    touched -- a check that can only ever report "everything was retained",
    which is the vacuous-pass failure mode this suite exists to avoid.

    Returns the graded result, the compaction evidence (compression_on only),
    and the retained / lost fact ids.
    """
    sandbox = _prepare_sandbox(fixtures_dir, case.fixture, case_id=case.id)
    try:
        messages = history_from_spec(case.history)
        evidence: CompactEvidence | None = None
        if variant == COMPRESSION_ON:
            messages, evidence = await _compact(messages, case=case)

        engine = build_engine(
            sandbox=sandbox, model=model, api_key=api_key, tool_profile="core",
        )

        # What the model is actually sent. Captured before `submit` because the
        # point of the experiment is the prompt, not the reply.
        recorder.record(normalize_messages_for_api(messages))

        async for _event in engine.submit(case.continuation_task, max_turns=case.max_turns):
            pass

        result = CaseResult(
            case_id=case.id,
            case_type="compression",
            passed=False,
            tags=list(case.tags),
            variant=variant,
        )
        passed, check_detail = case_passed(case.checks, Path(sandbox), mode=case.checks_mode)
        result.passed = passed
        result.detail = {
            "checks_mode": case.checks_mode,
            "checks": check_detail,
            "compaction_variant": variant,
        }

        # Judged inside the live sandbox (see the docstring).
        retained, lost, _, per_fact = await _judge_facts(case, Path(sandbox))
        result.detail["key_fact_verdicts"] = per_fact

        if evidence is not None:
            evidence.continuation_prompts = list(recorder.prompts)
            evidence.continuation_prompt_sha256 = [
                _short_hash(m) for m in recorder.prompts[-1]
            ]
            result.detail["compression"] = evidence.to_detail()
        return result, evidence, retained, lost
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


async def run_compression_case(
    case: CompressionCase,
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
) -> CompressionRun:
    """Run one case's baseline and candidate, in that order.

    The baseline runs first because it is the gate (contract §5.3): a case whose
    baseline cannot pass is excluded from the degradation denominator -- but the
    exclusion is recorded as a flag and a reason on the returned run, so the
    case stays visible in the failure report. Silently dropping it would shrink
    the denominator and inflate `PostCompressionSuccessRate`.
    """
    baseline, _, baseline_retained, baseline_lost = await _run_variant(
        case, model=model, api_key=api_key, fixtures_dir=fixtures_dir,
        variant=COMPRESSION_OFF, recorder=ContinuationRecorder(),
    )
    candidate, evidence, _, _ = await _run_variant(
        case, model=model, api_key=api_key, fixtures_dir=fixtures_dir,
        variant=COMPRESSION_ON, recorder=ContinuationRecorder(),
    )

    if not baseline.passed:
        return CompressionRun(
            case_id=case.id,
            baseline=baseline,
            candidate=candidate,
            key_facts=list(case.key_facts),
            retained_fact_ids=[],
            lost_fact_ids=[f.id for f in case.key_facts],
            num_facts=len(case.key_facts),
            retained_facts=0,
            excluded_from_denominator=True,
            exclusion_reason=REASON_BASELINE_FAILED,
            evidence=evidence,
            baseline_lost_fact_ids=baseline_lost,
            note=(
                "baseline failed; excluded from the compression-degradation "
                "denominator but kept in the failure report"
            ),
        )

    return CompressionRun(
        case_id=case.id,
        baseline=baseline,
        candidate=candidate,
        key_facts=list(case.key_facts),
        retained_fact_ids=baseline_retained,
        lost_fact_ids=baseline_lost,
        num_facts=len(case.key_facts),
        retained_facts=len(baseline_retained),
        excluded_from_denominator=False,
        exclusion_reason=None,
        evidence=evidence,
    )


async def run_compression_suite(
    cases: Iterable[CompressionCase],
    *,
    model: str,
    api_key: str,
    fixtures_dir: Path,
) -> list[CompressionRun]:
    """Run every case serially, baseline then candidate.

    Serial by contract (`evals/README.md` §4.6): parallel quality runs trip API
    rate limits and contaminate each other's latency, and the A/B must use the
    same machine and fixture for both sides.
    """
    runs: list[CompressionRun] = []
    for case in cases:
        runs.append(
            await run_compression_case(
                case, model=model, api_key=api_key, fixtures_dir=fixtures_dir,
            )
        )
    return runs


def aggregate_compression(runs: list[CompressionRun]) -> CompressionSummary:
    """Collapse per-case runs into the four contract metrics.

    Reading order matters, and it is the same trap the tool-calling metrics were
    split to avoid: `key_info_retention` counts **facts**, the two success rates
    count **eligible cases**. They are summed separately on purpose.

    Eligibility is the baseline gate. Both success rates use the SAME eligible
    set, because comparing a rate over all cases against a rate over the
    baseline-passing subset would fold a sampling difference into
    `SuccessDeltaPP` -- the exact confound the paired design exists to remove.
    """
    eligible = [r for r in runs if not r.excluded_from_denominator]

    # Facts: 5 per ELIGIBLE case. Including an excluded case's facts would let a
    # case whose baseline never worked depress the retention rate.
    facts_total = sum(r.num_facts for r in eligible)
    facts_retained = sum(r.retained_facts for r in eligible)

    ratios: dict[str, float | None] = {}
    for run in runs:
        ev = run.evidence
        ratios[run.case_id] = None if ev is None else ev.compression_ratio

    per_case_ratios = [v for v in ratios.values() if v is not None]
    mean_ratio = (
        sum(per_case_ratios) / len(per_case_ratios) if per_case_ratios else None
    )

    token_pairs = [
        (r.evidence.tokens_before, r.evidence.tokens_after)
        for r in runs
        if r.evidence is not None
    ]
    paired = paired_delta(
        [float(b) for b, _ in token_pairs],
        [float(a) for _, a in token_pairs],
        baseline_ids=[r.case_id for r in runs if r.evidence is not None],
        candidate_ids=[r.case_id for r in runs if r.evidence is not None],
    )

    baseline_ratio = Ratio.fraction(r.baseline.passed for r in eligible)
    candidate_ratio = Ratio.fraction(r.candidate.passed for r in eligible)

    return CompressionSummary(
        num_cases=len(runs),
        excluded_cases=len(runs) - len(eligible),
        eligible_cases=len(eligible),
        compression_ratio=mean_ratio,
        compression_ratio_per_case=ratios,
        key_info_retention=Ratio(facts_retained, facts_total),
        post_compression_success_rate=candidate_ratio,
        baseline_success_rate=baseline_ratio,
        success_delta_pp=percentage_points(baseline_ratio, candidate_ratio),
        paired_tokens=paired,
        per_case=[r.to_row() for r in runs],
        mean_tokens_before=(
            sum(b for b, _ in token_pairs) / len(token_pairs) if token_pairs else None
        ),
        mean_tokens_after=(
            sum(a for _, a in token_pairs) / len(token_pairs) if token_pairs else None
        ),
    )


__all__ = [
    "COMPRESSION_OFF",
    "COMPRESSION_ON",
    "COMPRESSION_SUMMARY_TAG",
    "CompactEvidence",
    "CompressionRun",
    "CompressionSummary",
    "ContinuationRecorder",
    "aggregate_compression",
    "history_from_spec",
    "measure_history_tokens",
    "run_compression_case",
    "run_compression_suite",
    "scripted_summariser",
]
