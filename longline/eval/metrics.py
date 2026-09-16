"""Statistical口径 (metric contract) for the evaluation suite.

Every proportion reported anywhere in `evals/` must be expressible as a
`Ratio` — numerator, denominator, value and a 95% Wilson score interval
(`evals/README.md` §4.1). Latency is reported as mean / p50 / p95, and paired
A/B comparisons as per-case deltas.

Deliberately dependency-free: no scipy, no numpy. The Wilson interval, the
percentile and the paired delta are all a handful of arithmetic operations, and
keeping them in pure Python means the numbers are auditable and the offline
test suite has no heavy imports.

Key rule: a zero denominator means **not measured**, which is `None` — never
`0.0`. Reporting `0.0` for "we ran zero cases" reads as "0% success" and would
silently corrupt a report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Two-sided 95% normal quantile. Hard-coded so the contract is reproducible
# without depending on a particular stdlib statistics version.
Z_95 = 1.959963984540054


def wilson_ci(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    The Wilson interval is preferred over the Wald (normal approximation)
    interval because it stays inside [0, 1] and does not collapse to a
    zero-width interval when the observed proportion is 0 or 1 — both of which
    happen constantly in a 40-case benchmark.

    Uses the continuity-free form:

        centre = (p + z^2 / 2n) / (1 + z^2 / n)
        half   = (z / (1 + z^2 / n)) * sqrt(p(1-p)/n + z^2 / 4n^2)

    Returns ``(0.0, 1.0)`` when ``total == 0`` (nothing measured). Values are
    returned unrounded; round at the presentation layer only.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")
    if total == 0:
        return (0.0, 1.0)

    n = float(total)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Ratio:
    """A proportion that always carries its own denominator.

    A percentage without a denominator is not a metric, it is a vibe. Every
    rate in `summary.json` is a Ratio so a reader can recompute it from
    `raw.jsonl`.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator < 0:
            raise ValueError(f"denominator must be >= 0, got {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"numerator must be >= 0, got {self.numerator}")
        if self.numerator > self.denominator:
            raise ValueError(
                f"numerator ({self.numerator}) cannot exceed denominator ({self.denominator})"
            )

    @property
    def value(self) -> float | None:
        """The proportion, or None when nothing was measured."""
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def ci95_wilson(self) -> tuple[float, float] | None:
        """The 95% Wilson interval, or None when nothing was measured.

        `wilson_ci` answers `(0.0, 1.0)` for a zero denominator, which is a
        defensible answer to "the interval covering everything" — but as a
        *report* it is a lie of the same size as `0.0`: it says an interval was
        measured, and that it spans 0-100%. Nothing was measured. The rule for
        `value` applies here too, so the interval cannot drift away from it.
        """
        if self.denominator == 0:
            return None
        return wilson_ci(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form: numerator, denominator, value, ci95_wilson.

        The key is always present, null when not measured, so a reader's shape
        does not change between a measured and an unmeasured ratio.
        """
        ci = self.ci95_wilson()
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "ci95_wilson": None if ci is None else [ci[0], ci[1]],
        }

    @classmethod
    def fraction(cls, outcomes: Iterable[object]) -> Ratio:
        """Build a Ratio from an iterable of truthy outcomes."""
        total = 0
        hits = 0
        for outcome in outcomes:
            total += 1
            if outcome:
                hits += 1
        return cls(numerator=hits, denominator=total)


def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile, q in [0, 100].

    Method: **linear interpolation between closest ranks** (equivalent to
    numpy's default `method="linear"` / R's type 7):

        h = (n - 1) * q / 100
        result = v[floor(h)] + (h - floor(h)) * (v[ceil(h)] - v[floor(h)])

    This is NOT the nearest-rank method — for `[1, 2, 3, 4]` nearest-rank p50
    would be 2 while this returns 2.5. The choice is pinned by tests so it
    cannot drift; the contract only mandates "p50 and p95", not the method.

    Returns None for an empty sequence (not measured).
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * q / 100.0
    lo_idx = math.floor(h)
    hi_idx = math.ceil(h)
    if lo_idx == hi_idx:
        return ordered[lo_idx]
    frac = h - lo_idx
    return ordered[lo_idx] + frac * (ordered[hi_idx] - ordered[lo_idx])


def ci95_percentile(values: Sequence[float]) -> tuple[float, float] | None:
    """Distribution-free 95% interval: the 2.5th and 97.5th percentiles.

    Latency is not a proportion, so a Wilson interval does not apply. The
    contract only requires mean / p50 / p95 for the point estimates; this
    helper exists for callers that want a spread alongside them.
    """
    if not values:
        return None
    lo = percentile(values, 2.5)
    hi = percentile(values, 97.5)
    assert lo is not None and hi is not None  # non-empty guaranteed above
    return (lo, hi)


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence."""
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class PairedDelta:
    """Per-case deltas between two aligned runs, plus their mean.

    Alignment is by position in the two sequences, which callers establish by
    sorting both runs by case id (or by passing the explicit id lists, which
    are then checked).
    """

    per_case: tuple[float, ...]
    mean: float | None
    case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        ids = self.case_ids or tuple(str(i) for i in range(len(self.per_case)))
        return {
            "per_case": [
                {"case_id": cid, "delta": delta}
                for cid, delta in zip(ids, self.per_case, strict=True)
            ],
            "mean": self.mean,
            "n_pairs": len(self.per_case),
        }


def paired_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    baseline_ids: Sequence[str] | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> PairedDelta:
    """Per-case `candidate - baseline` deltas for two aligned runs.

    Both sequences must be the same length. A mismatch is raised rather than
    silently truncated: dropping cases would bias the mean delta, which is
    exactly the number a compression / streaming / multi-agent A/B rests on.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired_delta length mismatch: baseline has {len(baseline)}, "
            f"candidate has {len(candidate)}; align by case id before calling"
        )
    if (baseline_ids is None) != (candidate_ids is None):
        raise ValueError("pass both baseline_ids and candidate_ids, or neither")
    if baseline_ids is not None and candidate_ids is not None:
        if len(baseline_ids) != len(candidate_ids):
            raise ValueError("baseline_ids and candidate_ids must be the same length")
        if list(baseline_ids) != list(candidate_ids):
            raise ValueError(
                "paired_delta case id mismatch: the two runs are not aligned "
                f"(baseline_ids={list(baseline_ids)!r}, candidate_ids={list(candidate_ids)!r})"
            )

    deltas = tuple(float(c) - float(b) for b, c in zip(baseline, candidate, strict=True))
    ids = tuple(baseline_ids) if baseline_ids is not None else ()
    return PairedDelta(per_case=deltas, mean=mean(list(deltas)), case_ids=ids)


def percentage_points(baseline: Ratio, candidate: Ratio) -> float | None:
    """Success-rate difference in **percentage points**.

    `84% vs 87%` is `-3 pp`. Never render this as "down 3%" — a relative
    percentage is a different and much larger-sounding number, and the contract
    (`evals/README.md` §4.3) forbids it.

    Returns None if either side was not measured.
    """
    b = baseline.value
    c = candidate.value
    if b is None or c is None:
        return None
    return (c - b) * 100.0


# --- pass@k and pass^k (reliability) ---
#
# Two estimators sharing one binomial-coefficient form, with opposite tails.
# Verified against tau-bench (Yao et al., arXiv:2406.12045), which states:
#
#     pass@k := 1 - E_task[ C(n - c, k) / C(n, k) ]
#     pass^k :=     E_task[ C(c,     k) / C(n, k) ]
#
# where n is the number of trials for a task, c the number that succeeded, and
# the expectation is taken across tasks.
#
# Why both matter: pass@1 (the mean success rate) cannot tell a stable agent
# from a lucky one. Three runs yielding 1/1/1 and 1/0/1 both average 0.667, but
# pass^3 gives 1.0 and 0.0 respectively. tau-bench's headline finding is exactly
# this gap -- gpt-4o scores pass^1 = 61.2% on tau-retail but pass^8 < 25%.
#
# Why the estimator rather than a naive ratio: E[C(c,k)/C(n,k)] is unbiased for
# the probability that k i.i.d. trials all succeed. Computing it as
# "fraction of tasks where c == k" is the same number only when n == k (there
# C(k,k)/C(k,k) = 1 and C(c,k)/C(n,k) = 0 for c < k); for n > k the two differ,
# so the general form is used throughout rather than special-cased.


def _comb(n: int, k: int) -> int:
    """Binomial coefficient; 0 when k > n (the term contributes nothing)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate of "at least one of k trials succeeds".

    Rising in k: a lucky hit counts. Returns 0.0 when successes == 0, and 1.0
    when successes > trials - k (fewer than k failures means every k-subset
    must contain a success).
    """
    if trials <= 0:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    denominator = _comb(trials, k)
    if denominator == 0:  # k > trials: cannot draw k distinct trials
        return 1.0 if successes > 0 else 0.0
    return 1.0 - _comb(trials - successes, k) / denominator


def pass_pow_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate of "all k trials succeed" -- the reliability metric.

    Falling in k: any flakiness is punished. At n == k this degenerates to the
    indicator `successes == k` (the fraction of tasks where every run passed),
    which is the practical reading of `pass^3` for a 3-repeat suite.
    """
    if trials <= 0:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    denominator = _comb(trials, k)
    if denominator == 0:  # k > trials: cannot draw k distinct trials
        return 0.0
    return _comb(successes, k) / denominator


def reliability_ratio(
    successes_per_case: Sequence[int],
    trials_per_case: Sequence[int],
    k: int,
) -> Ratio:
    """`pass^k` as a Ratio over cases, for reporting alongside the pass@1 mean.

    The numerator counts cases whose every trial passed, the denominator the
    cases considered -- so the pair reads as "N of M tasks passed all k runs".
    When n == k that is exactly the count; the general case sums the unbiased
    per-task estimate, which is a count only in expectation. Callers reporting a
    headcount should pass n == k.
    """
    if len(successes_per_case) != len(trials_per_case):
        raise ValueError(
            f"need one trial count per case: {len(successes_per_case)} successes "
            f"vs {len(trials_per_case)} trial counts"
        )
    numerator = sum(
        round(pass_pow_k(c, n, k)) for c, n in zip(successes_per_case, trials_per_case, strict=True)
    )
    return Ratio(numerator=numerator, denominator=len(successes_per_case))
