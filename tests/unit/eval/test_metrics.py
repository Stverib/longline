"""Unit tests for longline/eval/metrics.py — the statistical口径 module.

This is the contract module: every proportion in a report must be expressible
as numerator/denominator/value/Wilson CI, and a zero denominator must render as
"not measured" (None), never as 0.0 ("0% success").
"""

from __future__ import annotations

import math

import pytest

from longline.eval.metrics import (
    Ratio,
    ci95_percentile,
    mean,
    paired_delta,
    pass_at_k,
    pass_pow_k,
    percentage_points,
    percentile,
    reliability_ratio,
    wilson_ci,
)

# --- wilson_ci ---


def test_wilson_ci_zero_denominator_is_full_interval() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_all_successes_contains_one() -> None:
    lo, hi = wilson_ci(4, 4)
    # Wilson (unlike Wald) never collapses to a degenerate [1.0, 1.0].
    assert 0.0 < lo < 1.0
    assert hi == pytest.approx(1.0)


def test_wilson_ci_no_successes_contains_zero() -> None:
    lo, hi = wilson_ci(0, 4)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi < 1.0


def test_wilson_ci_known_value_3_of_10() -> None:
    # n=10, k=3, z=1.959964 -> (0.107791, 0.603222); centre = 0.3555065
    lo, hi = wilson_ci(3, 10)
    assert lo == pytest.approx(0.107791, abs=1e-6)
    assert hi == pytest.approx(0.603222, abs=1e-6)
    assert (lo + hi) / 2 == pytest.approx(0.3555065, abs=1e-6)


def test_wilson_ci_known_value_lower_bound_1_of_10() -> None:
    # n=10, k=1, z=1.959964 -> (0.017876, 0.404150); centre = 0.211013
    lo, hi = wilson_ci(1, 10)
    assert lo == pytest.approx(0.017876, abs=1e-6)
    assert hi == pytest.approx(0.404150, abs=1e-6)
    assert (lo + hi) / 2 == pytest.approx(0.211013, abs=1e-6)


def test_wilson_ci_brackets_point_estimate() -> None:
    for n in (5, 20, 50):
        for k in range(n + 1):
            lo, hi = wilson_ci(k, n)
            p = k / n
            assert lo <= p <= hi
            assert lo < hi  # strictly non-degenerate


def test_wilson_ci_hand_computed_closed_form() -> None:
    """Pin the formula itself, not just a table lookup."""
    z = 1.959963984540054
    n, k = 25, 18
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = wilson_ci(k, n)
    assert lo == pytest.approx(centre - half, abs=1e-12)
    assert hi == pytest.approx(centre + half, abs=1e-12)


def test_wilson_ci_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_ci(5, 4)
    with pytest.raises(ValueError):
        wilson_ci(-1, 4)


def test_wilson_ci_widens_as_n_shrinks() -> None:
    narrow = wilson_ci(50, 100)
    wide = wilson_ci(5, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# --- Ratio ---


def test_ratio_value_is_none_when_denominator_zero() -> None:
    r = Ratio(numerator=0, denominator=0)
    assert r.value is None
    # Not (0.0, 1.0): a full-width interval would read as "measured, spans
    # everything", which is exactly as wrong as reporting 0.0.
    assert r.ci95_wilson() is None


def test_ratio_value_and_ci_when_denominator_positive() -> None:
    r = Ratio(numerator=3, denominator=10)
    assert r.value == pytest.approx(0.3)
    lo, hi = r.ci95_wilson()
    assert lo == pytest.approx(0.107791, abs=1e-6)
    assert hi == pytest.approx(0.603222, abs=1e-6)


def test_ratio_is_frozen_and_hashable() -> None:
    from dataclasses import FrozenInstanceError

    r = Ratio(1, 2)
    with pytest.raises(FrozenInstanceError):
        r.numerator = 5  # type: ignore[misc]
    assert hash(r) == hash(Ratio(1, 2))


def test_ratio_to_dict_shape_is_recomputable() -> None:
    d = Ratio(2, 4).to_dict()
    assert d["numerator"] == 2
    assert d["denominator"] == 4
    assert d["value"] == pytest.approx(0.5)
    assert d["ci95_wilson"] == pytest.approx([wilson_ci(2, 4)[0], wilson_ci(2, 4)[1]])


def test_ratio_to_dict_zero_denominator_uses_null_value() -> None:
    d = Ratio(0, 0).to_dict()
    assert d["numerator"] == 0
    assert d["denominator"] == 0
    assert d["value"] is None
    # The interval must be null for the same reason the value is, and the key
    # must stay present so a dashboard's shape does not change with the data.
    assert "ci95_wilson" in d
    assert d["ci95_wilson"] is None


def test_ratio_to_dict_zero_denominator_value_and_ci_agree() -> None:
    """`value` and `ci95_wilson` are null together, never one without the other."""
    d = Ratio(0, 0).to_dict()
    assert (d["value"] is None) == (d["ci95_wilson"] is None)


def test_ratio_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        Ratio(-1, 10)


def test_ratio_fraction_helper() -> None:
    results = [True, True, False, True]
    r = Ratio.fraction(bool(x) for x in results)
    assert r.numerator == 3
    assert r.denominator == 4


def test_ratio_fraction_of_empty_is_unmeasured() -> None:
    r = Ratio.fraction(iter([]))
    assert r.denominator == 0
    assert r.value is None


# --- percentile ---


def test_percentile_empty_is_none() -> None:
    assert percentile([], 50) is None


def test_percentile_single_value() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_uses_linear_interpolation() -> None:
    # Documented method: linear interpolation between closest ranks
    # (numpy "linear" / R type 7), NOT nearest-rank.
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 0) == pytest.approx(1.0)
    assert percentile(data, 25) == pytest.approx(1.75)
    assert percentile(data, 50) == pytest.approx(2.5)
    assert percentile(data, 75) == pytest.approx(3.25)
    assert percentile(data, 100) == pytest.approx(4.0)


def test_percentile_95_of_ten_is_between_last_two() -> None:
    # rank = 0.95 * (10 - 1) = 8.55 -> between 9.0 (idx 8) and 10.0 (idx 9)
    data = [float(i) for i in range(1, 11)]
    assert percentile(data, 95) == pytest.approx(9.55)


def test_percentile_is_order_insensitive() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == pytest.approx(2.5)


def test_percentile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)
    with pytest.raises(ValueError):
        percentile([1.0], -1)


def test_ci95_percentile_pair_is_ordered() -> None:
    data = [float(i) for i in range(1, 101)]
    lo, hi = ci95_percentile(data)  # type: ignore[misc]
    assert lo == pytest.approx(3.475)
    assert hi == pytest.approx(97.525)


def test_ci95_percentile_empty_is_none() -> None:
    assert ci95_percentile([]) is None


# --- mean ---


def test_mean_empty_is_none() -> None:
    assert mean([]) is None


def test_mean_basic() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


# --- paired_delta ---


def test_paired_delta_basic() -> None:
    baseline = [0.0, 10.0, 20.0]
    candidate = [5.0, 5.0, 5.0]
    d = paired_delta(baseline, candidate)
    assert tuple(d.per_case) == (5.0, -5.0, -15.0)
    assert d.mean == pytest.approx(-5.0)


def test_paired_delta_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_delta([1.0, 2.0], [1.0])


def test_paired_delta_rejects_case_id_mismatch() -> None:
    with pytest.raises(ValueError, match="case id"):
        paired_delta([1.0, 2.0], [1.0, 2.0], baseline_ids=["a", "b"], candidate_ids=["a", "c"])


def test_paired_delta_of_empty_is_unmeasured() -> None:
    d = paired_delta([], [])
    assert list(d.per_case) == []
    assert d.mean is None


def test_paired_delta_accepts_matching_ids() -> None:
    d = paired_delta([1.0, 2.0], [3.0, 2.0], baseline_ids=["a", "b"], candidate_ids=["a", "b"])
    assert tuple(d.per_case) == (2.0, 0.0)


def test_paired_delta_to_dict_shape() -> None:
    d = paired_delta([0.0, 10.0], [10.0, 10.0], baseline_ids=["a", "b"], candidate_ids=["a", "b"])
    out = d.to_dict()
    assert out["per_case"] == [{"case_id": "a", "delta": 10.0}, {"case_id": "b", "delta": 0.0}]
    assert out["mean"] == pytest.approx(5.0)
    assert out["n_pairs"] == 2


# --- percentage_points ---


def test_percentage_points_is_a_pp_difference() -> None:
    # 84% vs 87% is -3 pp, never "down 3%". Signature is (baseline, candidate).
    assert percentage_points(Ratio(87, 100), Ratio(84, 100)) == pytest.approx(-3.0)
    assert percentage_points(Ratio(84, 100), Ratio(87, 100)) == pytest.approx(3.0)


def test_percentage_points_is_none_when_either_side_unmeasured() -> None:
    assert percentage_points(Ratio(0, 0), Ratio(3, 4)) is None
    assert percentage_points(Ratio(3, 4), Ratio(0, 0)) is None


# --- pass@k / pass^k (tau-bench estimators, Yao et al. arXiv:2406.12045) ---
#
# These pin the exact numbers so the formulas cannot drift. The values are
# computed by hand from C(c,k)/C(n,k); see the derivation in each docstring.


def test_pass_pow_k_at_n_equals_k_is_all_runs_passed() -> None:
    # C(c,3)/C(3,3): 1 when c == 3, else 0. This is the practical pass^3.
    assert pass_pow_k(3, 3, 3) == 1.0
    assert pass_pow_k(2, 3, 3) == 0.0
    assert pass_pow_k(0, 3, 3) == 0.0


def test_pass_pow_k_general_form_is_not_the_indicator() -> None:
    # n > k: C(3,2)/C(5,2) = 3/10. A naive "fraction of tasks with c == k"
    # would say 0 here; the unbiased estimator says 0.3.
    assert pass_pow_k(3, 5, 2) == pytest.approx(0.3)
    assert pass_pow_k(2, 4, 2) == pytest.approx(1 / 6)


def test_pass_at_k_general_form() -> None:
    # 1 - C(n-c,k)/C(n,k): 1 - C(2,2)/C(5,2) = 1 - 2/10 = 0.8? No:
    # 1 - C(5-3,2)/C(5,2) = 1 - C(2,2)/10 = 1 - 0.1 = 0.9
    assert pass_at_k(3, 5, 2) == pytest.approx(0.9)
    assert pass_at_k(0, 4, 2) == 0.0
    assert pass_at_k(3, 4, 2) == 1.0


def test_pass_at_k_and_pass_pow_k_are_dual() -> None:
    # Same binomial form, opposite tails: pass@k rises with success, pass^k
    # falls. For n == k they bracket: any c strictly between 0 and n gives
    # pass@k == 1 and pass^k == 0.
    for n in (2, 3, 4):
        for c in range(1, n):
            assert pass_at_k(c, n, n) == 1.0, (c, n)
            assert pass_pow_k(c, n, n) == 0.0, (c, n)


def test_pass_estimators_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        pass_at_k(3, 0, 1)
    with pytest.raises(ValueError):
        pass_pow_k(3, 3, 0)
    with pytest.raises(ValueError):
        pass_pow_k(4, 3, 1)


def test_reliability_ratio_counts_all_k_passers() -> None:
    # Three cases with 3 trials each: 3/3, 2/3, 0/3 -> only the first passes all.
    r = reliability_ratio([3, 2, 0], [3, 3, 3], 3)
    assert r.numerator == 1
    assert r.denominator == 3
    assert r.value == pytest.approx(1 / 3)


def test_reliability_ratio_distinguishes_stable_from_flaky() -> None:
    # This is the whole point of the metric: 1/1/1 vs 1/0/1 both mean 0.667
    # under pass@1, but only the first is reliable.
    stable = reliability_ratio([3], [3], 3)
    flaky = reliability_ratio([2], [3], 3)
    assert stable.value == 1.0
    assert flaky.value == 0.0


def test_reliability_ratio_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        reliability_ratio([1, 2], [3], 3)
