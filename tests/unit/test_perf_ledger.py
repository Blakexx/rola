"""The judgement over measurements: the paired test and the debounce.

Every check here plants the violation its instrument exists to catch and requires
the instrument to fire. A gate that cannot be shown failing is treated as absent,
so "it passed" is not evidence about either until the planted case is also shown
rejected.

No device: the test is arithmetic, and the debounce is a run-length rule.
"""
from __future__ import annotations

import pytest

from bench.regression import classify
from bench.stats import ALPHA, MIN_ROUNDS, classify_flags, paired_verdict

# ---------------------------------------------------------------------------
# THE PAIRED TEST (I-4)
# ---------------------------------------------------------------------------


def test_the_paired_test_detects_a_planted_shift_and_clears_an_unshifted_pair():
    """NON-VACUITY: the same n, one arm shifted and one not."""
    shifted = paired_verdict([0.05] * MIN_ROUNDS)
    assert shifted["verdict"] == "b_slower"
    assert shifted["p"] <= ALPHA and shifted["exact"]

    #: The planted null: differences that alternate sign with equal magnitude carry
    #: no evidence at all, and the test must say so rather than pick a direction.
    assert paired_verdict([0.05, -0.05] * (MIN_ROUNDS // 2))["verdict"] == "same"


def test_the_paired_test_sees_direction():
    assert paired_verdict([-0.05] * MIN_ROUNDS)["verdict"] == "b_faster"


def test_the_round_floor_is_derived_from_alpha_and_is_binding():
    """The floor is not a chosen number: one round fewer makes EVERY outcome
    non-significant, which is the definition of an undefined test rather than a
    strict one. Both sides are asserted, so the floor cannot drift off its alpha.
    """
    from bench.stats import _exact_two_sided_p, _signed_rank_statistic

    assert paired_verdict([0.05] * MIN_ROUNDS)["p"] <= ALPHA
    #: One round short, computed directly: the STRONGEST possible evidence (every
    #: difference the same way) still cannot reach alpha, which is why the harness
    #: refuses to report a verdict there at all.
    short = [0.05] * (MIN_ROUNDS - 1)
    w, ranks = _signed_rank_statistic(short)
    assert _exact_two_sided_p(w, ranks) > ALPHA
    assert paired_verdict(short)["verdict"] == "insufficient_data"


def test_a_shift_smaller_than_the_noise_is_not_called():
    """The instrument must not manufacture precision: a shift buried in scatter of
    its own size is `same`, not a quiet pass."""
    diffs = [0.01, -0.4, 0.35, -0.30, 0.28, -0.25, 0.22, -0.18, 0.30]
    assert paired_verdict(diffs)["verdict"] == "same"


def test_too_few_rounds_is_reported_as_undecided_never_as_no_regression():
    got = paired_verdict([0.05, 0.05])
    assert got["verdict"] == "insufficient_data"
    assert got["p"] is None


def test_the_exact_and_approximate_tails_agree_where_they_meet():
    """The EXACT_MAX_N boundary is structural, not a knob: the two computations must
    describe the same distribution at the size where the switch happens."""
    from bench.stats import _normal_two_sided_p, _signed_rank_statistic

    diffs = [0.3, -0.1, 0.25, 0.4, -0.05, 0.2, 0.35, 0.15, -0.2, 0.45,
             0.05, 0.5, -0.15, 0.28, 0.33]
    w, ranks = _signed_rank_statistic(diffs)
    exact = paired_verdict(diffs)["p"]
    assert exact == pytest.approx(_normal_two_sided_p(w, ranks), abs=0.02)


# ---------------------------------------------------------------------------
# THE DEBOUNCE (I-5)
# ---------------------------------------------------------------------------

def test_one_unlucky_run_is_not_a_regression_and_two_are():
    """PLANTED VIOLATION on both sides of the rule: the single thermal outlier the
    debounce exists to absorb, and the persistent pair it must still catch."""
    assert classify_flags([False, False, False, True]) == "no_regression"
    assert classify_flags([False, False, True, True]) == "regression"


def test_a_healed_run_is_suspicious_not_silent():
    assert classify_flags([True, True, True, False, False]) == "suspicious"


def test_a_short_history_is_undecided():
    assert classify_flags([True, True]) == "insufficient_data"


# ---------------------------------------------------------------------------
# THE THREE GATES TOGETHER
# ---------------------------------------------------------------------------

def _run(values):
    """A judged row: `classify` reads a row's samples and nothing else."""
    return {"values": values}


def _history(values, n=5):
    return [_run(values) for _ in range(n)]


def test_all_three_gates_must_fire_before_anything_is_called_a_regression():
    history = _history([1.00, 1.01, 0.99, 1.00])
    #: A candidate far above the derived limit, persistent, and significant.
    slow = _run([1.60, 1.61, 1.59, 1.60])
    got = classify(history + [slow], slow, paired_diffs=[0.6] * MIN_ROUNDS)
    assert got["verdict"] == "regression"
    assert got["median_ms"] > got["limit_ms"]


def test_an_effect_without_significance_is_flagged_but_not_confirmed():
    """The planted case the significance gate exists for: the median moved, the
    per-round differences do not agree that it did."""
    history = _history([1.00, 1.01, 0.99, 1.00])
    slow = _run([1.60, 1.61, 1.59, 1.60])
    got = classify(history + [slow], slow, paired_diffs=[0.6, -0.6, 0.6, -0.6])
    assert got["verdict"] == "flagged_not_confirmed"


def test_an_unchanged_kernel_does_not_fail_its_own_gate():
    history = _history([1.00, 1.01, 0.99, 1.00])
    same = _run([1.00, 1.00, 1.01, 0.99])
    assert classify(history, same, paired_diffs=[0.0] * 6)["verdict"] == "no_regression"


def test_a_series_too_short_to_judge_says_so():
    short = _history([1.0, 1.0], n=2)
    got = classify(short, short[-1], paired_diffs=[0.5] * 5)
    assert got["verdict"] == "insufficient_data"
