"""WHETHER a difference is real: the paired significance test and the debounce.

Three gates decide that an arm regressed, and all three must fire (docs/measurement.md
"The flagging rule"):

1. EFFECT SIZE -- :func:`bench.regression.threshold_ms`, derived from the cell's own
   measured spread.
2. SIGNIFICANCE -- :func:`paired_verdict` below: an exact Wilcoxon signed-rank test
   over the PER-ROUND differences. Paired, because a harness that pays for pairing
   is throwing that away with an unpaired test; exact,
   because the round counts here are single digits, where a normal approximation is
   an assumption rather than a shortcut.
3. DEBOUNCE -- :func:`classify_flags`: a violation is a regression only once it has
   PERSISTED. One thermally unlucky run on a box with logged power capping is not a
   kernel change.

No scipy. The exact null distribution of the signed-rank statistic is a subset sum
over at most a few thousand assignments at these sizes, and a dependency that exists
to compute one table is a dependency that has to be installed on every machine that
wants to read the ledger.
"""
from __future__ import annotations

import math
from itertools import product

#: Above this many non-zero differences the exact enumeration is 2**n terms and the
#: normal approximation is accurate to well past the third decimal of p. STRUCTURAL,
#: not a tuning knob: it is the point where the two computations agree.
EXACT_MAX_N = 20

#: The paired test's alpha. Relaxed from ASV's 0.002 because a landing run pairs
#: single-digit rounds, not twenty samples; stated once, here, rather than passed in
#: at each call site where it could drift.
ALPHA = 0.01

#: DERIVED FROM ALPHA, not chosen. The smallest two-sided exact p-value reachable
#: with `n` non-zero differences is `2 / 2**n` -- every sign the same way, counted in
#: both tails -- so below `ceil(log2(2 / ALPHA))` rounds NO outcome can be called
#: significant and the test is not underpowered but undefined. At ALPHA = 0.01 that
#: is 8 rounds, which is what a landing-tier run has to collect; a shorter run is
#: reported as `insufficient_data`, never as "no regression".
MIN_ROUNDS = math.ceil(math.log2(2 / ALPHA))


def _signed_rank_statistic(diffs) -> tuple[float, list[float]]:
    """W+ (the sum of ranks of the positive differences) and the ranks it used.

    Zero differences are DISCARDED, which is Wilcoxon's own treatment: a pair that
    did not move carries no evidence about the direction of the ones that did. Ties
    among the absolute differences take mid-ranks.
    """
    nonzero = [d for d in diffs if d != 0]
    order = sorted(range(len(nonzero)), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * len(nonzero)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    return w_plus, ranks


def _exact_two_sided_p(w_plus: float, ranks: list[float]) -> float:
    """P(|W+ - mean| >= |observed - mean|) under the null that each sign is a fair coin.

    Enumerated over the actual ranks, so mid-ranks from ties are handled by
    construction rather than by a correction term.
    """
    total = sum(ranks)
    mean = total / 2
    observed = abs(w_plus - mean)
    hit = 0
    trials = 0
    for signs in product((0, 1), repeat=len(ranks)):
        trials += 1
        s = sum(r for r, take in zip(ranks, signs) if take)
        if abs(s - mean) >= observed - 1e-12:
            hit += 1
    return hit / trials


def _normal_two_sided_p(w_plus: float, ranks: list[float]) -> float:
    """The large-sample approximation, with the tie correction the mid-ranks require."""
    n = len(ranks)
    mean = n * (n + 1) / 4
    counts: dict[float, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    ties = sum(t ** 3 - t for t in counts.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return 1.0
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def paired_verdict(diffs, *, alpha: float = ALPHA) -> dict:
    """Is the per-round difference between two arms distinguishable from zero?

    `diffs` are the PER-ROUND differences (arm B minus arm A within one round), the
    quantity the interleaved design produces. A drift step inside a round is common
    to both arms and cancels here, which is the whole reason the pairing was paid
    for.

    Returns `verdict` in `{insufficient_data, same, b_slower, b_faster}` with the
    p-value and the n it was computed on. `insufficient_data` is a distinct answer
    from `same`: the first says the design cannot decide, the second says it did.
    """
    diffs = list(diffs)
    if len(diffs) < MIN_ROUNDS:
        return {"verdict": "insufficient_data", "p": None, "n": len(diffs),
                "reason": f"{len(diffs)} rounds < MIN_ROUNDS={MIN_ROUNDS}"}
    w_plus, ranks = _signed_rank_statistic(diffs)
    if not ranks:
        return {"verdict": "same", "p": 1.0, "n": 0,
                "reason": "every paired difference was exactly zero"}
    p = (_exact_two_sided_p(w_plus, ranks) if len(ranks) <= EXACT_MAX_N
         else _normal_two_sided_p(w_plus, ranks))
    if p > alpha:
        verdict = "same"
    else:
        verdict = "b_slower" if sum(diffs) > 0 else "b_faster"
    return {"verdict": verdict, "p": p, "n": len(ranks), "w_plus": w_plus,
            "exact": len(ranks) <= EXACT_MAX_N}


def classify_flags(violations) -> str:
    """PyTorch HUD's debounce, over per-commit threshold violations in commit order.

    `violations` is oldest-first; the last element is HEAD.

    * `regression` -- the trailing run of violations is at least 2 long. A real
      regression persists; a thermal outlier does not.
    * `suspicious` -- a run of at least 3 exists somewhere in the history but not at
      HEAD. Something happened and then stopped happening, which is worth a look and
      is not a merge blocker.
    * `insufficient_data` -- fewer than 3 points on this fingerprint.
    * `no_regression` -- otherwise.
    """
    flags = [bool(v) for v in violations]
    if len(flags) < 3:
        return "insufficient_data"
    trailing = 0
    for v in reversed(flags):
        if not v:
            break
        trailing += 1
    if trailing >= 2:
        return "regression"
    run = best = 0
    for v in flags:
        run = run + 1 if v else 0
        best = max(best, run)
    return "suspicious" if best >= 3 else "no_regression"
