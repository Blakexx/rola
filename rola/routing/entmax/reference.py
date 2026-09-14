# CANONICAL fp64 reference for the entmax/sparsemax routing primitive.
#
# THE REFERENCE IS THE THIRD PARTY'S. `entmax` (PyPI, DeepSPIN) is written by the
# authors of the papers this primitive comes from -- Peters, Niculae & Martins 2019,
# "Sparse Sequence-to-Sequence Models" for the forward, Blondel et al. 2019 for the
# backward -- and this module is the thin, audited adapter between their function and
# the shapes this repository speaks.
#
# WHY THIS IS A REFERENCE AND OUR PREVIOUS ONE WAS ONLY A SECOND OPINION. The
# matching-the-naive rule says a gate that compares an implementation against code we
# also wrote proves self-consistency and nothing else. The hand-rolled version this
# replaces was 189 lines of our own numerics -- an 80-step bisection with a
# hand-proved bracket, a second sort-based algorithm existing only to cross-check the
# first, and a hand-derived analytic VJP -- all correct, and all ours. Reference code
# we did not write is the strongest available answer, and here it is also the more
# exact one: `entmax15` and `sparsemax` are CLOSED FORMS (sort/topk), not iterations,
# so the threshold is solved rather than approached.
#
# MEASURED before the swap (workflows/refactor/p07_audit.md section 3.2, fp64, over
# shapes (64,16) (8,256) (128,3) x logit scales 0.5/3/30): our forward agreed with
# theirs to 1.42e-14, our analytic VJP to 6.66e-16 at alpha=1.5 and BIT-IDENTICALLY at
# alpha=2, with 0 support-set mismatches in any cell. Their bisection lands within
# 1e-15 of their own exact path; our 80-step one landed 1.42e-14 away. So this swap
# tightens the reference while removing our authorship from it.
#
# PURE TORCH, and that is a contract, not an observation: the wheel is pure python
# with torch as its only dependency. Third-party CUDA is refused as a reference on the
# standing rule -- a reference must be readable, not fast.
#
# THE THREE STAGES the rest of the repository is written against are unchanged, and
# are recovered from the third party's output rather than recomputed:
#   Stage 1 - the threshold tau(z, alpha)   -- inverted from p, exactly (see entmax_tau)
#   Stage 2 - support = {i : p_i > 0}       -- read off p
#   Stage 3 - the values p                  -- theirs
#
# NOT ON THE PRODUCTION IMPORT PATH. `rola/routing/entmax/production.py` is RoLA's own
# Triton producer and imports nothing from here; `tests/integration/test_production_levels.py`
# asserts that by making this module's entry points raise if production reaches them,
# and separately greps that production imports nothing from this module at all.
#
# PINNED (`PIN.json`, `pin.py`). Every figure above is a property of (this adapter,
# entmax 1.3), and `pyproject.toml`'s `entmax==1.3` is only the coarse half of naming
# that: it stops the resolver from installing a different release, but not an editable
# or re-published install that keeps the version string over different bytes.
# `pin.gate()` runs at import time, below, and refuses BOTH kinds of drift -- the ONLY
# guard on either, because the dual-run protocol imports the same installed package on
# both its OLD and NEW sides and cannot see a bump itself.

from __future__ import annotations

import torch

from rola.routing.entmax import pin as _pin

_pin.gate()

from entmax import entmax15, entmax_bisect, sparsemax  # noqa: E402

__all__ = [
    "entmax",
    "entmax_forward",
    "entmax_tau",
]

#: The alphas with an EXACT closed form in the reference package. Everything else goes
#: to their bisection. Structural, not a threshold: the two named alphas are the ones
#: the papers solve in closed form, and `entmax_bisect` is the general solver they
#: ship for the rest.
_EXACT = {1.5: entmax15, 2.0: sparsemax}

#: Their bisection's iteration count when no closed form applies. Their default is 50
#: and lands within 1e-15 of their own exact path; this asks for more because a
#: reference pays for accuracy with time it does not care about.
_BISECT_ITERS = 100


def _masked(z: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Excluded entries become ``-inf``, which is the package's own masking contract
    (their tests assert equivalence between an ``-inf`` logit and a shortened row).
    A masked entry then leaves the support by construction rather than by a
    correction applied afterwards."""
    if mask is None:
        return z
    return z.masked_fill(~mask, float("-inf"))


def entmax(z: torch.Tensor, alpha: float, dim: int = -1,
           mask: torch.Tensor | None = None) -> torch.Tensor:
    """The differentiable entmax/sparsemax values. ``alpha=2`` is sparsemax.

    Autograd is the package's own analytic sparse Jacobian -- the authors' backward
    for the authors' forward, which is what makes the VJP an independent reference
    rather than our derivation checked against itself. dtype follows ``z``; gates
    hand it fp64.
    """
    masked = _masked(z, mask)
    exact = _EXACT.get(float(alpha))
    if exact is not None:
        return exact(masked, dim=dim)
    return entmax_bisect(masked, alpha=alpha, dim=dim, n_iter=_BISECT_ITERS)


def entmax_tau(z: torch.Tensor, alpha: float, dim: int = -1,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    """Stage 1's threshold, INVERTED FROM THE VALUES rather than solved again.

    The definition is ``p_i = [(alpha - 1)(z_i - tau)]_+^{1/(alpha-1)}``, so for any
    ``i`` in the support that relation reads backwards exactly:

        tau = z_i - p_i^(alpha - 1) / (alpha - 1)

    Every support entry gives the same ``tau``; this reads it off the ARGMAX entry,
    which is in the support of every non-degenerate row and carries the largest
    ``p_i``, so the division is the best-conditioned one available. One line of
    algebra against a closed-form ``p`` is a stronger construction than a second
    root-find: there is no iteration count to converge, and no bracket to prove.

    Returns ``[..., 1]``, broadcastable over ``dim``.
    """
    masked = _masked(z, mask)
    p = entmax(z, alpha, dim=dim, mask=mask)
    index = masked.argmax(dim=dim, keepdim=True)
    z_top = masked.gather(dim, index)
    p_top = p.gather(dim, index)
    return z_top - p_top.pow(alpha - 1.0) / (alpha - 1.0)


def entmax_forward(z: torch.Tensor, alpha: float, dim: int = -1,
                   mask: torch.Tensor | None = None):
    """All three stages: ``(p, support, tau)``, ``p``/``support`` shaped like ``z``.

    The support is ``p > 0`` -- read off the values rather than recomputed from a
    threshold comparison, so the two can never disagree about a boundary entry. Under
    a mask the excluded entries are ``-inf`` and their ``p`` is exactly zero, so they
    are outside the support without a second correction.

    Detached: this is the forward alone. :func:`entmax` is the differentiable entry.
    """
    with torch.no_grad():
        p = entmax(z, alpha, dim=dim, mask=mask)
        support = p > 0
        tau = entmax_tau(z, alpha, dim=dim, mask=mask)
    return p, support, tau
