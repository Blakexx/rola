"""THE ORACLE'S ADDRESS MAP IS ITS OWN, AND THIS IS WHERE THAT STOPS BEING A CLAIM.

Every Tier 1 gate compares the kernel against `rola/ops/naive.py`. The kernel's leaf
addressing comes from `Topology.radix_strides`; the oracle used to import the same
property, so a wrong stride convention would have been invisible to every one of
those gates -- oracle and kernel wrong together, agreeing perfectly. That is the
matching-the-naive anti-pattern in its purest form, and the audit named it F3.

The oracle now derives its own. Two claims follow, and both are tested here because
each alone is worthless:

1. **THEY AGREE** -- over the whole supported width space, so the independence costs
   no correctness.
2. **THEY WOULD DISAGREE IF ONE WERE WRONG** -- perturb the production property and
   the oracle's outputs must MOVE. Without this the first claim is equally consistent
   with the oracle having quietly kept importing production, which is exactly the
   state being left behind.

No device: the oracle is fp64 torch and the address map is arithmetic.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from rola.ops.naive import _leaf_capacity, _radix_strides, naive_rola
from rola.routing.types import (
    IndependentRouting,
    SoftmaxActivation,
    Topology,
)

_ROUTING = IndependentRouting(
    width=1, read=SoftmaxActivation(), write=SoftmaxActivation())

#: Every shape the two derivations must agree on: depth 1-4 (the validated range) over
#: widths spanning 1, the powers of two, and non-powers that make a stride product
#: something other than a shift.
_WIDTH_VALUES = (1, 2, 3, 4, 7, 8, 16, 17, 32, 64, 256)


def _topology(widths):
    return Topology(levels=tuple(_ROUTING.at(w) for w in widths))


# ---------------------------------------------------------------------------
# CLAIM 1 -- the two derivations agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", (1, 2, 3, 4))
def test_the_oracles_own_strides_equal_productions_at_every_supported_shape(depth):
    """The whole width space at this depth, not a sample of it.

    A width of 1 is in the set on purpose: it contributes a factor of 1 to every
    stride below it, so a derivation that confused "product of widths below" with
    "product of widths below, skipping ones" would pass a powers-of-two-only sweep.
    """
    checked = 0
    for widths in itertools.product(_WIDTH_VALUES, repeat=depth):
        if _leaf_capacity(widths) > 1 << 20:
            continue                      # the planner's own capacity ceiling
        topology = _topology(widths)
        assert _radix_strides(widths) == topology.radix_strides, widths
        assert _leaf_capacity(widths) == topology.N, widths
        checked += 1
    #: NON-VACUITY OF THE SWEEP ITSELF: a `continue` that swallowed everything would
    #: make this test pass by checking nothing.
    assert checked >= len(_WIDTH_VALUES), f"only {checked} shapes reached the assertion"


def test_the_convention_is_msb_first_and_the_last_level_varies_fastest():
    """The convention stated in the oracle's docstring, asserted rather than trusted.

    This is the check that would catch a REVERSED stride order -- the exact mutant
    the dual-run protocol injects -- without needing production's property at all, so
    the oracle's convention is anchored to the spec and not to the other side of the
    comparison.
    """
    assert _radix_strides((2, 3, 4)) == (12, 4, 1)
    assert _radix_strides((5,)) == (1,)
    #: leaf s = ((d0 * 3) + d1) * 4 + d2, so d0 is the most significant digit.
    widths = (2, 3, 4)
    strides = _radix_strides(widths)
    for leaf in range(_leaf_capacity(widths)):
        digits = [(leaf // stride) % width for stride, width in zip(strides, widths)]
        rebuilt = 0
        for digit, width in zip(digits, widths):
            rebuilt = rebuilt * width + digit
        assert rebuilt == leaf


# ---------------------------------------------------------------------------
# CLAIM 2 -- and they would disagree if production were wrong
# ---------------------------------------------------------------------------

def _oracle_outputs(topology):
    """One small deterministic `naive_rola` call, forward and state."""
    torch.manual_seed(11)
    B, T, H, d_v = 1, 5, 2, 32
    widths = topology.widths

    def simplex(width):
        x = torch.rand(B, T, H, width, dtype=torch.float64) + 1e-3
        return x / x.sum(dim=-1, keepdim=True)

    read = tuple(simplex(w) for w in widths)
    write = tuple(simplex(w) for w in widths)
    v = torch.randn(B, T, H, d_v, dtype=torch.float64)
    g_write = torch.rand(B, T, H, dtype=torch.float64) + 0.5
    return naive_rola(v, read, write, g_write, topology, None,
                      output_final_state=True)


def test_perturbing_productions_strides_makes_the_oracle_disagree(monkeypatch):
    """THE INDEPENDENCE PROOF. Reverse `Topology.radix_strides` -- the F3 mutant --
    and the oracle's output must be unchanged, because it no longer reads it.

    The two assertions are opposite halves of one claim and neither works alone:

    * the ORACLE is unchanged, so it is genuinely not consuming the property;
    * the PRODUCTION property really did move, so the perturbation was live and the
      first assertion is not passing because the monkeypatch failed to apply.

    Together they say: a wrong stride convention in production is now a DISAGREEMENT
    with the oracle rather than a shared assumption -- which is what makes every Tier
    1 comparison downstream of this file mean something.
    """
    topology = _topology((2, 3, 4))
    before = _oracle_outputs(topology)

    original = Topology.radix_strides.fget
    monkeypatch.setattr(Topology, "radix_strides",
                        property(lambda self: tuple(original(self)[::-1])))

    assert topology.radix_strides == (1, 4, 12), (
        "the perturbation did not take effect, so this test proves nothing")
    assert _radix_strides(topology.widths) == (12, 4, 1), (
        "the oracle's own derivation moved with production's; it is still borrowing")

    after = _oracle_outputs(topology)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])


def test_a_perturbed_oracle_derivation_does_move_the_oracle(monkeypatch):
    """The other direction, which is what makes the test above non-vacuous: the
    oracle's outputs DO depend on its own strides, so "unchanged" was a fact about
    the source of the map and not about the map being ignored."""
    import rola.ops.naive as naive

    topology = _topology((2, 3, 4))
    before = _oracle_outputs(topology)
    monkeypatch.setattr(naive, "_radix_strides",
                        lambda widths: tuple(_radix_strides(widths)[::-1]))
    after = _oracle_outputs(topology)
    assert not torch.equal(before[0], after[0]), (
        "reversing the ORACLE's own strides changed nothing, so the leaf address map "
        "is not reaching the result and neither test above is measuring anything")
