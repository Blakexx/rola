"""SHARED ORACLE FIXTURES — the tensors every oracle gate draws.

Every tier that compares a kernel launch to `rola.ops.naive.naive_rola` builds
the same three things: unit-sum simplex routing at a chosen density, a
`Topology` over some widths, and (on the decay arm) per-level rate dials. The
comparison itself is `tests.oracle.fixtures.assert_slots_close`, per slot.

Extracted here at P67 D2-b. The previous home was
the retired `test_consumer_vs_oracle.py`, the TILED consumer's oracle
gate, which retires with that consumer; four other files imported these helpers
from it. They are fixtures, not that gate's claims, so they outlive
it. VERBATIM: the bodies are unchanged from that file, which is what makes this a
move rather than a rewrite of a shared reference.

The numeric BAND lives next door in `tolerances.py`.
"""
from __future__ import annotations

import torch

from rola.routing.types import (
    IndependentRouting,
    SoftmaxActivation,
    Topology,
)

#: The routing every fixture here is built over: untied softmax on both sides.
_ROUTING = IndependentRouting(
    width=1, read=SoftmaxActivation(), write=SoftmaxActivation())


def _topology(widths):
    return Topology(levels=tuple(_ROUTING.at(w) for w in widths))


def _simplex(shape, p_nonzero, generator, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < p_nonzero
        x = x * keep
    # A token with an empty support at some level would have no route at all; the
    # producer cannot emit that (every activation is a simplex), so the fixture
    # repairs it rather than testing an unreachable input.
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    result = x / x.sum(-1, keepdim=True)
    if p_nonzero < 1.0:
        # The sparse/sparse-read-dense-write/dense-read-sparse-write fixture rows are
        # the density axis the whole exactness argument leans on (module docstring). A
        # seed, width or `p_nonzero` change could silently drive the surviving-zero
        # count to nothing and every test would stay green while testing the dense
        # case twice -- assert the fixture is actually sparse rather than assuming
        # the Bernoulli mask took.
        assert bool((result == 0).any()), (
            f"_simplex(p_nonzero={p_nonzero}) fixture is vacuously dense: no zero entries "
            "survived the Bernoulli mask + empty-row repair")
    return result


def _decay_dials(widths, H, generator, device, *, hot=False):
    """``delta in [0, 1)``, strict. ``hot`` drives the leaf rate product
    toward 1, where ``1 - rate`` cancels -- the regime the kernel's fp32
    accumulation requirement exists for."""
    lo, span = (0.90, 0.099) if hot else (0.05, 0.85)
    return tuple(
        lo + span * torch.rand(H, w, device=device, dtype=torch.float64, generator=generator)
        for w in widths)
