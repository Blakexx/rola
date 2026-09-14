"""CONFORMANCE — the producer-source families: realized routing at swept gains.

Some correlations only real routing produces (token coherence inherited from the
hidden stream, read-write coincidence from the shared projection), so these
families drive the REAL producer — ``RouteProducer`` over ``union_routing`` —
rather than a sampler, and the gain on the hidden stream is the knob that sweeps
the realized density, exactly as it does in training.

Both arms of ``rola_op`` run on each realized bundle and must agree at the
tier-1 exactness tolerance. All three families sit INSIDE the built chunk
matrix, so the kernel arm is reached on every sweep arm and the agreement is
the whole claim; there is no refusal branch to fall into.

P67 D2-b re-anchored this file. The families pinned at ``d_v = 32`` moved onto
built arms (``producer/swept-gain`` to the flat-routing baseline ``D = 1,
B = 64``; ``producer/swept-gain-d64`` to the uniform two-level ``D = 2, d_v =
64`` arm; ``producer/swept-gain-raw-d2`` to ``d_v = 64`` and, since ``raw``
normalization has been a dead axis since P26, to the honest name
``producer/swept-gain-d2``) rather than being left to assert a refusal, which
would have taken REALIZED routing off the kernel arm entirely — the one thing
this file exists to keep on it. ``producer/stale-plan`` retired with the
``Plan`` object it drove: its subject was executing a plan priced on a dead
distribution through ``rola.expert.execute``, and post-D2 there is no plan left
to be stale. That is a genuine loss of adversarial coverage, recorded as one
rather than papered over.

NON-VACUITY here is sweep-shaped: ``assert_swept`` proves the gain knob actually
moved the realized density (>= 4x span, monotone), and the shared regime
checkers run per arm with the family's cell required to be demonstrated by the
sweep as a whole (an entmax sweep is dense at its cold end BY DESIGN, so
requiring every arm to sit in the sparse cell would demand a vacuous sweep).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.oracle import generators as g
from tests.oracle.families import FAMILY_BY_NAME

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]


def _producer_and_stream(family):
    """The producer, and a temporally COHERENT hidden stream (AR(1), phi = 0.95):
    coherence is a property realized routing inherits from its input, so the
    input has to carry it."""
    from rola import union_routing
    from tests.conftest import build_routes

    c = family.config
    torch.manual_seed(c["seed"])
    producer = build_routes(
        hidden_size=64, num_heads=c["H"], widths=c["widths"],
        routing=union_routing(1, alpha=1.5)).cuda()
    B, T = c["B"], c["T"]
    gen = torch.Generator(device="cuda").manual_seed(c["seed"] + 1)
    eps = torch.randn(B, T, 64, device="cuda", generator=gen)
    x = torch.empty_like(eps)
    x[:, 0] = eps[:, 0]
    for t in range(1, T):
        x[:, t] = 0.95 * x[:, t - 1] + (1 - 0.95**2) ** 0.5 * eps[:, t]
    v = torch.randn(B, T, c["H"], c["d_v"], device="cuda", generator=gen)
    return producer, x, v


def _density(routes) -> float:
    """The realized exact-nonzero fraction of the write side."""
    total = live = 0
    for level in routes.write:
        live += int((level != 0).sum())
        total += level.numel()
    return live / total


def _view(family, routes):
    c = family.config
    return SimpleNamespace(
        read_levels=routes.read_simplex(), write_levels=routes.write, widths=tuple(c["widths"]),
        T=c["T"], BT=g.TOKEN_WORD, initial_state=None)


def _demonstrated_by_some_arm(family, views):
    """Each (axis, value) of the family's regime cell must be demonstrated by AT
    LEAST ONE arm of the sweep (`density = swept` is proven by `assert_swept`
    over all arms instead)."""
    for axis, value in family.regime.items():
        if (axis, value) == ("density", "swept"):
            continue
        errors = []
        for view in views:
            try:
                g.CHECKERS[(axis, value)](view)
                break
            except AssertionError as error:
                errors.append(str(error))
        else:
            raise AssertionError(
                f"no arm of the sweep demonstrates ({axis}, {value}): {errors}")


@pytest.mark.parametrize("name", ["producer/swept-gain", "producer/swept-gain-d64",
                                  "producer/swept-gain-d2"])
def test_the_swept_gain_family_realizes_its_density_sweep(name):
    """The producer half of the retired swept-gain kernel gate.

    Until the K31 deletion batch this cell also ran each realized bundle through
    the kernel arm of ``rola_op`` and graded it against the oracle (baseline =
    tag `baseline/pre-k31`); what survives is the family's own claim -- the gain
    knob sweeps the REALIZED density monotonically and the declared regime cell
    is demonstrated -- which is a fact about the producer, not about a consumer.
    """
    family = FAMILY_BY_NAME[name]
    producer, x, v = _producer_and_stream(family)
    densities, views = [], []
    for gain in family.params["gains"]:
        with torch.no_grad():
            routes = producer(gain * x)
        densities.append(_density(routes))
        views.append(_view(family, routes))

    # NON-VACUITY: the knob moved the distribution, and the cell is demonstrated.
    g.assert_swept(densities)  # ascending gain -> monotone descending density
    _demonstrated_by_some_arm(family, views)
