# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""TIER 2 — the decode path is BC-BLIND, stated as "no dependence", not "no literal".

Spec the stated enforcement for the tiled path is to parameterize over every
instantiated ``BC`` plus one hypothetical ``BC``, so a hardcoded quanta-shaped literal
fails the fake-``BC`` row. The decode path admits a stronger form of the same gate,
because ``BC`` is not an input to it at all: run the same fixture while the surrounding
prefill declares ``BC`` in ``{16, 32, 64}`` — 32 being the HYPOTHETICAL value, not
instantiated by the tiled consumer — and assert the decode state is BYTE-IDENTICAL and
``y`` equal to the order its sums are added in (docs/internals/decode/decode.md#determinism).

Contract class (a), engineered identity: ``torch.equal`` on the state. The claim is not that decode
*rounds the same* under different ``BC``; it is that ``BC`` cannot reach decode at all,
so the two runs execute the identical instruction stream on identical bytes. Anything
weaker than byte-equality here would be evidence of a dependence.

**WHY THE ENTRY STATE IS HELD FIXED, and why that is the honest formulation.** A prefill
at ``BC = 16`` and one at ``BC = 64`` produce entry states that differ in their low bits
(different summation trees), so feeding decode the two real prefill outputs would
compare two different inputs and prove nothing about ``BC``. The dependence being tested
is decode's, so the state is held identical and the ``BC`` is varied everywhere it could
possibly enter: the plan, the frozen carrier's construction, and the config object.

Run: ``bazel test //tests:integration --test_filter=test_decode_bc_blindness``
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from rola.engine.plan import DecodeGeometry
from rola.ops.decode import _decode_step, derive_decode_geometry
from rola.ops.paging import bytes_equal
from tests.oracle.fixtures import assert_reordered_close, state_pages, step_bundles, with_gain
from tests.oracle.oracle_fixtures import _simplex, _topology

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

#: Box capacities from a narrow box to a wide one, and one no prefill arm has ever built. Decode must be blind to a
#: number it will never see as thoroughly as to one it will.
_BCS = [32, 64, 128, 256, 1024]


def test_no_field_of_the_frozen_carrier_names_bc():
    """The structural half: ``BC`` is not a field, and cannot become one by accident.

    A field-level check rather than a comment, because the carrier is the one object
    that could plausibly acquire a ``BC`` (it is built from a plan that has one), and a
    field added there would be invisible to the numeric gate below until something
    started reading it.
    """
    names = {f.name.lower() for f in dataclasses.fields(DecodeGeometry)}
    assert not any(n == "bc" or n.startswith("bc_") or n.endswith("_bc") for n in names), (
        f"DecodeGeometry has acquired a BC-shaped field: {sorted(names)}")


@pytest.mark.parametrize("widths", [(64, 64), (16, 16, 16), (16, 16, 16, 16)])
def test_decode_is_blind_to_prefill_bc(widths):
    device = "cuda"
    d_v, B, H = 64, 2, 2
    N = math.prod(widths)
    # The mass column is unconditional (there is no normalization axis at
    # all), so `cols = d_v + 1`, always.
    cols = d_v + 1
    topology = _topology(widths)

    generator = torch.Generator(device=device).manual_seed(20260802)
    read_levels = tuple(_simplex((B, 1, H, w), 0.5, generator, device).float() for w in widths)
    write_levels = tuple(_simplex((B, 1, H, w), 0.4, generator, device).float() for w in widths)
    g_write = (torch.rand(B, 1, H, device=device, generator=generator) + 0.5).float()
    v = torch.randn(B, 1, H, d_v, device=device, generator=generator).float()
    m0 = 0.1 * torch.randn(B, H, N, cols, device=device, generator=generator).float()
    m0[..., d_v] = m0[..., d_v].abs()
    #: THE STATE IS HANDED TO THE KERNEL IN ITS STORED FORM (the int8 state pages).
    m0 = state_pages(m0)
    routes, values = step_bundles(v, read_levels, with_gain(write_levels, g_write), topology)

    results = []
    configs = []
    for BC in _BCS:
        # `BC` is here to be IGNORED: it belongs to a prefill's built arm and is
        # not host-selectable, so the loop varies the number a prefill COULD have
        # run at and asserts there is no channel through which it could reach the
        # frozen carrier. `derive_decode_geometry` is called inside the loop because
        # it must be a pure function of the topology, `d_v` and `decay`.
        assert BC in _BCS
        config = derive_decode_geometry(topology, d_v=d_v, decay=False)
        configs.append(config)
        state = m0.clone()
        y, state, _ = _decode_step(routes, values, config, state)
        results.append((y, state))

    assert configs[0] == configs[1] == configs[2], (
        "the frozen carrier moved with the prefill's BC; it must be a pure function of "
        "the topology, `d_v` and `decay`")
    for BC, (y, state) in zip(_BCS[1:], results[1:]):
        assert_reordered_close(y, results[0][0], f"decode's y at prefill BC={BC}")
        assert bytes_equal(results[0][1], state), (
            f"decode's state moved with prefill BC={BC}")
