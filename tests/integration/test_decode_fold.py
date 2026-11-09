# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE DECODE FOLD: the declared arithmetic, held to BIT-IDENTITY.

The step's operand pass lives INSIDE the step kernel
(``csrc/rola/src/decode/decode_fold.cuh``), so what this file gates it gates through the
step's own output. The fold reads a level's stored row -- u8 digits against the row's fp32
max -- as RAW values (a u8 digit as its integer, exact), and makes an amplitude
``raw * (scale / 255)``: the level as stored, on either side, whatever its sum
(docs/CONSTITUTION.md#r23).

**THE READOUT THE GATE IS BUILT ON.** With ``D = 1``, a read level that is nonzero
everywhere, no deposit, and a state whose rows are zero except at ONE leaf ``s0``, the
step's whole arithmetic collapses to

    ``y[c] = a[s0] * e[c] / (a[s0] * m + eps)``

-- one multiply per column and one divide, all in fp32 and all in a fixed order. So ``y``
is a function of the amplitude's last bits and of nothing else, and the reference below
computes exactly that expression.

Run: ``bazel test //tests:integration --test_filter=test_decode_fold``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.constants import READOUT_EPS
from rola.ops.decode import _decode_step, derive_decode_geometry, fold_levels
from rola.ops.lattice import permutation
from rola.ops.paging import decode_pages
from rola.routing import quant
from rola.routing.factors import RouteFactors
from rola.routing.types import Topology, UnionRouting
from rola.values import Values
from tests.oracle.fixtures import state_pages

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the fold is a CUDA kernel"),
]

_DV = 64


def logical_read_only(stored):
    """Stored pages ``[BH, N/16, page_bytes]`` as a read-only load sees them, ``[BH, N, d_v + 1]``."""
    return decode_pages(stored, _DV, hi_only=True).reshape(stored.shape[0], -1, _DV + 1)


def _single_leaf_state(BH: int, config, s0: int, device, seed: int):
    """A ``[BH, N, d_v + 1]`` logical state, STORED as int8 state pages -- zero everywhere but CANONICAL leaf ``s0``,
    in the plane's own LATTICE order. Returns ``(state, e, m)``: ``state`` in the stored form the step reads, and
    ``e``/``m`` the values a READ-ONLY load reconstructs (the hi plane at its group exponent, the fp32 mass) -- the
    reference must be built from what the kernel reads, not from the fp32 draw that never reaches it whole."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pi = permutation(config.widths, config.lattice_k, config.lattice_m, device=device)
    state = torch.zeros(BH, config.N, _DV + 1, dtype=torch.float32, device=device)
    row = torch.rand(BH, _DV + 1, dtype=torch.float32, device=device, generator=gen) + 0.5
    state[:, int(pi[s0])] = row
    stored = state_pages(state)
    row_hi = logical_read_only(stored)[:, int(pi[s0])]
    return stored, row_hi[:, :_DV], row_hi[:, _DV]


def _routes(rows, scale):
    """A one-level bundle over ``rows`` (read) and an empty write side, both sides dense (depth one: the read rows are
    strictly positive and the dead write rows are all-zero rows, which a dense level keeps zero). The level is
    union-routed, whose read duty is never on the simplex: the fold reads it as stored."""
    topology = Topology(levels=(UnionRouting(width=rows.shape[-1]),))
    return RouteFactors.from_rows(topology, (rows,), (quant.aligned_rows(*rows.shape, rows.device, quant.DENSE_DTYPE),),
                                  (scale,), (torch.zeros_like(scale),), sparse=((False,), (False,)))


def _y_of(rows, scale, *, seed, s0=0):
    """One step over the collapsing fixture. Returns ``(y[BH, d_v], e, m)``."""
    B, _T, H, width = rows.shape
    BH, device = B * H, rows.device
    routes = _routes(rows, scale)
    #: NO DEPOSIT: an all-zero write side leaves `supp(W)` empty, so every touched row is
    #: read-only and the state the readout sees is the state that was placed.
    values = Values.from_projection(torch.zeros(B, rows.shape[1], H, _DV, device=device))
    config = derive_decode_geometry(routes.topology, d_v=_DV, decay=False)
    state, e, m = _single_leaf_state(BH, config, s0, device, seed + 1)
    y, _state, _ws = _decode_step(routes, values, config, state)
    return y.reshape(BH, _DV), e, m


def _reference_y(rows, scale, *, e, m, s0=0):
    """The collapsed readout, computed from the DECLARED arithmetic and nothing else."""
    raw = fold_levels((rows,), (scale,), (_routes(rows, scale).form(0, "read"),))[0]
    #: a tensor divisor: torch divides by a SCALAR through its reciprocal, which is not the kernel's IEEE divide.
    scale = scale.reshape(-1)
    a = raw[:, s0] * (scale / torch.full_like(scale, quant.AMP_LEVELS))
    return (a.unsqueeze(-1) * e) / (a * m + READOUT_EPS).unsqueeze(-1)


def _read_rows(B, H, width, *, seed, device="cuda"):
    """``(rows, scale)``: a stored dense read level, f16 at the floor, off the simplex. Every entry is strictly
    positive, so `supp(R)` is the whole leaf space and the walk visits every row."""
    gen = torch.Generator(device=device).manual_seed(seed)
    level = 0.5 * (torch.rand(B, 1, H, width, device=device, generator=gen) + 0.25)
    return quant.DENSE.quantize(level)


# ---------------------------------------------------------------------------
# 1. the stored digit at its row's scale, and the address fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [16, 32, 64, 256])
def test_the_fold_is_the_raw_digit_at_its_scale(width):
    rows, scale = _read_rows(3, 2, width, seed=4)
    y, e, m = _y_of(rows, scale, seed=4)
    assert torch.equal(y, _reference_y(rows, scale, e=e, m=m))


def test_a_strided_view_folds_by_address():
    """A level may be a VIEW of one packed plane, and the fold indexes it there."""
    packed, scale = _read_rows(3, 2, 32, seed=3)
    sliced = packed[..., ::2]
    dense = quant.aligned_rows(*sliced.shape, sliced.device, sliced.dtype).copy_(sliced)
    y_view, e, m = _y_of(sliced, scale, seed=8)
    y_dense, e2, m2 = _y_of(dense, scale, seed=8)
    assert torch.equal(e, e2) and torch.equal(m, m2), "the two runs saw different states"
    assert torch.equal(y_view, y_dense)
    assert torch.equal(y_view, _reference_y(dense, scale, e=e, m=m))


# ---------------------------------------------------------------------------
# 2. the fold's refusals
# ---------------------------------------------------------------------------

def test_more_than_one_token_is_refused():
    rows, scale = _read_rows(2, 2, 16, seed=5)
    two = quant.aligned_rows(2, 2, 2, 16, rows.device, rows.dtype).copy_(rows.repeat(1, 2, 1, 1))
    with pytest.raises(ValueError, match="takes one token"):
        _y_of(two, scale.repeat(1, 2, 1), seed=5)


def test_a_row_dtype_outside_the_format_is_refused():
    rows, scale = _read_rows(2, 2, 16, seed=5)
    digits = quant.aligned_rows(*rows.shape, rows.device).copy_((rows.float() * quant.AMP_LEVELS).round().to(torch.uint8))
    with pytest.raises((TypeError, ValueError), match="torch.float16"):
        _routes(digits, scale)
