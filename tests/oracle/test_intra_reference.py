# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The intra reference IS the oracle, on the cell where the two coincide.

With a zero initial state and no decay, one window's whole recurrence is its
intra term, so `naive_rola`'s normalized readout must equal the reference's
``o / (den + eps)`` to fp64 round-off. That is the only claim this file makes,
and everything else in the K31 R2 battery gates against the reference.
"""

from __future__ import annotations

import torch

from rola.ops.constants import READOUT_EPS
from rola.ops.naive import naive_rola
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from tests.oracle.intra_reference import (
    DENSE_BOTH,
    LEVEL_WIDTH,
    READ_SPARSE,
    WINDOW,
    WRITE_SPARSE,
    intra_reference,
    make_cell,
)

_LEVEL = IndependentRouting(width=1, read=SoftmaxActivation(),
                            write=SoftmaxActivation())


def _oracle(pread, pwrite, gwrite, v_bh, levels, heads):
    bh, length, _ = pread.shape
    batch = bh // heads
    topology = Topology(levels=tuple(_LEVEL.at(LEVEL_WIDTH) for _ in range(levels)))

    def split(plane):
        return tuple(
            plane[..., l * LEVEL_WIDTH:(l + 1) * LEVEL_WIDTH]
            .view(batch, heads, length, LEVEL_WIDTH).permute(0, 2, 1, 3).contiguous()
            for l in range(levels))

    v = v_bh.view(batch, heads, length, v_bh.shape[-1]).permute(0, 2, 1, 3).contiguous()
    g = gwrite.view(batch, heads, length).permute(0, 2, 1).contiguous()
    y, _ = naive_rola(v, split(pread), split(pwrite), g, topology, None)
    return y


def _cases():
    yield "dense", (DENSE_BOTH, DENSE_BOTH), None
    yield "alt", (WRITE_SPARSE, READ_SPARSE), 4


def test_reference_matches_the_oracle_on_one_window():
    torch.manual_seed(0)
    heads, bh, levels = 2, 2, 2
    for name, modes, k_tok in _cases():
        pread, pwrite, gwrite, v_bh = make_cell(bh, WINDOW, levels, modes,
                                                k_tok=k_tok if k_tok else 4, seed=7)
        o_ref, den_ref = intra_reference(pread, pwrite, gwrite, v_bh, levels)
        y_ref = (o_ref / (den_ref.unsqueeze(-1) + READOUT_EPS))
        y_oracle = _oracle(pread, pwrite, gwrite, v_bh, levels, heads)
        y_oracle = y_oracle.permute(0, 2, 1, 3).reshape(bh, WINDOW, -1)
        err = (y_ref - y_oracle).abs().max().item()
        assert err < 1e-9, f"{name}: reference and oracle disagree by {err}"
