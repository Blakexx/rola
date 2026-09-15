# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31-i-d: the owner BALLOT ROWS at the lattice's spans, against a torch reference.

Symbols, each at first use: ``widths`` the per-level digit counts (``B`` each),
``D`` levels, ``k`` the uniform sub-box span per level, ``m`` the capacity
multiplier, ``s_l`` the owner's span at level ``l``, ``g_l = B / s_l`` the
owner-grid extent at level ``l``, ``k_tok`` a token's nonzeros per level (a
property of the DRAW and of nothing in the kernel), ``T`` the sequence length.

WHAT IS UNDER TEST.  The facts pass is templated on the PLAN TYPE since K31-i-d,
so a plan whose spans are GIVEN -- the lattice's ``(8, 16)`` -- votes the
same rows the lex plan votes for its own spans, and it is templated on ONE PLAN PER
SIDE since K35 S1b, so the table is the read plan's rows followed by the write plan's.
This diagnostic asks for the same grain on both sides, so the two blocks are the same
size and the table reads as ``[BH, 2, entries, L/32]``.  The claim is bit-level: row
``(side, prefix_entries(l) + o_l)``'s word ``w`` bit ``i`` is exactly "token
``32w + i``'s level-``l`` support on that side has a nonzero in the owner group
``o_l``'s digit run".  A reference that recomputed the kernel's packing would only
prove self-consistency (the matching-the-naive antipattern); this one is written
from the CLAIM -- a magnitude test on the same bf16 bytes, packed by position.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops import carry as carry_ops
from tests.oracle.fixtures import Cell, assert_fresh_binary, built_arms

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: the bf16 magnitude bits; the only zeros bf16 has are +0 and -0.
MAGNITUDE = 0x7FFF


def require_topology(widths, k, m):
    """SKIP unless a built DENSE_BOTH row carries this ``(D, B, k, m)``.

    ``owner_rows`` is a diagnostic over the PLAN alone, so the seam dispatches it on the
    topology and the box over the rows declared DENSE_BOTH -- the window, the value width
    and the declaration are not part of its key.
    """
    D, B = len(widths), widths[0]
    if not any(row[:4] == (D, B, k, m) and row[8] == 0 for row in built_arms()):
        pytest.skip(
            f"this binary carries no DENSE_BOTH carry arm at (D={D}, B={B}, k={k}, m={m}), "
            f"which is what `owner_rows` dispatches on.")


def reference_rows(plane: torch.Tensor, widths, spans) -> torch.Tensor:
    """``[BH, entries, L/32]`` int32 ballot words for one side's packed plane.

    ``plane`` is ``[BH, L, sum_l B_l]`` bf16 -- the SAME bytes the kernel reads.
    """
    bh, length, _ = plane.shape
    bits = plane.view(torch.int16).to(torch.int32) & MAGNITUDE
    words = (length + 31) // 32
    rows = []
    col = 0
    for width, span in zip(widths, spans):
        for group in range(width // span):
            run = bits[:, :, col + group * span:col + (group + 1) * span]
            rows.append((run != 0).any(-1))
        col += width
    live = torch.stack(rows, dim=1).to(torch.int64)          # [BH, entries, L]
    pad = words * 32 - length
    if pad:
        live = torch.nn.functional.pad(live, (0, pad))
    live = live.reshape(bh, len(rows), words, 32)
    place = (1 << torch.arange(32, device=plane.device, dtype=torch.int64))
    return (live * place).sum(-1).to(torch.int32)


@pytest.mark.parametrize("widths,T,k_tok", [((64, 64), 256, 4),
                                            ((64, 64), 320, None),
                                            ((256, 256), 128, 4)])
def test_the_lattice_spans_vote_the_rows_the_claim_names(widths, T, k_tok):
    assert_fresh_binary()
    k, m = 4, 8
    require_topology(widths, k, m)
    _, spans, groups, bc, _ = carry_ops.box_shape(widths, k, m)
    assert spans == [8, 16], f"the lattice's spans are (8, 16); got {spans}"
    cell = Cell(widths, T, k_tok=k_tok, seed=11)

    got = carry_ops.owner_rows(cell.read_bf, cell.write_bf, widths, k=k, m=m)
    #: the two sides' blocks are consecutive, each `entries` rows: this diagnostic's two
    #: plans coincide, so the concatenation reshapes back to the per-side view.
    assert got.shape == (1, 2 * sum(groups), (T + 31) // 32), got.shape
    got = got.reshape(1, 2, sum(groups), (T + 31) // 32)

    for side, levels in ((0, cell.read_bf), (1, cell.write_bf)):
        want = reference_rows(carry_ops.pack_side(levels), widths, spans)
        assert torch.equal(got[:, side], want), (
            f"side {side}: the owner rows disagree with the rectangle test at "
            f"{int((got[:, side] != want).sum())} of {want.numel()} words")


def test_the_table_is_sigma_shaped_not_product_shaped():
    """``entries = sum_l B/s_l``, which is why the table does not grow with ``N``."""
    for widths in ((64, 64), (256, 256)):
        _, spans, groups, bc, owners = carry_ops.box_shape(widths, 4, 8)
        entries = sum(groups)
        assert entries == sum(w // s for w, s in zip(widths, spans))
        assert entries < owners, (
            f"{widths}: {entries} rows for {owners} owners -- the table is the SUM "
            f"over levels, never the product")
