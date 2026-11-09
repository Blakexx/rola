# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALE PASS AGAINST ITS fp64 BOUND: each token's scale byte over a whole call, on every central oracle cell, and
the bound against the inter numerator it must cover.

The pass's byte is the carry's rule (`unit_byte`) on the write-aligned bound over the carry's factor
(`tests/oracle/scale_reference.py`); a byte holds where it is the least that covers the reference's bound within
fp32's sums. The bound itself must cover the recurrence's numerator on the value scales -- every channel's bound -- over
the reader's factor, which the second test states without the kernel.
RED BY PLAN: a cell the pass does not serve (`rola.ops.scale.serves`) fails by name.
"""
from __future__ import annotations

import pytest
import torch

from measure.cells import carry_cells, realize
from measure.subjects import bundle_of, lists_form_of
from rola.ops import facts as facts_ops
from rola.ops import scale as scale_ops
from rola.ops.prefill import carry_factor, reader_factor
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.scale_reference import bound, byte_holds

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def operands(spec):
    """``(routes, values, facts)`` over the cell."""
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    facts = facts_ops.facts(routes, values, layout, *statics, lists_form_of(spec))
    return routes, values, facts


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_every_byte_is_the_rules_on_the_fp64_bound(spec):
    """Every token's byte over the whole call, against the rule on the reference's bound."""
    routes, values, facts = operands(spec)
    factor = carry_factor(routes)
    grid = scale_ops.scale(routes, facts, factor)
    torch.cuda.synchronize()
    T = routes.tokens
    assert (grid[:, T:] == -100).all(), f"{spec.name}: a partial window's tail holds a byte"
    got = grid[:, :T]
    p = bound(routes, values, window_of(), scale_ops.level_s(routes)) / factor
    ok = byte_holds(got, p)
    bad = (~ok).nonzero()
    assert ok.all(), (f"{spec.name}: {bad.shape[0]} of {ok.numel()} bytes off the rule; the first at "
                      f"{tuple(bad[0].tolist())}: byte {int(got[tuple(bad[0])])}, bound {float(p[tuple(bad[0])]):.6g}")


@pytest.mark.parametrize("spec", [c for c in ORACLE_CELLS if c.state == "fresh" and c.D == 2],
                         ids=[c.name for c in ORACLE_CELLS if c.state == "fresh" and c.D == 2])
def test_the_bound_covers_the_inter_numerator(spec):
    """The bound over the reader's factor covers the recurrence run on each writing token's value scale -- the
    largest a channel's numerator can be -- at every token (no kernel: the bound's own claim)."""
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    read, write = routes.dequantized()
    num, _den, _ = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), values.scale.double().unsqueeze(-1), routes.topology.widths, window_of())
    B, T, H = num.shape[:3]
    exact = num[..., 0].permute(0, 2, 1).reshape(B * H, T)
    r = reader_factor(routes).double()
    b = bound(routes, values, window_of(), scale_ops.level_s(routes))
    covered = (b * r >= exact.abs() * (1 - 1e-9)) | (exact.abs() <= 1e-300)
    assert covered.all(), f"{spec.name}: {int((~covered).sum())} tokens above the bound"
