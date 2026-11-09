# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A hypothesis-drawn cell space over the carry oracle's REGISTRY.

A SUPPLEMENT to `test_carry_vs_oracle.py`'s fixed named cells (the fixed cells stay the default battery), not a
replacement: an oracle red yields a minimal failing (widths, k_tok, seed, cohort, d_v) by shrinking, rather than only at
whichever fixed point someone named. Opt-in (the `hypothesis_cells` marker, deselected by default beside `bench`:
`pyproject.toml`'s `addopts`), so its search runs on request, never inside the default battery's budget.

The strategy draws only over shapes the registry declares (widths, tokens and value width together) and over
properties of the DRAW the kernel is never told (``k_tok``, ``seed``, ``cohort``). RED BY PLAN with the rest of the
tier where the carry has no arm for a drawn carve.
"""
from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from measure.cells import CarryCell
from measure.cells import carry_cells as registry_cells
from measure.subjects import bundle_of
from rola.engine.carve import plan_carve
from tests.oracle.test_carry_vs_oracle import check, run

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
    pytest.mark.hypothesis_cells,
]

#: THE SHAPES THE REGISTRY DECLARES, read off it: a topology or a value width added there is picked up here
_SHAPES = sorted({(cell.widths, cell.tokens, cell.dv) for cell in registry_cells(tier="oracle")})
#: the shapes a sparse declaration is drawn on: those where one of the registry's cells declares a sparse side
_SPARSE_SHAPES = sorted({(cell.widths, cell.tokens, cell.dv) for cell in registry_cells(tier="oracle")
                         if any(read or write for read, write in cell.declared_sparsity())})


@st.composite
def drawn_cells(draw) -> CarryCell:
    """One drawn `CarryCell` on the central registry's own record shape, so a shrunk failure is reportable by name
    and can be pasted into rola-devtools' `carry.json` as a fixed cell."""
    widths, tokens, dv = draw(st.sampled_from(_SHAPES))
    k_tok = draw(st.sampled_from([2, 4, 8, 16, None]) if (widths, tokens, dv) in _SPARSE_SHAPES else st.none())
    seed = draw(st.integers(min_value=0, max_value=2 ** 16))
    #: a cohort divides the length exactly and means something only once k_tok narrows the support
    cohort = draw(st.none() if k_tok is None else st.sampled_from([None, 32, 64]))
    return CarryCell(name=f"hypothesis-{widths}-{dv}-{k_tok}-{cohort}-{seed}", seed=seed, widths=widths, dv=dv,
                     tokens=tokens, draw="dense" if k_tok is None else ("cohort" if cohort else "alt"), k_tok=k_tok,
                     cohort=cohort, support=1.0, backing="dense", state="fresh", tier="oracle", regime=None)


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(spec=drawn_cells())
def test_the_inter_numerator_is_the_fp64_reference_on_a_drawn_cell(spec):
    """The tier's own comparison over a drawn cell: the same `run` and `check`, per slot against the same envelopes --
    only the cell is drawn, never the assertion."""
    check(spec, *run(spec))


@settings(deadline=None, max_examples=25)
@given(spec=drawn_cells())
def test_a_drawn_cell_is_a_lawful_shape(spec):
    """The strategy draws only declarations the carve rule admits: a drawn cell the planner refused would exercise
    the rule's refusals, not the carry."""
    from measure.cells import realize

    routes, values, _layout, _statics = bundle_of(spec, realize(spec))
    plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
