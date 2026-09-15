# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A hypothesis-drawn cell space over the carry oracle's REGISTRY.

A SUPPLEMENT to `test_carry_vs_oracle.py`'s fixed named cells (`rola-oracle-tests`:
"never invent a new one ad hoc" -- the fixed cells stay the default battery), not a
replacement: this file exists so an oracle red yields a minimal failing
(widths, k_tok, seed, cohort, DV) automatically, via shrinking, rather than only at
whichever fixed point someone happened to name. Opt-in (`hypothesis_cells` marker,
deselected by default alongside `bench` -- `pyproject.toml`'s `addopts`) so its search
runs on request, never inside the default battery's runtime budget.

The strategy draws only over axes the KERNEL BOUNDARY admits (the descriptor's widths
and a shipped DV) or that are properties of the DRAW and never told to the kernel
(``k_tok``, ``seed``, ``cohort``). Drawing an unlawful shape would only ever exercise
the surface's own refusal, which is `tests/unit/test_carry_surface.py`'s claim.

RED BY DESIGN with the rest of the tier: the kernel has no implementation on this line.
"""
from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.cells import CellSpec
from benchmarks.cells import carry_cells as registry_cells
from rola.ops import carry as carry_ops
from tests.oracle.test_carry_vs_oracle import check, run

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
    pytest.mark.hypothesis_cells,
]

#: THE SHAPES THE REGISTRY DECLARES, read off it rather than restated: a topology or a
#: value width added there is picked up here without editing this file (the same "read
#: from the declaration, don't mirror it" discipline the census gates apply).
_TOPOLOGIES = sorted({(cell.widths, cell.tokens) for cell in registry_cells("oracle")})
_DV = sorted({cell.dv for cell in registry_cells("oracle")})


@st.composite
def drawn_cells(draw) -> CellSpec:
    """One drawn `CellSpec`, on the registry's own record shape.

    It is a registry record like any other -- the same fields, the same validation, the
    same descriptor and launch derivation -- so a shrunk failure is reportable BY NAME
    and can be pasted into `carry_cells.json` as a fixed cell.
    """
    widths, tokens = draw(st.sampled_from(_TOPOLOGIES))
    dv = draw(st.sampled_from(_DV))
    k_tok = draw(st.sampled_from([2, 4, 8, 16, None]))
    seed = draw(st.integers(min_value=0, max_value=2 ** 16))
    #: `cohort` must divide the length exactly (`clustered`'s reshape) and is only
    #: meaningful once `k_tok` narrows the support at all.
    cohort = draw(st.none() if k_tok is None else st.sampled_from([None, 32, 64]))
    return CellSpec(name=f"hypothesis-{widths}-{dv}-{k_tok}-{cohort}-{seed}",
                    widths=widths, dv=dv, tokens=tokens, warps_per_cta=8,
                    draw="dense" if k_tok is None else ("cohort" if cohort else "alt"),
                    k_tok=k_tok, cohort=cohort, support=1.0, backing="dense",
                    state="fresh", tier="oracle")


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(spec=drawn_cells())
def test_the_inter_term_is_the_fp64_reference_on_a_drawn_cell(spec):
    """The tier's own comparison, over a drawn cell instead of a fixed one: the same
    `run` and `check`, per slot against the same envelopes -- only the cell is drawn, never the
    assertion."""
    drawn, num, den, plane = run(spec)
    check(spec, drawn, num, den, plane)


@given(spec=drawn_cells())
def test_a_drawn_cell_is_a_lawful_shape(spec):
    """The strategy may only draw shapes the kernel boundary ADMITS: a drawn cell whose
    descriptor, launch shape or geometry block is refused would be exercising the
    surface's refusals, which is `tests/unit/test_carry_surface.py`'s claim, not this
    file's."""
    carry_ops.geometry_block(spec.descriptor(), spec.launch())
