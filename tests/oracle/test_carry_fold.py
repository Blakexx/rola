"""Gate for THE CARRY'S FOLD (csrc/rola/src/carry/ampere/fold.cuh): the state at a call's end, read through the probe, is the
fp64 reference of the fold's declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`) per slot, at both
orders of the even box's levels, and its den state the reference's mass per leaf (docs/internals/carry/ampere/fold.md#den).

Symbols: ``W`` the window (the device's), ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import carve_at
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from tests.oracle import carry_plan
from tests.oracle.carry_fixtures import (
    FORM_CELLS,
    FORM_CELLS64,
    box64,
    box64_dense,
    box64_same,
    listed_bundle,
    read_listed_bundle,
)
from tests.oracle.fold_reference import fold_state


#: THE BOUND, relative to each slot's envelope (sum |A| |v|, absolute): a slot takes at most one fp32 accumulation a
#: 16-token step, ``L / 16`` of them, each off by at most 2^-23 (the MMA's accumulate truncates); the rescales exact.
def _rtol(L: int) -> float:
    return L / 16 * 2.0 ** -23


#: A DENSE FOLD'S DEN WEIGHT is rounded twice (`fold_den_dense`: the slot's `f16(255 m 2^(4 - E))`, then the coefficient
#: times it by one HMUL2) where the reference's c_m rounds once: three f16 unit roundoffs a term at most, every term
#: positive (docs/internals/carry/ampere/fold.md#den; ruled 2026-11-03)
DENSE_DEN_WEIGHT = 3 * 2.0 ** -11


def _den_ratio(den: torch.Tensor, mass: torch.Tensor, L: int, weight: float = 0.0) -> float:
    """The den state's largest leaf distance from the reference's mass over its bound: the steps' fp32 accumulation as
    the state's and one fp32 add a window into the den state, and the form's own rounding of each term's weight
    (``weight``, relative; every term positive: the mass is its own envelope)."""
    bound = mass * (_rtol(L) + (L / window_of() + 1) * 2.0 ** -23 + weight)
    return ((den.double() - mass).abs() / bound.clamp_min(1e-300)).max().item()


def _probed(routes, values, facts, carve, unit, forced=None):
    """``(probe, den)``: one launch's final state and den state through the probes."""
    widths = routes.topology.widths
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    probe = torch.zeros((BH, N, d_v), dtype=torch.float32, device="cuda")
    den = torch.zeros((BH, N), dtype=torch.float32, device="cuda")
    carry_ops.probe_bind(probe, den)
    try:
        carry_ops.carry(routes, values, facts, carve, unit, forced=forced)
        torch.cuda.synchronize()
    finally:
        carry_ops.probe_bind(None)
    return probe, den


def _reference(routes, values, facts, level_s, BH, N, d_v):
    """``(state, env, mass)`` the reference's, in the probes' leaf order."""
    state, env, mass, _ = fold_state(routes, values, facts.list_exp, level_s, window_of())
    if level_s == 1:
        state, env, mass = state.permute(0, 2, 1, 3), env.permute(0, 2, 1, 3), mass.permute(0, 2, 1)
    return state.reshape(BH, N, d_v), env.reshape(BH, N, d_v), mass.reshape(BH, N)


#: THE EVEN BOX, 32 x 32 x 32 (two channel boxes at d_v 64)
EVEN_BOX = ((32, 32), 32)


def _fold_ratio(widths, L, write_sparse, growth, *, live=4, forced=None, box=None, dense_write=False, same=False):
    """``(ratio, plans)``: the call's largest slot distance from the reference over its bound, and its boxes' plans; at
    the planner's box, or at ``box`` (extents a level, channels); ``dense_write`` a write side dense at every level and
    the read side listing level ``write_sparse``."""
    if dense_write:
        routes, values = read_listed_bundle(widths, L, write_sparse, live=live)
    else:
        routes, values = listed_bundle(widths, L, write_sparse, live=live, same=same)
    at = torch.arange(L, device="cuda") // window_of()
    routes.scale_write[0].mul_(torch.exp2(growth * at.float())[None, :, None])
    padded, d_v = tuple(routes.topology.padded_widths), values.v.shape[-1]
    #: the 32-channel box unless named (at d_v 64 the planner's is the 64-channel one, gated by its own tests)
    carve = carve_at(padded, *routes.sparse, *(box or EVEN_BOX), d_v)
    assert carry_ops.arm_key(carve) in carry_ops.arms(), carry_ops.arm_key(carve)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.lists_form(carry_ops.arm_key(carve)))
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    unit = torch.zeros((BH, L), dtype=torch.int8, device="cuda")
    probe, den = _probed(routes, values, facts, carve, unit, forced)
    #: the reference's level S is the fold's rows: the other level on either side form (the alternating box's read list
    #: level, the same split's dense row level)
    want, env, mass = _reference(routes, values, facts, 1 - write_sparse, BH, N, d_v)
    err = (probe.double() - want).abs()
    assert (env > 0).float().mean() > 0.5, "most slots unwritten: the cell is vacuous"
    weight = DENSE_DEN_WEIGHT if dense_write else 0.0
    ratio = max((err / (env * _rtol(L)).clamp_min(1e-300)).max().item(), _den_ratio(den, mass, L, weight))
    return ratio, carry_plan.plans(routes, facts, carve, forced)


#: fresh cells whose write side has no list level (every level dense)
#: the dense cells, each at its own length or (`tokens`) at another: a partial last window whatever the device's W
#: (1,792 tokens: three and a half windows of 512; flat-small-dense's 256 one partial window)
DENSE_CELLS = (("flagship-dense", None), ("flat-small-multiwindow", None), ("corner-window-flip", None),
               ("flagship-dense", 1792), ("flat-small-dense", None))


def _cell_at(name: str, tokens: int | None):
    """The oracle cell ``name``, at ``tokens`` when given (its declared tail recomputed against the cells' window)."""
    import dataclasses

    from rola_devtools.cells.carry import WINDOW

    from measure.cells import carry_cells

    spec = next(c for c in carry_cells(tier="oracle") if c.name == name)
    if tokens is None:
        return spec
    tail = "ragged" if tokens % WINDOW else "divisible"
    regime = tuple((k, tail if k == "tail" else v) for k, v in spec.regime)
    return dataclasses.replace(spec, tokens=tokens, regime=regime)


@pytest.mark.parametrize("name,tokens", DENSE_CELLS, ids=[f"{n}-L{t}" if t else n for n, t in DENSE_CELLS])
def test_a_write_side_with_no_list_level_folds_to_the_reference(name, tokens):
    """A dense write side's final state per slot within the bound of the reference's: each token's factor F in its
    coefficient, every list at its window's bound (`dense_exponents`), a partial last window folded with the rest."""
    from measure.subjects import carry_operands

    spec = _cell_at(name, tokens)
    routes, values, facts, carve, unit = carry_operands(spec)
    assert not carve.fold.lists, f"{name}'s write side lists"
    widths, L = routes.topology.widths, routes.tokens
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    probe, den = _probed(routes, values, facts, carve, unit)
    want, env, mass = _reference(routes, values, facts, routes.level_s, BH, N, d_v)
    assert (env > 0).float().mean() > 0.5, "most slots unwritten: the cell is vacuous"
    ratio = ((probe.double() - want).abs() / (env * _rtol(L)).clamp_min(1e-300)).max().item()
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
    den_ratio = _den_ratio(den, mass, L, DENSE_DEN_WEIGHT)
    assert den_ratio <= 1.0, f"a leaf's den is {den_ratio:.3g}x its bound"


@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_the_fold_is_the_reference_per_slot(write_sparse, growth):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64): every leaf and channel of the final state within the bound
    of the reference's over four windows -- the write gain steady, or scaled 2^growth a window, so the units climb
    and rescale the state, or are held under the bound's headroom."""
    ratio, _ = _fold_ratio((64, 64), 2048, write_sparse, growth)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_the_64_channel_box_folds_to_the_reference(write_sparse, growth):
    """The 64-channel box (one m-tile of 16 rows by eight n-tiles, two den words a lane): every leaf and channel of the
    final state and every leaf's den within its bound of the reference's, the units steady, climbing or held."""
    ratio, _ = _fold_ratio((64, 64), 2048, write_sparse, growth, box=box64(write_sparse))
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("box", [((32, 32), 32), "box64"], ids=["32", "64"])
@pytest.mark.parametrize("read_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_a_dense_write_side_folds_to_the_reference(read_sparse, growth, box):
    """A dense write side (its fold's coefficients the read side's list level, the same split) at the 32-channel box
    and the 64-channel one (two fold lists a warp, the den's quads): every leaf's state and den within its bound."""
    box = box64_dense(read_sparse) if box == "box64" else box
    ratio, _ = _fold_ratio((64, 64), 2048, read_sparse, growth, box=box, dense_write=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("box", [((32, 32), 32), "box64"], ids=["32", "64"])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_the_same_split_folds_to_the_reference(level, growth, box):
    """Both sides listing one level (the fold's lists the readout's digits) at the 32-channel box and the 64-channel one
    (two fold lists a warp): every leaf's state and den within its bound."""
    box = box64_same(level) if box == "box64" else box
    ratio, _ = _fold_ratio((64, 64), 2048, level, growth, box=box, same=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS64)
def test_every_plan_form_folds_at_the_64_channel_box(widths, live, forced, forms):
    """Each of the plan's forms at the 64-channel box: the final state within its bound per slot."""
    ratio, plans = _fold_ratio(widths, 2048, 0, 8, live=live, forced=forced, box=box64(0))
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS)
def test_every_plan_form_folds_to_the_reference(widths, live, forced, forms):
    """Each of the plan's forms (`tests/oracle/carry_fixtures.py` `FORM_CELLS`), its cell taking them by the plans'
    model: the final state within its bound per slot, the units climbing 2^8 a window."""
    ratio, plans = _fold_ratio(widths, 2048, 0, 8, live=live, forced=forced)
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
