# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INT8 CARRY'S DECLARED ARITHMETIC AGAINST THE fp64 OP, and the report beside it.

Three claims. (1) The Hadamard rotation is exact for the op: it is orthogonal and the op is linear in the values, so
un-rotating the op's output on rotated values IS the op on the values. (2) The declared recurrence is the op's: with
every rounding turned off, `int_reference` reproduces `inter_reference` to fp64 on every cell it is run on -- the
structure is judged before any rounding is. (3) THE REPORT: on the oracle cells, the declared arithmetic's error against
the unquantized op, per slot over the slot's envelope and per token in norm, at one and two readout digits, in the rotated and
the real basis. It is PRINTED and never gated (`docs/internals/...`: the truth oracle reports, the fidelity gate is the
op gate on the same quantized inputs, quality is the layer oracle's). The one property asserted is monotone: two digits
never lose to one on the same draw.
"""
from __future__ import annotations

import pytest
import torch

from measure.cells import by_name, realize
from rola.ops.window import window_of
from tests.oracle import int_reference as iref
from tests.oracle import reference
from tests.oracle.fixtures import with_gain

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

STRUCTURE_CELLS = ["flat-small-alt-k4", "one-box-short", "deep3-alt-k4"]
REPORT_CELLS = ["flat-small-dense", "flat-small-alt-k4", "flat-small-alt-k16", "one-box-alt-k4",
                "flagship-alt-k4", "flagship-cohort-k16", "deep3-alt-k4"]


def _sparsity(cell):
    return cell.declared_sparsity()


def test_the_hadamard_is_orthogonal_and_commutes_with_the_op():
    dev = "cuda"
    h = iref.hadamard(64, dev)
    eye = torch.eye(64, device=dev, dtype=torch.float64)
    assert torch.allclose(h @ h.T, eye, atol=1e-13)
    cell = by_name("flat-small-alt-k4")
    drawn = realize(cell, device=dev)
    read, write, gain, v = drawn.doubles()
    num, den, state = reference.inter_reference(read, with_gain(write, gain), v, cell.widths, window_of())
    num_r, den_r, state_r = reference.inter_reference(read, with_gain(write, gain), iref.rotate(v, h), cell.widths, window_of())
    assert torch.allclose(iref.unrotate(num_r, h), num, rtol=1e-12, atol=1e-12 * num.abs().max())
    assert torch.allclose(den_r, den)
    assert torch.allclose(iref.unrotate(state_r[..., :-1], h), state[..., :-1], rtol=1e-12,
                          atol=1e-12 * state.abs().max())


@pytest.mark.parametrize("name", STRUCTURE_CELLS)
def test_the_declared_recurrence_is_the_op_when_nothing_is_rounded(name):
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    read, write, gain, v = drawn.doubles()
    want = reference.inter_reference(read, with_gain(write, gain), v, cell.widths, window_of())
    got = iref.int_reference(read, with_gain(write, gain), v, cell.widths, _sparsity(cell), window_of(), exact=True)
    for a, b, what in zip(got, want, ("num", "den", "state")):
        scale = b.abs().max().clamp_min(1e-300)
        assert torch.allclose(a, b, rtol=1e-11, atol=1e-11 * float(scale)), f"{name}: {what} is not the op's"


@pytest.mark.parametrize("name", REPORT_CELLS)
def test_the_quantization_report(name, capsys):
    """REPORTED, NEVER GATED: the declared int arithmetic's error against the unquantized op. The two properties
    asserted are structural: two readout digits everywhere never lose to one (read with the s32 panel, so the packed
    panel's step does not mask them), and the packed s16 panel's running sum stays inside its 32767 units on every
    cell (the no-wrap claim behind two channels an ATOMS.ADD.32)."""
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    lines = [f"{name}: widths={cell.widths} tokens={cell.tokens} draw={cell.draw} k={cell.k_tok}"]
    for basis, rot in (("rotated-basis-read", False), ("real-basis-after-unrotate", True)):
        rows = iref.quantization_report(drawn, _sparsity(cell), window_of(), rotate_values=rot)
        for r in rows:
            lines.append(f"  {basis:26s} eps={r['readout_eps']:<8.3g} flagged {100*r['flag_rate']:5.1f}%: "
                         f"num per-slot err / (2^-8 env): max {r['slot_max']:8.2f}, frac>1 {100*r['slot_frac_over']:5.1f}% | "
                         f"per-token norm err/env: max {r['token_max']:.2e} mean {r['token_mean']:.2e} | "
                         f"state per-slot max {r['state_slot_max']:8.2f} frac>1 {100*r['state_slot_frac_over']:5.1f}% | "
                         f"den rel {r['den_max_rel']:.1e} | panel peak {r['panel_peak']:.0f}")
            assert r["panel_peak"] < iref.PANEL_LEVELS, "the packed panel's running sum left its s16 half"
    #: the readout's own monotonicity, read with the panel step off so its rounding does not mask the digits
    one, two = iref.quantization_report(drawn, _sparsity(cell), window_of(), knobs=(1.0, 0.0),
                                        panel=False)
    lines.append(f"  s32 panel: one readout digit mean {one['token_mean']:.2e} -> two {two['token_mean']:.2e}")
    assert two["token_mean"] <= one["token_mean"] * 1.0001, "two readout digits lost to one on the same draw"
    with capsys.disabled():
        print("\n" + "\n".join(lines))


#: the pre-pass gates need a token that reads a WRITTEN state: cells of more than one window
PREPASS_CELLS = ["one-box-alt-k4", "flagship-alt-k4", "flagship-cohort-k16"]


@pytest.mark.parametrize("name", PREPASS_CELLS)
def test_the_prepass_reference_is_the_op_when_nothing_is_rounded(name):
    """The pre-pass's structure: with every rounding off its ``den`` is the fp64 op's ``den`` exactly."""
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    read, write, gain, v = drawn.doubles()
    _, want_den, _ = reference.inter_reference(read, with_gain(write, gain), v, cell.widths, window_of())
    den, _ = iref.prepass_reference(read, with_gain(write, gain), v, cell.widths, _sparsity(cell), window_of(), exact=True)
    assert torch.allclose(den, want_den, rtol=1e-11, atol=1e-11 * float(want_den.abs().max()))


@pytest.mark.parametrize("name", PREPASS_CELLS)
def test_the_prepass_den_is_within_its_digits_and_its_scale_is_a_bound(name, capsys):
    """THE TWO GATES OF THE PRE-PASS. ``den``: per token, its relative error is the u8 READ ROW's -- den multiplies
    the same 255-level amplitude row num does, so it is exactly as precise as num (about 2^-9 on the cells) and that
    rounding is common-mode between the two in the ratio; the 15-bit mass column sits under it. The bound asserted
    is the format's: 2^-8. ``scale``: a BOUND -- never below the readout's own |partials| sum on the same quantized
    state (an under-read would let the packed panel wrap); the report prints how far above."""
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    read, write, gain, v = drawn.doubles()
    sp = _sparsity(cell)
    _, want_den, _ = reference.inter_reference(read, with_gain(write, gain), v, cell.widths, window_of())
    env_num, _, _ = reference.inter_reference(read, with_gain(write, gain), v.abs(), cell.widths, window_of())
    den, scale = iref.prepass_reference(read, with_gain(write, gain), v, cell.widths, sp, window_of())
    live = want_den > 0
    rel = ((den - want_den).abs() / want_den.clamp_min(1e-300))[live]
    assert float(rel.max()) <= 2.0 ** -8, f"den left the amplitude row's precision: max rel {float(rel.max()):.2e}"
    #: the readout's |partials| sum is at most the fp64 op on |v| (the envelope's own sum), so the bound must clear it
    env_sum = env_num.abs().amax(-1)
    assert bool((scale[live] * (1 + 1e-9) >= env_sum[live]).all()), "scale_t under-reads the readout's envelope"
    over = (scale[live] / env_sum[live].clamp_min(1e-300))
    with capsys.disabled():
        print(f"\n{name}: den rel err max {float(rel.max()):.2e} mean {float(rel.mean()):.2e} | "
              f"scale / envelope-max median {float(over.median()):.1f}x  p99 {float(over.quantile(0.99)):.1f}x")


# --------------------------------------------------------------- the intra's declared arithmetic, on the bundle

INTRA_CELLS = ["flat-small-alt-k4", "one-box-short", "deep3-alt-k4"]


def _bundle(name):
    """The cell's draw in the bundle's format: u8 rows, f16 values, each level sparse or dense as the cell declares."""
    from rola.routing.factors import RouteFactors
    from rola.routing.types import OpaqueRouting, Topology
    from rola.values import Values

    cell = by_name(name)
    drawn = realize(cell)
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in cell.widths))
    sparse = tuple(tuple(level[side] for level in cell.declared_sparsity()) for side in (0, 1))
    routes = RouteFactors.from_levels(topology, drawn.read, with_gain(drawn.write, drawn.gain.float()), sparse=sparse)
    return cell, routes, Values.from_projection(drawn.v)


@pytest.mark.parametrize("name", INTRA_CELLS)
def test_the_intra_reference_is_the_intra_on_the_bundle_when_nothing_is_rounded(name):
    """Unrounded, the declared intra times the reader's factor IS the fp64 intra of the bundle's own dequantized
    levels and values: the structure (the Grams, the writer factor, the causal window) before any rounding."""
    cell, routes, values = _bundle(name)
    num, den = iref.intra_bundle_reference(routes, values, window_of())
    read, write = routes.dequantized()
    want_num, want_den = reference.intra_reference(read, write, values.dequantized(), cell.widths, window_of())
    envelope, _ = reference.intra_reference(read, write, values.dequantized().abs(), cell.widths, window_of())
    reader = torch.ones_like(den)
    for scale in routes.scale_read:
        reader = reader * scale.double()
    #: per slot against the slot's envelope: the dequantized operands are fp32, a few units of 2^-24 each
    assert ((num * reader[..., None] - want_num).abs() <= 2.0 ** -20 * envelope).all()
    assert ((den * reader - want_den).abs() <= 2.0 ** -20 * want_den).all()


