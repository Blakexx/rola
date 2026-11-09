# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""PAD (K50 "PADDING AT THE PRODUCER" / "d_v PADDING IS ALSO HOST-SIDE"): a model with
any level width and any value width is served by host-side padding, with no kernel
change and no oracle arithmetic change.

**TWO CONTRACTS, KEPT SEPARATE (CONSTITUTION §R13).** The API accepts any shape and
pads; the kernel boundary accepts only its own strict shape and refuses the rest
(`test_box_shape_is_a_second_kernel_boundary_that_refuses_unpadded_widths`).

**PREFILL (no kernel on this line, C_CLEAN_SLATE):** `naive_rola` is already
dimension-generic (it derives `N`/strides from whatever `Topology` it is handed and
consumes whatever per-level tensors it is given), so the prefill-side proof is a pure
fp64 equivalence: the SAME learned routing, padded with `rola.ops.padding.pad_routes`
to the next power of two >= 16 per level, produces the identical `y` a model built
directly at that padded width (with the extra digits simply never drawn on) would.

**DECODE has a real kernel on this line**, so its half is graded on the stored state bit
for bit, and on `y` to its sums' order, through
`rola.ops.decode._decode_step`/`derive_decode_geometry`: a `d_v` decode's own arm table
does not ship is padded to the smallest one that is, and the result must equal running
the same weights directly at that shipped `DV` with the pad columns of `v` held at
zero (K50: "Kernels never see a width other than their own").
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops.decode import _decode_step, derive_decode_geometry
from rola.ops.lattice import box_shape, to_canonical, to_lattice
from rola.ops.naive import naive_rola
from rola.ops.padding import pad_routes, pad_v
from rola.ops.paging import bytes_equal
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from tests.oracle.fixtures import assert_reordered_close, logical_state, state_pages, step_bundles, with_gain
from tests.oracle.oracle_fixtures import _simplex

# ---------------------------------------------------------------------------
# PREFILL: the fp64 naive is the ONE ground truth on this line (no kernel body)
# ---------------------------------------------------------------------------


def _dense_topology(widths):
    return Topology(levels=tuple(
        IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation())
        for w in widths))


def _padded_case(widths, d_v, *, B=2, T=5, H=2, seed=0):
    """`(y_logical, y_padded)`: the same learned routes, run at `widths`/`d_v` and
    again after `pad_routes`/`pad_v` widen them to the padded shape -- fp64 throughout,
    CPU, no extension involved. `naive_rola` never sees the padding decision; it just
    consumes whichever tensors and `Topology` it is handed (K50: "no oracle arithmetic
    change")."""
    generator = torch.Generator().manual_seed(seed)
    topology = _dense_topology(widths)
    read_levels = tuple(_simplex((B, T, H, w), 1.0, generator, "cpu") for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 1.0, generator, "cpu") for w in widths)
    g_write = torch.rand(B, T, H, dtype=torch.float64, generator=generator) + 0.5
    v = torch.randn(B, T, H, d_v, dtype=torch.float64, generator=generator)

    y_logical, _ = naive_rola(v, read_levels, with_gain(write_levels, g_write), topology, decay=None)

    padded_topology = topology.padded()
    padded_read = pad_routes(read_levels, padded_topology.widths)
    padded_write = pad_routes(write_levels, padded_topology.widths)
    y_padded, _ = naive_rola(pad_v(v, d_v), padded_read, with_gain(padded_write, g_write), padded_topology, decay=None)
    return y_logical, y_padded


@pytest.mark.parametrize("widths", [(31, 45), (17, 16, 33)],
                        ids=["depth2-nonpow2", "depth3-nonpow2"])
def test_prefill_padded_widths_are_bit_identical_to_the_unpadded_model(widths):
    """Item 5(a): widths (31, 45) at D=2 and (17, 16, 33) at D=3 -- ``y`` bit for bit
    against the same model run at the unpadded, logical widths (the extra padded
    states unreachable: no digit ever draws mass there, so a zero-padded input can
    never make one live)."""
    y_logical, y_padded = _padded_case(widths, d_v=8)
    assert torch.equal(y_logical, y_padded), (
        "a zero-column pad on the routing levels must not move a single value: "
        "softmax over b real + (B-b) dead columns IS the b-way softmax")


def test_prefill_padded_d_v_is_bit_identical_with_a_zero_tail():
    """Item 5(b): d_v = 48 vs d_v = 64 with a zero tail -- y[.., :48] bit for bit."""
    y_logical, y_padded = _padded_case((32, 64), d_v=48)
    assert y_padded.shape[-1] == 48, "pad_v pads v, not the readout width itself here"
    assert torch.equal(y_logical, y_padded)


def test_prefill_padding_is_the_ruled_mechanism_not_an_approximation_of_it():
    """The `-inf`-logit-then-softmax mechanism K50 states, and the zero-pad-the-solved-
    level mechanism this seam actually runs, are the SAME tensor: prove it directly on
    one level rather than only end to end."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 31, dtype=torch.float64, generator=generator)
    padded_logits = torch.cat([logits, torch.full((3, 1), float("-inf"), dtype=torch.float64)], dim=-1)
    solved_then_padded = pad_routes((torch.softmax(logits, dim=-1),), (32,))[0]
    padded_then_solved = torch.softmax(padded_logits, dim=-1)
    assert torch.equal(solved_then_padded, padded_then_solved)


# ---------------------------------------------------------------------------
# TWO CONTRACTS: the API pads, the kernel refuses (CONSTITUTION §R13)
# ---------------------------------------------------------------------------


def test_box_shape_is_a_second_kernel_boundary_that_refuses_unpadded_widths():
    """`box_shape` (this file's own lattice-capacity gate, unrelated to and
    independent of the descriptor above) refuses the same unpadded shape too --
    a second, independently-arrived-at boundary agreeing with the first."""
    with pytest.raises(ValueError):
        box_shape((31, 45), k=4, m=8)


# ---------------------------------------------------------------------------
# DECODE: a real kernel on this line -- graded against itself, not by the oracle band
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel")
def test_decode_padded_d_v_matches_the_shipped_dv_with_a_zero_tail():
    """Item (i) of the ADDITION: a padded decode run at d_v=48 must equal the SAME
    model at the shipped DV=64 with the extra 16 value columns unreachable (zero):
    the stored state bit for bit, `y` to its sums' order.

    Decode's own arm key is `(DV, D, decay)` only (`rola.ops.decode.arms`); per-level
    widths carry no floor of their own (`derive_decode_geometry`'s own docstring) --
    `(64, 64)` is used here as an ALREADY-SHIPPED topology so the row isolates the
    ``d_v`` axis alone, rather than re-covering decode's any-width tolerance
    (`test_decode_vs_oracle.py`'s `_TOPOLOGIES` already covers that).
    """
    device = "cuda"
    widths = (64, 64)
    B, H, d_v_logical, DV = 1, 2, 48, 64
    topology = _dense_topology(widths)
    N = math.prod(widths)
    generator = torch.Generator(device=device).manual_seed(7)

    read_levels = tuple(_simplex((B, 1, H, w), 1.0, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, 1, H, w), 1.0, generator, device) for w in widths)
    g_write = torch.rand(B, 1, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    v_logical = torch.randn(B, 1, H, d_v_logical, device=device, dtype=torch.float64,
                            generator=generator)

    config = derive_decode_geometry(topology, d_v=d_v_logical, decay=False)
    assert config.d_v == DV, "d_v=48 must pad up to the shipped DV=64, not run at 48 directly"

    def run(v):
        #: THE STATE PLANE IS ALWAYS SIZED AT THE PADDED `config.cols` (K50: pages
        #: carry the padded width) -- whether `v` here is the caller's logical
        #: d_v=48 or the test's own hand-padded DV=64, the plane the kernel writes
        #: into is the SAME shape either way, which is exactly the claim under test.
        cols = config.cols
        state = state_pages(
            to_lattice(torch.zeros(B, H, N, cols, device=device, dtype=torch.float32),
                      widths, config.lattice_k, config.lattice_m))
        routes, values = step_bundles(v, read_levels, with_gain(write_levels, g_write), topology)
        y, state, _ = _decode_step(routes, values, config, state, decay=None)
        logical = logical_state(state, config.d_v).reshape(B, H, N, cols)
        return y, to_canonical(logical, widths, config.lattice_k, config.lattice_m)

    #: THE PADDED CALL: v at the caller's logical d_v=48.
    y_padded, state_padded = run(v_logical)
    #: THE EXPLICIT CALL: the identical v, hand-padded to DV=64 with a zero tail --
    #: exactly what the seam does internally, done here by the TEST instead, so the
    #: two calls are independent of each other but for that one shared step.
    v_explicit = pad_v(v_logical, DV)
    y_explicit, state_explicit = run(v_explicit)

    #: y is summed by atomics in queue order, so two runs agree to summation reordering; the state is exact.
    assert_reordered_close(y_padded, y_explicit[..., :d_v_logical],
                           "a d_v=48 decode step against the DV=64 step's first 48 output columns")
    assert bytes_equal(state_padded[..., :d_v_logical], state_explicit[..., :d_v_logical]), (
        "the stored state's real d_v columns must agree bit for bit")
    assert bytes_equal(state_padded[..., d_v_logical:DV], state_explicit[..., d_v_logical:DV]), (
        "the padded state columns are dead in BOTH calls (nothing ever wrote a nonzero v there)"
    )


# ---------------------------------------------------------------------------
# INTRA: item (ii) of the ADDITION -- graded against its own oracle
# ---------------------------------------------------------------------------
