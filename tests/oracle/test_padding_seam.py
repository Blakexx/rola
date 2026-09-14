# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""PAD (K50 "PADDING AT THE PRODUCER" / "d_v PADDING IS ALSO HOST-SIDE"): a model with
any level width and any value width is served by host-side padding, with no kernel
change and no oracle arithmetic change.

**TWO CONTRACTS, KEPT SEPARATE (KERNEL_STANDARDS §R13).** The API accepts any shape and
pads; the kernel accepts only its own descriptor's strict shape and refuses the rest.
`test_the_api_boundary_pads_and_the_kernel_boundary_refuses` is both halves, side by
side: the API is `rola.ops.prefill`'s `_descriptor`/`_pack_operands` operand-packing
seam (C2b2), which pads through `rola.ops.padding` onto the carry launch surface
(`rola.ops.carry`, C1c) this line's kernel has no body for yet.

**PREFILL (no kernel on this line, C_CLEAN_SLATE):** `naive_rola` is already
dimension-generic (it derives `N`/strides from whatever `Topology` it is handed and
consumes whatever per-level tensors it is given), so the prefill-side proof is a pure
fp64 equivalence: the SAME learned routing, padded with `rola.ops.padding.pad_routes`
to the next power of two >= 16 per level, produces the identical `y` a model built
directly at that padded width (with the extra digits simply never drawn on) would.

**DECODE has a real kernel on this line**, so its half is graded bit for bit through
`rola.ops.decode._decode_step`/`derive_decode_geometry`: a `d_v` decode's own arm table
does not ship is padded to the smallest one that is, and the result must equal running
the same weights directly at that shipped `DV` with the pad columns of `v` held at
zero (K50: "Kernels never see a width other than their own").
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops.carry import box_shape
from rola.ops.decode import _decode_step, derive_decode_geometry
from rola.ops.intra import DENSE_BOTH, WINDOW_SMALL, intra_forward_padded
from rola.ops.lattice import to_canonical, to_lattice
from rola.ops.naive import naive_rola
from rola.ops.padding import pad_routes, pad_v
from rola.ops.paging import bytes_equal, from_split_planes, to_split_planes
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from tests.oracle.oracle_fixtures import _simplex
from tests.oracle.tolerances import BF16_RTOL

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

    y_logical, _ = naive_rola(v, read_levels, write_levels, g_write, topology, decay=None)

    padded_topology = topology.padded()
    padded_read = pad_routes(read_levels, padded_topology.widths)
    padded_write = pad_routes(write_levels, padded_topology.widths)
    y_padded, _ = naive_rola(pad_v(v, d_v), padded_read, padded_write, g_write,
                             padded_topology, decay=None)
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
# TWO CONTRACTS: the API pads, the kernel refuses (KERNEL_STANDARDS §R13)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the launch surface is the extension's boundary")
def test_the_api_boundary_pads_and_the_kernel_boundary_refuses():
    """(i) API: `rola.ops.prefill`'s `_descriptor`/`_pack_operands` -- the operand-
    packing seam onto the carry launch surface -- take a genuinely unpadded
    shape ((31, 45), d_v=8) without ever raising a SHAPE error: they pad, and the only
    failure `carry_forward` can reach past them is the HONEST ABSENCE of the carry
    kernel body (`C_CLEAN_SLATE`: carry is not built on this line). (ii) KERNEL: the
    descriptor the state addresses bytes with (`rola._state.StateFormat`) refuses that
    same unpadded shape directly, by name -- the boundary a real kernel launch sits
    behind.
    """
    from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
    from rola.ops import carry as carry_ops
    from rola.ops.prefill import _conservative_activity, _descriptor, _liveness_words, _pack_operands

    widths = (31, 45)
    B, T, H, d_v = 1, 2, 1, 8
    device = "cuda"
    read_levels = tuple(torch.rand(B, T, H, w, dtype=torch.float64, device=device) for w in widths)
    write_levels = tuple(torch.rand(B, T, H, w, dtype=torch.float64, device=device) for w in widths)
    g_write = torch.ones(B, T, H, dtype=torch.float64, device=device)
    v = torch.randn(B, T, H, d_v, dtype=torch.float64, device=device)

    #: (i) THE API PADS: no ValueError about shape reaches the caller. `_pack_operands`
    #: widens (31, 45) to (32, 64) and d_v=8 to the smallest shipped DV (32) before the
    #: call ever names `carry_forward`, so the only failure past it is the HONEST
    #: ABSENCE of the carry kernel body -- not a shape excuse.
    pread, pwrite, gwrite, v_bh, v_bthd = _pack_operands(read_levels, write_levels, g_write, v)
    bh, dv = v_bh.shape[0], v_bh.shape[2]
    descriptor = _descriptor(widths, dv, bh)
    assert descriptor.B == (32, 64) and descriptor.DV == 32, (
        "the descriptor names the PADDED shape, never the caller's logical one")
    launch = carry_ops.LaunchShape()
    geometry = carry_ops.geometry_block(descriptor, launch)
    liveness = _liveness_words(pread, pwrite, descriptor)
    activity = _conservative_activity(descriptor, bh, v_bh.device)
    routes = carry_ops.RoutePlanes(read=pread, write=pwrite, gain=gwrite)
    with pytest.raises(RuntimeError, match="no implementation on this line"):
        carry_ops.carry_forward(routes, v_bh, descriptor=descriptor, geometry=geometry,
                                liveness=liveness, activity=activity, launch=launch)

    #: (ii) THE KERNEL REFUSES: the unpadded widths, presented directly to the
    #: descriptor a real launch would be checked against, are refused by name.
    with pytest.raises(ValueError, match="level 0 is 31"):
        StateFormat(D=2, B=widths, order=CANONICAL_ORDER, page_bits=PAGE_RECTANGLE_BITS,
                   DV=32, dtype=SPLIT_PLANE_DTYPE, ids=1)


def test_box_shape_is_a_second_kernel_boundary_that_refuses_unpadded_widths():
    """`box_shape` (this file's own lattice-capacity gate, unrelated to and
    independent of the descriptor above) refuses the same unpadded shape too --
    a second, independently-arrived-at boundary agreeing with the first."""
    with pytest.raises(ValueError):
        box_shape((31, 45), k=4, m=8)


# ---------------------------------------------------------------------------
# DECODE: a real kernel on this line -- graded bit for bit, not by the oracle band
# ---------------------------------------------------------------------------

pytestmark_decode = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel")
def test_decode_padded_d_v_matches_the_shipped_dv_with_a_zero_tail_bit_for_bit():
    """Item (i) of the ADDITION: a padded decode run at d_v=48 must equal the SAME
    model at the shipped DV=64 with the extra 16 value columns unreachable (zero),
    bit for bit -- both `y` and the stored state.

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

    config = derive_decode_geometry(topology, d_v=d_v_logical, decay=False, BH=B * H,
                                    device=device)
    assert config.d_v == DV, "d_v=48 must pad up to the shipped DV=64, not run at 48 directly"

    def run(v):
        #: THE STATE PLANE IS ALWAYS SIZED AT THE PADDED `config.cols` (K50: pages
        #: carry the padded width) -- whether `v` here is the caller's logical
        #: d_v=48 or the test's own hand-padded DV=64, the plane the kernel writes
        #: into is the SAME shape either way, which is exactly the claim under test.
        cols = config.cols
        state = to_split_planes(
            to_lattice(torch.zeros(B, H, N, cols, device=device, dtype=torch.float32),
                      widths, config.lattice_k, config.lattice_m))
        y, state, _ = _decode_step(
            v.float(), tuple(t.float() for t in read_levels), tuple(t.float() for t in write_levels),
            g_write.float(), config, state, decay=None)
        return y, to_canonical(from_split_planes(state), widths, config.lattice_k, config.lattice_m)

    #: THE PADDED CALL: v at the caller's logical d_v=48.
    y_padded, state_padded = run(v_logical)
    #: THE EXPLICIT CALL: the identical v, hand-padded to DV=64 with a zero tail --
    #: exactly what the seam does internally, done here by the TEST instead, so the
    #: two calls are independent of each other but for that one shared step.
    v_explicit = pad_v(v_logical, DV)
    y_explicit, state_explicit = run(v_explicit)

    assert torch.equal(y_padded, y_explicit[..., :d_v_logical]), (
        "a d_v=48 decode step must equal the DV=64 step's first 48 output columns, bit for bit")
    assert bytes_equal(state_padded[..., :d_v_logical], state_explicit[..., :d_v_logical]), (
        "the stored state's real d_v columns must agree bit for bit")
    assert bytes_equal(state_padded[..., d_v_logical:DV], state_explicit[..., d_v_logical:DV]), (
        "the padded state columns are dead in BOTH calls (nothing ever wrote a nonzero v there)"
    )


# ---------------------------------------------------------------------------
# INTRA: item (ii) of the ADDITION -- graded against its own oracle
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="intra is a CUDA kernel")
def test_intra_padded_nonpow2_levels_and_d_v_match_the_oracle():
    """A D=2 topology at (17, 19) -- neither a power of two, neither the shipped
    LEVEL_WIDTH=256 -- and d_v=48 (VALUE_WIDTH=64 shipped), through
    `intra_forward_padded`. With a zero initial state and no decay, one window's
    whole recurrence is its intra term (`test_intra_reference.py`'s own claim), so
    the padded kernel's ``o / (den + eps)`` must land in the same bf16 band every
    other intra gate is held to against the UNPADDED fp64 oracle.
    """
    from rola.ops.constants import READOUT_EPS

    device = "cuda"
    widths = (17, 19)
    B, H, d_v = 2, 2, 48
    window = WINDOW_SMALL
    generator = torch.Generator(device=device).manual_seed(3)

    read_levels = tuple(_simplex((B, window, H, w), 1.0, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, window, H, w), 1.0, generator, device) for w in widths)
    g_write = (torch.rand(B, window, H, device=device, dtype=torch.float64, generator=generator)
              + 0.5)
    v = torch.randn(B, window, H, d_v, device=device, dtype=torch.float64, generator=generator)

    topology = _dense_topology(widths)
    y_oracle, _ = naive_rola(v, read_levels, write_levels, g_write, topology, decay=None)

    def bf16(t):
        return t.to(torch.bfloat16)

    gwrite_bh = g_write.permute(0, 2, 1).reshape(B * H, window).to(torch.bfloat16).contiguous()
    v_token_major = bf16(v).contiguous()
    o, den = intra_forward_padded(
        tuple(bf16(t) for t in read_levels), tuple(bf16(t) for t in write_levels),
        gwrite_bh, v_token_major, (DENSE_BOTH,) * len(widths), window=window)
    y_kernel = (o / (den.unsqueeze(-1) + READOUT_EPS)).view(B, H, window, d_v).permute(0, 2, 1, 3)

    err = float((y_kernel.double() - y_oracle).abs().max() / max(1e-30, float(y_oracle.abs().max())))
    assert err < BF16_RTOL, f"padded intra leaves the bf16 band at {err:.3e}"
