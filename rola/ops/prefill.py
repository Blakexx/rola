# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The box-native PREFILL: the carry and the intra as one call.

THE TWO KERNELS ARE ONE OPERATOR, and this module is the seam that says so. The
window grid is GLOBAL and FIXED, identical for both: the carry computes every
pair whose write fell in a STRICTLY EARLIER window than its read, and the intra
computes every pair INSIDE one window. Together they are the whole recurrence,
with no pair counted twice and none dropped — the study's exactness clause, which
`tests/oracle/test_carry_vs_oracle.py::test_m2_decomposition_is_exact` states
independently in fp64.

THE CARRY LEG IS `rola.ops.carry`: this module builds that surface's five
arguments — the descriptor, the geometry block, the liveness words, the activity
bits and the launch shape — from what it already has (the packed planes and the
widths), and calls `carry_forward`. The box is a launch
consequence and the window is a kernel constant (`carry_ops.WINDOW`); neither is a
parameter of this operator.

Symbols, each at first use: ``widths`` the per-level digit counts, ``D`` levels,
``W`` the window length in tokens (the carry's constant), ``d_v`` value channels,
``modes`` the per-level support-mode declaration the intra's zero certificate reads.

WHY THE INTRA GOES SECOND AND REDUCES.  The intra kernel's write-back is a RED
(``fan_in_add``), not a store, so it ADDS into whatever ``o``/``den`` already
hold. The carry allocates and fills them; the intra then folds its term into the
same buffers on the same stream. Nothing is summed on the host, and there is no
second output pair to reconcile — which is the whole reason the intra was built
as a reducer rather than as a returner.

WHAT THIS OPERATOR REFUSES, and why each refusal is structural rather than a
missing feature:

* ``L % W != 0`` — the intra's tile grid is a COMPILE-TIME count per arm, so a
  partial trailing window has no kernel. (The carry alone tolerates one; the
  combined operator cannot, and a host-side patch for the tail would be a second
  code path for the case the window grid exists to remove.)
* any topology but ``(256, 256)`` — the intra's level width is 256 by
  instantiation. The carry's ``(64, 64)`` arm therefore has no intra partner and
  is an INTER-ONLY conformance cell; see `docs/internals/intra/intra_kernel.md`.
"""
from __future__ import annotations

import torch

from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.facts.liveness import LivenessLayout, side_statics
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops import padding as padding_ops
from rola.ops._ext import extension
from rola.ops.liveness import liveness_words

#: THE WINDOW, restated so a caller has one import. It is the CARRY KERNEL's constant
#: (the axis law), and the intra family must carry an arm at the same number, because
#: the two kernels run ONE window grid.
WINDOW = carry_ops.WINDOW

#: the level widths the intra kernel is instantiated at, and therefore the ones the
#: combined operator has.  A topology is admitted when BOTH families carry an arm at
#: its ``(D, B, W)``; the widths are uniform per topology, as everywhere else.
LEVEL_WIDTH = intra_ops.LEVEL_WIDTH
LEVEL_WIDTHS = (intra_ops.LEVEL_WIDTH, intra_ops.LEVEL_WIDTH_DEEP3, intra_ops.LEVEL_WIDTH_DEEP4)


def _padded_shape(widths, dv: int) -> padding_ops.PaddedShape:
    """This call's `rola.ops.padding.PaddedShape` -- every level widened to
    `padded_level_width` and `dv` widened to the smallest of `carry_ops.SHIPPED_DV`
    that admits it (KERNEL_STANDARDS §the "two contracts": the carry surface's
    kernel boundary refuses anything but its own strict shape, so this seam's
    descriptor and operand packing pad BEFORE either helper builds anything the
    boundary would see). A caller already at a shipped shape pays nothing here --
    `PaddedShape.of` is a no-op past an already-padded width.
    """
    return padding_ops.PaddedShape.of(tuple(widths), int(dv), shipped_dv=carry_ops.SHIPPED_DV)


def _descriptor(widths, dv: int, bh: int) -> StateFormat:
    """The call's own state descriptor -- the same fields `rola._state._format_of`
    derives for a routed call, built here from what this operator already has.
    `widths`/`dv` are padded first (:func:`_padded_shape`), so the descriptor always
    names a shape the carry surface's kernel boundary accepts -- a caller's logical
    shape never reaches `StateFormat` directly."""
    shape = _padded_shape(widths, dv)
    n = 1
    for width in shape.padded_widths:
        n *= width
    return StateFormat(D=len(shape.padded_widths), B=shape.padded_widths, order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=shape.DV, dtype=SPLIT_PLANE_DTYPE,
                       ids=int(bh) * (n >> PAGE_RECTANGLE_BITS))


def _liveness_words(pread: torch.Tensor, pwrite: torch.Tensor,
                    descriptor: StateFormat) -> carry_ops.LivenessWords:
    """THE CLASS-1 WORDS, from the ONE pass over the planes this call already built.

    There is one producer of liveness and this is a CALLER of it, never a second
    derivation: `rola.ops.liveness.liveness_words` votes every level from its own
    amplitudes, which is what the static width mask reduces to on an unpadded
    topology, and the layout is the contract's.
    """
    layout = LivenessLayout.of(descriptor, pread.shape[-2])
    statics = side_statics(layout, (False,) * layout.D, layout.B)
    return carry_ops.LivenessWords(
        words=liveness_words(pread, pwrite, layout, statics, statics))


def _conservative_activity(descriptor: StateFormat, bh: int, device) -> torch.Tensor:
    """Every page READ and WRITTEN: the conservative statement this seam makes when
    no facts-pass output is available for the call. It asserts nothing about the
    draw and therefore skips nothing (`benchmarks.cells.conservative_activity`
    states the same fact for a cell)."""
    pages = descriptor.N // carry_ops.PAGE_LEAVES
    return torch.full((bh, pages), carry_ops.ACTIVITY_READ | carry_ops.ACTIVITY_WRITTEN,
                      dtype=torch.uint8, device=device)


def _pack_operands(read_levels, write_levels, g_write, v):
    """The planes both kernels read, packed ONCE: `(pread, pwrite, gwrite, v_bh, v_bthd)`.

    ``v_bh`` is `[BH, L, DV]`, the carry surface's own layout; ``v_bthd`` is the
    intra kernel's token-major `[B, T, H, DV]`. Both are views of the one cast, never
    two draws of it.

    THE LEVELS AND ``v`` ARE PADDED FIRST (:func:`_padded_shape`, `rola.ops.padding`):
    `pad_routes` widens each level's trailing dimension to its `PaddedShape.
    padded_widths` entry with exact-zero columns before `carry_ops.pack_side` packs
    it, and `pad_v` widens `v`'s value axis to `PaddedShape.DV` the same way -- a
    logical `(31, 45)`/`d_v=48` call is packed at the padded shape the carry surface's
    kernel boundary accepts, and an already-shipped call pays nothing (`pad_routes`/
    `pad_v` are no-ops past an already-padded width).
    """
    widths = tuple(level.shape[-1] for level in read_levels)
    shape = _padded_shape(widths, v.shape[-1])
    pread = carry_ops.pack_side(padding_ops.pad_routes(read_levels, shape.padded_widths))
    pwrite = carry_ops.pack_side(padding_ops.pad_routes(write_levels, shape.padded_widths))
    gwrite = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    v_bthd = padding_ops.pad_v(v.to(torch.bfloat16).contiguous(), shape.DV)
    Bn, T, H, dv = v_bthd.shape
    v_bh = v_bthd.permute(0, 2, 1, 3).contiguous().reshape(Bn * H, T, dv)
    return pread, pwrite, gwrite, v_bh, v_bthd


def prefill(read_levels, write_levels, g_write, v, widths, *, modes, sread, swrite,
            state_in: torch.Tensor | None = None,
            state_out: torch.Tensor | None = None,
            page_table: torch.Tensor | None = None,
            atom_bits: torch.Tensor | None = None):
    """The whole forward term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32 and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order -- the
    caller's own ``state_out`` plane, or ``None`` where the call stores no state.
    The readout a caller wants is ``num / (den + READOUT_EPS)``; this operator
    does not divide, because the state and the mass are what a chained call
    needs and a ratio is not.

    THE STATE PLANES ARE THE CALLER'S (`docs/internals/state.md` section 4).
    THE FOUR SHAPES, the kernel's own, and this operator allocates no plane: neither
    plane is the NULL-STATE call (nothing in, nothing out); ``state_in`` alone is
    READ-ONLY (the readout against a frozen state, the folds never stored);
    ``state_out`` alone is a FRESH sequence (a zero state in, no entry sweep, stored
    out); both, the SAME tensor, ADVANCE it in place. A plane is the arena's, with its
    ``page_table`` for the paged backing, or a dense one from
    :func:`rola.ops.carry.state_plane`. ``page_table`` is the
    ``[BH, N/16]`` int32 slot table, and ``None`` is the dense backing, where the slot
    IS the atom. ``atom_bits`` is the per-atom ACTIVITY byte (bit 0 WRITTEN, bit 1
    READ); a call that passes none states the conservative fact
    (:func:`_conservative_activity`) rather than a derivation this seam does not own.

    ``modes`` is one :mod:`rola.ops.intra` support mode per level. It is a
    DECLARATION about the routing, not a switch: the intra's slab gate reads
    all-ones for an undeclared side and therefore cannot move a number, so
    ``DENSE_BOTH`` everywhere is always correct and only ever slower.

    ``sread``/``swrite`` are the frozen support words of the two sides
    (``[BH, L/32, sum(widths)]`` int32). THEY ARE AN INPUT, NOT A DERIVATION: the
    routing producer emits exactly this word from the solve that produced the
    amplitudes (``entmax/factor.cu:219-241``), so deriving them here would be a
    second pass over the planes to answer a question already answered. A caller
    that drew its planes instead of solving them has
    :func:`rola.ops.intra.pack_support`.
    """
    D = len(widths)
    if len(modes) != D:
        raise ValueError(f"{D} levels want {D} support modes; got {len(modes)}")
    if len(set(widths)) != 1 or widths[0] not in LEVEL_WIDTHS:
        raise ValueError(
            f"the combined prefill is built at one uniform level width out of "
            f"{LEVEL_WIDTHS} (the intra kernel's instantiations); got widths "
            f"{tuple(widths)}. A topology neither family carries is an inter-only "
            f"cell -- call rola.ops.carry.carry_forward directly.")
    length = read_levels[0].shape[1]
    if length % WINDOW:
        raise ValueError(
            f"L = {length} is not a whole number of {WINDOW}-token windows; the "
            f"intra kernel's tile grid is a compile-time count per arm.")

    #: THE LOGICAL VALUE WIDTH, read before `v` is padded -- `slice_y` narrows `num`
    #: back to this width on the way out, so a caller at an unshipped `d_v` never
    #: sees the padded tail (`rola.ops.padding`, "TWO CONTRACTS, ONE TRANSLATOR").
    dv_logical = v.shape[-1]

    #: THE OPERANDS ARE PACKED ONCE.  Both kernels read the SAME planes, so a second
    #: pack of either side is how a rounding difference becomes a conformance mystery.
    pread, pwrite, gwrite, v_bh, v_bthd = _pack_operands(read_levels, write_levels,
                                                         g_write, v)
    bh, dv = v_bh.shape[0], v_bh.shape[2]

    descriptor = _descriptor(widths, dv, bh)
    launch = carry_ops.LaunchShape()
    geometry = carry_ops.geometry_block(descriptor, launch,
                                        intra_ops.pack_modes(tuple(modes)))
    liveness = _liveness_words(pread, pwrite, descriptor)
    activity = (atom_bits if atom_bits is not None
               else _conservative_activity(descriptor, bh, v_bh.device))
    #: THE STATE PLANES ARE THE CALLER'S, both of them, and this operator allocates none:
    #: the kernel's four shapes pass through as the caller names them (state.md section 4).

    routes = carry_ops.RoutePlanes(read=pread, write=pwrite, gain=gwrite)
    num, den = carry_ops.carry_forward(
        routes, v_bh, descriptor=descriptor, geometry=geometry, liveness=liveness,
        activity=activity, launch=launch, state_in=state_in, state_out=state_out,
        page_table=page_table)
    #: SAME STREAM, SAME BUFFERS.  The intra REDs into `num`/`den`; the stream
    #: order is the only synchronization either kernel needs, and there is no
    #: host round trip between them.
    extension().intra_forward(pread, pwrite, gwrite, v_bthd, sread, swrite, num, den, D,
                              int(widths[0]), intra_ops.pack_modes(tuple(modes)),
                              int(WINDOW))
    return padding_ops.slice_y(num, dv_logical), den, state_out


__all__ = ["LEVEL_WIDTH", "LEVEL_WIDTHS", "WINDOW", "prefill"]

