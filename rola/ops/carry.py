# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY LAUNCH SURFACE -- the contract, written before the body exists.

Card `development/queue/C_CLEAN_SLATE.md`: the interfaces state the FINAL form of the
line, not a bring-up form to be widened later. The forward entry reaches the built body;
the reverse entry reaches the extension's refusing stub and says so by name.

THE TWO CONTRACTS (KERNEL_STANDARDS §R13 addendum, "TWO CONTRACTS, ONE TRANSLATOR").
The API above this module accepts any shape -- any logical level width ``b_l >= 2``,
any logical value width ``d_v``, any length ``L`` -- and serves it by PADDING at the
routing producer and at the value projection. This module is the KERNEL boundary: it
accepts only the descriptor's strict shape and REFUSES everything else. It never pads,
never masks and never widens a call to fit an arm.

Symbols, each at first use. ``D`` = routing depth (levels). ``B_l`` = the PADDED digit
count of level ``l`` (a power of two at or above 16; the descriptor's ``B``).
``N = prod_l B_l`` = leaf capacity. ``DV`` = the padded value width (a shipped one).
``BH`` = batch times heads. ``L`` = tokens. ``cols = DV + 1``, the mass column riding
beside the value columns. ``warps_per_cta`` = the launch shape's one free field, the
warps one CTA runs (``{8, 4}``, 8 shipped and 4 the latency diagnostic).
``leaves_per_warp`` = the warp sub-box's leaf count -- register shape, and therefore a
consequence of ``DV`` rather than a dial. ``BC`` = ``leaves_per_warp * warps_per_cta``,
the CTA box, derived at launch and never an axis. The window ``W``, the private token
tile ``C`` and the stream count ``S`` are NOT on this surface: the axis law makes
``W`` a kernel constant, ``C`` a file constant and ``S`` derived, so a caller cannot
name them and a short sequence runs a partial window inside the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola._state import MIN_LEVEL_WIDTH, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.facts.liveness import LivenessLayout
from rola.ops import lattice
from rola.ops._ext import extension

#: The value widths a shipped binary carries (``DV`` is a FREE AXIS because it is
#: the register shape). A logical ``d_v`` outside the set is served by padding at the
#: value projection, above this boundary -- never by a kernel that masks.
SHIPPED_DV: tuple[int, ...] = (32, 64, 128)

#: The launch shapes the arm key admits (``warps_per_sm`` splits into
#: ``ctas_per_sm x warps_per_cta``). 8 is the shipped shape -- one CTA per SM, the
#: register file filled exactly; 4 is the two-CTA latency diagnostic and ships nowhere.
WARPS_PER_CTA: tuple[int, ...] = (8, 4)
SHIPPED_WARPS_PER_CTA = 8

#: The state a warp holds is the register file's share of it, so the leaf count per
#: warp is the budget divided by the value width: 32 leaves at ``DV = 64`` (two m-tiles,
#: the ldmatrix granule) and half that at ``DV = 128``.
_LEAVES_PER_WARP_BUDGET = 32 * 64

#: THE KERNEL'S WINDOW, in tokens (the axis law: ``W`` is CONSTANT and kernel
#: internal; a short sequence runs a partial window). It is readable, and never an
#: argument: the inter/intra decomposition is DEFINED on this grid, so the fp64
#: reference has to mirror the same number or the two terms would not be one recurrence.
WINDOW = 512

#: One page is the trailing ``log2(16)`` canonical leaf bits, so a page is 16 leaf rows
#: and every lawful ``N`` is a whole number of them.
PAGE_LEAVES = lattice.MMA_K_QUANTUM

#: The activity byte's two bits, per page, as the facts pass emits them and the kernel
#: reads them: a page is loaded iff resident and (READ or WRITTEN), stored iff
#: WRITTEN.
ACTIVITY_WRITTEN = 1
ACTIVITY_READ = 2


class CarryRefusal(ValueError):
    """A KERNEL-BOUNDARY refusal: this call names a shape the kernel does not address.

    Distinct from the API's own envelope refusal, which is where a servable-by-padding
    shape is translated. Reaching this class means the translator did not run, or
    produced something the descriptor does not describe.
    """


def leaves_per_warp(DV: int) -> int:
    """The warp sub-box's leaf count at this value width -- register shape, derived."""
    return _LEAVES_PER_WARP_BUDGET // int(DV)


def box_leaves(DV: int, warps_per_cta: int) -> int:
    """``BC``: the CTA's box, ``leaves_per_warp * warps_per_cta``. Never an axis."""
    return leaves_per_warp(DV) * int(warps_per_cta)


@dataclass(frozen=True, slots=True)
class LaunchShape:
    """The launch's one free field, and what the arm key reads off it.

    ``warps_per_cta`` is COMPILE-TIME (the 2x4 ruling): the CTA's box is the
    ownership unit and is carved per side among ITS warps, so the member set is
    ``f(D, leaves_per_warp, warps_per_cta)`` and the value is part of the arm key
    ``(D, DV, warps_per_cta)``, not a runtime argument.
    """

    warps_per_cta: int = SHIPPED_WARPS_PER_CTA

    def __post_init__(self) -> None:
        if self.warps_per_cta not in WARPS_PER_CTA:
            raise CarryRefusal(
                f"warps_per_cta={self.warps_per_cta} is not a launch shape this family "
                f"builds; the set is {WARPS_PER_CTA} (8 = one CTA per SM, the "
                f"shipped shape; 4 = the two-CTA latency diagnostic).")

    @property
    def threads(self) -> int:
        return self.warps_per_cta * 32


@dataclass(frozen=True, slots=True)
class CarrySchedule:
    """THE LAUNCH DIAL -- the primitive ledger's class 3, and nothing else.

    ``order`` is the ORDER POLICY: how the kernel ranks a window's tokens before it tiles
    them, per side, from the box words alone (docs/internals/carry/carry_kernel.md#order).
    ``ORDER_FIRST_BOX`` is a counting sort by first live box with the dead tokens last, the
    tiles over the live prefix -- the default, and the reap (P81 §1: 3-4x fewer (tile, box)
    pairs than token order at alt-k4). ``ORDER_IDENTITY`` keeps token order and tiles every
    token: the same arithmetic, the A/B of the sort.
    """

    order: str = "first"

    def __post_init__(self) -> None:
        if self.order not in ORDER_POLICIES:
            raise CarryRefusal(
                f"order={self.order!r} is not an order policy; the set is "
                f"{tuple(ORDER_POLICIES)}.")

    def flat(self) -> list[int]:
        """The one entry the kernel boundary takes: the order policy's code."""
        return [ORDER_POLICIES[self.order]]


#: THE ORDER POLICIES, as `csrc/rola/src/carry/params.cuh` numbers them.
ORDER_FIRST_BOX: str = "first"
ORDER_IDENTITY: str = "identity"
ORDER_POLICIES: dict[str, int] = {ORDER_FIRST_BOX: 0, ORDER_IDENTITY: 1}

#: THE MODE WORD'S ENCODING, as `csrc/rola/src/common/geom.cuh` spells it and
#: `rola.ops.intra.pack_modes` packs it: two bits per level, bit 0 the READ side is sparse
#: there, bit 1 the WRITE side. Named here rather than imported because `rola.ops.intra`
#: imports this module.
SIDE_SPARSE: tuple[int, int] = (1, 2)
MAX_LEVELS: int = 4


def select_schedule(descriptor: StateFormat, level_modes: int = 0) -> CarrySchedule:
    """THE SELECTION RULE: the first-live-box order at every density.

    The order study (P81, 2026-09-08) puts it within 1.3x of the ILP optimum's (tile, box)
    count at alt-k4 and at 3-4x below token order; at dense every token is live in every
    box and the two policies tile the same work. A caller that has measured a cell where
    another policy wins names the schedule outright; ``level_modes`` is accepted so the rule
    keeps the seam a density-dependent rule would need.
    """
    del descriptor, level_modes
    return CarrySchedule(order=ORDER_FIRST_BOX)


@dataclass(frozen=True, slots=True)
class RoutePlanes:
    """The routing operands as the kernel reads them: two packed planes and the gain.

    ``read``/``write`` are ``[BH, L, sum_l B_l]`` bf16 -- the levels concatenated in
    canonical order, which is the layout `pack_side` produces and both kernel families
    consume. ``gain`` is ``[BH, L]`` bf16, the write side's per-token scale.
    """

    read: torch.Tensor
    write: torch.Tensor
    gain: torch.Tensor


@dataclass(frozen=True, slots=True)
class LivenessWords:
    """The class-1 liveness words, in the ONE layout the pass emits and every fold reads.

    ``words`` is ``[BH, 2, rows, ceil(L / 32)]`` int32 with ``rows = sum_l B_l``, side 0
    read and side 1 write, ``row = row_base(l) + digit`` in canonical order and
    ``bit = token % 32`` -- :class:`rola.engine.facts.liveness.LivenessLayout`, which is
    the contract this surface does not restate and cannot disagree with. Sparse levels
    carry the amplitudes' support; a dense level carries its static width mask; a PAD
    digit is dead, because the producer pads its logits with -inf and the amplitude is
    then exactly zero. :func:`rola.ops.liveness.liveness_words` produces it.

    The type exists so a bare tensor cannot reach the surface by accident; the LAYOUT
    is the contract's and is checked against the descriptor the call already carries.
    """

    words: torch.Tensor


@dataclass(frozen=True, slots=True)
class GeometryBlock:
    """The addressing block, derived once from ``(descriptor, launch)`` and CHECKED.

    The kernel derives its own copy at the prologue from the descriptor it is handed
    ("each kernel derives its addressing from (descriptor, its own shape) at the
    prologue and REFUSES on mismatch"), so this object is not an operand -- it is the
    HOST's independent derivation of the same law, and handing it to the surface is
    what makes a disagreement a refusal here instead of mis-addressed bytes inside a
    launch. `geometry_block` derives it; ``fields`` is the shipped C++ derivation read
    back, so the two legs that must agree are both present in one object.
    """

    D: int
    B: tuple[int, ...]
    DV: int
    warps_per_cta: int
    BC: int
    fields: dict
    #: the per-level support declaration this block was derived at -- the carve order's
    #: input, and the SELECTION RULE's (the reaping bound is a function of it).
    level_modes: int = 0

    @property
    def carve_order(self) -> list[int]:
        """THE CARVE ORDER this block was derived at: both sides' levels ranked, read side
        then write side, rank 0 innermost -- the flat form the kernel boundary takes.

        It is READ OFF the shipped derivation rather than re-derived here: the order is a
        runtime input to `derive_carry_geom`, the block is that function's output, and a
        second Python sort of the level modes would be a second derivation of the same law.
        """
        return list(self.fields["r"]["level_at"]) + list(self.fields["w"]["level_at"])

    @property
    def leaves(self) -> int:
        n = 1
        for width in self.B:
            n *= width
        return n


def geometry_block(descriptor: StateFormat, launch: LaunchShape,
                   level_modes: int = 0) -> GeometryBlock:
    """The host's derivation of the addressing block for this ``(descriptor, launch)``.

    The block itself comes from the ONE shipped derivation in
    ``csrc/rola/src/common/geom.cu``; what this adds is the binding of its inputs to
    the descriptor's fields and the launch's shape, so a block a caller hands
    `carry_forward` cannot have been derived from a different state.

    ``level_modes`` is the per-level support declaration, two bits per level in
    `rola.ops.intra`'s encoding. It is what the CARVE ORDER is derived from: a side
    carves its sparse levels first, so declaring a level sparse ranks it outermost and
    makes the warp boxes uniform in activity rather than only in size. Zero -- every
    level dense on both sides -- is the canonical order and is always lawful.
    """
    _refuse_descriptor(descriptor)
    _refuse_launch(launch)
    BC = box_leaves(descriptor.DV, launch.warps_per_cta)
    fields = geometry(descriptor.B, level_modes=int(level_modes), bc=BC, d_v=descriptor.DV)
    return GeometryBlock(D=descriptor.D, B=tuple(descriptor.B), DV=descriptor.DV,
                         warps_per_cta=launch.warps_per_cta, BC=BC, fields=fields,
                         level_modes=int(level_modes))


def _refuse_descriptor(descriptor) -> None:
    if not isinstance(descriptor, StateFormat):
        raise CarryRefusal(
            f"the carry surface is driven by the STATE's format descriptor "
            f"(rola._state.StateFormat), got {type(descriptor).__name__}. The state owns "
            f"its format (KERNEL_STANDARDS.md §R13) and the kernel derives its addressing "
            f"from it; there is no second place to name a shape.")
    for level, width in enumerate(descriptor.B):
        if width & (width - 1) or width < MIN_LEVEL_WIDTH:
            raise CarryRefusal(
                f"level {level} is {width} digits wide. THE KERNEL DOES NOT PAD: B_l is a "
                f"power of two at or above {MIN_LEVEL_WIDTH}, and a logical width outside "
                f"that is served by padding the producer's logits with -inf above this "
                f"boundary (KERNEL_STANDARDS.md §R13 addendum, the two contracts).")
    if descriptor.DV not in SHIPPED_DV:
        raise CarryRefusal(
            f"DV={descriptor.DV} is not a shipped value width; the set is {SHIPPED_DV}. "
            f"An irregular d_v is zero-padded to the next shipped DV at the value "
            f"projection and y is sliced back at the seam -- the kernel never sees a "
            f"width other than its own.")
    if descriptor.N % PAGE_LEAVES:
        raise CarryRefusal(
            f"N={descriptor.N} is not a whole number of {PAGE_LEAVES}-leaf pages. THERE "
            f"IS NO RAGGED STATE: a lawful state is prod_l B_l with every B_l a "
            f"power of two at or above {MIN_LEVEL_WIDTH}, hence page aligned by "
            f"construction; a ragged one is refused, never padded.")
    if descriptor.dtype != SPLIT_PLANE_DTYPE:
        raise CarryRefusal(
            f"the state's dtype is {descriptor.dtype!r}; the carry family reads and "
            f"writes {SPLIT_PLANE_DTYPE!r} pages ([16 x DV hi][16 x DV lo][mass]).")


def _refuse_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype is not torch.bfloat16:
        raise CarryRefusal(
            f"{name} is {tensor.dtype}; the carry family's operands are bf16 and only "
            f"bf16 (KERNEL_STANDARDS.md §R17, one form -- the float operand paths are "
            f"deleted). Cast above this boundary, where the cast is visible.")


def _refuse_launch(launch) -> None:
    if not isinstance(launch, LaunchShape):
        raise CarryRefusal(
            f"the launch shape is a rola.ops.carry.LaunchShape, got "
            f"{type(launch).__name__}. The window, the token tile and the stream count "
            f"are not on this surface (the axis law): W is a kernel constant, C a file "
            f"constant and S derived.")


def _refuse_geometry(block, descriptor: StateFormat, launch: LaunchShape) -> None:
    if not isinstance(block, GeometryBlock):
        raise CarryRefusal(
            f"the geometry block is a rola.ops.carry.GeometryBlock derived from this "
            f"call's descriptor (geometry_block(descriptor, launch)), got "
            f"{type(block).__name__}.")
    presented = (block.D, tuple(block.B), block.DV, block.warps_per_cta, block.BC)
    bound = (descriptor.D, tuple(descriptor.B), descriptor.DV, launch.warps_per_cta,
             box_leaves(descriptor.DV, launch.warps_per_cta))
    if presented != bound:
        raise CarryRefusal(
            f"the geometry block was derived for {presented} and this call presents "
            f"{bound}. The kernel derives the SAME block at its prologue from the "
            f"descriptor it is handed, so a block from another shape would address "
            f"somebody else's leaves (refuse on mismatch).")


def _refuse_operands(routes, v, descriptor: StateFormat, liveness, activity):
    if not isinstance(routes, RoutePlanes):
        raise CarryRefusal(
            f"the routing operands reach this surface as a rola.ops.carry.RoutePlanes "
            f"bundle (two packed [BH, L, sum_l B_l] bf16 planes and the [BH, L] gain), "
            f"got {type(routes).__name__}.")
    for name, plane in (("routes.read", routes.read), ("routes.write", routes.write),
                        ("routes.gain", routes.gain), ("v", v)):
        _refuse_bf16(name, plane)

    wtot = sum(descriptor.B)
    for name, plane in (("routes.read", routes.read), ("routes.write", routes.write)):
        if plane.ndim != 3 or plane.shape[2] != wtot:
            raise CarryRefusal(
                f"{name} is {tuple(plane.shape)}; the packed routing plane is "
                f"[BH, L, {wtot}] -- the descriptor's levels {tuple(descriptor.B)} "
                f"concatenated in canonical order. A plane of a different total width is "
                f"a different topology, not a servable one.")

    bh, length = int(routes.read.shape[0]), int(routes.read.shape[1])
    if tuple(routes.write.shape) != (bh, length, wtot):
        raise CarryRefusal(
            f"the two sides disagree: read is {tuple(routes.read.shape)} and write is "
            f"{tuple(routes.write.shape)}. One call is one token addressing.")
    if tuple(routes.gain.shape) != (bh, length):
        raise CarryRefusal(
            f"routes.gain is {tuple(routes.gain.shape)}; the write side's per-token "
            f"scale is [BH, L] = {(bh, length)}.")
    if v.ndim != 3 or tuple(v.shape) != (bh, length, descriptor.DV):
        raise CarryRefusal(
            f"v is {tuple(v.shape)}; the kernel reads [BH, L, DV] = "
            f"{(bh, length, descriptor.DV)} at the descriptor's PADDED width. A logical "
            f"d_v is padded at the projection, which is a zero-copy write of the first "
            f"d_v columns -- the kernel never sees a partial tile.")
    if not isinstance(liveness, LivenessWords):
        raise CarryRefusal(
            f"the liveness words reach this surface as a rola.ops.carry.LivenessWords, "
            f"got {type(liveness).__name__}.")
    layout = LivenessLayout.of(descriptor, length)
    want = (bh, 2, layout.rows, layout.words)
    if liveness.words.dtype is not torch.int32 or tuple(liveness.words.shape) != want:
        raise CarryRefusal(
            f"liveness.words is {tuple(liveness.words.shape)} {liveness.words.dtype}; the "
            f"class-1 block is [BH, 2, sum_l B_l, ceil(L / 32)] = {want} int32 -- one bit "
            f"per digit per token per side in canonical order "
            f"(rola.engine.facts.liveness.LivenessLayout).")
    pages = descriptor.N // PAGE_LEAVES
    if activity.dtype is not torch.uint8 or tuple(activity.shape) != (bh, pages):
        raise CarryRefusal(
            f"the activity bits are {tuple(activity.shape)} {activity.dtype}; the "
            f"per-page byte is [BH, N / {PAGE_LEAVES}] = {(bh, pages)} uint8 (bit 0 "
            f"WRITTEN, bit 1 READ). Residency says WHERE a page is, activity says "
            f"WHETHER this call touches it.")
    return bh, length


def _refuse_state(name: str, plane, descriptor: StateFormat, bh: int) -> None:
    if plane is None:
        return
    want = (bh, descriptor.N // PAGE_LEAVES, PAGE_LEAVES, descriptor.cols)
    if tuple(plane.shape) != want:
        raise CarryRefusal(
            f"{name} is {tuple(plane.shape)}; a state plane is "
            f"[BH, N / {PAGE_LEAVES}, {PAGE_LEAVES}, DV + 1] = {want} in CANONICAL leaf "
            f"order, stored as split bf16 planes.")


def carry_forward(routes: RoutePlanes, v: torch.Tensor, *, descriptor: StateFormat,
                  geometry: GeometryBlock, liveness: LivenessWords,
                  activity: torch.Tensor, launch: LaunchShape,
                  schedule: CarrySchedule | None = None,
                  state_in: torch.Tensor | None = None,
                  state_out: torch.Tensor | None = None,
                  page_table: torch.Tensor | None = None):
    """THE INTER TERM, undivided: ``(num, den)``, with ``state_out`` advanced in place.

    ``num`` is ``[BH, L, DV]`` fp32 and ``den`` is ``[BH, L]`` fp32; the readout a
    caller wants is ``num / (den + READOUT_EPS)``, and this surface does not divide
    because a chained call needs the mass and a ratio is not it.

    Every shape argument is the DESCRIPTOR's. ``page_table`` is the ``[BH, N / 16]``
    int32 slot table; ``None`` is the dense backing, where the slot IS the page, so the
    two backings are one kernel, one ABI and one accumulation order. Passing neither
    state plane is the NULL-STATE call.

    The refusals below are the API's own envelope and run first; the extension's entry
    then applies the KERNEL boundary's shape law to the same call.
    """
    _refuse_descriptor(descriptor)
    _refuse_launch(launch)
    _refuse_geometry(geometry, descriptor, launch)
    bh, length = _refuse_operands(routes, v, descriptor, liveness, activity)
    _refuse_state("state_in", state_in, descriptor, bh)
    _refuse_state("state_out", state_out, descriptor, bh)

    #: ZEROED, not empty: the readout REDUCES into these planes (one contribution per
    #: owner box per token), so they are an accumulator the kernel adds into and never a
    #: buffer it fills.
    num = torch.zeros((bh, length, descriptor.DV), dtype=torch.float32, device=v.device)
    den = torch.zeros((bh, length), dtype=torch.float32, device=v.device)
    if schedule is None:
        schedule = select_schedule(descriptor, geometry.level_modes)
    extension().carry_forward(
        routes.read, routes.write, routes.gain, v, num, den,
        list(descriptor.B), int(descriptor.DV), int(descriptor.page_bits),
        int(launch.warps_per_cta), geometry.carve_order, schedule.flat(),
        liveness.words, activity, state_in, state_out, page_table)
    return num, den


def carry_backward(routes: RoutePlanes, v: torch.Tensor, d_num: torch.Tensor,
                   d_den: torch.Tensor, *, descriptor: StateFormat,
                   geometry: GeometryBlock, liveness: LivenessWords,
                   activity: torch.Tensor, launch: LaunchShape,
                   state_in: torch.Tensor | None = None,
                   d_state_out: torch.Tensor | None = None,
                   page_table: torch.Tensor | None = None):
    """THE CROSS-WINDOW REVERSE PASS: ``(d_read, d_write, d_gain, d_v, d_state_in)``.

    ``d_num``/``d_den`` are the seeds the op seam forms from the division's backward,
    shaped like the forward's ``num``/``den`` and fp32. ``d_read``/``d_write`` are
    ``[BH, L, sum_l B_l]`` fp32 on the packed amplitude layout, ``d_gain`` is
    ``[BH, L]``, ``d_v`` is ``[BH, L, DV]`` and ``d_state_in`` is a state plane.

    Same descriptor, same geometry block, same liveness words and same activity bits as
    the forward it reverses -- a reverse pass that re-derived any of them would be a
    second derivation of the schedule, which is the drift shape this family has already
    paid for once. THERE IS NO REVERSE IMPLEMENTATION ON THIS LINE.
    """
    _refuse_descriptor(descriptor)
    _refuse_launch(launch)
    _refuse_geometry(geometry, descriptor, launch)
    bh, length = _refuse_operands(routes, v, descriptor, liveness, activity)
    for name, seed, want in (("d_num", d_num, (bh, length, descriptor.DV)),
                             ("d_den", d_den, (bh, length))):
        if seed.dtype is not torch.float32 or tuple(seed.shape) != want:
            raise CarryRefusal(
                f"{name} is {tuple(seed.shape)} {seed.dtype}; the reverse seeds are the "
                f"forward's outputs, {want} fp32.")
    _refuse_state("state_in", state_in, descriptor, bh)
    _refuse_state("d_state_out", d_state_out, descriptor, bh)

    return extension().carry_backward(
        routes.read, routes.write, routes.gain, v, d_num, d_den,
        list(descriptor.B), int(descriptor.DV), int(descriptor.page_bits),
        int(launch.warps_per_cta), geometry.carve_order, liveness.words, activity,
        state_in, d_state_out, page_table)


def arms():
    """``[(D, DV, warps_per_cta)]`` -- the arms THIS binary carries.

    The binary's own answer, never the source tree's: an empty list is the honest report
    of a build that selected no arm, and a short list the honest report of an iteration
    build (`ROLA_CARRY_ARMS`), which is not shippable.
    """
    return [tuple(row) for row in extension().carry_arms()]


def build_stamp():
    """The carry family's DEVICE build stamp -- a fact only a BUILT kernel can produce.

    The four device facts as one number: the page granule with the warps per SM, the
    capability digest the device code compiled against, the state's share of the register
    file and the cluster size, all read back off a kernel this binary compiled. A path or a
    hash could be produced by a stale binary; this cannot
    (`extension-trap-device-side-check`).
    """
    return extension().carry_build_stamp()


def sm_clock_ghz() -> float | None:
    """The SM's EFFECTIVE clock under load, read off the device beside a measurement.

    The driver reports one number on this host whatever the silicon runs at; the kernel
    runs at either of two states minutes apart (journal section 37). A harness records
    this with every row so a millisecond carries its state. A binary from before the
    probe reports None.
    """
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())


def census():
    """One row per built arm: what the compiler did, then what the box plan says it should.

    ``(index, D, DV, warps_per_cta, regs, local_bytes, smem_bytes, max_threads,
    binary_version, threads, box_leaves, atoms, boxes, state_elems, read_bytes,
    pool_bytes)``. The first group is
    ``cudaFuncGetAttributes`` on the arm's own kernel and the second is the derivation;
    having both in one row is what makes a disagreement visible instead of inferred.
    """
    return [tuple(row) for row in extension().carry_census()]


def pack_side(levels) -> torch.Tensor:
    """The per-level ``[B, L, H, B_l]`` tuple as one ``[BH, L, sum_l B_l]`` bf16 plane.

    The layout both kernel families read. It is a REPACK, not a translation: the levels
    arrive already padded to their ``B_l`` by the producer, so nothing here widens
    anything.
    """
    Bn, T, H, _ = levels[0].shape
    out = torch.empty((Bn, H, T, sum(x.shape[-1] for x in levels)),
                      dtype=torch.bfloat16, device=levels[0].device)
    off = 0
    for level in levels:
        w = level.shape[-1]
        out[:, :, :, off:off + w].copy_(level.permute(0, 2, 1, 3))
        off += w
    return out.reshape(Bn * H, T, -1)


def route_planes(read_levels, write_levels, g_write) -> RoutePlanes:
    """The three routing operands packed into the bundle the surface takes."""
    gain = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    return RoutePlanes(read=pack_side(read_levels), write=pack_side(write_levels),
                       gain=gain.reshape(gain.shape[0] * gain.shape[1], -1))


def state_plane(descriptor: StateFormat, BH: int, device="cuda") -> torch.Tensor:
    """A DENSE state plane for this descriptor: ``[BH, N / 16, 16, DV + 1]``.

    The dense BACKING, page for page the atom-major view of ``[BH, N, cols]``
    (`docs/internals/state.md`). It is the caller's to allocate because the planes are
    the caller's: a continuation hands the SAME tensor in and out and advances it in
    place, which is what keeps a stateful chain to ONE plane per sequence.
    """
    return torch.zeros((BH, descriptor.N // PAGE_LEAVES, PAGE_LEAVES, descriptor.cols),
                       dtype=torch.float32, device=device)


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box, with the atom law ASSERTED.

    The derivation is `rola.ops.lattice`'s -- the ONE host authority over the leaf
    order, shared with the decode family, because the two families write and read one
    plane and two derivations disagree exactly when one is wrong. What this wrapper adds
    is the carry family's own gate: an owner's box is a whole number of pages, so a fold
    bucket never straddles two of them.
    """
    out = lattice.box_shape(widths, k, m)
    assert out[3] % PAGE_LEAVES == 0, (
        f"an owner box of {out[3]} leaves must be a whole number of "
        f"{PAGE_LEAVES}-leaf pages")
    return out


def lattice_of_canonical(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    A permutation, and only a permutation -- a fixed BIT PERMUTATION of the leaf index,
    so this tensor is a convenience for the host seam and never a table the device
    consults.
    """
    box_shape(widths, k, m)
    return lattice.permutation(widths, k, m, device=device)


#: THE ADDRESSING BLOCK'S FIELD ORDER, mirrored from `csrc/rola/src/common/geom.cu`'s
#: `geometry` -- the ONE reader of that positional vector.
_GEOM_MAX_LEVELS = 4
_GEOM_HEAD = ("D", "bc")
_GEOM_LEVEL = ("width", "span", "grid", "col_base", "run_shift", "owner_div", "weight")
_GEOM_SCALAR = ("wtot", "owners", "leaves", "local_bits", "min_width", "identity_map")
_SIDE_LEVEL = ("level_at", "rank", "row_span", "grid", "grid_div", "row_g", "row_prefix",
               "sub_bits", "sub_shift")
_SIDE_SCALAR = ("carves", "rows", "leaf", "streams", "assign")


def geometry(widths, level_modes=0, bc=128, nsr=1, nsw=1, d_v=64):
    """THE SHIPPED DERIVATION, read back as a dict of dicts.

    ``{scalars..., "r": {side fields}, "w": {...}, "xch": {"w": [...], "r": [...]}}`` --
    every per-level field is a list of length ``D``, the padding levels the block carries
    past ``D`` dropped here because they are a C-array convenience and not part of the
    law. ``xch`` is each side's transit row map over the whole region, ``[stream][slot]``.
    See ``docs/internals/common/geom.md``.
    """
    flat = list(extension().carry_geometry(list(widths), int(level_modes), int(bc), int(nsr),
                                           int(nsw), int(d_v)))
    it = iter(range(len(flat)))

    def take(n):
        return [flat[next(it)] for _ in range(n)]

    out = {}
    for name in _GEOM_HEAD:
        out[name] = take(1)[0]
    D = out["D"]
    for name in _GEOM_LEVEL:
        out[name] = take(_GEOM_MAX_LEVELS)[:D]
    for name in _GEOM_SCALAR:
        out[name] = take(1)[0]
    for side in ("r", "w"):
        block = {}
        for name in _SIDE_LEVEL:
            block[name] = take(_GEOM_MAX_LEVELS)[:D]
        for name in _SIDE_SCALAR:
            block[name] = take(1)[0]
        out[side] = block
    xch = {}
    for side in ("w", "r"):
        rows = out[side]["streams"], out[side]["leaf"]
        xch[side] = [take(rows[1]) for _ in range(rows[0])]
    out["xch"] = xch
    assert len(list(it)) == 0, "the addressing block's flattened field order drifted"
    return out


def sub_boxes(depth, box_leaves_, workers):
    """The GENERATED set of warp sub-box shapes, as spans -- `geom_api.md#sub-box-set`."""
    return [tuple(row) for row in
            extension().carry_sub_boxes(int(depth), int(box_leaves_), int(workers))]


__all__ = [
    "ACTIVITY_READ",
    "ACTIVITY_WRITTEN",
    "ORDER_FIRST_BOX",
    "ORDER_IDENTITY",
    "ORDER_POLICIES",
    "PAGE_LEAVES",
    "SHIPPED_DV",
    "SHIPPED_WARPS_PER_CTA",
    "WARPS_PER_CTA",
    "WINDOW",
    "CarryRefusal",
    "CarrySchedule",
    "GeometryBlock",
    "LaunchShape",
    "LivenessWords",
    "RoutePlanes",
    "arms",
    "box_leaves",
    "box_shape",
    "build_stamp",
    "carry_backward",
    "carry_forward",
    "census",
    "geometry",
    "geometry_block",
    "lattice_of_canonical",
    "leaves_per_warp",
    "pack_side",
    "route_planes",
    "select_schedule",
    "sm_clock_ghz",
    "state_plane",
    "sub_boxes",
]
