# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode family's BODIES: the geometry, the scratch, the launches.

The decode DAG (:mod:`rola.engine.dags.decode_dag`) declares these as nodes and composes the
step; the facade lives there with it. What is here is one function per thing that is
done, and the raw-tensor entry :func:`_decode_step` is those same functions in the
step's order, for a caller holding tensors rather than a bundle.

THE STEP IS ONE LAUNCH. Every per-token operand is folded inside the step kernel, from
the producer's own views, so there is no operand plane between them
(docs/internals/decode/decode_fold.md).

The bypass is architectural rather than a cheap default
(docs/internals/decode/decode.md#no-stage-2), the support is a Kronecker product the
kernel WALKS from its factors rather than materializing
(docs/internals/decode/decode.md#kronecker-support), and decode's decay domain is
strictly larger than the chunk arm's as a CONSEQUENCE
(docs/internals/decode/decode.md#decay-domain).

FORWARD ONLY. The decode kernels have no backward; the guard below is defence in depth
behind the layer's own envelope check (docs/internals/decode/decode.md#forward-only).

PAGED AND DENSE ARE ONE WALK: the page granule is the ATOM, which is what makes the
paged address BC-free, so the whole difference is one indirection resolved once per
touched atom (docs/internals/decode/decode.md#the-atom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.plan import DecodeGeometry, DecodePlan
from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.decay import leaf_rate_dials
from rola.ops.lattice import box_shape, derive_lattice, permutation
from rola.ops.padding import pad_v, padded_value_width, slice_y
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.types import Topology

__all__ = [
    "C_DECODE_UNITS_PER_CTA",
    "DecodeScratch",
    "arms",
    "derive_decode_geometry",
    "derive_n_split",
    "leaf_rate_dials_for",
    "step",
    "step_plane",
]


def arms() -> tuple[tuple[int, int, bool], ...]:
    """The decode arms THIS BINARY carries, as ``(d_v, D, decay)`` rows.

    A full build carries the whole declared matrix (``d_v x D x decay`` = 16 rows). An
    ITERATION build names a subset with ``ROLA_DECODE_ARMS`` and is NOT SHIPPABLE; this
    is how such a binary identifies itself, and the launch refuses any arm not listed
    here rather than mis-launching it (``docs/build.md``).
    """
    return tuple((int(dv), int(d), bool(decay)) for dv, d, decay in extension().rola_decode_arms())


#: THE SMALLEST UNIT SHARE worth a CTA's prologue. A CTA pays the fold and the factor
#: tables before it can walk anything, so splitting past this point buys parallelism the
#: replicated prologue immediately spends. It bounds the split on SMALL topologies, where
#: the wave below is not the binding constraint.
C_DECODE_UNITS_PER_CTA = 16


def _shipped_decode_dv(D: int, decay: bool, device) -> tuple[int, ...]:
    """The ``DV`` values this binary's decode arms ship at this ``(D, decay)``.

    Read off :func:`arms` (the device's own declared matrix) rather than a literal
    set: decode's shipped ``d_v`` axis is wider than carry's single ``D_V`` (a full
    build carries ``d_v x D x decay``), and a literal tuple here would drift from
    whatever the binary actually carries the moment either side changed.
    """
    dv_at_shape = tuple(sorted({dv for dv, d, dec in arms() if d == D and dec == decay}))
    if not dv_at_shape:
        raise ValueError(
            f"no shipped decode arm at D={D}, decay={decay} -- there is no DV this "
            "call could be padded to")
    return dv_at_shape


def _admit(widths: tuple[int, ...]) -> None:
    """DECODE'S OP ENTRY, THE SAME DESCRIPTOR CHECK AS PREFILL (K53).

    K50 declared ONE admission law for both arms; until now decode's own entry
    never asked it -- :class:`~rola._state.StateFormat` is what
    ``RoLAState._kernel_entry`` builds a call's presented descriptor through, and
    this is the SAME class, so a topology narrower than its floor (every ``B_l`` a
    power of two at or above :data:`~rola._state.MIN_LEVEL_WIDTH`) is refused by
    name here exactly as it would be at prefill's seam, rather than only wherever a
    call happens to route through a bound :class:`~rola._state.RoLAState`.

    Only the levels are this call's business: ``DV`` and ``ids`` are placeholders
    that satisfy the descriptor's OTHER fields (an even ``DV``, a page-aligned
    ``N``, which every lawful ``widths`` already is) without asking
    :func:`_shipped_decode_dv` -- that is decode's own, separate law, checked next,
    and it alone needs the extension.
    """
    StateFormat(D=len(widths), B=widths, order=CANONICAL_ORDER, page_bits=PAGE_RECTANGLE_BITS,
                DV=2, dtype=SPLIT_PLANE_DTYPE, ids=1)


def derive_decode_geometry(
    topology: Topology,
    *,
    d_v: int,
    decay: bool,
    BH: int,
    device: torch.device | str,
    eps: float = READOUT_EPS,
    n_split: int | None = None,
) -> DecodeGeometry:
    """Build the carrier once, at the boundary.

    HOST-SIDE PADDING ("d_v padding is also host-side"): ``d_v`` is the caller's
    LOGICAL value width, and this is the SEAM -- the geometry this returns carries
    the padded ``DV`` (:func:`_shipped_decode_dv`), which is what the scratch sizing
    and the launch address. Decode's arm key is ``(DV, D, decay)`` ONLY (:func:`arms`):
    ``DV`` is padded to a shipped value, but the per-level widths carry no such
    shipped set of their own, so ``topology.widths`` travels to the kernel exactly as
    the caller holds them -- past :func:`_admit`'s floor, which REFUSES rather than
    pads (R13: the kernel never pads or masks).

    ``n_split`` is FROZEN here and never selected per step: re-deriving it would
    reintroduce exactly the per-step selection the design dissolves
    (:func:`derive_n_split`).
    """
    widths = tuple(topology.widths)
    _admit(widths)
    d_v = padded_value_width(d_v, shipped=_shipped_decode_dv(len(widths), decay, device))
    D = len(widths)
    k, m = derive_lattice(widths)
    row_offset = []
    acc = 0
    for w in widths:
        row_offset.append(acc)
        acc += w

    if n_split is None:
        n_split = derive_n_split(widths, k, m, decay=decay, BH=BH, device=device)

    return DecodeGeometry(
        D=D, widths=widths, level_row_offset=tuple(row_offset), lattice_k=k, lattice_m=m,
        d_v=int(d_v), has_decay=decay, eps=eps, n_split=int(n_split))


def derive_n_split(widths, lattice_k: int, lattice_m: int, *, decay: bool, BH: int,
                   device) -> int:
    """CTAs per batch-head: ONE WAVE of the step's own declared residency, work permitting.

    THE SPLIT IS A WAVE, not a target. Every CTA pays the fold and the factor tables before
    it can walk anything, so a second wave re-pays that whole prologue for work the first
    wave could have finished -- which is why over-launching is not free the way a
    `beyond-realized CTA exits early` argument suggests. One wave is where the replicated
    prologue is paid exactly once per SM slot.

    The residency comes from :func:`rola.ops._ext.extension().rola_decode_residency`, i.e.
    off the same ``constexpr`` the kernel's ``__launch_bounds__`` are built from: a python
    mirror of a launch bound is exactly the constant that drifts.

    ``n_split`` is FROZEN by the caller for the sequence and never re-selected per step.
    """
    D = len(widths)
    #: THE UNIT CEILING, from the box's spans: a walk unit's leaves are the innermost
    #: level's run, so ``N / s_{D-1}`` is the unit count exactly wherever the atom
    #: rectangle is one level and an OVER-count where it spans more -- more CTAs, never
    #: fewer, which is the direction a split may err in (a CTA with no units exits after
    #: its prologue).
    _, spans, _, _, _ = box_shape(widths, lattice_k, lattice_m)
    unit_ceiling = math.prod(widths) // spans[-1]
    dev = torch.device(device)
    if dev.type != "cuda":
        return 1
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    residency = extension().rola_decode_residency(decay=bool(decay), levels=D)
    wave = max(1, (sms * int(residency)) // max(1, BH))
    supply = max(1, math.ceil(unit_ceiling / C_DECODE_UNITS_PER_CTA))
    return min(supply, wave)


@dataclass(slots=True)
class DecodeScratch:
    """Per-sequence scratch, allocated once and reused across steps.

    ``ctr``, ``growth_ctr`` and ``done`` are zeroed exactly once here: every one of them
    is reset by the last-arriving CTA that reads it, so no ``memset`` launch ever stands
    between two decode steps and the whole step stays capturable in a CUDA graph.
    """

    ws: torch.Tensor
    ctr: torch.Tensor
    #: `[BH]` int32, THE PER-BATCH-HEAD VERDICT, and the `[1]` int32 OR of it that the
    #: host reads through a PINNED mirror -- pinned so the read is one DMA rather than a
    #: staging copy, and held here so a steady step allocates nothing
    #: (docs/internals/decode/decode.md#per-bh-verdict).
    growth: torch.Tensor
    growth_any: torch.Tensor
    #: The reducing CTA's self-resetting arrival counter.
    growth_ctr: torch.Tensor
    growth_host: torch.Tensor
    #: `[BH]` int32, THE DONE FLAGS: which batch-heads have already deposited this step.
    #: A replay re-walks a done batch-head for its `y` and deposits nothing, which is what
    #: makes the caller's replay advance every sequence EXACTLY ONCE
    #: (docs/internals/decode/decode.md#done-flags).
    done: torch.Tensor
    #: `[BH, N/16]` bool, the step's own write-atom set, condensed on the device from the
    #: expansion the verdict was read off. A host admission plans from THIS -- the host
    #: derives nothing: docs/internals/decode/decode.md#atom-bits
    atom_bits: torch.Tensor

    @staticmethod
    def build(config: DecodeGeometry, BH: int, device) -> DecodeScratch:
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        cuda = torch.device(device).type == "cuda"
        return DecodeScratch(
            ws=torch.empty(BH, config.n_split, config.cols, **f32),
            ctr=torch.zeros(BH, **i32),
            growth=torch.zeros(BH, **i32),
            growth_any=torch.zeros(1, **i32),
            growth_ctr=torch.zeros(1, **i32),
            growth_host=torch.empty(1, dtype=torch.int32, pin_memory=cuda),
            done=torch.zeros(BH, **i32),
            atom_bits=torch.empty(BH, config.N // MMA_K_QUANTUM, dtype=torch.bool,
                                  device=device))


def _fold(t: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W] -> [B*H, W]`` with ``bh = b * H + h``.

    The fold order is the consumer's own (``h = bh % H`` in the kernel), restated here
    rather than imported because decode shares no schedule object with it — and a fold
    that disagreed would misroute every head's dials.
    """
    B, T, H, W = t.shape
    if T != 1:
        raise ValueError(f"the decode op consumes exactly one token, got T={T}")
    return t.permute(0, 2, 1, 3).reshape(B * H, W).contiguous()


def fold_levels(levels) -> list[torch.Tensor]:
    """The per-level amplitude stream, in torch: ``[BH, width_l]`` fp32.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. The shipped path folds inside
    the step kernel, from the producer's own views; a gate that fed the kernel its own
    fold's output would prove nothing about the fold
    (docs/internals/decode/decode_fold.md#the-gate).
    """
    return [_fold(t).float() for t in levels]


def guard_no_grad(operands) -> None:
    if not torch.is_grad_enabled():
        return
    if any(t is not None and torch.is_tensor(t) and t.requires_grad for t in operands):
        from rola.interface import BackwardNotImplemented

        raise BackwardNotImplemented(
            "rola decode is forward-only: the single-token arm has no reverse pass -- "
            "the two-pass backward is the chunk arm's. Reaching it from a "
            "grad-requiring context would return a tensor with no grad_fn and silently "
            "sever the graph (the x.grad-is-None failure class). Use eval mode / "
            "torch.no_grad.")


def step_plane(v: torch.Tensor, config: DecodeGeometry, decay,
               state: torch.Tensor, page_table: torch.Tensor | None) -> torch.Tensor:
    """The plane the kernel indexes, and the per-step agreement checks that find it.

    The two backings have ONE shape rule each. Under a table the plane holds only the
    slots a plan committed, so its leading extent is the arena's and not ``BH``; without
    one the plane IS the leaf space. Both are handed to the ABI in the shape the kernel
    indexes, so the reshape below is the whole difference.
    """
    if v.dim() != 4:
        raise ValueError(f"v must be [B, 1, H, d_v], got {tuple(v.shape)}")
    B, _T, H, d_v = v.shape
    BH = B * H
    if d_v != config.d_v:
        raise ValueError(f"v's d_v={d_v} disagrees with the frozen config's {config.d_v}")
    if (decay is not None) != config.has_decay:
        raise ValueError(
            f"the frozen config says has_decay={config.has_decay} but decay is "
            f"{'absent' if decay is None else 'present'}; the config is frozen for the "
            "sequence and the decay arm is part of it")
    if page_table is None:
        #: PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the dense
        #: plane is the same page sequence with every page resident, and `N` is page
        #: aligned by construction (docs/internals/state.md#format).
        if state.shape != (B, H, config.N, config.cols):
            raise ValueError(
                f"a dense state is [B, H, N, d_v + 1] = "
                f"{(B, H, config.N, config.cols)}, got {tuple(state.shape)}")
        return state.reshape(BH, config.N, config.cols)
    if state.dim() != 3 or state.shape[1:] != (MMA_K_QUANTUM, config.cols):
        raise ValueError(
            f"under a page table the state is the arena's plane, "
            f"[slots, {MMA_K_QUANTUM}, d_v + 1], got {tuple(state.shape)}")
    if tuple(page_table.shape) != (BH, config.N // MMA_K_QUANTUM):
        raise ValueError(
            f"page_table must be [BH, N / {MMA_K_QUANTUM}] = "
            f"{(BH, config.N // MMA_K_QUANTUM)}, got {tuple(page_table.shape)}")
    return state


def leaf_rate_dials_for(decay, config: DecodeGeometry, device) -> torch.Tensor | None:
    """``[H, sum_l width_l]`` fp32 -- the decay arm's per-HEAD constant, or ``None``.

    It is the ONE operand the step kernel does not fold: a per-HEAD constant rather than a
    per-token one, and its fp32 cast carries a refusal the host has to raise
    (:func:`rola.ops.decay.leaf_rate_dials`).
    """
    if decay is None:
        return None
    return leaf_rate_dials(decay, config).to(device).float()


def step(plan: DecodePlan) -> torch.Tensor:
    """THE STEP, in ONE launch, in place on ``plan.state_plane``. ``y`` as ``[B, 1, H, d_v]``.

    Fold, expand and walk are one kernel: the per-token operands are gathered from the
    producer's own views straight into the CTA's shared memory, the support bitmap is
    built there, and the walk consumes both without either ever reaching DRAM
    (docs/internals/decode/decode.md#per-cta-prologue).

    Every touched row has exactly one owning CTA, warp and lane, so the update needs no
    atomic and is bitwise reproducible.

    THE ADMISSION QUESTION IS PART OF IT: the write map the walk needs is the map the
    residency test reads, so the test is a pass over shared memory rather than a launch,
    and the verdict it publishes is per batch-head
    (docs/internals/decode/decode.md#absorbed-verdict).
    """
    config = plan.config
    pool = plan.pool
    y = extension().rola_decode_forward(
        read=list(plan.read),
        write=list(plan.write),
        normalize=[int(bool(flag)) for flag in plan.normalize],
        dials=plan.dials,
        g_write=plan.g_write,
        v=plan.v,
        state=plan.state_plane,
        ws=plan.scratch.ws,
        ctr=plan.scratch.ctr,
        growth=plan.scratch.growth,
        growth_any=plan.scratch.growth_any,
        growth_ctr=plan.scratch.growth_ctr,
        done=plan.scratch.done,
        atom_bits=plan.scratch.atom_bits,
        level_widths=list(config.widths),
        level_row_offsets=list(config.level_row_offset),
        lattice_k=config.lattice_k,
        lattice_m=config.lattice_m,
        n_split=config.n_split,
        eps=config.eps,
        page_table=plan.page_table,
        pool_slots=None if pool is None else pool.pool_slots,
        pool_cursor=None if pool is None else pool.pool_cursor,
        pool_map=None if pool is None else pool.pool_map,
    )
    B, _T, H, _d_v = plan.v.shape
    return y.view(B, H, 1, config.d_v).permute(0, 2, 1, 3).contiguous()


def _write_atom_bitmap(write_levels, config: DecodeGeometry) -> torch.Tensor:
    """``[BH, N / 16]`` bool: the atoms this ONE-TOKEN step deposits into, EXACTLY.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. Deleting it would delete the
    gate's reference; calling it on a shipped path would delete the gate's independence
    (docs/internals/decode/decode.md#second-derivation). It derives the CANONICAL leaf
    support digit by digit and crosses into lattice order through ``pi``, so it shares no
    step of the kernel's own enumeration.
    """
    device = write_levels[0].device
    index = torch.arange(config.N, device=device)
    live = None
    for level, width in enumerate(config.widths):
        digit = (index // math.prod(config.widths[level + 1:])) % width
        column = _fold(write_levels[level])[:, digit] != 0
        live = column if live is None else (live & column)
    pi = permutation(config.widths, config.lattice_k, config.lattice_m, device=device)
    lattice_live = torch.empty_like(live)
    lattice_live.index_copy_(1, pi, live)
    return lattice_live.view(-1, config.N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)


def _decode_step(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_write: torch.Tensor,
    config: DecodeGeometry,
    state: torch.Tensor,
    *,
    decay=None,
    normalize: tuple[bool, ...] | None = None,
    workspace: DecodeScratch | None = None,
    page_table: torch.Tensor | None = None,
    pool=None,
) -> tuple[torch.Tensor, torch.Tensor, DecodeScratch]:
    """The RAW-TENSOR entry: the step's own bodies, in the step's order.

    ``v`` is ``[B, 1, H, d_v]``; the level tensors are ``[B, 1, H, width_l]``;
    ``g_write`` is ``[B, 1, H]``. There is no read-side gain at all: the readout
    is a ratio, which fixes it to 1 identically. ``state`` is the dense
    ``[B, H, N, d_v + 1]`` plane, or the arena's ``[slots, 16, d_v + 1]`` one under
    ``page_table``.

    ``normalize`` is the per-read-level answer to "does this level still carry its own
    mass"; ``None`` is the ordinary case here, a caller already on the simplex. The
    bundle entry reads it off :meth:`~rola.routing.factors.RouteFactors.read_needs_normalization`
    instead.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]``.

    ``config`` ALREADY CARRIES THE PADDED ``DV`` (:func:`derive_decode_geometry`),
    which is what makes this the SEAM for decode's value width: ``v`` here is still
    the caller's own logical ``d_v``, padded to ``config.d_v``, and the returned
    ``y`` is sliced back to it (``rola.ops.padding``). Per-level widths are NOT
    padded -- :func:`derive_decode_geometry` already refused ``config`` past
    :func:`_admit`'s floor rather than pad it, so ``read_levels``/``write_levels``
    reach the kernel exactly as the caller holds them.
    """
    guard_no_grad((v, g_write, *read_levels, *write_levels,
                   *(decay.dials if decay is not None else ())))
    logical_d_v = v.shape[-1]
    v = pad_v(v, config.d_v)
    plane = step_plane(v, config, decay, state, page_table)
    BH = v.shape[0] * v.shape[2]
    if workspace is None:
        workspace = DecodeScratch.build(config, BH, v.device)
    y = step(DecodePlan(
        config=config, heads=v.shape[2], read=read_levels, write=write_levels,
        normalize=(False,) * config.D if normalize is None else tuple(normalize),
        dials=leaf_rate_dials_for(decay, config, v.device),
        g_write=g_write, v=v, state_plane=plane, page_table=page_table,
        scratch=workspace, pool=pool))
    return slice_y(y, logical_d_v), state, workspace
