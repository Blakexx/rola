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
from rola.ops.lattice import derive_lattice, permutation
from rola.ops.padding import pad_v, padded_value_width, slice_y
from rola.ops.paging import MMA_K_QUANTUM, page_layout
from rola.routing import quant
from rola.routing.types import Topology

__all__ = [
    "DecodeScratch",
    "arms",
    "derive_decode_geometry",
    "leaf_rate_dials_for",
    "step",
    "step_plane",
]


def arms() -> tuple[tuple[int, int, bool], ...]:
    """The decode arms THIS BINARY carries, as ``(d_v, D, decay)`` rows.

    A full build carries the whole declared matrix (``d_v x D x decay`` = 16 rows). An
    ITERATION build names a subset with ``--//:decode_arms`` and is NOT SHIPPABLE; this
    is how such a binary identifies itself, and the launch refuses any arm not listed
    here rather than mis-launching it (``docs/build.md``).
    """
    return tuple((int(dv), int(d), bool(decay)) for dv, d, decay in extension().rola_decode_arms())


#: THE PHASE LEDGER'S SHAPE: `[ctas][warps][phases]`, the kernel's own `DecodePhase` order.
LEDGER_PHASES = ("fold", "factor", "admit", "walk", "combine")


def ledger_warps() -> int:
    """The phase ledger's warps: the step's plan's at the current card's build, as the C++ entry derives them."""
    return int(extension().rola_decode_ledger_warps())


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/decode/decode.md#phase-ledger)
    """Bind the phase ledger the next steps book into, ``[ctas, ledger_warps(), 5]`` int64 cycles, or ``None``."""
    extension().rola_decode_ledger_bind(ledger)


def _shipped_decode_dv(D: int, decay: bool) -> tuple[int, ...]:
    """The ``DV`` values this binary's decode arms ship at this ``(D, decay)``.

    Read off :func:`arms` (the binary's own declared arm set) rather than a literal
    set: decode's shipped ``d_v`` axis is wider than carry's single ``D_V`` (a full
    build carries ``d_v x D x decay``), and a literal tuple here would drift from
    whatever the binary carries.
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
    eps: float = READOUT_EPS,
    write_dense: bool | None = None,
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

    ``ctas`` is ``None``: the step sizes its grid itself from the card, the same every step (one wave of the
    declared residency, at most the heads' unit supply; docs/internals/decode/decode.md#the-split-policy), and the work
    queue balances it inside the launch; a gate forces a count by replacing it.
    """
    widths = tuple(topology.widths)
    _admit(widths)
    d_v = padded_value_width(d_v, shipped=_shipped_decode_dv(len(widths), decay))
    D = len(widths)
    k, m = derive_lattice(widths)
    row_offset = []
    acc = 0
    for w in widths:
        row_offset.append(acc)
        acc += w

    #: the bundle's flag where the caller has one (the step's graph); a caller holding a topology alone takes the
    #: production rule
    if write_dense is None:
        write_dense = not topology.activation_properties(D - 1, "write").exact_zeros
    return DecodeGeometry(
        D=D, widths=widths, level_row_offset=tuple(row_offset), lattice_k=k, lattice_m=m,
        d_v=int(d_v), has_decay=decay, eps=eps, ctas=None, write_dense=write_dense)


@dataclass(slots=True)
class DecodeScratch:
    """Per-sequence scratch, allocated once and reused across steps.

    ``yacc``, ``queue``, ``growth_ctr`` and ``done`` are set exactly once here: every one of them
    is reset by the kernel that last reads it (a head's finisher, the last CTA out), so no
    ``memset`` launch ever stands between two decode steps and the whole step stays capturable
    in a CUDA graph.
    """

    #: `[BH, cols]` fp32, a head's readout sums, added into once by every CTA visit that walks it.
    yacc: torch.Tensor
    #: `[4 BH + 2]` int32, THE WORK QUEUE: per head the positions claimed, the positions finished,
    #: the position count (-1 until published) and the admission state, then the unstarted-head
    #: cursor and the exit count (docs/internals/decode/decode.md#queue).
    queue: torch.Tensor
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
            yacc=torch.zeros(BH, config.cols, **f32),
            queue=_fresh_queue(BH, device),
            growth=torch.zeros(BH, **i32),
            growth_any=torch.zeros(1, **i32),
            growth_ctr=torch.zeros(1, **i32),
            growth_host=torch.empty(1, dtype=torch.int32, pin_memory=cuda),
            done=torch.zeros(BH, **i32),
            atom_bits=torch.empty(BH, config.N // MMA_K_QUANTUM, dtype=torch.bool,
                                  device=device))


def _fresh_queue(BH: int, device) -> torch.Tensor:
    """The queue as a step finds it: counters zero, every position count unpublished (-1)."""
    queue = torch.zeros(4 * BH + 2, dtype=torch.int32, device=device)
    queue[2 * BH:3 * BH] = -1
    return queue


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


def fold_levels(levels, scales, forms) -> list[torch.Tensor]:
    """The per-level RAW coefficient values, in torch: ``[BH, width_l]`` fp32, each stored entry on the u8 scale (a u8
    digit as its integer, a dense level's f16 coefficient times 255; a form with a floor at it in a live row), the
    values the step kernel's fold sums and scales; ``forms`` each level's (`rola.routing.quant.Form`).

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. The shipped path folds inside
    the step kernel, from the producer's own views; a gate that fed the kernel its own
    fold's output would prove nothing about the fold
    (docs/internals/decode/decode_fold.md#the-gate).
    """
    out = []
    for rows, scale, form in zip(levels, scales, forms, strict=True):
        raw = _fold(rows).float() * (quant.AMP_LEVELS / form.top)
        if form.floor is not None:
            live = _fold(scale.unsqueeze(-1)) > 0
            raw = torch.where(live, raw.clamp_min(quant.AMP_LEVELS * form.floor), raw)
        out.append(raw)
    return out


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
    page_bytes = page_layout(d_v).bytes
    if page_table is None:
        #: PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the dense
        #: plane is the same page sequence with every page resident, and `N` is page
        #: aligned by construction (docs/internals/state.md#format).
        expected = (BH, config.N // MMA_K_QUANTUM, page_bytes)
        if tuple(state.shape) != expected:
            raise ValueError(f"a dense state is [BH, N / {MMA_K_QUANTUM}, page_bytes] = {expected}, "
                             f"got {tuple(state.shape)}")
        return state
    if state.dim() != 2 or state.shape[1] != page_bytes:
        raise ValueError(
            f"under a page table the state is the arena's plane, [slots, {page_bytes}], got {tuple(state.shape)}")
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
    producer's own views straight into the CTA's shared memory, the digit masks and owner
    lists are built there, and the walk consumes both without either ever reaching DRAM
    (docs/internals/decode/decode.md#per-cta-prologue).

    Every written leaf has exactly one depositing lane group, so the state update needs no
    atomic and is bitwise reproducible; ``y`` is reproducible to the order the visits add
    its sums in (docs/internals/decode/decode.md#determinism).

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
        read_scale=list(plan.read_scale),
        write_scale=list(plan.write_scale),
        floor_read=plan.floor[0],
        floor_write=plan.floor[1],
        dials=plan.dials,
        v=plan.v,
        v_scale=plan.v_scale,
        state=plan.state_plane,
        yacc=plan.scratch.yacc,
        queue=plan.scratch.queue,
        growth=plan.scratch.growth,
        growth_any=plan.scratch.growth_any,
        growth_ctr=plan.scratch.growth_ctr,
        done=plan.scratch.done,
        atom_bits=plan.scratch.atom_bits,
        level_widths=list(config.widths),
        level_row_offsets=list(config.level_row_offset),
        lattice_k=config.lattice_k,
        lattice_m=config.lattice_m,
        write_dense=config.write_dense,
        ctas=-1 if config.ctas is None else config.ctas,
        eps=config.eps,
        page_table=plan.page_table,
        pool_slots=None if pool is None else pool.pool_slots,
        pool_cursor=None if pool is None else pool.pool_cursor,
    )
    B, _T, H, _d_v = plan.v.shape
    return y.view(B, H, 1, config.d_v).permute(0, 2, 1, 3).contiguous()


def _write_atom_bitmap(write_live, config: DecodeGeometry) -> torch.Tensor:
    """``[BH, N / 16]`` bool: the atoms this ONE-TOKEN step deposits into, EXACTLY, from each write level's live
    digits (:meth:`~rola.routing.factors.RouteFactors.live`).

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. Deleting it would delete the
    gate's reference; calling it on a shipped path would delete the gate's independence
    (docs/internals/decode/decode.md#second-derivation). It derives the CANONICAL leaf
    support digit by digit and crosses into lattice order through ``pi``, so it shares no
    step of the kernel's own enumeration.
    """
    device = write_live[0].device
    index = torch.arange(config.N, device=device)
    live = None
    for level, width in enumerate(config.widths):
        digit = (index // math.prod(config.widths[level + 1:])) % width
        column = _fold(write_live[level])[:, digit]
        live = column if live is None else (live & column)
    pi = permutation(config.widths, config.lattice_k, config.lattice_m, device=device)
    lattice_live = torch.empty_like(live)
    lattice_live.index_copy_(1, pi, live)
    return lattice_live.view(-1, config.N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)


def _decode_step(
    routes,
    values,
    config: DecodeGeometry,
    state: torch.Tensor,
    *,
    decay=None,
    workspace: DecodeScratch | None = None,
    page_table: torch.Tensor | None = None,
    pool=None,
) -> tuple[torch.Tensor, torch.Tensor, DecodeScratch]:
    """THE BUNDLE ENTRY for one token: the step's own bodies, in the step's order.

    ``routes`` is a one-token :class:`~rola.routing.factors.RouteFactors` and ``values`` a one-token
    :class:`~rola.values.Values`; the kernel reads their rows, scales and values as they are. ``state`` is the dense
    ``[BH, N / 16, page_bytes]`` plane, or the arena's ``[slots, page_bytes]`` one under ``page_table``.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]`` fp32.

    ``config`` ALREADY CARRIES THE PADDED ``DV`` (:func:`derive_decode_geometry`), which is what makes this the SEAM
    for decode's value width: the values are padded to ``config.d_v`` with zero columns and the returned ``y``
    is sliced back to the caller's ``d_v`` (``rola.ops.padding``). Per-level widths are NOT padded --
    :func:`derive_decode_geometry` already refused ``config`` past :func:`_admit`'s floor rather than pad it, so the
    rows reach the kernel exactly as the bundle holds them.
    """
    guard_no_grad((*routes.tensors(), *values.tensors(), *(decay.dials if decay is not None else ())))
    routes.refuse_grouped("the decode step")
    if routes.tokens != 1:
        raise ValueError(f"the decode step takes one token, and a one-token bundle is all rows: the routes are "
                         f"[B, T={routes.tokens}, H]")
    logical_d_v = values.d_v
    v = pad_v(values.v, config.d_v)
    plane = step_plane(v, config, decay, state, page_table)
    BH = values.batch * values.heads
    if workspace is None:
        workspace = DecodeScratch.build(config, BH, v.device)
    y = step(DecodePlan(
        config=config, heads=values.heads, read=routes.amp_read, write=routes.amp_write,
        read_scale=routes.scale_read, write_scale=routes.scale_write,
        floor=routes.floor_bits(), dials=leaf_rate_dials_for(decay, config, v.device), v=v, v_scale=values.scale,
        state_plane=plane, page_table=page_table, scratch=workspace, pool=pool))
    return slice_y(y, logical_d_v), state, workspace
