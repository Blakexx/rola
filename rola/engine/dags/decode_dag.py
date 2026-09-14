# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode family's DAG, read top to bottom, its context, and its four entries.

M3 (decode-dense) and M4 (decode-paged) are ONE tuple, and after the step absorbed its
own residency question they are the same tuple: what the mask still separates is the
host-visible verdict band of :data:`DECODE_VERDICT`, which a dense step has no fact for.
A GROWTH STEP IS NOT A THIRD GRAPH -- it is this same tuple walked twice around an
admission, which is what makes the replay self-correcting and the step capturable
(docs/internals/engine/decode_dag.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rola.engine import (
    DecodePlan,
    Mask,
    node,
    rules,
    run,
    validate_capturable,
    validate_dag,
)
from rola.ops import decode as bodies

_SEEDS = ("routes", "v", "decay", "state", "expert")

#: BOTH decode masks. A node carrying only `Mask.M4` is one the dense backing has no
#: work for; there is no node the paged backing skips.
_DECODE = (Mask.M3, Mask.M4)


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """One decode step's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~rola.RoLAState`, which owns its own backing and
    therefore answers the one question the mask folds.
    """

    routes: Any
    v: torch.Tensor
    state: Any
    decay: Any = None
    expert: Any = None


@node(gives=("widths", "d_v", "has_decay", "BH", "device", "paged"),
      needs=("routes", "v", "decay", "state"), masks=_DECODE, capturable=True)
def decode_call(f):
    routes, state = f["routes"], f["state"]
    return (tuple(routes.topology.widths), f["v"].shape[-1], f["decay"] is not None,
            routes.batch * routes.heads, routes.device, bool(state.paged))


@node(gives=("geometry",),
      needs=("routes", "widths", "d_v", "has_decay", "BH", "device"),
      pure_of=("widths", "d_v", "has_decay", "BH", "device"), masks=_DECODE,
      capturable=True)
def geometry(f):
    """The frozen carrier. Memoized on the CONFIG half alone, which is what it is."""
    return bodies.derive_decode_geometry(
        f["routes"].topology, d_v=f["d_v"], decay=f["has_decay"], BH=f["BH"],
        device=f["device"])


@node(gives=("scratch",), needs=("state", "geometry", "BH", "device"),
      per_state=True, pure_of=("geometry", "BH", "device"), masks=_DECODE,
      capturable=True)
def scratch(f):
    """The per-sequence buffers, held WEAKLY against the state that owns them.

    The weakness is required rather than incidental: a dropped sequence must drop its
    scratch, and two concurrent sequences must never share a buffer. Being memoized is
    also what hoists the allocation and the one-time zeroing of the counters and flags above
    any captured region (docs/internals/decode/decode.md#capture).
    """
    return bodies.DecodeScratch.build(f["geometry"], f["BH"], f["device"])


@node(gives=("carried_plane", "page_table", "pool"), needs=("state", "routes"),
      stream="main", masks=_DECODE, capturable=True)
def backing(f):
    """The state's own plane, table and slack pool, over the residency it already has.

    A step admits from its POOL inside the kernel and from nowhere else: the admission a
    verdict the pool could not cover calls for is :func:`admit_growth`, between two walks,
    because that verdict is a host fact.
    """
    return f["state"]._decode_entry(f["routes"])


@node(gives=("state_plane",),
      needs=("v", "geometry", "decay", "carried_plane", "page_table"),
      masks=_DECODE, capturable=True)
def state_plane(f):
    """The plane the kernel indexes, and the per-step agreement checks that find it."""
    return bodies.step_plane(f["v"], f["geometry"], f["decay"],
                             f["carried_plane"], f["page_table"])


@node(gives=("plan",),
      needs=("geometry", "routes", "v", "decay", "state_plane", "page_table", "scratch",
             "pool"),
      stream="main", masks=_DECODE, capturable=True)
def build_plan(f):
    """The launch's arguments, over the producer's OWN views.

    It carries a stream because the decay arm's dials are derived here: a per-HEAD
    constant, the one operand the step kernel does not fold
    (docs/internals/decode/decode_fold.md).
    """
    routes, geometry = f["routes"], f["geometry"]
    return DecodePlan(
        config=geometry, heads=routes.heads, read=routes.read, write=routes.write,
        normalize=tuple(routes.read_needs_normalization()),
        dials=bodies.leaf_rate_dials_for(f["decay"], geometry, f["v"].device),
        g_write=routes.g_write, v=f["v"], state_plane=f["state_plane"],
        page_table=f["page_table"], scratch=f["scratch"], pool=f["pool"])


@node(gives=("y",), needs=("plan",), stream="main", masks=_DECODE, capturable=True)
def step(f):
    """THE STEP: fold, expand, walk and the residency verdict, in one launch."""
    return bodies.step(f["plan"])


@node(gives=("grow",), needs=("scratch", "y"), masks=(Mask.M4,), host_sync=True)
def read_verdict(f):
    """The ONE 4-byte device-to-host read, and it CANNOT precede the step any more.

    The verdict is now the step's OWN output, published by the last CTA of its single
    launch, so there is nothing left to enqueue the read ahead of: the step is the last
    launch either way and the copy is behind it by data dependence, not by policy
    (docs/internals/decode/decode.md#per-bh-verdict).
    """
    workspace = f["scratch"]
    workspace.growth_host.copy_(workspace.growth_any)
    return bool(workspace.growth_host[0])


@node(gives=("atom_bits",), needs=("routes", "geometry"), masks=_DECODE,
      reference_only=True)
def write_atom_bitmap(f):
    """The gate's INDEPENDENT second derivation of the write-atom set.

    Declared so that `validate_dag`'s refusal of a reference_only node has something to
    refuse: the shipped path derives this set on the device, inside the step, and a gate
    that compared the step against itself would prove nothing
    (docs/internals/decode/decode.md#second-derivation).
    """
    return bodies._write_atom_bitmap(f["routes"].write, f["geometry"])


#: THE CAPTURED RECIPE. `decode_step_gated` is exactly this tuple, and a CUDA graph
#: captured over it replays correctly across an admission because nothing in it reads a
#: device value on the host and nothing in it allocates.
DECODE_STEP = (
    decode_call,
    rules.decode_mask.node,
    geometry,
    scratch,
    backing,
    state_plane,
    build_plan,
    step,
)

#: The same tuple with the host-visible verdict read appended. ONE node, and it is the
#: only place a decode step touches the host at all.
DECODE_VERDICT = DECODE_STEP + (read_verdict,)

validate_dag(DECODE_STEP, _SEEDS)
validate_dag(DECODE_VERDICT, _SEEDS)
validate_capturable(DECODE_STEP, "DECODE_STEP")


def validate_context(ctx: DecodeContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a decode step can be refused for here.

    It runs before the DAG's first node, so nothing here can refuse a step whose
    operands a launch has already consumed. The per-step SHAPE agreements are NOT here:
    they are :func:`rola.ops.decode.step_plane`'s, checked against the plane the step
    actually indexes.
    """
    ctx.state._require_carrying()
    if ctx.routes.tokens != 1:
        raise ValueError(
            f"the decode op consumes exactly one token, got T={ctx.routes.tokens}")
    bodies.guard_no_grad((ctx.v, ctx.routes.g_write,
                          *ctx.routes.read, *ctx.routes.write,
                          *(ctx.decay.dials if ctx.decay is not None else ())))


def _facts(ctx: DecodeContext) -> dict:
    return {"routes": ctx.routes, "v": ctx.v, "decay": ctx.decay, "state": ctx.state,
            "expert": ctx.expert}


def _scratch_of(state):
    """The buffers a step of `state` left behind, or `None` before its first step."""
    table = scratch.memo.get(state)
    if not table:
        return None
    if len(table) > 1:
        raise RuntimeError(
            f"{len(table)} decode scratches are alive for one state: the launch config "
            "is FROZEN for the sequence, so a second one means the decay arm changed "
            "between two steps of the same sequence. Run the sequence with one arm.")
    return next(iter(table.values()))


def decode_step_gated(routes, v: torch.Tensor, state, *, decay=None, expert=None):
    """ONE decode step with NO host participation -- the CAPTURE-SAFE entry.

    Same kernel in the same order as :func:`decode_forward`, minus the verdict read: each
    batch-head is gated on the step's own device-side answer, so one whose write set it
    can neither address nor admit is a NO-OP that leaves the state exactly as the
    admission needs it. The CALLER owns the verdict, at whatever synchronization it
    already has:

        y, state = decode_step_gated(...)   # or graph.replay()
        if growth_pending(state):
            admit_growth(state)
            y, state = decode_step_gated(...)   # or graph.replay(); now it runs

    A REPLAY IS SELF-CORRECTING because the verdict is re-derived inside it: after the
    admission it is 0 and the same captured graph performs the step. It advances every
    sequence EXACTLY ONCE across the pair -- a batch-head that ran in the first walk
    carries a done flag into the second and re-reads for its `y` without depositing again
    (docs/internals/decode/decode.md#done-flags).
    """
    ctx = DecodeContext(routes=routes, v=v, state=state, decay=decay, expert=expert)
    validate_context(ctx)
    return run(DECODE_STEP, _facts(ctx))["y"], state


def growth_pending(state) -> bool:
    """Did the last gated step skip itself for want of an atom? ONE 4-byte read."""
    workspace = _scratch_of(state)
    if workspace is None or not state.paged:
        return False
    workspace.growth_host.copy_(workspace.growth_any)
    return bool(workspace.growth_host[0])


def admit_growth(state) -> None:
    """Admit the pending step's write atoms, from the step's own condensation."""
    workspace = _scratch_of(state)
    if workspace is None:
        raise RuntimeError("admit_growth wants the workspace a gated step leaves behind")
    state._decode_entry_arena(workspace.atom_bits).wait()


def pool_headroom(state) -> int:
    """Slack slots the thinnest batch-head still has. A HOST READ, AT LEISURE.

    Pool-covered growth raises no verdict -- the step admitted itself and there is nothing
    urgent for the host -- so this is the level a serving loop polls to decide when to
    :func:`refill_pool`, on its own schedule and off the step's path.
    """
    return 0 if state._arena is None else state._arena.pool_headroom()


def refill_pool(state) -> int:
    """Replace the slack slots the device claimed, and return how many. HOST, OFF PATH."""
    if state._arena is None:
        raise RuntimeError("a pool refill wants a paged state; this one is dense")
    return state._arena.refill_pool()


def decode_forward(routes, v: torch.Tensor, state, *, decay=None, expert=None):
    """ONE decode step over a routing bundle and a carried state -- the whole surface.

    ``routes`` is a one-token :class:`~rola.routing.factors.RouteFactors`; ``v`` is
    ``[B, 1, H, d_v]``; ``state`` is a :class:`~rola.RoLAState` that CARRIES
    (a step continues a sequence, so an empty state has nothing to decode from).
    Returns ``(y, state)`` -- ``y`` shaped ``[B, 1, H, d_v]`` and the SAME state
    object, updated in place, exactly as :func:`rola.rola_op` returns it.

    This is :data:`DECODE_STEP` with the verdict read spliced in, and a growth step is
    that same walk again after the admission the verdict called for -- never a second
    graph (docs/internals/decode/decode.md#growth).
    """
    ctx = DecodeContext(routes=routes, v=v, state=state, decay=decay, expert=expert)
    validate_context(ctx)
    facts = run(DECODE_VERDICT, _facts(ctx))
    if not facts["grow"]:
        return facts["y"], state
    #: The batch-heads that could not advance deposited nothing and read nothing, so the
    #: state is exactly where the admission needs it, and `scratch.atom_bits` is the
    #: step's own condensation -- the host derives no support
    #: (docs/internals/decode/decode.md#second-derivation).
    admit_growth(state)
    return run(DECODE_STEP, _facts(ctx))["y"], state
