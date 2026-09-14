# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The chunk family's DAG, read top to bottom, and the context it runs over.

The stateless and stateful prefill bands are ONE tuple: the stateless is the stateful with the
residency band masked out, which is what makes the two readable against each other
(docs/internals/engine/chunk_dag.md).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from rola.engine import SUPPLIABLE, ChunkPlan, node, rules, run, validate_dag
from rola.engine.facts import nodes
from rola.engine.facts.call import require_envelope

_SEEDS = ("routes", "v", "decay", "state", "expert", "want_state", "given_state_in",
          "given_arena")


@dataclass(frozen=True, slots=True)
class ChunkContext:
    """One chunk call's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~rola.RoLAState`, which owns its own residency;
    `state_in` and `arena` are the same two facts handed in directly, which is how a
    caller drives the kernel arm without the state object.
    """

    routes: Any
    v: torch.Tensor
    decay: Any = None
    state: Any = None
    expert: Any = None
    state_in: torch.Tensor | None = None
    arena: Any = None
    want_state: bool = False
    supplied: Mapping[str, Any] = field(default_factory=dict)


@node(gives=("plan",),
      needs=("call_class", "arm", "widths", "d_v", "v", "v_packed", "routes",
             "read_plane", "write_plane", "g_write", "read_mass", "union_table",
             "block_bits", "page_table", "atom_bits", "state_in", "state_out",
             "run_table"))
def build_plan(f):
    from rola.ops.constants import READOUT_EPS

    v_packed = f["v_packed"]
    return ChunkPlan(
        call_class=f["call_class"], arm=f["arm"], widths=f["widths"], d_v=f["d_v"],
        v_row_bytes=v_packed.shape[2] * v_packed.shape[3] * v_packed.element_size(),
        eps=READOUT_EPS, run_table=f["run_table"], y_dtype=f["v"].dtype,
        routes=f["routes"],
        read_plane=f["read_plane"], write_plane=f["write_plane"],
        g_write=f["g_write"], read_mass=f["read_mass"], v=v_packed,
        union_table=f["union_table"], block_bits=f["block_bits"],
        page_table=f["page_table"], atom_bits=f["atom_bits"], state_in=f["state_in"],
        state_out=f["state_out"])


CHUNK_DAG = (
    rules.paging_preference.node,
    nodes.topology,
    nodes.classify_call,
    rules.mask.node,
    nodes.manifest,
    rules.arm.node,
    nodes.pack_planes,
    nodes.pack_v,
    nodes.facts_tables,
    nodes.residency,
    nodes.join,
    nodes.backing,
    nodes.block_bits,
    rules.run_table.node,
    build_plan,
)

validate_dag(CHUNK_DAG, _SEEDS)

if set(nodes.facts_tables.gives) != {"union_table", "atom_bits"}:
    raise ValueError(
        "the union table and the atom bitmap came apart. They are ONE pass over the "
        "amplitude planes -- the plane is read once and voted at both grains -- so a "
        "DAG that produces them from two nodes has re-read it and paid the bytes "
        "twice (docs/internals/engine/chunk_dag.md).")


def validate_context(ctx: ChunkContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a chunk call can be refused for.

    It runs before the DAG's first node, so nothing here can refuse a call whose
    operands a launch has already consumed. Configuration, operand types, the
    suppliable set and the reverse pass's narrower arm matrix are all checkable at
    this point, and a check that is checkable here belongs here and nowhere else.
    """
    from rola._state import RoLAState
    from rola.routing.factors import RouteFactors

    illegal = sorted(set(ctx.supplied) - SUPPLIABLE)
    if illegal:
        raise ValueError(
            f"a ChunkContext may not supply {illegal}: the suppliable set is closed at "
            f"{sorted(SUPPLIABLE)}. Everything below the amplitude boundary is one "
            f"canonical implementation, gated once, and substituting it would fork "
            f"fact production.")
    routes, v, decay = ctx.routes, ctx.v, ctx.decay
    if type(routes) is not RouteFactors:
        raise TypeError(
            "rola_op's first argument must be a RouteFactors bundle, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair; "
            "build one with RouteProducer, or construct a RouteFactors directly -- its "
            "invariants are checked either way.")
    if not isinstance(v, torch.Tensor) or v.ndim != 4:
        raise ValueError(f"v must be [B, T, H, d_v], got {getattr(v, 'shape', type(v).__name__)}")
    #: THE LAST DIM IS NOT CHECKED, IT IS READ. `d_v` is the value stream's own width
    #: and the routing carries no value geometry, so there is nothing to agree with;
    #: what must match the bundle is the token addressing, `[B, T, H]`.
    expected = (routes.batch, routes.tokens, routes.heads)
    if tuple(v.shape[:3]) != expected:
        raise ValueError(
            f"v must be [B, T, H, d_v] with [B, T, H]={expected} to match the routing "
            f"bundle, got {tuple(v.shape)}")
    if v.device != routes.device:
        raise ValueError(f"v is on {v.device} but the routing bundle is on {routes.device}")
    if decay is not None:
        decay.validate_against(routes.topology)
    if ctx.state is not None and type(ctx.state) is not RoLAState:
        raise TypeError(
            "rola_op's `state` argument is a rola.RoLAState from rola.state(), got "
            f"{type(ctx.state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the shape it is bound to.")
    require_envelope(routes, v, decay, ctx.state)


def build_chunk_plan(ctx: ChunkContext) -> ChunkPlan:
    """Walk the chunk DAG over `ctx` and return the plan its terminal node built."""
    validate_context(ctx)
    facts = {"routes": ctx.routes, "v": ctx.v, "decay": ctx.decay, "state": ctx.state,
             "expert": ctx.expert, "want_state": ctx.want_state,
             "given_state_in": ctx.state_in, "given_arena": ctx.arena}
    facts.update(ctx.supplied)
    return run(CHUNK_DAG, facts)["plan"]
