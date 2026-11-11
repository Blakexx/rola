# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The prefill family's DAG, read top to bottom, and the context it runs over.

ONE SEGMENT in, its readout out: the facts pass over the segment's bundle, the
carry and the intra, then the post pass ``y = num / (den + eps / r)``. The stateless and
stateful bands are one tuple, the stateless being the stateful with the residency band
masked out. Segmentation is the LAYER's (docs/internals/engine/prefill_dag.md): this
DAG runs whatever segment it is handed, against whatever state it is handed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from rola.engine import SUPPLIABLE, Mask, node, rules, run, validate_dag
from rola.engine.facts import liveness, nodes
from rola.engine.facts.call import require_envelope

_SEEDS = ("routes", "values", "decay", "state", "expert", "want_state", "given_state_in", "given_arena")


@dataclass(frozen=True, slots=True)
class PrefillContext:
    """One segment's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~rola.RoLAState`, which owns its own residency;
    `state_in` and `arena` are the same two facts handed in directly, which is how a
    caller drives the kernels without the state object.
    """

    routes: Any
    values: Any
    decay: Any = None
    state: Any = None
    expert: Any = None
    state_in: torch.Tensor | None = None
    arena: Any = None
    want_state: bool = False
    supplied: Mapping[str, Any] = field(default_factory=dict)


@node(gives=("facts", "carve", "layout", "statics"), needs=("routes", "values"), stream="main")
def facts(f):
    """THE FACTS PASS over the bundle, one launch: both sides' class-1 words and their tile unions, each side's
    lists at its list group and the writers' factor products, with the carve
    the planner picks, the layout the words are read in -- the topology's padded widths -- and the two sides' statics,
    their dense levels unread (`rola.ops.facts`)."""
    from rola.engine.carve import plan_carve
    from rola.ops import carry as carry_ops
    from rola.ops import facts as facts_ops

    routes = f["routes"]
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, f["values"].v.shape[-1])
    out = facts_ops.facts(routes, f["values"], layout, *statics, carry_ops.lists_form(carry_ops.arm_key(carve)))
    return out, carve, layout, statics


@node(gives=("atom_bits",), needs=("facts", "layout", "statics"), stream="main", masks=(Mask.M2,))
def activity(f):
    """The per-atom ACTIVITY byte, ``WRITTEN | READ``, folded from the words: the state's write set
    (`rola.engine.facts.liveness.page_bits`)."""
    return liveness.page_bits(f["facts"].words, f["layout"], *f["statics"])


@node(gives=("bc",), needs=("d_v",), masks=(Mask.M2,))
def box(f):
    """The carry's box, the arena's allocation extent: allocator policy, never an address."""
    from rola.ops import carry as carry_ops
    from rola.ops import padding as padding_ops

    dv = padding_ops.padded_value_width(f["d_v"], shipped=carry_ops.SHIPPED_DV)
    return carry_ops.box_leaves(dv, carry_ops.LaunchShape().warps_per_cta)


@node(gives=("state_in", "arena"),
      needs=("state", "given_state_in", "given_arena", "routes", "d_v", "atom_bits", "paging", "bc"),
      stream="side", masks=(Mask.M2,), host_sync=True)
def residency(f):
    """The state's entry plane and arena, this call's write atoms committed (`RoLAState._kernel_entry`)."""
    state = f["state"]
    if state is None:
        return f["given_state_in"], f["given_arena"]
    return state._kernel_entry(f["routes"], f["atom_bits"], d_v=f["d_v"], paging=f["paging"], BC=f["bc"])


@node(gives=("num", "den"), needs=("routes", "values", "facts", "carve", "layout", "statics"), stream="main")
def launch(f):
    """The carry at its bytes and the intra into the same planes: ``num [BH, T, d_v]`` and
    ``den [BH, T]``, fp32, each over the reader factor (`rola.ops.prefill.prefill`)."""
    from rola.ops import prefill as prefill_ops

    return prefill_ops.prefill(f["routes"], f["values"], f["facts"], f["carve"], f["layout"], *f["statics"])


@node(gives=("y",), needs=("num", "den", "routes"), stream="main")
def post(f):
    """THE POST PASS: ``y = num / (den + eps)``, ``[B, T, H, d_v]`` fp32 (`rola.ops.prefill.readout`)."""
    from rola.ops import prefill as prefill_ops

    return prefill_ops.readout(f["num"], f["den"], f["routes"])


PREFILL_DAG = (
    rules.paging_preference.node,
    nodes.topology,
    nodes.classify_call,
    rules.mask.node,
    facts,
    activity,
    box,
    residency,
    nodes.join,
    nodes.backing,
    launch,
    post,
)

validate_dag(PREFILL_DAG, _SEEDS)


def validate_context(ctx: PrefillContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a prefill call can be refused for.

    It runs before the DAG's first node, so nothing here can refuse a call whose
    operands a launch has already consumed.
    """
    from rola._state import RoLAState
    from rola.routing.factors import RouteFactors
    from rola.values import Values

    illegal = sorted(set(ctx.supplied) - SUPPLIABLE)
    if illegal:
        raise ValueError(
            f"a PrefillContext may not supply {illegal}: the suppliable set is closed at "
            f"{sorted(SUPPLIABLE)}. Everything below the amplitude boundary is one "
            f"canonical implementation, gated once, and substituting it would fork "
            f"fact production.")
    routes, values, decay = ctx.routes, ctx.values, ctx.decay
    if type(routes) is not RouteFactors:
        raise TypeError(
            "rola_op's first argument must be a RouteFactors bundle, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair; "
            "a layer builds one with its producer, and RouteFactors.from_levels builds one from any levels -- "
            "its invariants are checked either way.")
    if type(values) is not Values:
        raise TypeError(
            "rola_op's second argument must be a Values bundle, got "
            f"{type(values).__name__}. The value stream reaches the op in the internal format; "
            "build one with Values.from_projection(v).")
    #: THE LAST DIM IS NOT CHECKED, IT IS READ. `d_v` is the value stream's own width
    #: and the routing carries no value geometry, so there is nothing to agree with;
    #: what must match the bundle is the token addressing, `[B, T, H_w]`, a write head's.
    expected = (routes.batch, routes.tokens, routes.write_heads)
    if (values.batch, values.tokens, values.heads) != expected:
        raise ValueError(
            f"values must be [B, T, H_w, d_v] with [B, T, H_w]={expected} (a write head's) to match the routing "
            f"bundle, got {tuple(values.v.shape)}")
    if values.device != routes.device:
        raise ValueError(f"values are on {values.device} but the routing bundle is on {routes.device}")
    if decay is not None:
        decay.validate_against(routes.topology)
    if ctx.state is not None and type(ctx.state) is not RoLAState:
        raise TypeError(
            "rola_op's `state` argument is a rola.RoLAState from rola.state(), got "
            f"{type(ctx.state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the shape it is bound to.")
    require_envelope(routes, values, decay, ctx.state)


def run_prefill(ctx: PrefillContext) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Walk the prefill DAG over a VALIDATED `ctx` (:func:`validate_context`, which the caller runs in its own
    grad mode): ``(y, state_out)``, the plane the call stored into or ``None``.

    The facts pass and the intra read the bundle; the carry leg's floating operands are `rola.ops.prefill`'s,
    dequantized there once a call until the carry reads the rows.
    """
    facts = {"routes": ctx.routes, "values": ctx.values, "decay": ctx.decay, "state": ctx.state,
             "expert": ctx.expert, "want_state": ctx.want_state, "given_state_in": ctx.state_in,
             "given_arena": ctx.arena}
    facts.update(ctx.supplied)
    done = run(PREFILL_DAG, facts)
    return done["y"], done["state_out"]
