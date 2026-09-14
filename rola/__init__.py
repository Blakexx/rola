# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RoLA — Routed Linear Attention.

RoLA is linear attention under a structured feature map. A head's recurrent state is
expanded into many sub-states that share the head's projections, and a learned routing
decides which sub-states each token reads and writes. That reduction IS this API: a
user who knows linear attention meets exactly ONE new concept here, the feature map,
and everything else is linear attention's own contract.

    rola_op(routes, v, decay)

``routes`` is the ``(q, k)`` pair after the map -- a ``RouteFactors`` bundle -- ``v`` is
the value stream, ``decay`` is the recurrence dial. The bundle comes from a producer,
and a producer is ANY callable that returns one; the shipped one takes the routing
LEVELS, each carrying its own width:

    routes = RouteProducer(uniform(2, union_routing(64, alpha=1.5)),
                           hidden_size=1024, num_heads=8)
    layer = RoLA(routes, d_v=64, decay=LearnedDecay(widths=routes.widths, num_heads=8))

**No public name says `v3`.** This repository IS that generation; encoding it in the
API would force a break at the next one for a reason that is not an API reason.

**What is deliberately NOT here.** Plans, schedules, launch geometry, numeric arms,
allocator caps -- every one of them is how the op RUNS rather than what the model IS,
and the op decides how it runs. They live in :mod:`rola.expert`, which is a marked door
and not part of the contract.

| Group | Names | Stability |
|---|---|---|
| op | `rola_op` | STABLE, semver-major to break |
| feature map | `RouteFactors`, `RouteProducer`, `dense_routing`, `split_routing`, `tied_routing`, `union_routing`, `uniform` | STABLE |
| recurrence dials | `ConstantDecay`, `LearnedDecay`, `DecaySource`, `LeafMassDecay` | STABLE |
| layer | `RoLA`, `LayerContinuation` | STABLE |
| state | `state()`, `RoLAState` | STABLE |
| configuration | `Topology`, `softmax()`, `entmax(alpha)`, `IndependentRouting`, `TiedRouting`, `UnionRouting` | STABLE |
| oracle | `naive_rola` | STABLE signature, EXACT semantics — it is the definition |
"""

__version__ = "0.1.0.dev0"

#: `rola.state` is the CONSTRUCTOR and nothing else. The class and the function live
#: in the private `rola._state` (renamed from `rola/state.py`, whose public
#: name the function was shadowing): docs/internals/state.md
from rola._state import RoLAState, state
from rola.interface import rola_op
from rola.layer import LayerContinuation, RoLA
from rola.ops.naive import naive_rola
from rola.routing.decay import ConstantDecay, DecaySource, LearnedDecay
from rola.routing.factors import RouteFactors
from rola.routing.producer import (
    RouteProducer,
    dense_routing,
    split_routing,
    tied_routing,
    uniform,
    union_routing,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
    entmax,
    softmax,
)

__all__ = [
    # the op — the single autograd boundary (rola/interface.py)
    "rola_op",
    # the feature map: the bundle (the whole producer contract) and the shipped producer
    "RouteFactors",
    "RouteProducer",
    "dense_routing",
    "split_routing",
    "tied_routing",
    "uniform",
    "union_routing",
    # the decay-source slot — where the recurrence's rates come from
    "DecaySource",
    "ConstantDecay",
    "LearnedDecay",
    "LeafMassDecay",
    # layer
    "RoLA",
    "LayerContinuation",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "state",
    "RoLAState",
    # oracle — ships in the wheel; it is the definition (docs/testing.md)
    "naive_rola",
    # configuration
    "Topology",
    "IndependentRouting",
    "TiedRouting",
    "UnionRouting",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors — users write these, never property metadata
    "softmax",
    "entmax",
    "__version__",
]
