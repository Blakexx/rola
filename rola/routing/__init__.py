"""Ops-internal RoLA routing implementation.

This package is intentionally not a root ``rola.ops`` export surface.
The canonical routing types are re-exported here for routing internals.
"""

from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    RouteDescriptor,
    RoutingActivation,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
)

__all__ = [
    "EntmaxActivation",
    "IndependentRouting",
    "LevelRouting",
    "ResolvedRouting",
    "RouteDescriptor",
    "RoutingActivation",
    "SoftmaxActivation",
    "TiedRouting",
    "UnionRouting",
]
