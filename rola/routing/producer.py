# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ROUTING A LAYER COMPUTES, DECLARED: its levels, the write gain and the projection bias.

RoLA is linear attention under a structured feature map, and the feature map is declared, not supplied: a
:class:`Routing` is DATA, the layer owns the weights it implies, and the producer (:func:`rola.ops.produce.produce`)
computes the bundle from them in one call. The format and its producer are ours, so a routing the registry does not
hold is added to the registry -- its activation, its solve, its reference and its oracle -- never plugged in beside
it (docs/api.md section 2).

**THE LEVEL LIST IS THE CONFIGURATION.** ``Routing([IndependentRouting(width=64, read=softmax(),
write=entmax(1.5)), ...])``: each element is one level's WHOLE spec, width included, outermost level first. Depth,
leaf count, packed router width, logit offsets and the topology are all DERIVED from that list -- none of them is a
second input that could disagree with it. A uniform stack is spelled with :func:`uniform`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch
import torch.nn as nn

from rola.routing.side_gain import GainConfig
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    RoutingActivation,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
)

__all__ = [
    "Routing",
    "dense_routing",
    "split_routing",
    "tied_routing",
    "uniform",
    "union_routing",
]


def dense_routing(width: int) -> IndependentRouting:
    """Softmax on both duties, untied -- the full-support level, at ``width``."""
    return IndependentRouting(width=width, read=SoftmaxActivation(), write=SoftmaxActivation())


def union_routing(width: int, alpha: float = 1.5) -> UnionRouting:  #: reserved: the solve's order (2.0: sparsemax)
    """One exact entmax solve determining a shared support, split across duties. ``alpha`` is reserved: the solve's
    order, 1.5 or 2.0 (sparsemax), a declaration's to choose."""
    return UnionRouting(width=width, alpha=alpha)


def tied_routing(width: int, activation: RoutingActivation = EntmaxActivation(1.5)) -> TiedRouting:
    """One activation, one solve -- the read factor IS the write factor, at ``width``.

    Distinct from :func:`union_routing`: union routing splits ONE shared support
    across two roles, so the read and write factors differ (a share each) even though
    they came from one solve. Tied routing hands both duties the SAME tensor outright --
    there is no split to perform, because there is only one duty's worth of output.
    """
    return TiedRouting(width=width, op=activation)


def split_routing(width: int, alpha: float = 1.5, *,
                  sparse_duty: str = "write") -> IndependentRouting:
    """The alternating scheme's level: entmax on one duty, softmax on the other.

    Named because the split-level scheme is a configuration the campaign runs and the
    support-symmetry argument is about -- not because the producer has a code path for
    it. It is two registry keys in a level config, like every other level.
    """
    if sparse_duty not in ("read", "write"):
        raise ValueError(f"sparse_duty must be 'read' or 'write', got {sparse_duty!r}")
    sparse, dense = EntmaxActivation(alpha), SoftmaxActivation()
    return IndependentRouting(
        width=width,
        read=sparse if sparse_duty == "read" else dense,
        write=sparse if sparse_duty == "write" else dense)


def uniform(depth: int, level: LevelRouting) -> list[LevelRouting]:
    """``depth`` levels all spelled the same way -- the uniform stack, written once.

    Replaces the broadcast a single routing tag used to get: a list is what the producer
    takes, so the way to say "the same thing D times" is to build the list, not to teach
    the constructor a second admissible shape for its first argument.
    """
    if type(depth) is not int or depth < 1:
        raise ValueError(f"uniform depth must be a positive exact int, got {depth!r}")
    return [level] * depth


@dataclass(frozen=True)
class Routing:
    """THE DECLARATION: the levels, the write gain and the projection bias. Data only.

    Args:
        levels: one :class:`~rola.routing.types.IndependentRouting`,
            :class:`~rola.routing.types.TiedRouting` or
            :class:`~rola.routing.types.UnionRouting` per routing level, outermost
            first. Each carries its own ``width``; :func:`uniform` builds a uniform
            stack. The tags name activations from the closed registry
            (``rola/routing/activations.py``) -- users never supply property metadata.
        gain: whether the write-side gain is learned. ``False`` removes its projection
            column rather than pinning it to 1, so no dead parameter is optimized.
        gain_bias_init: initialize the gain bias so an all-zero-logit gain starts at
            this value. ``None`` leaves it at zero (unit gain at zero logits).
        gain_config: the research surface behind ``gain``, reachable from
            :mod:`rola.expert` and deliberately not part of the contract.
        bias: a learned bias on the routing projection.

    Construction validates the whole declaration, so an unbuildable model fails here and never inside a forward.
    """

    levels: tuple[LevelRouting, ...]
    gain: bool = True
    gain_bias_init: float | None = None
    gain_config: GainConfig | None = None
    bias: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.levels, (IndependentRouting, TiedRouting, UnionRouting)):
            raise TypeError(
                "Routing takes a LIST of level configs, one per routing level, and got a single "
                f"{type(self.levels).__name__}. A one-level model is `[level]`; a uniform stack is "
                "`uniform(D, level)`.")
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("Routing needs at least one routing level")
        for index, level in enumerate(levels):
            if type(level) not in (IndependentRouting, TiedRouting, UnionRouting):
                raise TypeError(
                    f"levels[{index}] must be an IndependentRouting, TiedRouting or UnionRouting "
                    f"carrying its own width, got {level!r}")
        if type(self.gain) is not bool:
            raise TypeError(f"gain must be a bool, got {self.gain!r}")
        if type(self.bias) is not bool:
            raise TypeError(f"bias must be a bool, got {self.bias!r}")
        object.__setattr__(self, "levels", levels)

    @cached_property
    def topology(self) -> Topology:
        """The routing contract the bundle and its consumers read."""
        return Topology(levels=self.levels)

    @cached_property
    def resolved(self) -> ResolvedRouting:
        """The packed-projection layout the solve reads: one function of the same level list as the topology."""
        return ResolvedRouting(levels=self.levels)

    @property
    def widths(self) -> tuple[int, ...]:
        """One branch width per level, in level order."""
        return self.topology.widths

    @property
    def gain_layout(self) -> GainConfig:
        return self.gain_config if self.gain_config is not None else GainConfig.from_public(self.gain)

    @property
    def gain_columns(self) -> int:
        """The projection columns the write gain takes, per head: its trailing span of ``route_W``."""
        return len(self.gain_layout.layout)

    @property
    def columns(self) -> int:
        """``route_W``'s columns per head: every level's slots, then the gain's."""
        return self.resolved.packed_router_width + self.gain_columns


def _init_packed_router(W: nn.Parameter) -> None:
    """Init ``[H, hidden, packed_router_width]`` like an ``nn.Linear(hidden, cols)``.

    The routing logit is ``h @ W[head]``, so the init must be a TRANSPOSED linear
    weight with ``fan_in == hidden``. ``kaiming_uniform_(a=sqrt(5))`` on ``[out,
    hidden]`` draws ``U(-1/sqrt(hidden), +1/sqrt(hidden))`` per element -- the bound
    does not depend on ``out`` -- so every level gets exactly the per-level fan-in
    semantics the projection GEMM has, one dot product over the full hidden dim per
    column.
    """
    H, hidden, cols = W.shape
    wt = torch.empty(H * cols, hidden, dtype=W.dtype, device=W.device)
    nn.init.kaiming_uniform_(wt, a=5 ** 0.5)
    W.data.copy_(wt.view(H, cols, hidden).transpose(-1, -2))
