# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The shipped route producer: one module, one bundle out.

RoLA is linear attention under a structured feature map, so the feature map is the one
slot the public surface has. **THE DUCK IS THE BUNDLE.** A producer is anything callable
that turns a source stream into a well-formed
:class:`~rola.routing.factors.RouteFactors`; the library never asks what a producer is,
and correctness is producer-blind by construction, because every support bit is derived
from the returned tensors and never from anything a producer says about itself
(``docs/api.md``). There is no per-level producer tier, no assembly gateway and no
registration: those were three boundaries where one is load-bearing.

:class:`RouteProducer` is the one this package ships. It owns the whole job --

    projection -> per-level solves -> mass fold -> side gain -> bundle

-- because every one of those steps is a function of the SAME level list, and splitting
them across modules bought a negotiation protocol (which module computes the gain, who
packs which columns) whose only content was that the split existed.

**THE LEVEL LIST IS THE CONFIGURATION.** ``RouteProducer([IndependentRouting(width=64,
read=softmax(), write=entmax(1.5)), ...], hidden_size=512, num_heads=8)``: each element
is one level's WHOLE spec, width included, outermost level first. Depth, leaf count,
packed router width, logit offsets and the topology are all DERIVED from that list --
none of them is a second input that could disagree with it. A uniform stack is spelled
with :func:`uniform`; there is no broadcast sugar, because a broadcast is a rule the
reader has to know before they can read the widths off the call.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.factors import RouteFactors
from rola.routing.projection import _packed_router_logits
from rola.routing.side_gain import GainConfig, SideGain
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
    "RouteProducer",
    "dense_routing",
    "split_routing",
    "tied_routing",
    "uniform",
    "union_routing",
]


def dense_routing(width: int) -> IndependentRouting:
    """Softmax on both duties, untied -- the full-support level, at ``width``."""
    return IndependentRouting(width=width, read=SoftmaxActivation(), write=SoftmaxActivation())


def union_routing(width: int, alpha: float = 1.5) -> UnionRouting:
    """One exact entmax solve determining a shared support, split across duties."""
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


class RouteProducer(nn.Module):
    """The shipped producer: a learned packed router projection, exact solves, one bundle.

    ONE projection and one batched solve group cover EVERY level: the levels share a
    packed ``route_W`` and are dispatched per level inside the kernel with no privileged
    level, so a model pays one GEMM and one solve group for the whole topology.

    Args:
        levels: one :class:`~rola.routing.types.IndependentRouting`,
            :class:`~rola.routing.types.TiedRouting` or
            :class:`~rola.routing.types.UnionRouting` per routing level, outermost
            first. Each carries its own ``width``; :func:`uniform` builds a uniform
            stack. The tags name activations from the closed registry
            (``rola/routing/activations.py``) -- users never supply property metadata.
        hidden_size / num_heads: the source stream's width and the head count.
        bias: a learned bias on the routing projection.
        gain: whether the write-side gain is a learned projection. ``False`` removes the
            column rather than pinning it to 1, so no dead parameter is optimized.
        gain_bias_init: initialize the gain bias so an all-zero-logit gain starts at
            this value. ``None`` leaves it at zero (unit gain at zero logits).
        gain_config: the research surface behind ``gain``, reachable from
            :mod:`rola.expert` and deliberately not part of the contract.

    Construction validates the whole assembly, so an unbuildable model fails here and
    never inside a forward.
    """

    def __init__(
        self,
        levels,
        *,
        hidden_size: int,
        num_heads: int,
        bias: bool = False,
        gain: bool = True,
        gain_bias_init: float | None = None,
        gain_config: GainConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(levels, (IndependentRouting, TiedRouting, UnionRouting)):
            raise TypeError(
                "RouteProducer takes a LIST of level configs, one per routing level, and "
                f"got a single {type(levels).__name__}. A one-level model is `[level]`; a "
                "uniform stack is `uniform(D, level)`.")
        _levels = tuple(levels)
        if not _levels:
            raise ValueError("RouteProducer needs at least one routing level")
        for index, level in enumerate(_levels):
            if type(level) not in (IndependentRouting, TiedRouting, UnionRouting):
                raise TypeError(
                    f"levels[{index}] must be an IndependentRouting, TiedRouting or "
                    f"UnionRouting carrying its own width, got {level!r}")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive exact int, got {hidden_size!r}")
        if type(num_heads) is not int or num_heads < 1:
            raise ValueError(f"num_heads must be a positive exact int, got {num_heads!r}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bool(bias)
        self.levels = _levels
        #: The two derived views of ONE list: `routing` is the packed-projection layout
        #: the solve reads, `topology` is the routing contract the bundle and the
        #: consumer read. Both are functions of `levels` alone, so neither can disagree
        #: with the other about a width.
        self.routing = ResolvedRouting(levels=self.levels)
        self.topology = Topology(levels=self.levels)

        self.side_gain = SideGain(
            num_heads=num_heads,
            config=(gain_config if gain_config is not None else GainConfig.from_public(gain)),
            bias_init=gain_bias_init)
        #: ONE PARAMETER, and the gain's columns are its trailing span. The gain is a
        #: function of the read stream and so are the routing logits, so a second
        #: `[hidden -> H]` projection was a second GEMM and a second cast of the whole
        #: source stream for one column per head.
        columns = self.routing.packed_router_width + self.side_gain.columns
        self.route_W = nn.Parameter(torch.empty(num_heads, hidden_size, columns))
        _init_packed_router(self.route_W)
        if self.bias:
            self.route_bias = nn.Parameter(torch.zeros(num_heads, columns))
        else:
            self.register_parameter("route_bias", None)

    @property
    def widths(self) -> tuple[int, ...]:
        """One branch width per level, in level order."""
        return self.topology.widths

    # --- production ----------------------------------------------------------

    def forward(
        self,
        read_stream: torch.Tensor,
        write_stream: torch.Tensor | None = None,
    ) -> RouteFactors:
        """One source stream in, one validated :class:`RouteFactors` out.

        ``read_stream``/``write_stream`` are ``[B, T, hidden_size]``; passing only one
        means both duties read the same stream, which takes the single-projection path.
        """
        if not isinstance(read_stream, torch.Tensor) or read_stream.ndim != 3:
            raise ValueError(
                "read_stream must be [B, T, hidden_size], got "
                f"{getattr(read_stream, 'shape', type(read_stream).__name__)}")
        if read_stream.shape[-1] != self.hidden_size:
            raise ValueError(
                f"read_stream is {read_stream.shape[-1]} wide but this RouteProducer was "
                f"built for hidden_size={self.hidden_size}")
        if write_stream is not None and write_stream.shape != read_stream.shape:
            raise ValueError("write_stream must have the same shape as read_stream")

        B, T, _ = read_stream.shape
        H = self.num_heads
        read_bh, write_bh, gain_logits = self._solve(read_stream, write_stream)

        #: THE READ MASS IS CARRIED, NOT APPLIED: the readout is a ratio, so a per-token
        #: scale common to every leaf cancels between numerator and denominator, and the
        #: consumer that takes it scales the readout's floor by it instead of paying a
        #: pass over the levels. The WRITE mass is APPLIED, because the decay clock is
        #: defined on the stored level; it rides the write gain.
        read_levels, read_mass = _fold(read_bh, B=B, H=H, skip=self._skip("read"),
                                       carry_mass=True)
        write_levels, write_mass = _fold(write_bh, B=B, H=H, skip=self._skip("write"),
                                         carry_mass=False)
        #: THE GAIN IS A ROUTING FACTOR AND CARRIES THE FACTORS' OWN DTYPE (Blake,
        #: 2026-08-29): it followed the CALLER'S input dtype before, so an fp32 layer
        #: input produced bf16 levels and an fp32 `g_write` -- two dtypes out of one
        #: producer, which the raw decode seam narrowed and the planned seam did not.
        #: The producer is the one place that knows the operand dtype.
        gains = self.side_gain(gain_logits, batch=B, tokens=T, dtype=read_levels[0].dtype,
                               device=read_stream.device)
        return RouteFactors(
            topology=self.topology,
            read=read_levels, write=write_levels,
            g_write=gains["g_write"] if write_mass is None else gains["g_write"] * write_mass,
            read_mass=read_mass)

    def _skip(self, duty: str) -> tuple[bool, ...]:
        """The levels whose fold is an identity, from the topology's declared columns.

        A DECLARATION, never a measurement: a runtime ``sum(...) == 1`` test would be
        the very reduction the skip exists to delete, and would make the bundle's
        arithmetic depend on its data rather than on its configuration.
        """
        return tuple(self.topology.duty_simplex_output(level, duty)
                     for level in range(self.topology.D))

    def _project(self, x: torch.Tensor, x_w: torch.Tensor | None):
        """Packed router logits in the domain the caller's dtype selects.

        fp64 in, fp64 out (the oracle-equivalence gate runs the whole layer there). On
        CUDA the projection is **bf16 operands, fp32 accumulate, bf16 out** -- the
        industry-standard class -- which is a tensor-core GEMM rather than
        the fp32 SIMT one, measured 2.76x faster at the pinned cell. The solve widens
        each logit in-register, so nothing in the tree ever narrows a solved value to buy
        it. Off CUDA there are no tensor cores to reach, so the torch lowering keeps
        fp32.
        """
        if x.dtype == torch.float64:
            projection_dtype = torch.float64
        elif x.is_cuda:
            projection_dtype = torch.bfloat16
        else:
            projection_dtype = torch.float32
        #: TOKEN-MAJOR `[B, T, H, W]` for the hand-CUDA solve: that is the sgemm's own
        #: output order, so the head axis is free here and the solve addresses `b` and
        #: `h` as two strides; the flat `[B*H, T, W]` form costs a whole pass over the
        #: plane to make `(B, H)` one axis
        #: (docs/internals/entmax/entmax.md#stream-split).
        #:
        #: The PREDICATE is the one `production_routing_factor_levels` selects the CUDA
        #: solve on, and it is here for the same reason: every other consumer -- the
        #: torch lowering, and the fp64 reference solve the oracle-equivalence gate
        #: swaps in -- reads a level by slicing, with no stride tuple to read a head
        #: axis through, so handing them the flat plane is what keeps this an
        #: implementation choice rather than a change to what a reference accepts.
        cuda_solve = projection_dtype is torch.bfloat16
        return _packed_router_logits(
            x, self.route_W, self.routing, route_bias=self.route_bias, h_w=x_w,
            projection_dtype=projection_dtype, token_major=cuda_solve,
            gain_columns=self.side_gain.columns)

    def _solve(self, x: torch.Tensor, x_w: torch.Tensor | None):
        read_logits, write_logits, gain_logits = self._project(x, x_w)
        read, write = production_routing_factor_levels(read_logits, write_logits, self.routing)
        return read, write, gain_logits


# --- helpers ----------------------------------------------------------------


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


def _fold(levels: tuple[torch.Tensor, ...], *, B: int, H: int, skip: tuple[bool, ...],
          carry_mass: bool):
    """Factor every level into ``(simplex, mass)`` and return the mass product, or ``None``.

    ``skip[l]`` is the topology's DECLARED ``simplex_output`` for that cell, never a
    runtime test: a ``sum(...) == 1`` check is the reduction the skip exists to delete.
    ``carry_mass`` leaves the mass ON the levels and returns the product for the caller
    to apply -- legal only where the consumer's use of the levels is homogeneous of
    degree zero. What the two duties then do with it, and the one consequence for the
    decay clock: ``docs/api.md`` section 2.2.

    The divide runs in the solve's ``[BH, T, width]`` domain because the public
    ``[B, T, H, width]`` layout is a permute of it and only one of the two can be the
    materialized tensor.
    """
    normalized: list[torch.Tensor] = []
    mass_product: torch.Tensor | None = None
    for folded, unit_sum in zip(levels, skip):
        if not unit_sum:
            #: THE SUM IS AT LEAST fp32, whatever the level is stored in: a width-256
            #: sum of bf16 addends accumulated in bf16 loses three digits of the mass
            #: the readout's floor is scaled by, and an fp64 level's sum must not be
            #: narrowed to buy that. The addends stay stored -- only the accumulator is
            #: widened -- so this reads the plane once and materializes nothing.
            mass = folded.sum(dim=-1, dtype=torch.promote_types(folded.dtype, torch.float32))
            if not carry_mass:
                folded = folded / mass.unsqueeze(-1).to(folded.dtype)
            mass_product = mass if mass_product is None else mass_product * mass
        width = folded.shape[-1]
        normalized.append(folded.reshape(B, H, -1, width).permute(0, 2, 1, 3))
    if mass_product is None:
        return tuple(normalized), None
    return tuple(normalized), mass_product.reshape(B, H, -1).permute(0, 2, 1)
