# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The RoLA layer: a value projection, the feature-map slot, and the op.

    RoLA(routes, d_v=..., decay=..., layer_idx=...)

``routes`` is the feature-map slot: any callable producing a
:class:`~rola.routing.factors.RouteFactors` bundle that also declares the topology.
:class:`~rola.routing.producer.RouteProducer` is the one this
package ships.
It also carries ``hidden_size`` and ``num_heads``, so the layer reads them off it.
``d_v`` is the per-head value width, which is the LAYER's: it sizes ``v_proj`` and
``o_proj``, and the routing carries no value geometry at all. ``decay`` is a decay
SOURCE (:mod:`rola.routing.decay`), the slot that decides where the recurrence's
forgetting rates come from. Everything else the layer used to take was either one of
those facts spelled out flat, or a launch decision the op makes.

**Continuation is a container the layer owns** (:class:`LayerContinuation`). It drops
``value_conv``/``route_conv`` and the conv-ring half of the old ``(state, conv_rings)``
pair (docs/internals/DELETIONS.md) -- they were never reviewed as a mechanism and are
not paper-1 scope. What is left of the pair is one field, ``state``, and the container
exists anyway rather than collapsing back to a bare :class:`~rola._state.RoLAState`:
future layer-level recurrences become fields on this container, never on the op's, and
the membership rule (docs/internals/state.md §7) needs a place to land them.

**Execution backends: two arms, and the choice is STRUCTURAL.** ``'cuda'`` is the
prefill arm through :func:`rola.rola_op` — DELETED pending the rebuild (the op raises,
naming the ruling; baseline = tag ``baseline/pre-k31``); ``'decode'`` is the
``T = 1`` sibling kernel, which bypasses selection architecturally (at one token
there is no tile to choose, no split, a statistics population of one) and carries
its own envelope -- BC-blind and general in the topology, so it is never gated on
the prefill matrix.

**THE BUILT ENVELOPE IS THE WHOLE SURFACE** (docs/api.md §1). Everything outside it
raises: training mode and any grad-bearing call (the backward arrives with the native
backward), a CPU device, decay, a jagged topology, ``d_v != 64``, an unbuilt
``(D, B)``. The layer's own clause is ``self.training``, which
has no meaning for a free function -- reentrant checkpointing runs its first pass under
``torch.no_grad()`` with ``module.training`` still ``True``, and that pass must refuse
too rather than produce a forward no backward can follow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from rola.engine import PlanOverrides
from rola.engine.dags.decode_dag import decode_forward
from rola.interface import (
    BackwardNotImplemented,
    operands_of,
    requires_backward,
    rola_op,
)
from rola.routing.decay import validate_decay_source

if TYPE_CHECKING:
    from rola._state import RoLAState

__all__ = ["LayerContinuation", "RoLA"]


@dataclass(frozen=True)
class LayerContinuation:
    """The LAYER's own recurrence bundle -- everything ``RoLA.forward`` needs handed
    back to continue a sequence, and nothing the op already owns.

    ``state`` is the op's paged tensor (:class:`~rola._state.RoLAState`) or ``None``
    for a fresh sequence. It is the only field today; a future layer-level recurrence
    is a field on THIS container, never on the op's state, per the membership rule
    (docs/internals/state.md §7): an object holds exactly the recurrences its own
    level introduced.
    """

    state: RoLAState | None = None


def _unpack_continuation(continuation):
    """The op's state out of the pair a forward returns, or ``None``.

    A bare :class:`~rola.RoLAState` and the retired ``(state, conv_rings)`` tuple are
    both refused by name rather than adopted: :class:`LayerContinuation` is the
    contract in both directions, so accepting a lookalike would make it silently
    optional to carry.
    """
    if continuation is None:
        return None
    if not isinstance(continuation, LayerContinuation):
        raise TypeError(
            "RoLA.forward's `continuation` is a LayerContinuation -- the container a "
            f"previous forward returned -- got {type(continuation).__name__}. A bare "
            "rola.state() is the OP's half: pass LayerContinuation(state=rola.state()) "
            "for a fresh sequence, and hand back what forward returned for a "
            "continuation. The old (state, conv_rings) tuple is retired "
            "(docs/internals/DELETIONS.md).")
    return continuation.state


class RoLA(nn.Module):
    """Routed linear attention.

    Args:
        routes: the feature-map slot -- any callable returning a
            :class:`~rola.routing.factors.RouteFactors`, typically a
            :class:`~rola.routing.producer.RouteProducer`. It owns the topology (levels,
            widths) and the side gains, because those are facts
            about the map rather than about the layer
            wrapping it, and it owns ``hidden_size``/``num_heads``, which the layer
            reads off it rather than taking a second declaration of.
        d_v: per-head value width -- the width of the state each leaf holds. It sizes
            ``v_proj`` and ``o_proj``, which is why it is the layer's; the default is
            the width the chunk matrix is built at.
        decay: a decay source (:mod:`rola.routing.decay`), or ``None`` for a
            non-decaying recurrence. Validated against the topology at construction.
        layer_idx: index into the recurrent cache.
        expert: :class:`rola.expert.PlanOverrides`. Everything it pins is an axis the
            op decides; it is quarantined because a caller pinning one is not
            describing a model.
    """

    def __init__(
        self,
        routes,
        *,
        d_v: int = 64,
        decay=None,
        layer_idx: int | None = None,
        expert=None,
    ) -> None:
        super().__init__()
        #: THE PRODUCER IS A DUCK, so this is a STRUCTURAL check and not an isinstance
        #: one: the layer reads three facts off the slot and calls it, and anything that
        #: answers those three and returns a well-formed bundle is a feature map. The
        #: check exists so a missing attribute is named here rather than surfacing as an
        #: AttributeError inside a forward.
        missing = [name for name in ("hidden_size", "num_heads", "topology")
                   if not hasattr(routes, name)]
        if missing or not callable(routes):
            raise TypeError(
                "RoLA's `routes` argument is the feature-map slot: a callable producing a "
                "RouteFactors bundle, which also declares hidden_size, num_heads and topology. "
                f"{type(routes).__name__} is missing {missing or ['__call__']}. "
                "rola.RouteProducer is the one this package ships.")
        hidden_size = routes.hidden_size
        num_heads = routes.num_heads
        if type(d_v) is not int or d_v < 1:
            raise ValueError(f"d_v must be a positive exact int, got {d_v!r}")
        if decay is not None:
            validate_decay_source(decay, routes.topology, num_heads)

        overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_v = d_v
        self.routes = routes
        self.decay = decay
        self.layer_idx = layer_idx
        self.expert = overrides

        self.value_dim = num_heads * d_v
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        #: Which kernel the last forward took. A plain attribute, not a buffer:
        #: diagnostic state must not enter a state_dict and must not be cast by
        #: `_apply`. It is PUBLISHED for the measurement guards -- a benchmark process
        #: cannot spy on the call it is timing, and wall-clock cannot tell two arms
        #: apart.
        self.last_execution_backend: str | None = None

    # --- accessors ----------------------------------------------------------

    @property
    def topology(self):
        """The routing topology, which is the feature map's, not the layer's."""
        return self.routes.topology

    @property
    def head_v_dim(self) -> int:
        return self.d_v

    def validate_decode_envelope(self, *operation_inputs) -> None:
        """THE DECODE ARM'S OWN CLAUSE: it is forward-only and has no node.

        `decode_forward` is a bare extension call with no autograd node on it, so a
        grad-bearing continuation step would return a tensor whose graph is severed
        and say nothing -- the layer-dispatch autograd gap, on the one path that
        still has it. Training mode alone is enough: reentrant checkpointing runs its
        first pass under `no_grad` with `training` still True, and a decode step
        taken there must refuse too.
        """
        if self.training:
            raise BackwardNotImplemented(
                "the RoLA DECODE arm is forward-only and this layer is in TRAINING "
                "mode. Single-token continuation has no reverse pass -- the two-pass "
                "backward is the chunk arm's -- so call `.eval()` for inference, or "
                "take the prefill arm.")
        if requires_backward(*[t for t in operation_inputs if t is not None]):
            raise BackwardNotImplemented(
                "a RoLA DECODE step carrying a gradient: the single-token arm is "
                "forward-only and has no autograd node, so this would sever the "
                "graph silently. Run under torch.no_grad(), or freeze the operands.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        continuation: LayerContinuation | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, LayerContinuation]:
        """``(y, LayerContinuation)`` -- ALWAYS the container. CONTINUATION IS LAYERED.

        ``continuation`` is ``None`` (a stateless call) or the container a previous
        forward returned. The op's paged tensor lives on it as ``.state``; a bare
        :class:`~rola.RoLAState` or the retired ``(state, conv_rings)`` tuple is
        refused by name (docs/api.md §1.2).
        """
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "attention_mask must be a [batch, seq_len] 0-1 padding matrix; "
                "[batch, seq_len, seq_len] masks are not supported.")
        for key in kwargs:
            # READ WHAT YOU SUPPORT, RAISE ON THE REST. A swallowed keyword is a knob
            # that silently does nothing, which is how a caller comes to believe a model
            # was configured a way it never was.
            raise TypeError(
                f"RoLA.forward got an unsupported keyword {key!r}. Variable-length packing "
                "(`cu_seqlens`) and attention-probability outputs are not implemented; RoLA "
                "has no attention matrix to return.")

        state = _unpack_continuation(continuation)
        x = hidden_states
        B, L, _ = x.shape
        H = self.num_heads

        #: `v` FLOWS bf16 into the op unconditionally (docs/api.md §4.1).
        v = nn.functional.linear(x.to(torch.bfloat16), self.v_proj.weight.to(torch.bfloat16))
        stateful = state is not None

        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        routes = self.routes(x)
        decay = None if self.decay is None else self.decay()
        #: a state that carries CONTENTS, which is what makes a step a continuation.
        #: A bound-but-empty state is a fresh sequence, and takes the prefill arm.
        carrying = stateful and state.carries

        if L == 1 and carrying:
            #: decode's envelope is its own -- BC-blind, general in the topology -- so
            #: it is never asked the PREFILL matrix's questions.  Its GRADIENT clause
            #: is its own too, and is asked here rather than above so that prefill,
            #: which now trains, is not gated on decode's forward-only-ness.
            self.validate_decode_envelope(*operands_of(routes, v, decay))
            out, state = decode_forward(routes, v, state, decay=decay, expert=self.expert)
            backend = "decode"
        else:
            out, state = rola_op(routes, v, decay, state=state, expert=self.expert)
            backend = "cuda"

        self.last_execution_backend = backend
        out = out.reshape(B, L, H * self.head_v_dim)
        projected = nn.functional.linear(
            out.to(torch.bfloat16), self.o_proj.weight.to(torch.bfloat16)).to(x.dtype)
        return projected, LayerContinuation(state=state)

    # --- state accounting (zoology's Hybrid.state_size reads these) ----------

    def get_stats(self):
        cols = self.head_v_dim + 1  # the mass column is unconditional
        return {
            "d_v": self.head_v_dim,
            "n_heads": self.num_heads,
            "num_chunks": self.topology.N,
            "state_floats": self.num_heads * self.topology.N * cols,
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()["state_floats"]
