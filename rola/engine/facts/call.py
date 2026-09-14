# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""One call's facts, read off the routing bundle and the value stream, and the
envelope those facts fold into: whether the built arm has an arm for this call.

The rules fold VALUES -- that is what makes them testable with no device present --
so something has to read those values off the objects a caller actually holds. This
is that adapter and nothing else: it finds facts and applies the rules over them, and
it decides nothing of its own (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rola.engine import BackwardNotImplemented, rules
from rola.engine.facts import manifest
from rola.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    import torch

    from rola.engine import Arm
    from rola.routing.factors import RouteFactors


def widths_of(routes) -> tuple[int, ...]:
    """The per-level widths of this bundle's topology, in level order."""
    return tuple(level.width for level in routes.topology.levels)


def arm_of(routes) -> Arm | None:
    """`(C, BC)` -- the built arm this routing runs on, or None if it has none.

    The facade needs `BC` before the launch, because the page arena's extent policy
    is stated in owners; it is the one launch parameter that leaves the plan.
    """
    return rules.arm(widths_of(routes), manifest.chunk_arms())


def envelope_clauses(routes, decay: Any = None, *, d_v: int) -> tuple:
    """The facts :func:`rola.engine.rules.envelope.arm_envelope` folds, for this call.

    `d_v` is the value stream's per-head width, which is a fact about the operand and
    not about the routing -- the caller reads it off `v` and states it here. The arm
    is the ArmRule's own verdict: the envelope refuses the ABSENCE of an arm and
    never re-derives one.
    """
    widths = widths_of(routes)
    return (widths, d_v, decay,
            rules.arm(widths, manifest.chunk_arms()))


def arm_refusal(routes, decay: Any = None, *, d_v: int) -> str | None:
    """Why the chunk arm refuses this call, or None if it runs."""
    return rules.arm_envelope(*envelope_clauses(routes, decay, d_v=d_v))


def envelope_refusal(routes: RouteFactors, v: torch.Tensor, decay: Any = None) -> str | None:
    """Why the built kernel arm has no arm for this call, or ``None`` if it runs.

    THE ONE AUTHORITY on the envelope's CONFIGURATION clauses, so a caller can ask
    before calling and get the sentence the refusal will carry. It is a CALLER of the
    rule layer: the sentences are :mod:`rola.engine.rules.envelope`'s and this function reads
    the facts they fold. The clauses that exist only on the grad path are
    :func:`require_envelope`'s.
    """
    reason = rules.envelope(tuple(routes.topology.unratified_activations), routes.device)
    if reason is not None:
        return reason
    return arm_refusal(routes, decay, d_v=v.shape[-1])


def require_envelope(routes: RouteFactors, v: torch.Tensor, decay: Any = None,
                     state: Any = None) -> None:
    """Refuse anything the built arm cannot run. THE FAST-FAIL, in one place.

    There is no second arm to route to: the fp64 oracle is the executable spec, not a
    fallback, and a caller who wants it calls it (:func:`rola.ops.naive.naive_rola`).

    **THE GRADIENT IS NO LONGER A CLAUSE HERE.** It was, while the operator was
    forward-only; the two-pass backward is built and gated, so a grad-bearing call
    now runs. What remains is the CONFIGURATION envelope, which refuses whether or
    not a gradient is attached, plus one clause that exists only on the grad path:
    a carried STATE. Training in v1 is stateless -- the state plane is
    caller-owned and mutated in place, and `d initial_state` is not in the scans --
    so `state=` and a gradient are mutually exclusive, by name, at the boundary
    rather than silently.
    """
    grad = requires_backward(*operands_of(routes, v, decay))
    stateless = rules.training_envelope(grad, state)
    if stateless is not None:
        raise BackwardNotImplemented(stateless)
    reason = envelope_refusal(routes, v, decay)
    if reason is not None:
        raise NotImplementedError(
            f"rola_op runs the BUILT envelope only and there is no arm for this call: "
            f"{reason}. Nothing else is reachable through this surface "
            "(docs/api.md §1).")
