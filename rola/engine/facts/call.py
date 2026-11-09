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
from rola.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    from rola.routing.factors import RouteFactors
    from rola.values import Values


def widths_of(routes) -> tuple[int, ...]:
    """The per-level widths of this bundle's topology, in level order."""
    return tuple(level.width for level in routes.topology.levels)


def prefill_clauses(routes, decay: Any = None, *, d_v: int, stateful: bool = False) -> tuple:
    """The facts :func:`rola.engine.rules.envelope.prefill_envelope` folds, for this segment: the call's shape,
    whether the intra reads its widths, the carve's arm where the binary's own census carries it, the scale pass's
    refusal and whether a state rides the call."""
    from rola.engine.carve import plan_carve
    from rola.ops import carry as carry_ops
    from rola.ops import intra as intra_ops
    from rola.ops import scale as scale_ops

    widths = widths_of(routes)
    key = None
    if len(widths) == 2:
        try:
            key = carry_ops.arm_key(plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, d_v))
        except ValueError:
            key = None
        key = key if key is not None and tuple(key) in {tuple(arm) for arm in carry_ops.arms()} else None
    return (widths, d_v, decay, intra_ops.admits(widths, intra_ops.VALUE_WIDTH),
            intra_ops.VALUE_WIDTH, key, scale_ops.serves(routes), stateful)


def arm_refusal(routes, decay: Any = None, *, d_v: int, stateful: bool = False) -> str | None:
    """Why the prefill kernels refuse this segment, or None if they run."""
    return rules.prefill_envelope(*prefill_clauses(routes, decay, d_v=d_v, stateful=stateful))


def envelope_refusal(routes: RouteFactors, values: Values, decay: Any = None, *, stateful: bool = False) -> str | None:
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
    return arm_refusal(routes, decay, d_v=values.d_v, stateful=stateful)


def require_envelope(routes: RouteFactors, values: Values, decay: Any = None,
                     state: Any = None) -> None:
    """Refuse anything the built kernels cannot run. THE FAST-FAIL, in one place.

    There is no second arm to route to: the fp64 oracle is the executable spec, not a
    fallback, and a caller who wants it calls it (:func:`rola.ops.naive.naive_rola`).

    A GRADIENT IS REFUSED, loudly: the prefill has no reverse pass until the native
    backward is rebuilt, and a forward reached from a grad context must raise rather
    than sever the graph silently (the layer-dispatch autograd gap).
    """
    if requires_backward(*operands_of(routes, values, decay)):
        raise BackwardNotImplemented(
            "a RoLA PREFILL carrying a gradient: the prefill has no reverse pass until the native backward is "
            "rebuilt, and running it would sever the graph silently. Run under torch.no_grad(), freeze the "
            "operands, or train through the fp64 oracle (rola.ops.naive.naive_rola).")
    reason = envelope_refusal(routes, values, decay, stateful=state is not None)
    if reason is not None:
        raise NotImplementedError(
            f"rola_op runs the BUILT envelope only and there is no kernel for this call: "
            f"{reason}. Nothing else is reachable through this surface "
            "(docs/api.md §1).")
