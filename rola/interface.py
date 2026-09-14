# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op — and, until the kernel lands, its refusal.

    rola_op(routes, v, decay)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~rola.routing.factors.RouteFactors` bundle -- ``v`` is the value stream, and
``decay`` is the recurrence dial. No plan, no configuration, no statistics appear on
it, because none of them are model facts: they are how the op runs, and the op decides
how it runs.

**THE PREFILL EXECUTION ARM IS DELETED** (the pre-build deletion batch, ruling
2026-08-19; baseline = tag ``baseline/pre-k31``). The shipped chunk consumer and
its two-scan backward were removed ahead of the box-native rebuild, so this
op currently REFUSES every call, loudly, naming the ruling. What survives of the
family is its fact layer — the union-table pass, the atom bitmap and the
liveness epilogue (:mod:`rola.engine.facts`) — the decode arm
(:mod:`rola.engine.dags.decode_dag`), and the fp64 oracle
(:mod:`rola.ops.naive`), which remains the executable spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rola._state import RoLAState
from rola.engine import BackwardNotImplemented
from rola.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    from rola.routing.factors import RouteFactors
    from rola.routing.types import DecayConfig

__all__ = [
    "BackwardNotImplemented",
    "operands_of",
    "requires_backward",
    "rola_op",
]


def rola_op(
    routes: RouteFactors,
    v: torch.Tensor,
    decay: DecayConfig = None,
    *,
    state: RoLAState | None = None,
    expert: Any = None,
) -> tuple[torch.Tensor, RoLAState | None]:
    """REFUSED: the prefill execution arm is deleted pending the rebuild.

    The signature is the contract the rebuild re-implements — ``(y, state)`` over one
    routing bundle and one value stream — and it is kept so callers break HERE,
    by name, rather than on an import. The fp64 oracle
    (:func:`rola.ops.naive.naive_rola`) is the executable spec and still runs
    every configuration; the decode arm still continues a carried state at
    ``T = 1`` (:class:`rola.RoLA`'s dispatch).
    """
    raise NotImplementedError(
        "the prefill arm is deleted; see docs/internals/DELETIONS.md. The shipped "
        "chunk consumer and its backward were removed from the master line "
        "(baseline = tag baseline/pre-k31); the box-native kernel replaces "
        "them. Run the fp64 oracle (rola.ops.naive.naive_rola) for semantics, "
        "or check out the baseline tag for the retired kernel.")
