# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""What the OPERATION consumes, and whether it must be differentiable.

Two facts about one call's operands, read before any decision folds them: the flat
operand list, and the grad bit the call class carries (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from rola.routing.factors import RouteFactors


def operands_of(routes: RouteFactors, v: torch.Tensor, decay: Any = None) -> tuple:
    """Every tensor the OPERATION consumes, flat -- the predicate's argument list."""
    return (v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))


def requires_backward(*tensors: Any) -> bool:
    """Whether a call over these operands must produce a differentiable result.

    Evaluated on the tensors the OPERATION consumes -- ``v``, the routing factors, the
    side gains, the decay dials -- and never on a module's parameters. A
    frozen-parameter forward under ``torch.no_grad()`` has no graph to sever and
    correctly takes the kernel; a dial-only unfrozen finetune has exactly one
    grad-requiring operand and must not.
    """
    if not torch.is_grad_enabled():
        return False
    return any(_any_requires_grad(t) for t in tensors)


def _any_requires_grad(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    if isinstance(obj, (tuple, list)):
        return any(_any_requires_grad(item) for item in obj)
    return False
