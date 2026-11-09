# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A parameter's copy in the dtype and layout a step with no gradient reads, made once and reused while the parameter
is unchanged (docs/api.md section 2.3, projection-weight).

Casting a weight on every call is a whole pass over it, and at one token a step that pass costs more than the work
it feeds: the routing projection's cast of `route_W` measured ~55 us a decode step at `H = 16`, the value and output
projections' casts ~12 us more. The copy is keyed on the parameter's version counter, storage, device and the dtype
asked for, so an in-place update (an optimizer step, `load_state_dict`) or a move rebuilds it. A step a gradient flows
through never reads it: its cast is part of the graph.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class InferenceCopy:
    """One parameter's no-grad copy: ``copy(W, dtype)`` is ``lay(W.to(dtype))`` (``lay=None``: as stored), rebuilt only
    when ``W`` moved. Every copy states its layout."""

    def __init__(self, *, lay: Callable[[torch.Tensor], torch.Tensor] | None) -> None:
        self._lay = lay
        self._entry: tuple[tuple, torch.Tensor] | None = None

    def __call__(self, W: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        key = (W._version, W.data_ptr(), W.device, dtype)
        if self._entry is None or self._entry[0] != key:
            copy = W.detach().to(dtype)
            self._entry = (key, copy if self._lay is None else self._lay(copy))
        return self._entry[1]


def reads_copy(W: torch.Tensor) -> bool:
    """A step reads the copy exactly when no gradient can flow to ``W``."""
    return not (torch.is_grad_enabled() and W.requires_grad)
