# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTRA, as a caller reaches it: every pair inside one window, over the bundle, reduced into the output planes.
No intra kernel is built on this line: :func:`intra_forward` raises :class:`rola.ops._ext.KernelNotBuilt`. What stands
is the shape law the envelope reads (:func:`reads`, :func:`admits`).
"""
from __future__ import annotations

import torch

from rola.engine.facts.liveness import LivenessLayout, SideStatics
from rola.ops._ext import not_built

#: The value width the kernel reads.
VALUE_WIDTH = 64
#: THE ROWS' LEVELS: the built arms serve depth <= 2, each level a whole number of 16-digit granules up to 256.
MAX_LEVELS = 2
WIDTH_GRANULE = 16
MAX_LEVEL_WIDTH = 256

__all__ = ["MAX_LEVELS", "MAX_LEVEL_WIDTH", "VALUE_WIDTH", "WIDTH_GRANULE", "admits", "intra_forward", "reads"]


def intra_forward(routes, values, facts, layout: LivenessLayout, read_statics: SideStatics,
                  write_statics: SideStatics, num: torch.Tensor, den: torch.Tensor, **kwargs) -> None:
    """REDUCE the within-window pairs of the bundle into ``num`` and ``den``: not built on this line
    (:func:`rola.ops._ext.not_built`)."""
    not_built("intra")


def reads(widths) -> bool:
    """Whether the kernel reads levels of these widths: one or two, each a whole number of `WIDTH_GRANULE`-digit
    granules up to `MAX_LEVEL_WIDTH` (no device asked)."""
    widths = tuple(int(width) for width in widths)
    return 1 <= len(widths) <= MAX_LEVELS and all(
        width % WIDTH_GRANULE == 0 and WIDTH_GRANULE <= width <= MAX_LEVEL_WIDTH for width in widths)


def admits(widths, d_v: int) -> bool:
    """Whether the intra runs a call at these level widths and value width: widths it :func:`reads` and ``d_v`` the
    `VALUE_WIDTH`. Every width it reads fits every card's shared memory in every form: each build's walk is sized for the
    widest rows (`Walk<B>`, asserted at compile time; docs/internals/intra/intra.md#block)."""
    return d_v == VALUE_WIDTH and reads(widths)
