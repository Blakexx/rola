# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Host-side padding: the ONE translator between any ``b_l``/``d_v`` and a kernel's own
shape: padding happens at the producer, and the value width is padded host-side too.

**TWO CONTRACTS, ONE TRANSLATOR (KERNEL_STANDARDS §R13).** The API accepts any level
width ``b_l >= 1`` and any value width ``d_v``; a kernel accepts only its own
descriptor's shape (a level's width a power of two at or above 16, the value width one
of the shipped ``DV``s) and refuses the rest. This module is that translator, and lives
in ``rola.ops`` because every op's call surface (``rola.ops.carry``, ``rola.ops.decode``,
``rola.ops.intra``) is the seam: the kernel body itself never learns padding exists,
and neither does the routing producer or the layer -- both hand an op exactly the
tensors and widths a caller configured, unpadded.

``pad_routes``/``pad_v`` are ZERO-COLUMN pads, not ``-inf``-logit pads: the descriptor states the
mechanism as padding the LOGITS with ``-inf`` before entmax/softmax, and a zero-padded
SOLVED level is the identical tensor, because both softmax and entmax normalize over,
and threshold against, only the finite entries of a row -- an ``exp(-inf) == 0.0``
addend changes neither a softmax's normalizer nor an entmax's support-selecting sort.
Padding the solved level with an exact ``0.0`` is that same mechanism without ever
materializing a ``-inf`` operand (and so without the ``0 * -inf`` NaN a backward pass
through a literal ``-inf`` logit would risk). No parameter reads a pad column, so
autograd hands it exactly ``0.0`` for its gradient too -- trivially, not by
cancellation, because nothing computed it from an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing.topology import padded_level_width

__all__ = [
    "PaddedShape",
    "pad_routes",
    "pad_v",
    "padded_value_width",
    "slice_y",
]


@dataclass(frozen=True, slots=True)
class PaddedShape:
    """One call's logical shape beside the padded one it is served by.

    The source of truth for "logical vs padded" at an op's call surface: every op
    that pads builds one of these before it touches a tensor, and every field on it
    is a pure function of the caller's own ``widths``/``d_v`` -- never a second input
    that could disagree with them.
    """

    widths: tuple[int, ...]
    padded_widths: tuple[int, ...]
    d_v: int
    DV: int

    @classmethod
    def of(cls, widths: tuple[int, ...], d_v: int, *,
          shipped_dv: tuple[int, ...]) -> PaddedShape:
        return cls(widths=tuple(widths),
                   padded_widths=tuple(padded_level_width(w) for w in widths),
                   d_v=int(d_v), DV=padded_value_width(d_v, shipped=shipped_dv))

    @property
    def is_padded(self) -> bool:
        """Whether this call needed no padding at all -- the dense fast path."""
        return self.widths == self.padded_widths and self.d_v == self.DV


def padded_value_width(d_v: int, *, shipped: tuple[int, ...]) -> int:
    """The smallest of an op's shipped value widths that is ``>= d_v``.

    Raises rather than silently widening past the shipped ceiling: a ``d_v`` past the
    largest shipped width names no kernel this build carries, so this is a refusal by
    construction (KERNEL_STANDARDS §R16: unknown descriptors are admitted by
    refusal-checked derivation, never by enumeration) rather than a fourth width
    nobody built.
    """
    for width in shipped:
        if d_v <= width:
            return width
    raise ValueError(
        f"d_v={d_v} exceeds every shipped value width {shipped} for this op -- there "
        "is no padded width it can serve; a larger value width is a new arm, not a "
        "bigger pad.")


def pad_routes(levels: tuple[torch.Tensor, ...],
               padded_widths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Per level, ``[..., b_l]`` -> ``[..., B_l]``, the trailing columns exact zero.

    A no-op per level where ``b_l`` already equals ``B_l`` -- an already-shipped
    topology pays nothing through this function.
    """
    if len(levels) != len(padded_widths):
        raise ValueError(
            f"pad_routes got {len(levels)} levels and {len(padded_widths)} padded "
            "widths -- one padded width per level is required")
    return tuple(_pad_last_dim(level, width)
                for level, width in zip(levels, padded_widths, strict=True))


def pad_v(v: torch.Tensor, DV: int) -> torch.Tensor:
    """``[..., d_v]`` -> ``[..., DV]``, the trailing ``DV - d_v`` columns exact zero."""
    return _pad_last_dim(v, DV)


def slice_y(y: torch.Tensor, d_v: int) -> torch.Tensor:
    """``[..., DV]`` -> ``[..., d_v]``, a VIEW (the pad tail is simply not read)."""
    return y[..., :d_v]


def _pad_last_dim(t: torch.Tensor, width: int) -> torch.Tensor:
    current = t.shape[-1]
    if current == width:
        return t
    if current > width:
        raise ValueError(
            f"cannot pad a width-{current} tensor down to {width} -- padding only "
            "ever widens")
    return torch.nn.functional.pad(t, (0, width - current))
