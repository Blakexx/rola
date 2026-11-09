# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALE PASS, as a caller reaches it: three launches over a call's windows, from the bundle's rows and the facts
pass's ``(F, m)``, each token's scale byte for the carry's f16 sum across CTAs (docs/internals/scale/scale_api.md).

THE BOUND (ruled 2026-11-02, the read-row form): a token ``t`` of window ``w`` is bounded by ``(sum_r x_R(t, r)) (sum_a
x_S(t, a) G_w(a))``, ``G_w(a) = sum_{t' in windows < w} w_S(t', a) max_r w_R(t', r) F(t')`` -- ``x``, ``w`` the stored
unit rows (a u8 digit over 255, a dense f16 at least the floor), ``F = m s`` the facts pass's, the reader's per-token
row scales left out as every kernel leaves them -- which bounds every channel of the carry's ``num``; over the carry's
factor it is the byte rule's ``P`` (`rola.ops.prefill.scale_bytes`). Symbols: ``B`` batch, ``L`` tokens, ``H_w`` write
heads, ``g`` query heads a write head, ``BH = B H_w``, level ``S`` the carry's (`rola.routing.factors.carry_level_s`,
level 1 where none), ``R`` the other.
"""
from __future__ import annotations

import torch

from rola.ops import facts as facts_ops
from rola.ops._ext import extension

__all__ = ["level_s", "scale", "serves"]


def level_s(routes) -> int:
    """The bound's level S: the carry's, else level 1."""
    return routes.level_s if routes.level_s >= 0 else 1


def serves(routes) -> str | None:
    """Why the scale pass refuses this bundle, or ``None``: depth two."""
    widths = routes.topology.padded_widths
    if len(widths) != 2:
        return f"depth {len(widths)}: the scale pass is built for depth two"
    return None


def scale(routes, facts: facts_ops.Facts, factor: float) -> torch.Tensor:
    """``[BH g, windows W]`` int8: each token's scale byte on the whole window grid (a partial window's tail
    `UNIT_NONE`), the bound over ``factor`` (the carry's partials' units, `rola.ops.prefill.carry_factor`) by the
    carry's byte rule."""
    why = serves(routes)
    if why is not None:
        raise NotImplementedError(f"the scale pass does not serve this bundle: {why}")
    return extension().scale_forward(list(routes.amp_read), list(routes.amp_write), facts.fm,
                                     list(routes.topology.padded_widths), level_s(routes), routes.group, 1.0 / factor)
