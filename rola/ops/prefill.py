# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL COMPOSITION: one segment's kernels over the bundle and the epilogue that joins them.

THE KERNELS ARE ONE OPERATOR on one window grid (the device's card's, which each entry reads): the carry computes every
pair whose write fell in a strictly earlier window than its read -- its ``num`` and, as the v+1 channel, its ``den`` --
and the intra every pair inside one window. Together they are the whole recurrence, no pair counted twice and none
dropped (`tests/oracle/test_carry_vs_oracle.py`'s decomposition test states it in fp64). The scale pass gives the
carry's f16 sum across CTAs each token's scale byte (`rola.ops.scale`) before the carry runs.

THE UNITS. Every kernel leaves each token's READER FACTOR ``r`` (its read levels' scales) out of its sums, and the
readout divides by it (:func:`readout`): ``y = num / (den + eps / r)`` is the defined readout exactly. The intra's
``num`` and ``den`` are already in those units; the carry's take its state's unit and its read rows'
(:func:`carry_factor`), ``num`` times the f16 sum's scale ``2^(s + 1)`` where it is summed as f16 pairs, ``s`` each
token's scale byte -- every CTA's partials meet at it.

Symbols, each at first use: ``W`` the window, ``L`` tokens, ``BH`` batch x heads, ``d_v`` value channels, level ``S``
the carry's (`rola.routing.factors.carry_level_s`) and ``R`` the other.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops import scale as scale_ops
from rola.routing import quant

__all__ = ["UNIT_HEADROOM", "UNIT_LEVELS", "UNIT_NONE", "carry_factor", "prefill", "reader_factor", "readout",
           "unpack_num"]

#: THE f16 SUM's SCALE BYTE, the kernels' `unit_byte` rule (`csrc/rola/src/carry/carry.cuh`): the smallest ``s`` with
#: ``1.02 P <= 2^(s + 15)`` for the token's bound ``P``, held to int8, `UNIT_NONE` where the token reads nothing
UNIT_LEVELS, UNIT_HEADROOM, UNIT_NONE = 15, 1.02, -100


def carry_factor(routes) -> float:
    """What the carry's partials at a token's unit are multiplied by to be the recurrence's numerator over ``r``: the
    state's byte scale (the fold's operand a dense write level's f16, its coefficient on the byte scale: ``1 / 255``,
    `fold.cuh` `kByteScale`) and the read rows' storage -- level S's coefficient a byte over 256 (``256 / 255`` of its
    unit value), level R's row a byte (``1 / 255``)."""
    ls = routes.level_s
    lr = 1 - ls
    kappa = 1.0 / quant.AMP_LEVELS
    k_s = (quant.AMP_LEVELS + 1.0) / quant.AMP_LEVELS if routes.form(ls, "read") is quant.SPARSE else 1.0
    k_r = 1.0 / quant.AMP_LEVELS if routes.form(lr, "read") is quant.SPARSE else 1.0
    return kappa * k_s * k_r


def unpack_num(num: torch.Tensor, factor: torch.Tensor | float = 1.0) -> torch.Tensor:
    """``[BH, L, d_v]`` fp32 in channel order, times ``factor`` (``[BH, L]`` a token's, or one float): the carry's
    ``num`` as the drain wrote it -- fp32, or int32 words of f16 pairs, word ``k`` channels ``2 k`` (its low half) and
    ``2 k + 1`` (`docs/internals/carry/panel.md#drain`) -- read as its f16 halves and scaled in one fp32 multiply."""
    halves = num if num.dtype == torch.float32 else num.view(torch.float16)
    scale = (factor.unsqueeze(-1) if isinstance(factor, torch.Tensor)
             else torch.full((1,), float(factor), device=num.device))
    return halves * scale


def prefill(routes, values, facts, carve, layout, read_statics, write_statics, *, accumulate: str | None = None,
            reduce_dtype: torch.dtype | None = None, intra_accumulate: str | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(num, den)`` fp32 ``[BH, L, d_v]`` and ``[BH, L]`` over ``r`` (:func:`readout` divides by it): the scale pass,
    the carry at its bytes (its ``num`` and its ``den``), and the intra reduced into the same planes. ``facts`` is the
    facts pass's output over this bundle in ``layout`` with the carve's read row (`rola.ops.carry.form`) and these
    statics. ``accumulate`` and ``reduce_dtype`` are reserved: the gates' choice of the MMAs' accumulation and the
    carry's cross-CTA dtype, each the device's or the carve's rule by default; ``intra_accumulate`` names the intra's
    apart from the carry's (``accumulate``'s by default), for a measurement that turns one leg at a time."""
    unit = carry_factor(routes)
    scale = scale_ops.scale(routes, facts, unit)
    num_c, den_c, _, _ = carry_ops.carry(routes, values, facts, carve, scale, accumulate=accumulate,
                                      reduce_dtype=reduce_dtype)
    #: the f16 sum's tokens at ``2^(s + 1)`` times the unit, an fp32 sum's at the unit: one pass with the unpack; the
    #: scale bytes lie on the whole window grid, the call's tokens first
    factor = (torch.exp2(scale[:, :routes.tokens].float() + 1.0) * unit if num_c.dtype != torch.float32
              else unit)
    num = unpack_num(num_c, factor)
    den = (den_c * unit).contiguous()
    intra_ops.intra_forward(routes, values, facts, layout, read_statics, write_statics, num, den,
                            accumulate=intra_accumulate or accumulate)
    return num, den


def reader_factor(routes) -> torch.Tensor:
    """``[BH, L]``: each token's read scales multiplied -- the per-token factor the kernels leave out of ``num`` and
    ``den``."""
    r = routes.scale_read[0]
    for scale in routes.scale_read[1:]:
        r = r * scale
    return r.permute(0, 2, 1).reshape(routes.batch * routes.heads, routes.tokens)


def readout(num: torch.Tensor, den: torch.Tensor, routes) -> torch.Tensor:
    """THE READOUT, ``y = num / (den + eps)`` of the full sums, ``[B, T, H, d_v]`` fp32. The kernels' sums leave out
    each token's reader factor ``r`` (it cancels in the ratio), so the floor divides by it: ``num' / (den' + eps / r)``
    is the defined readout exactly, and a token that reads nothing (``r = 0``) reads zero."""
    from rola.ops.constants import READOUT_EPS

    y = num / (den + READOUT_EPS / reader_factor(routes)).unsqueeze(-1)
    return y.reshape(routes.batch, routes.heads, routes.tokens, -1).permute(0, 2, 1, 3)
