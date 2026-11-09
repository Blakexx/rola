# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALE PASS'S REFERENCE: the write-aligned bound in fp64 from the bundle's own rows, and the byte rule.

``bound(t) = (sum_r x_R(t, r)) (sum_a x_S(t, a) G_w(a))``, ``G_w(a) = sum_{t' in windows < w} w_S(t', a) max_r w_R(t',
r) F(t')``, ``x`` and ``w`` each level's unit rows (their form's `dequantize` at a unit scale: a u8 digit over
255, a dense f16 at least the floor), ``F = prod_l scale_write_l s`` -- in the recurrence's units over the reader's
per-token factor ``r``. It bounds the inter numerator over ``r`` on every channel; over the carry's factor it is the
byte rule's ``P``. Symbols: ``W`` the window, ``L`` tokens, ``g`` query heads a write head.
"""
from __future__ import annotations

import torch

from rola.routing.factors import level_form

__all__ = ["UNIT_HEADROOM", "UNIT_LEVELS", "bound", "byte_holds"]

#: the byte rule's headroom and levels (`csrc/rola/src/carry/carry.cuh`'s `unit_byte`), stated here as the definition
UNIT_HEADROOM, UNIT_LEVELS = 1.02, 15


def _units(rows, sparse: bool) -> torch.Tensor:
    return level_form(sparse).dequantize(rows, torch.ones(rows.shape[:-1], device=rows.device)).double()


def bound(routes, values, window: int, level_s: int) -> torch.Tensor:
    """``[BH g, L]`` fp64: the write-aligned bound of every token, window by window."""
    ls, lr = level_s, 1 - level_s
    xs, xr = (_units(routes.amp_read[lv], routes.sparse[0][lv]) for lv in (ls, lr))     # [B, T, H g, w]
    ws, wr = (_units(routes.amp_write[lv], routes.sparse[1][lv]) for lv in (ls, lr))   # [B, T, H, w]
    F = values.scale.double()
    for scale in routes.scale_write:
        F = F * scale.double()
    c = wr.amax(-1) * F                                                                # [B, T, H]
    B, T, Hq, _ = xs.shape
    g = Hq // ws.shape[2]
    G = torch.zeros(B, ws.shape[2], ws.shape[-1], dtype=torch.float64, device=xs.device)
    out = torch.zeros(B, T, Hq, dtype=torch.float64, device=xs.device)
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        out[:, sl] = xr[:, sl].sum(-1) * torch.einsum("btha,bha->bth", xs[:, sl], G.repeat_interleave(g, 1))
        G += torch.einsum("btha,bth->bha", ws[:, sl], c[:, sl])
    return out.permute(0, 2, 1).reshape(B * Hq, T)


def byte_holds(byte: torch.Tensor, p: torch.Tensor, rel: float = 1e-5) -> torch.Tensor:
    """Bool, a token's: ``byte`` is the rule's for ``p`` within ``rel`` of fp32's sums -- none exactly where ``p`` is
    zero, else ``1.02 p <= 2^(byte + 15)`` and the byte below it fails, held to int8."""
    b = byte.double()
    q = UNIT_HEADROOM * p
    covers = q * (1 - rel) <= torch.exp2(b + UNIT_LEVELS)
    least = (b == -128) | (torch.exp2(b + UNIT_LEVELS - 1) < q * (1 + rel))
    clamped = (b == 127) & (q > torch.exp2(b + UNIT_LEVELS))
    live = (covers | clamped) & least
    return torch.where(p > 0, live & (byte != -100), byte == -100)
