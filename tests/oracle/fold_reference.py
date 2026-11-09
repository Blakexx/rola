# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the fp16 fold's DECLARED arithmetic, beginning with its scales.

THE LIST EXPONENTS (the facts pass writes them; docs/internals/facts/facts.md#lists): per window ``w`` and digit ``r``
of the list level, ``K[w, r, col] = ceil(log2(max over t in w of b(t, r) x col(t)))``, floored at -126 -- ``b`` the
token's write coefficient at ``r`` on the byte scale (a sparse level's u8 digit, a dense level's f16 times 255, its
floor stored), ``col`` either ``m(t)``
(the write gain times every write level's scale, column 0) or ``m(t) s(t)`` (times the value scale, column 1). The fold
puts each list's largest coefficient at a fixed power of two by them.

Symbols: ``W`` the window, ``L`` tokens, ``B_l`` level ``l``'s padded width.
"""
from __future__ import annotations

import numpy as np
import torch

from rola.routing import quant

#: the exponents' floor: the smallest normal fp32's
EXPONENT_FLOOR = -126


def list_columns(routes, values) -> torch.Tensor:
    """``[B, T, H, 2]`` fp64: each token's ``m`` and ``m s``."""
    m = routes.scale_write[0].double()
    for scale in routes.scale_write[1:]:
        m = m * scale.double()
    return torch.stack((m, m * values.scale.double()), -1)


def list_maxima(routes, values, level: int, window: int) -> torch.Tensor:
    """``[B, H, windows, B_level, 2]`` fp64: per window and digit of ``level``, the largest ``b x col`` over its
    tokens, ``b`` the write coefficient on the byte scale (a dense level's f16 times 255), zero past the rows' width."""
    rows = write_bytes(routes, level).double()
    B, T, H, width = rows.shape
    cols = list_columns(routes, values)
    windows = -(-T // window)
    pad = windows * window - T
    prod = rows.unsqueeze(-1) * cols.unsqueeze(-2)                                     # [B, T, H, width, 2]
    prod = torch.cat((prod, prod.new_zeros(B, pad, H, width, 2)), 1)
    return prod.reshape(B, windows, window, H, width, 2).amax(2).permute(0, 2, 1, 3, 4)


def _exponents(top: torch.Tensor) -> torch.Tensor:
    return torch.ceil(torch.log2(top.clamp_min(2.0 ** EXPONENT_FLOOR))).to(torch.int64)


def _padded_maxima(routes, values, level: int, window: int, padded_width: int) -> torch.Tensor:
    top = list_maxima(routes, values, level, window)
    return torch.nn.functional.pad(top, (0, 0, 0, padded_width - top.shape[-2]))


def list_exponents(routes, values, level: int, window: int, padded_width: int) -> torch.Tensor:
    """``[B, H, windows, padded_width, 2]`` int: :func:`list_maxima`'s ceil-log2, floored at -126."""
    return _exponents(_padded_maxima(routes, values, level, window, padded_width))


def exponent_bounds(routes, values, level: int, window: int, padded_width: int, rtol: float):
    """``(lo, hi)``: the exponents of the maxima moved by ``rtol`` either way -- a kernel computing the products in
    fp32 lands in between, the exponent changing only where a maximum sits within ``rtol`` of a power of two."""
    top = _padded_maxima(routes, values, level, window, padded_width)
    return _exponents(top * (1 - rtol)), _exponents(top * (1 + rtol))


# --------------------------------------------------------------- the fold (docs/internals/carry/ampere/fold.md)

#: THE UNITS' CONSTANTS, the kernel's: c's top 2^4, a state held under 2^100 of its unit, the lowest unit, a list never
#: live, a window's largest add in units (512 tokens x 16)
COEF_TOP = 4
UNIT_HEADROOM = 100
UNIT_MIN = COEF_TOP - 127
UNIT_NONE = -1000
WINDOW_ADD = np.float32(512 * 16)


def unit_scale(routes, level_s: int) -> float:
    """THE STATE'S ABSOLUTE UNIT: ``c`` carries ``b_R`` on the byte scale and ``x`` level S's coefficient -- a dense
    level's stored f16 itself, a sparse level's byte over 256 -- so the state is ``st x 2^(unit - 4)`` times ``1/255``
    or ``256/255^2``."""
    return (1.0 if is_half(routes, level_s) else 256.0 / quant.AMP_LEVELS) / quant.AMP_LEVELS


def is_half(routes, level: int) -> bool:
    """Whether ``level``'s write rows are f16: a dense level's (docs/internals/facts/liveness_contract.md#floor)."""
    return routes.amp_write[level].dtype == torch.float16


def write_bytes(routes, level: int) -> torch.Tensor:
    """``level``'s write coefficients on the byte scale, fp32: a sparse level's u8, a dense level's f16 times 255
    (exact)."""
    rows = routes.amp_write[level].float()
    return rows * quant.AMP_LEVELS if is_half(routes, level) else rows


def _ceil_exp(x: np.ndarray) -> np.ndarray:
    """The smallest ``e`` with ``x <= 2^e`` (``UNIT_NONE`` for zero), elementwise."""
    mant, e = np.frexp(x.astype(np.float64))
    return np.where(x == 0, UNIT_NONE, np.where(mant == 0.5, e - 1, e)).astype(np.int64)


def next_units(k: np.ndarray, unit: np.ndarray, bound: np.ndarray):
    """``(unit', bound', f)``: a window's units from its exponents ``k``, the state's rescale ``f`` -- the kernel's
    `next_unit` in its fp32 arithmetic."""
    held = unit + _ceil_exp(bound) - UNIT_HEADROOM
    nxt = np.maximum(np.where(k <= EXPONENT_FLOOR, unit, np.maximum(k, held)), UNIT_MIN)
    d = unit - nxt
    f = np.where(d < -126, 0.0, np.exp2(np.clip(d, -126, 127).astype(np.float64))).astype(np.float32)
    return nxt, (bound * f + WINDOW_ADD).astype(np.float32), f


def gains_shift(top: np.ndarray) -> np.ndarray:
    """The facts pass's window shift of a largest gain (docs/internals/facts/facts.md#gains): ``-4 - floor(log2 top)``,
    the largest in [2^-4, 2^-3) after it, clamped to [-126, 127], 0 for a zero largest."""
    _mant, e = np.frexp(top.astype(np.float64))
    return np.where(top == 0, 0, np.clip(-4 - (e - 1), -126, 127)).astype(np.int64)


def dense_exponents(m: torch.Tensor, F: torch.Tensor, window: int, w_r: int) -> np.ndarray:
    """``[B, H, windows, w_r, 2]``: a write side with no list level's exponents (`fold.cuh` `dense_exp`), every list's
    the window's bound ``5 - shift`` of its largest ``(m, F)`` -- a dense byte under 2^8, the factor under 2^(-3 - shift)."""
    B, T, H = F.shape
    windows = -(-T // window)
    #: a partial last window's bound over the tokens it holds (the pad's zeros under every largest)
    mf = torch.nn.functional.pad(torch.stack((m, F), -1), (0, 0, 0, 0, 0, windows * window - T))
    top = mf.reshape(B, windows, window, H, 2).amax(2)  # [B, windows, H, 2]
    e = 5 - gains_shift(top.permute(0, 2, 1, 3).cpu().numpy())
    return np.broadcast_to(e[:, :, :, None, :], (B, H, windows, w_r, 2)).copy()


def fold_state(routes, values, list_exp, level_s: int, window: int, at_window=None):
    """THE FOLD'S DECLARED ARITHMETIC over a call (one batch-head at a time): ``(state, env, mass, units)`` -- the
    absolute state ``[BH, B_S, B_R, C]`` and its envelope (``sum |A| |v|``, absolute), the absolute mass ``[BH, B_S,
    B_R]`` and the final units ``[BH, B_R, 2]`` (values', mass's), from the facts pass's ``list_exp``. ``at_window(w,
    state, env)``, when given, sees each window's absolute state and envelope ``[B, H, B_S, B_R, C]`` before its writes.

    ``F = ((g sw_0) sw_1) vs`` and ``m = (g sw_0) sw_1`` in fp32; ``c = f16(f16(fp32(b_R F) 2^(4 - E)) 2^(E - u))``
    -- the facts pass's record at the list's window exponent ``E`` (floored at ``UNIT_MIN``), moved to the state's unit
    ``u`` by an f16 power of two (zero below 2^-24) -- ``b_R`` on the byte scale (:func:`write_bytes`); ``x`` level S's stored f16 (a dense level) or its byte over 256 (exact); ``A =
    f16(x c)``; the state gains ``sum_t A v`` in its list's unit (exact here), the mass ``sum_t x c_m``; each window
    first rescales both to the units `next_units` gives. A dense level's floor is in its stored rows."""
    level_r = 1 - level_s
    m = routes.scale_write[0].float() * routes.scale_write[1].float()
    F = m * values.scale.float()
    b_r = write_bytes(routes, level_r)
    x = routes.amp_write[level_s] if is_half(routes, level_s) else (routes.amp_write[level_s].float() / 256).half()
    v = values.v.double()
    scale = unit_scale(routes, level_s)
    B, T, H, w_r = b_r.shape
    w_s, C = x.shape[-1], v.shape[-1]
    #: every window the call touches, a partial last one included (its slice ends at the call's end)
    windows = -(-T // window)
    if list_exp.numel():
        k = list_exp.view(B, H, windows, -1, 2)[..., :w_r, :].cpu().numpy().astype(np.int64)
    else:
        k = dense_exponents(m, F, window, w_r)
    dev = b_r.device
    state = torch.zeros(B, H, w_s, w_r, C, dtype=torch.float64, device=dev)
    env, mass = torch.zeros_like(state), torch.zeros(B, H, w_s, w_r, dtype=torch.float64, device=dev)
    u = np.full((B, H, w_r), UNIT_NONE, np.int64)
    v_unit = u.copy()
    bu = np.zeros((B, H, w_r), np.float32)
    bv = bu.copy()
    for w in range(windows):
        sl = slice(w * window, (w + 1) * window)
        if at_window is not None:
            at_window(w, state * torch.tensor(np.exp2(u - COEF_TOP) * scale, device=dev)[:, :, None, :, None], env)
        u, bu, fu = next_units(k[:, :, w, :, 1], u, bu)
        v_unit, bv, fv = next_units(k[:, :, w, :, 0], v_unit, bv)
        e_v = np.maximum(k[:, :, w, :, 1], UNIT_MIN)
        pe = torch.tensor(np.exp2(COEF_TOP - e_v).astype(np.float32), device=dev)       # [B, H, w_r]
        move = torch.tensor(np.where(e_v - u < -24, 0.0, np.exp2(e_v - u)).astype(np.float16), device=dev)
        pv = torch.tensor(np.exp2(COEF_TOP - v_unit).astype(np.float32), device=dev)
        cv = ((b_r[:, sl] * F[:, sl, :, None]) * pe[:, None]).half()                    # [B, Tw, H, w_r]
        cv = (cv.float() * move[:, None].float()).half()
        cm = ((b_r[:, sl] * m[:, sl, :, None]) * pv[:, None]).half()
        a = (x[:, sl, :, :, None].float() * cv[:, :, :, None, :].float()).half().double()  # [B, Tw, H, w_s, w_r]
        unit_abs = torch.tensor(np.exp2(u - COEF_TOP) * scale, device=dev)               # [B, H, w_r]
        state = state * torch.tensor(fu, device=dev).double()[:, :, None, :, None] \
            + torch.einsum("bthds,bthc->bhdsc", a, v[:, sl])
        env += torch.einsum("bthds,bthc->bhdsc", a.abs(), v[:, sl].abs()) * unit_abs[:, :, None, :, None]
        mass = mass * torch.tensor(fv, device=dev).double()[:, :, None, :] \
            + torch.einsum("bthd,bths->bhds", x[:, sl].double(), cm.double())
        del a
    state = state * torch.tensor(np.exp2(u - COEF_TOP) * scale, device=dev)[:, :, None, :, None]
    mass = mass * torch.tensor(np.exp2(v_unit - COEF_TOP) * scale, device=dev)[:, :, None, :]
    units = np.stack((u, v_unit), -1)
    return (state.reshape(B * H, w_s, w_r, C), env.reshape(B * H, w_s, w_r, C), mass.reshape(B * H, w_s, w_r),
            units.reshape(B * H, w_r, 2))

