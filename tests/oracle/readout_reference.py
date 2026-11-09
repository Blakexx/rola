# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the carry's READOUT (docs/internals/carry/ampere/readout.md): each token's readout, from the fold's
declared state before its window (`fold_reference.fold_state`), and the bound of the value-carve readout's arithmetic.

THE OUTPUT: ``P(t, c) = sum_{d, r} x(t, d) a(t, r) S(d, r, c) / kappa`` over every box -- ``S`` the absolute state
before ``t``'s window (level S's digit ``d``, level R's list ``r``), ``x`` level S's read coefficient as stored (a sparse
level's byte / 256, a dense level's f16), ``a`` level R's read row as stored (f16), ``kappa`` the state's byte scale
(``unit_scale`` at the fold's operand level, :func:`fold_level`). The read side's per-token scales and ``kappa`` are common to ``num`` and ``den`` and cancel in the
epilogue.

THE BOUND, a slot's distance from the kernel (:func:`readout`): the image holds 11 bits a value at its digit's power of
two and its subnormals 2^-25 there; the MMA accumulates in fp32 or (GeForce) fp16; the fold's state is within
``(L / 16) 2^-23`` of its envelope; each (token, digit) piece of a box is rounded once into the panel's dtype and each
add into the panel once more (f16 pairs at the token's scale byte, or fp32); the CTAs' sums are added in the reduce
dtype.

Symbols: ``W`` the window, ``L`` tokens, ``B_l`` level ``l``'s padded width, ``C`` the value channels.
"""
from __future__ import annotations

import torch

from rola.routing import quant
from tests.oracle.fold_reference import fold_state, unit_scale

#: THE IMAGE'S TOP (`parts.cuh` `kImageTop`) and c's (`fold.cuh` `kCoefTop`)
IMAGE_TOP = 10
COEF_TOP = 4
#: a unit holds a final under 2^15 with 2% headroom; a token with nothing to read takes the least unit byte
UNIT_LEVELS = 15
UNIT_HEADROOM = 1.02
UNIT_NONE = -100
#: the MMA's accumulation, relative to the sum of its products' magnitudes
ACCUMULATE = {"fp32": 2.0 ** -18, "fp16": 2.0 ** -9}


def _read_rows(routes, level_s: int):
    """``(x, a)``: level S's coefficients ``[B, T, H, B_S]`` and level R's rows ``[B, T, H, B_R]``, fp64, as stored."""
    level_r = 1 - level_s
    xs = routes.amp_read[level_s]
    x = xs.double() if xs.dtype == torch.float16 else (xs.double() / 256).half().double()
    return x, routes.amp_read[level_r].double()


def fold_level(routes, level_s: int) -> int:
    """The level whose write rows are the fold's operand, a dense level: level S where the sides list different levels,
    level R where the write side's coefficients are level S -- both sides listing it (the same split), or a write side
    listing none against a read side listing it (its coefficients the read side's list level: `rola.engine.carve`'s
    `_sides`) -- its state taken back to the readout's levels; kappa is its byte scale (``unit_scale``)."""
    write, read = routes.sparse[1], routes.sparse[0]
    coefficients_s = write[level_s] or (not any(write) and read[level_s])
    return 1 - level_s if coefficients_s else level_s


def _windows(routes, values, list_exp, level_s: int, window: int):
    """Each window's absolute state over kappa and the fold's envelope, ``[B, H, B_S, B_R, C]`` fp64 each (the
    readout's digit, its list), before its writes."""
    fold_s = fold_level(routes, level_s)
    kappa = unit_scale(routes, fold_s)
    turn = (lambda t: t.transpose(2, 3)) if fold_s != level_s else (lambda t: t)
    out = []
    fold_state(routes, values, list_exp, fold_s, window,
               at_window=lambda w, state, env: out.append((turn(state) / kappa, turn(env) / kappa)))
    return out


def units(routes, values, list_exp, level_s: int, window: int) -> torch.Tensor:
    """``[BH, T]`` int8: each token's unit byte from the bound ``num(t) = sum_{d, r} x a max_c |S| / kappa``,
    ``ceil(log2(1.02 num)) - 15`` (``UNIT_NONE`` where ``num`` is 0)."""
    x, a = _read_rows(routes, level_s)
    B, T, H, _ = x.shape
    num = torch.zeros(B, T, H, dtype=torch.float64, device=x.device)
    for w, (state, _env) in enumerate(_windows(routes, values, list_exp, level_s, window)):
        sl = slice(w * window, (w + 1) * window)
        top = state.abs().amax(-1)                                                          # [B, H, B_S, B_R]
        num[:, sl] = torch.einsum("bthd,bthr,bhdr->bth", x[:, sl], a[:, sl], top)
    u = torch.where(num > 0, torch.ceil(torch.log2(UNIT_HEADROOM * num.clamp_min(1e-300))) - UNIT_LEVELS,
                    torch.full_like(num, UNIT_NONE))
    return u.clamp(-128, 127).to(torch.int8).permute(0, 2, 1).reshape(B * H, T).contiguous()


def _ceil_exp(x: torch.Tensor) -> torch.Tensor:
    """The smallest ``e`` with ``x <= 2^e`` (``x > 0``), as the kernel takes it from a float's bits."""
    m, e = torch.frexp(x)
    return torch.where(m == 0.5, e - 1, e)


def readout(routes, values, list_exp, level_s: int, window: int, scale: torch.Tensor, accumulate: str,
            reduce_dtype: torch.dtype, *, box: tuple[int, int]):
    """``(P, bound)``: ``[BH, T, C]`` fp64 each slot's readout in the kernel's frame (the state over kappa) and its bound,
    for an MMA accumulating in ``accumulate`` (``"fp32"`` or ``"fp16"``) and the panel and cross-CTA sum in
    ``reduce_dtype``. THE DECLARED ARITHMETIC: a CTA a box of ``box`` = (level-S digits, level-R lists) -- the carve's
    extents at the two levels -- x its channels; its
    image a digit at the power of two over its largest |state| in the box (rounded to 11 bits, subnormals at 2^-25
    there); a (token, digit)'s piece the MMA's sum over the box's lists times the coefficient, rounded once into the
    panel's dtype -- f16 at the token's scale byte ``2^(scale + 1)``, or fp32 -- and added into the token's panel word,
    a rounding an add; the CTAs' panels added into ``num`` in the reduce dtype, each add half an ulp of the slot's
    magnitude ``(n + 1) 2^-11`` for ``n`` CTAs (f16) or ``2^-24`` (fp32), and the f16 subnormal floor a piece, an add and
    a CTA. A DENSE READ SIDE (readout_dense.md): each (token, digit)'s coefficient rewritten at the token's exponent
    ``e_t`` over the box's digits and its rows scaled by it, both rounded to f16 (``2^-11`` each, their subnormal floors
    ``2^(e_t - e_d - 25)`` of the state there), one fp32 accumulator a box, its exit at most two panel roundings (an
    image half's store and the other's add)."""

    x, a = _read_rows(routes, level_s)
    B, T, H, B_S = x.shape
    B_R = a.shape[-1]
    L = T
    C = values.v.shape[-1]
    ES, ER = box
    KS, KR = B_S // ES, B_R // ER
    P = torch.zeros(B, T, H, C, dtype=torch.float64, device=x.device)
    bound = torch.zeros_like(P)
    s_bytes = scale.double().reshape(B, H, T).permute(0, 2, 1)                                 # [B, T, H]
    f16 = reduce_dtype == torch.float16
    eps = 2.0 ** -11 if f16 else 2.0 ** -24
    dense = not any(routes.sparse[0])
    acc = ACCUMULATE["fp32" if dense else accumulate]

    for w, (state, env) in enumerate(_windows(routes, values, list_exp, level_s, window)):
        sl = slice(w * window, (w + 1) * window)
        xs, as_ = x[:, sl], a[:, sl]
        t = xs.shape[1]
        P[:, sl] = torch.einsum("bthd,bthr,bhdrc->bthc", xs, as_, state)
        mag = torch.einsum("bthd,bthr,bhdrc->bthc", xs.abs(), as_.abs(), state.abs())
        fold = torch.einsum("bthd,bthr,bhdrc->bthc", xs.abs(), as_.abs(), env) * (L / 16 * 2.0 ** -23)
        #: the image's subnormal floor at each digit's power of two: its largest |S| over every list and channel, doubled
        top = state.abs().amax(dim=(-1, -2))                                                 # [B, H, B_S]
        floor = torch.einsum("bthd,bthr,bhd->bth", xs.abs(), as_.abs(), 2 * top) * 2.0 ** (-25 - IMAGE_TOP)
        #: THE PANEL: each box's pieces and adds, one rounding each a digit the token reads there
        xb, ab = xs.reshape(B, t, H, KS, ES), as_.reshape(B, t, H, KR, ER)
        Sb = state.reshape(B, H, KS, ES, KR, ER, C)
        mag_box = torch.einsum("bthsd,bthkr,bhsdkrc->bthskc", xb.abs(), ab.abs(), Sb.abs())
        adds = (xb != 0).double().sum(-1)[:, :, :, :, None] * (ab != 0).any(-1).double()[:, :, :, None, :]
        scaling = torch.zeros_like(mag)
        if dense:
            #: one exit a box (at most two panel roundings); OPERAND SCALING: c and the scaled rows at f16 and their floors
            adds = (adds > 0).double()
            top_b = Sb.abs().amax(dim=(5, 6))                                                  # [B, H, KS, ES, KR]
            ed = torch.where(top_b > 0, _ceil_exp(top_b.clamp_min(1e-300)).double(), -torch.inf)
            ex = torch.where(xb != 0, _ceil_exp(xb.abs().clamp_min(1e-300)).double(), -torch.inf)  # [B, t, H, KS, ES]
            et = (ed[:, None] + ex[..., None]).amax(dim=4)                                     # [B, t, H, KS, KR]
            sc = torch.exp2(et[:, :, :, :, None, :] - ed[:, None] - 25)                        # [B, t, H, KS, ES, KR]
            sc = torch.where(torch.isfinite(sc), sc, 0.0)
            scaling = (2 * 2.0 ** -11 * mag
                       + torch.einsum("bthsdk,bthkr,bhsdkrc->bthc", sc, ab.abs(), Sb.abs())
                       + torch.einsum("bthsdk,bthkr,bhsdkrc->bthc", sc, (ab != 0).double(), Sb.abs()))
        panel = (eps * (1 + adds)[..., None] * mag_box).sum(dim=(3, 4))                       # [B, t, H, C]
        n = (adds > 0).double().sum(dim=(3, 4))                                               # [B, t, H]
        frame = torch.exp2(s_bytes[:, sl] + 1)
        subnormal = (((1 + adds) * (adds > 0)).sum(dim=(3, 4)) + n) * 2.0 ** -24 * frame if f16 else torch.zeros_like(n)
        bound[:, sl] = ((2.0 ** -11 + acc) * mag + fold + floor[..., None] + panel + scaling
                        + ((n + 1) * eps)[..., None] * mag + subnormal[..., None])

    return P.permute(0, 2, 1, 3).reshape(B * H, T, C), bound.permute(0, 2, 1, 3).reshape(B * H, T, C)


def num_absolute(num: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``[BH, T, C]`` fp64: the carry's ``num`` in the kernel's frame -- fp32 as written, f16 pairs times ``2^(scale +
    1)`` (``scale`` ``[BH, T]`` the den pass's bytes)."""
    got = unpack_num(num).double()
    return got if num.dtype == torch.float32 else got * torch.exp2(scale[:, :got.shape[1]].double() + 1)[..., None]


def unpack_num(num: torch.Tensor) -> torch.Tensor:
    """``[BH, T, C]``: the carry's ``num`` per channel -- fp32 ``[BH, T, C]`` as the drain wrote it (reduce fp32), or
    int32 ``[BH, T, C / 2]`` words of f16 pairs (reduce f16), word ``k`` channels ``2 k`` and ``2 k + 1``, as fp32."""
    if num.dtype == torch.float32:
        return num
    lo = (num & 0xFFFF).to(torch.int16).view(torch.float16).float()
    hi = ((num >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    return torch.stack([lo, hi], -1).flatten(-2)


def truth_scale(routes, unit: torch.Tensor, level_s: int) -> torch.Tensor:
    """``[B, T, H]`` fp64: what a token's partials at its unit are multiplied by to be the recurrence's numerator, the
    per-token factor the epilogue cancels against ``den``: ``2^u(t)``, ``kappa`` and each read level's row scale, a u8
    level's stored value scaled to its digit over 255 (level S's coefficient is the byte over 256, level R's row the
    byte), an f16 level's the value itself -- a read side with no list level's two levels both (level S its
    coefficient level, level R its row level: `carry_level_s`)."""
    if level_s < 0:
        raise NotImplementedError("the carry's level S is defined at depth two (`carry_level_s`)")
    level_r = 1 - level_s
    stored = {level_s: quant.AMP_LEVELS + 1, level_r: 1.0}
    factor = unit_scale(routes, fold_level(routes, level_s))
    for level in (level_s, level_r):
        rows = routes.amp_read[level]
        k = 1.0 if rows.dtype == torch.float16 else stored[level] / quant.AMP_LEVELS
        factor = factor * k * routes.scale_read[level].double()
    B, T, H = factor.shape
    return factor * torch.exp2(unit[:, :T].double().reshape(B, H, T).permute(0, 2, 1))


def num_in_truth(num: torch.Tensor, routes, unit: torch.Tensor) -> torch.Tensor:
    """``[B, T, H, C]`` fp64: the carry's ``num`` (:func:`unpack_num`) times :func:`truth_scale` -- the inter term's
    numerator, comparable with `tests.oracle.reference.inter_reference`'s. ``unit`` is the f16 sum's scale bytes (the
    den pass's): f16 pairs hold ``num / 2^(scale + 1)`` (a bit of range, `parts.cuh` `drain`); an fp32 ``num`` carries
    each CTA's unit already, so only the per-token factor remains."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    got = unpack_num(num[:, :T]).double().reshape(B, H, T, -1).permute(0, 2, 1, 3)
    unit = torch.zeros_like(unit) if num.dtype == torch.float32 else unit + 1
    return got * truth_scale(routes, unit, routes.level_s)[..., None]
