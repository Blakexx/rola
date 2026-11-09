# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the int8 carry's DECLARED arithmetic, and the quantization-error report.

THE OP ORACLE MEASURES THAT THE MATH IS IMPLEMENTED; THE LAYER ORACLE MEASURES QUALITY (ruled
2026-09-26). This module is the first half's reference for the int8 form: the same windowed
recurrence as :func:`tests.oracle.reference.inter_reference`, run in fp64 on the OPERANDS THE
KERNEL DECLARES. The FOLD: per-token unit-max f16 values (the producer's), and per leaf an A row in u8 -- the token's dense write amplitude times its sparse
amplitude and scale (a producer's gain in them), against the leaf's biggest writer that window -- into an s32-exact
state. The READOUT (ruled 2026-09-27): the state is read per BLOCK, one read-sparse digit times
32 read-dense digits, quantized against the block's max |S| at 127 levels, or at 32767 when the
block's spread (max over smallest live leaf max) exceeds ``R = round(254 * readout_eps)``; A is
the token's unit-max dense read row in u8; the per-(token, block) factor multiplies the block's
partial, which is then rounded to the token's PANEL unit ``scale_t / 2^14`` (the packed s16
panel; ``scale_t`` is the pre-pass's bound readout) before the blocks are summed. Every product
of two quantized operands is exact in fp64, so the panel step is the only rounding after the
operands. ``exact=True`` skips every rounding and must reproduce ``inter_reference`` exactly:
that is the test that the recurrence's STRUCTURE is right before any rounding is judged.

The REPORT (:func:`quantization_report`) is comparison B: the declared arithmetic against the
unquantized op on the oracle cells, per slot against each slot's envelope and per token in norm,
reported and never gated (the doctrine's truth-oracle half), with the fraction of (block, window)
pairs that took the second digit. The Hadamard rotation is here so the report can show the same
error in the rotated and the real basis.

Symbols, each at first use: ``D`` levels, ``B_l`` widths, ``N = prod B_l`` leaves, ``dv`` value
channels, ``W`` the window, ``s_t`` a token's value scale (its max |channel|), ``a`` an amplitude,
``c`` a leaf coefficient (the product of its digits' amplitudes).
"""
from __future__ import annotations

import math

import torch

from rola.routing import quant
from tests.oracle.fixtures import with_gain
from tests.oracle.reference import inter_reference, leaf_product, radix_strides

TINY = 1e-300
#: The IMMA's K: a read block is one read-sparse digit x 32 consecutive read-dense digits of the
#: owner set (docs/internals/common/geom.md#tall-owner-set).
BLOCK_LEAVES = 32
#: The packed s16 panel: a token's unit is ``scale_t (1 + R/254) / PANEL_LEVELS`` -- the pre-pass's bound
#: times the most a one-digit block at the declared spread can exceed it, over the s16 half's range.
PANEL_LEVELS = 32767.0
HI_LEVELS, TWO_DIGIT_LEVELS = 127.0, 32767.0


def hadamard(n: int, device, dtype=torch.float64) -> torch.Tensor:
    """The normalized Sylvester Hadamard matrix of order ``n`` (a power of two): orthogonal, entries ±1/sqrt(n)."""
    assert n & (n - 1) == 0 and n >= 1
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(n)).to(device)


def rotate(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Channels are the last axis; the layer folds ``h`` into W_V and ``h^T`` into W_O."""
    return x @ h


def unrotate(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return y @ h.T


def quantize_u8_unit(p: torch.Tensor) -> torch.Tensor:
    """An amplitude in [0, 1] as u8: ``round(255 p) / 255``."""
    return torch.round(p * 255.0) / 255.0


def unit_max(v: torch.Tensor):
    """``(v_hat, s_t)``: the per-token unit-max form, ``|v_hat| <= 1`` with the largest channel at ±1."""
    s = v.abs().amax(-1, keepdim=True).clamp_min(TINY)
    return v / s, s


def group_index(widths, levels) -> torch.Tensor:
    """For every canonical leaf, its mixed-radix index over the digits of ``levels`` (in level order); zeros when
    ``levels`` is empty."""
    N = math.prod(widths)
    strides = radix_strides(widths)
    idx = torch.zeros(N, dtype=torch.long)
    ell = torch.arange(N)
    for lvl in levels:
        idx = idx * widths[lvl] + (ell // strides[lvl]) % widths[lvl]
    return idx


def block_order(widths, read_sparse, read_dense):
    """``(order, leaves)``: the permutation that lists the canonical leaves block by block, and a block's leaf count.
    A block is one read-sparse digit tuple and ``BLOCK_LEAVES`` consecutive read-dense digit tuples; a read-dense group
    narrower than that (a small cell, its K zero-padded in the kernel) is one block."""
    idx_s, idx_d = group_index(widths, read_sparse), group_index(widths, read_dense)
    n_d = int(idx_d.max().item()) + 1
    leaves = min(BLOCK_LEAVES, n_d)
    assert n_d % leaves == 0, "a read-dense group must tile into blocks"
    return torch.argsort(idx_s * n_d + idx_d, stable=True), leaves


def group_product(levels_all, widths, levels, sl, u8=None):
    """``(product, row_scale)``: the chosen levels' amplitudes multiplied over their digit tuples, ``[B, t, H, prod
    widths[levels]]``, and the per-token scalar factored out of it, ``[B, t, H, 1]``. With ``u8`` each level's row is
    first put in UNIT-MAX form (its max digit at 1) and quantized by ``u8``; the row maxima's product is the scalar --
    a per-token factor, which the fold migrates onto the values and the readout onto the output row. Without ``u8`` the
    scalar is one. Ones when ``levels`` is empty."""
    x, scale = None, None
    for lvl in levels:
        a = levels_all[lvl][:, sl].double()
        if u8 is not None:
            m = a.amax(-1, keepdim=True).clamp_min(TINY)
            a = u8(a / m)
            scale = m if scale is None else scale * m
        x = a if x is None else (x.unsqueeze(-1) * a.unsqueeze(-2)).flatten(-2)
    b, t, h = levels_all[0][:, sl].shape[:3]
    ones = torch.ones(b, t, h, 1, device=levels_all[0].device, dtype=torch.float64)
    return (ones if x is None else x), (ones if scale is None else scale)


def int_reference(read_levels, write_levels, v, widths, sparsity, window, *,
                  readout_eps: float = 0.0, exact: bool = False, state_in=None,
                  panel: bool = True, telemetry: dict | None = None):
    """``(num, den, state)`` in fp64 under the DECLARED int arithmetic; ``sparsity[l] = (read sparse, write sparse)``.

    Fold, per write-sparse digit ``s`` and window: ``A[leaf, t] = a_dense[t, d] x a_sparse[t, s] x s_t x g_t`` in
    u8 against ``sigma_leaf,w`` (the leaf's largest writer that window), ``state[leaf] += sum_t A x v_hat[t]`` with
    ``v_hat`` the token's unit-max f16 value; the bound ``B[leaf] += sum_t A``
    beside it (the pre-pass's recurrence). Readout, per block: ``sigma_b = max |S|`` over the block, the block's
    spread ``sigma_b / min live leaf max``; ``S_op = q(S / sigma_b) sigma_b`` at 127 levels, or 32767 when the spread
    exceeds ``R = round(254 readout_eps)`` (``readout_eps`` in [0, 1]: 0 is two digits everywhere, 1 one digit
    wherever the spread is under 254); ``A`` the unit-max dense read row in u8; the (token, block) partial times
    ``a_sparse x m_t`` rounded to the token's panel unit ``scale_t (1 + R/254) / 32767`` with ``scale_t = sum_leaf r
    x B`` (the pre-pass's bound readout; ``panel=False`` skips this rounding, the s32 panel); ``den`` from the exact
    amplitudes and the mass. ``exact`` turns every rounding into the identity. ``telemetry`` receives ``blocks``,
    ``flagged`` (block-window pairs and how many took two digits) and ``panel_peak`` (the largest |partial sum| in
    panel units, which the packed panel needs below 32767).
    """
    Bn, T, H, dv = v.shape
    D = len(widths)
    N = math.prod(widths)
    dev = v.device
    r_sp = [lvl for lvl in range(D) if sparsity[lvl][0]]
    r_dn = [lvl for lvl in range(D) if not sparsity[lvl][0]]
    w_sp = [lvl for lvl in range(D) if sparsity[lvl][1]]
    w_dn = [lvl for lvl in range(D) if not sparsity[lvl][1]]
    idx_rs, idx_rd = group_index(widths, r_sp).to(dev), group_index(widths, r_dn).to(dev)
    idx_ws, idx_wd = group_index(widths, w_sp).to(dev), group_index(widths, w_dn).to(dev)
    n_ws = int(idx_ws.max().item()) + 1
    order, blk = block_order(widths, r_sp, r_dn)
    order = order.to(dev)
    n_blocks = N // blk
    if not 0.0 <= readout_eps <= 1.0:
        raise ValueError("readout_eps is a relative step in [0, 1]")
    threshold = float(round(254.0 * readout_eps))
    u8 = None if exact else quantize_u8_unit
    q_fold = (lambda x: x) if exact else (lambda x: x.to(torch.float16).double())

    v_hat, s_t = unit_max(v.double())                    # [B,T,H,dv], [B,T,H,1]
    v_hat = q_fold(v_hat)                                 # the producer's f16 V-hat
    state = (torch.zeros(Bn, H, N, dv, device=dev, dtype=torch.float64) if state_in is None
             else state_in.double()[..., :dv].clone())
    mass = (torch.zeros(Bn, H, N, device=dev, dtype=torch.float64) if state_in is None
            else state_in.double()[..., dv].clone())
    bound = state.abs().amax(-1) if state_in is not None else torch.zeros(Bn, H, N, device=dev, dtype=torch.float64)
    num = torch.zeros(Bn, T, H, dv, device=dev, dtype=torch.float64)
    den = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)
    blocks_seen = flagged = 0
    panel_peak = 0.0
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        # --- readout against the state at the window start, block by block ------------------------------------
        a_rd, m_rd = group_product(read_levels, widths, r_dn, sl, u8)     # [B,t,H,n_rd] unit-max u8 rows, row max
        a_rs, _ = group_product(read_levels, widths, r_sp, sl)             # [B,t,H,n_rs] the sparse amps, fp
        r_q = (a_rd[..., idx_rd] * (a_rs * m_rd)[..., idx_rs])[..., order]  # [B,t,H,N] block order
        r_q = r_q.reshape(Bn, sl.stop - sl.start, H, n_blocks, blk)
        s_blk = state[:, :, order].reshape(Bn, H, n_blocks, blk, dv)
        leaf_max = s_blk.abs().amax(-1)                                    # [B,H,blocks,32]
        sigma_b = leaf_max.amax(-1, keepdim=True)                          # [B,H,blocks,1]
        live = leaf_max > 0
        min_live = torch.where(live, leaf_max, torch.inf).amin(-1, keepdim=True)
        two = (sigma_b > threshold * min_live) & (live.sum(-1, keepdim=True) > 0)
        levels = torch.where(two, TWO_DIGIT_LEVELS, HI_LEVELS)             # [B,H,blocks,1]
        unit = (sigma_b / levels).clamp_min(TINY).unsqueeze(-1)            # [B,H,blocks,1,1]
        s_op = s_blk if exact else torch.round(s_blk / unit) * unit
        partial = torch.einsum("bthnk,bhnkc->bthnc", r_q, s_op)            # [B,t,H,blocks,dv]
        r_fp = leaf_product(read_levels, widths, sl)                       # [B,t,H,N] exact amplitudes
        if not exact:
            blocks_seen += two.numel()
            flagged += int(two.sum())
            if panel:
                scale_t = torch.einsum("bthn,bhn->bth", r_fp, bound)       # the pre-pass's bound readout
                p_unit = (scale_t * (1.0 + threshold / 254.0) / PANEL_LEVELS).clamp_min(TINY)[..., None, None]
                partial = torch.round(partial / p_unit) * p_unit
                panel_peak = max(panel_peak, float((partial.sum(3).abs() / p_unit[..., 0, :]).max()))
        num[:, sl] = partial.sum(3)
        den[:, sl] = torch.einsum("bthn,bhn->bth", r_fp, mass)
        del r_q, r_fp, s_op, s_blk, partial
        # --- fold ------------------------------------------------------------------------------------------------
        a_wd, m_wd = group_product(write_levels, widths, w_dn, sl, u8)    # [B,t,H,n_wd] unit-max u8, its row scale
        a_ws, _ = group_product(write_levels, widths, w_sp, sl)            # [B,t,H,n_ws]
        scal = a_ws * s_t[:, sl, :, 0].unsqueeze(-1) * m_wd                                # [B,t,H,n_ws]
        vh = v_hat[:, sl]                                                  # [B,t,H,dv] one f16 row a token
        for s in range(n_ws):
            leaves = torch.nonzero(idx_ws == s, as_tuple=False).flatten()
            a_full = a_wd[..., idx_wd[leaves]] * scal[..., s].unsqueeze(-1)   # [B,t,H,n_wd] the leaves' A rows
            sigma = a_full.amax(dim=1, keepdim=True).clamp_min(TINY)           # sigma_leaf,w: the biggest writer
            a_q = a_full if exact else quantize_u8_unit(a_full / sigma) * sigma
            state[:, :, leaves] += torch.einsum("bthd,bthc->bhdc", a_q, vh)
            bound[:, :, leaves] += a_q.sum(1)
        w_fp = leaf_product(write_levels, widths, sl)
        mass += w_fp.sum(1)
        del a_wd, a_ws, scal, w_fp
    if telemetry is not None:
        telemetry.update(blocks=blocks_seen, flagged=flagged, panel_peak=panel_peak)
    full = torch.cat((state, mass.unsqueeze(-1)), -1)
    return num, den, full


def fold_zeta(s_t, m_w, window):
    """``[B, H, windows, 2]`` fp64: the fold's per-window fixed-point scales -- over each window's tokens the largest
    ``m_w`` and ``s_t m_w`` (``s_t`` the value scale, ``m_w`` the write levels' maxima multiplied, ``m_ws m_wd``: the
    write side's shared scale), at least ``TINY``. The facts pass writes them (docs/internals/facts/facts.md#zeta)."""
    col = m_w.double()
    cols = torch.stack((col, col * s_t.double()), -1)                                          # [B, T, H, 2]
    B, T, H, _ = cols.shape
    windows = -(-T // window)
    cols = torch.cat((cols, cols.new_zeros(B, windows * window - T, H, 2)), 1)
    return cols.reshape(B, windows, window, H, 2).amax(2).permute(0, 2, 1, 3).clamp_min(TINY)


def unit_rows(rows, scales, sparse) -> tuple[torch.Tensor, ...]:
    """A bundle side's rows as fp64 unit values: a u8 digit over 255 (a dense write level's f16 coefficient as it is),
    a dense level's zeros in a live row at the floor (docs/internals/facts/liveness_contract.md#floor)."""
    out = []
    for r, scale, is_sparse in zip(rows, scales, sparse, strict=True):
        unit = r.double() if r.dtype is quant.DENSE_DTYPE else r.double() / 255.0
        if not is_sparse:
            unit = torch.where((scale > 0).unsqueeze(-1), unit.clamp_min(quant.DENSE_FLOOR), unit)
        out.append(unit)
    return tuple(out)


def value_units(values) -> torch.Tensor:
    """``q``: a value's f16 unit, fp64 ``[B, T, H, dv]``, the bundle's value being ``scale q``."""
    return values.v.double()


def intra_bundle_reference(routes, values, window):
    """``(num, den)``, fp64 ``[B, T, H, dv]`` and ``[B, T, H]``: the INTRA on the bundle as stored, nothing rounded.

    Per window, reader ``t`` and writer ``t' <= t`` in it: ``G_l[t, t'] = sum_d r_l[t, d] w_l[t', d]`` over the
    rows' unit values (:func:`unit_rows`); ``A[t, t'] = f(t') prod_l G_l[t, t']`` with the writer factor
    ``f(t') = prod_l sw_l(t')`` (the write levels' scales: the write side's shared scale). The reader's factor
    ``prod_l sr_l(t)`` is not applied: it is per token and cancels in ``num / den`` (the output contract, carry
    and intra alike). ``den[t] = sum_t' A`` and ``num[t] = sum_t' A s(t') q(t')`` over the value's f16 units
    (:func:`value_units`) and scale ``s``. A grouped call's write side, values and factors are its write heads',
    each read by ``H / H_w`` consecutive query heads.
    """
    group = routes.group

    def heads(x):
        return x.repeat_interleave(group, dim=2)

    reads = unit_rows(routes.amp_read, routes.scale_read, routes.sparse[0])
    writes = tuple(heads(w) for w in unit_rows(routes.amp_write, routes.scale_write, routes.sparse[1]))
    f = routes.scale_write[0].double()
    for scale in routes.scale_write[1:]:
        f = f * scale.double()
    f, s, q = heads(f), heads(values.scale.double()), heads(value_units(values))
    Bn, T, H, dv = q.shape
    num = torch.zeros(Bn, T, H, dv, device=q.device, dtype=torch.float64)
    den = torch.zeros(Bn, T, H, device=q.device, dtype=torch.float64)
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        n = sl.stop - sl.start
        A = f[:, sl].permute(0, 2, 1)[:, :, None, :]                                    # [B,H,1,n] the writers
        for r, w in zip(reads, writes, strict=True):
            A = A * torch.einsum("bthd,bshd->bhts", r[:, sl], w[:, sl])
        A = A * torch.tril(torch.ones(n, n, device=q.device, dtype=torch.float64))
        den[:, sl] = A.sum(-1).permute(0, 2, 1)
        P = A * s[:, sl].permute(0, 2, 1)[:, :, None, :]
        num[:, sl] = (P @ q[:, sl].permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return num, den


def prepass_reference(read_levels, write_levels, v, widths, sparsity, window, *, exact: bool = False,
                      state_in=None, telemetry: dict | None = None):
    """``(den, scale)`` per token, ``[B, T, H]`` fp64, under the PRE-PASS's DECLARED arithmetic: the carry's
    skeleton at two columns a leaf. FOLD, per window and write-sparse digit ``s``, column ``m`` in {1, s_t}:
    ``zeta_m`` = the window's largest ``col_m(t) m_ws(t) m_wd(t)`` (the two write levels' per-token maxima,
    :func:`fold_zeta`);
    ``P_m(t) = rint(2^23 col_m m_ws m_wd / (255 zeta_m))``; ``V_m(t, d) = (P_m(t) a_wd[t, d] + 128) >> 8`` (a
    15-bit integer, ``a_wd`` the u8 write-dense digit); ``acc_m(s, d) = sum_t a_ws[t, s] V_m(t, d)`` (the u8
    write-sparse digit, exact in s32); the leaf's column gains ``acc_m zeta_m / (2^15 255)``. Both u8 levels are the
    bundle's own; the one added rounding is the 15-bit digit a (token, write-dense digit). READOUT, per block (the carry's blocks): B = [mass, bound]
    each at 15 bits against the block's max, A = the unit-max dense read row in u8, the (token, block) factor
    ``a_sparse x m_t`` on the partial; the two sums in fp64 here (fp32 RED.F32 in the kernel). ``den`` is the inter
    denominator against the window-start mass; ``scale`` is the packed panel's per-token bound, which the kernel
    needs to be >= every running sum of |partials| and reads as ``scale x (1 + R/254) / 32767``. ``exact`` turns
    every rounding into the identity, reproducing the fp64 op's ``den``. ``telemetry`` receives the final
    ``mass`` and ``bound`` planes ``[B, H, N]`` when given."""
    Bn, T, H, dv = v.shape
    D = len(widths)
    N = math.prod(widths)
    dev = v.device
    r_sp = [lvl for lvl in range(D) if sparsity[lvl][0]]
    r_dn = [lvl for lvl in range(D) if not sparsity[lvl][0]]
    w_sp = [lvl for lvl in range(D) if sparsity[lvl][1]]
    w_dn = [lvl for lvl in range(D) if not sparsity[lvl][1]]
    idx_rs, idx_rd = group_index(widths, r_sp).to(dev), group_index(widths, r_dn).to(dev)
    idx_ws, idx_wd = group_index(widths, w_sp).to(dev), group_index(widths, w_dn).to(dev)
    order, blk = block_order(widths, r_sp, r_dn)
    order = order.to(dev)
    n_blocks = N // blk
    u8 = None if exact else quantize_u8_unit
    q15 = (lambda x: x) if exact else (lambda x: torch.round(x * TWO_DIGIT_LEVELS) / TWO_DIGIT_LEVELS)

    _, s_t = unit_max(v.double())
    if state_in is None:
        mass = torch.zeros(Bn, H, N, device=dev, dtype=torch.float64)
        bound = torch.zeros(Bn, H, N, device=dev, dtype=torch.float64)
    else:
        mass = state_in.double()[..., dv].clone()
        bound = state_in.double()[..., :dv].abs().amax(-1)

    den = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)
    scale = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)

    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        # --- readout of mass and bound against the window-start state, block by block ---------------------------
        a_rd, m_rd = group_product(read_levels, widths, r_dn, sl, u8)
        a_rs, _ = group_product(read_levels, widths, r_sp, sl)
        r_q = (a_rd[..., idx_rd] * (a_rs * m_rd)[..., idx_rs])[..., order].reshape(Bn, sl.stop - sl.start, H,
                                                                                  n_blocks, blk)
        cols = torch.stack((mass, bound), -1)[:, :, order].reshape(Bn, H, n_blocks, blk, 2)   # [B,H,blocks,leaves,2]
        sigma = cols.amax(-2, keepdim=True).clamp_min(TINY)                                     # per block, per column
        cols_q = q15(cols / sigma) * sigma
        partial = torch.einsum("bthnk,bhnkc->bthnc", r_q, cols_q).sum(3)                         # [B,t,H,2]
        den[:, sl] = partial[..., 0]
        scale[:, sl] = partial[..., 1]

        # --- fold: the two digit rows against the write-dense u8 row, one 15-bit digit a (token, list) ----------
        a_wd, m_wd = group_product(write_levels, widths, w_dn, sl, u8)                          # [B,t,H,n_wd]
        a_ws, m_ws = group_product(write_levels, widths, w_sp, sl, u8)                          # [B,t,H,n_ws]
        gs = s_t[:, sl, :, 0]                                                                    # [B,t,H]
        g = torch.ones_like(gs)
        f = (m_ws * m_wd)[..., 0]
        zetas = fold_zeta(gs, f, window)[:, :, 0]                                               # [B,H,2]
        a_ws_digit = a_ws if exact else torch.round(a_ws * 255.0)                                  # the u8 bytes
        for m, (col, target) in enumerate(((g, mass), (gs, bound))):
            zeta = zetas[..., m].unsqueeze(1)                                                    # [B,1,H]
            if exact:
                V = (col * f)[..., None] * a_wd
                unit = 1.0
            else:
                P = torch.round(2.0 ** 23 * (col * f) / (255.0 * zeta))                          # [B,t,H]
                V = torch.floor((P[..., None] * torch.round(a_wd * 255.0) + 128.0) / 256.0)     # [B,t,H,n_wd]
                unit = zeta[:, 0] / (2.0 ** 15 * 255.0)                                          # [B,H]
            acc = torch.einsum("bths,bthd->bhsd", a_ws_digit, V)                                 # [B,H,n_ws,n_wd]
            target += acc[:, :, idx_ws, idx_wd] * (unit if exact else unit[..., None])

    if telemetry is not None:
        telemetry["mass"] = mass.clone()
        telemetry["bound"] = bound.clone()
    return den, scale


def per_token_norm_ratio(err: torch.Tensor, envelope: torch.Tensor) -> torch.Tensor:
    """``||err[t]|| / ||envelope[t]||`` over the channel axis: the basis-free fidelity number."""
    return err.norm(dim=-1) / envelope.norm(dim=-1).clamp_min(TINY)


#: The report's readout eps settings: one readout digit wherever the spread allows, the declared 2^-6 and 2^-7, and
#: two digits everywhere.
REPORT_KNOBS = (1.0, 2.0 ** -6, 2.0 ** -7, 0.0)


def quantization_report(drawn, sparsity, window, knobs=REPORT_KNOBS, rotate_values=False, panel=True):
    """Comparison B on one drawn cell: the declared arithmetic against the unquantized op. Returns one row per knob
    setting: the per-slot error over the slot's envelope (max and the fraction above 2^-8), the per-token norm ratio
    (max, mean) for ``num``, the state's per-slot max over its envelope, the fraction of (block, window) pairs that
    took the second digit and the panel's peak in units. With ``rotate_values`` the values are rotated by the Hadamard
    before quantization and the outputs unrotated, so the errors are read in the real basis."""
    read, write, gain, v = drawn.doubles()
    widths = drawn.cell.widths
    h = hadamard(v.shape[-1], v.device) if rotate_values else None
    v_in = rotate(v, h) if rotate_values else v
    true_num, true_den, true_state = inter_reference(read, with_gain(write, gain), v, widths, window)
    env_num, _, env_state = inter_reference(read, with_gain(write, gain), v.abs(), widths, window)
    rows = []
    for eps in knobs:
        tele = {}
        num, den, state = int_reference(read, with_gain(write, gain), v_in, widths, sparsity, window,
                                        readout_eps=eps, panel=panel, telemetry=tele)
        if rotate_values:
            num, state = unrotate(num, h), torch.cat((unrotate(state[..., :-1], h), state[..., -1:]), -1)
        e_num = (num - true_num).abs()
        slot = e_num / (env_num * 2.0 ** -8).clamp_min(TINY)
        tok = per_token_norm_ratio(num - true_num, env_num)
        e_state = (state[..., :-1] - true_state[..., :-1]).abs() / (env_state[..., :-1] * 2.0 ** -8).clamp_min(TINY)
        rows.append(dict(readout_eps=eps,
                         flag_rate=tele["flagged"] / max(tele["blocks"], 1), panel_peak=tele["panel_peak"],
                         slot_max=float(slot.max()), slot_frac_over=float((slot > 1).double().mean()),
                         token_max=float(tok.max()), token_mean=float(tok.mean()),
                         state_slot_max=float(e_state.max()), state_slot_frac_over=float((e_state > 1).double().mean()),
                         den_max_rel=float(((den - true_den).abs() / true_den.abs().clamp_min(TINY)).max())))
    return rows
