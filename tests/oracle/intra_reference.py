# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The fp64 intra reference, and the operand cells the K31 R2 battery runs it on.

The oracle (`rola.ops.naive`) exposes only the NORMALIZED readout, so the
un-normalized numerator and mass the intra kernel REDs cannot be read off it
directly. :func:`intra_reference` states the intra term from its definition
instead, in fp64, and `test_intra_reference.py` gates that statement against
`naive_rola` on a single-window cell where the two must agree exactly up to the
readout's division: zero initial state and no decay make the oracle's whole
output the intra term.
"""

from __future__ import annotations

import torch

#: the flagship window -- `rola.ops.intra.WINDOW`, restated here for the same
#: reason the reference derives its own leaf addresses: a reference that imported
#: production's constant would make a wrong window agree with itself.
WINDOW = 384
LEVEL_WIDTH = 256
VALUE_WIDTH = 64

DENSE_BOTH = 0
READ_SPARSE = 1
WRITE_SPARSE = 2
BOTH_SPARSE = 3


def intra_reference(pread, pwrite, gwrite, v_bh, levels, window=WINDOW,
                    level_width=LEVEL_WIDTH):
    """``o[t] = sum_{t' <= t in w} A[t, t'] v[t']`` and ``den[t] = sum_{t'} A[t, t']``.

    ``A[t, t'] = gwrite[t'] * prod_l <pread_l[t], pwrite_l[t']>`` -- the gain is a
    per-COLUMN scalar applied after the level product, so the support factors are
    ungained, matching the shipped kernel's gain law.

    ``pread``/``pwrite``: ``[BH, L, levels*level_width]``; ``gwrite``: ``[BH, L]``;
    ``v_bh``: ``[BH, L, d_v]``. All are consumed at float64.
    """
    pread = pread.to(torch.float64)
    pwrite = pwrite.to(torch.float64)
    gwrite = gwrite.to(torch.float64)
    v_bh = v_bh.to(torch.float64)
    bh, length, _ = pread.shape
    o = torch.zeros((bh, length, v_bh.shape[-1]), dtype=torch.float64, device=v_bh.device)
    den = torch.zeros((bh, length), dtype=torch.float64, device=v_bh.device)
    causal = torch.tril(torch.ones((window, window), dtype=torch.float64, device=v_bh.device))
    for start in range(0, length, window):
        cut = slice(start, start + window)
        a = torch.ones((bh, window, window), dtype=torch.float64, device=v_bh.device)
        for level in range(levels):
            span = slice(level * level_width, (level + 1) * level_width)
            a = a * (pread[:, cut, span] @ pwrite[:, cut, span].transpose(-1, -2))
        a = a * gwrite[:, cut].unsqueeze(1) * causal
        o[:, cut] = a @ v_bh[:, cut]
        den[:, cut] = a.sum(-1)
    return o, den


def _simplex(shape, k_tok, generator, device, clustered=False):
    """A row-normalized nonnegative factor plane; ``k_tok`` nonzero digits per row.

    ``clustered`` is the COHORT form the slab gate exists for: a run of
    ``clustered`` consecutive tokens draws ONE contiguous digit window, so a
    tile's digit union is a run rather than a scatter. ``clustered=True`` means a
    per-token run, which is the weakest cohort there is.
    """
    width = shape[-1]
    x = torch.rand(shape, dtype=torch.float64, device=device, generator=generator)
    if k_tok is not None and k_tok < width:
        if clustered:
            cohort = 1 if clustered is True else int(clustered)
            groups = shape[:-2] + (shape[-2] // cohort, 1, 1)
            starts = torch.randint(0, width, groups, device=device, generator=generator)
            starts = starts.expand(*groups[:-2], cohort, 1).reshape(*shape[:-1], 1)
            offsets = torch.arange(k_tok, device=device).view(*([1] * (len(shape) - 1)), k_tok)
            index = (starts + offsets) % width
        else:
            index = torch.argsort(
                torch.rand(shape, dtype=torch.float64, device=device, generator=generator),
                dim=-1)[..., :k_tok]
        keep = torch.zeros(shape, dtype=torch.float64, device=device)
        keep.scatter_(-1, index, 1.0)
        x = x * keep
    return x / x.sum(-1, keepdim=True)


def make_cell(bh, length, levels, modes, k_tok=4, seed=0, device="cuda", clustered=False,
              level_width=LEVEL_WIDTH):
    """Operand planes for one conformance cell, in the kernel's bf16 storage form.

    A level's mode decides which SIDE carries the sparse support; BOTH_SPARSE
    draws ONE support and gives both sides that same support, which is the law's
    union-routing case and must run on the same machinery.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    read_parts, write_parts = [], []
    for level in range(levels):
        mode = modes[level]
        shape = (bh, length, level_width)
        if mode == BOTH_SPARSE:
            support = _simplex(shape, k_tok, generator, device, clustered) > 0
            read = torch.rand(shape, dtype=torch.float64, device=device, generator=generator)
            read = read * support
            write = torch.rand(shape, dtype=torch.float64, device=device, generator=generator)
            write = write * support
            read_parts.append(read / read.sum(-1, keepdim=True))
            write_parts.append(write / write.sum(-1, keepdim=True))
            continue
        read_k = k_tok if mode == READ_SPARSE else None
        write_k = k_tok if mode == WRITE_SPARSE else None
        read_parts.append(_simplex(shape, read_k, generator, device, clustered))
        write_parts.append(_simplex(shape, write_k, generator, device, clustered))
    pread = torch.cat(read_parts, dim=-1).to(torch.bfloat16)
    pwrite = torch.cat(write_parts, dim=-1).to(torch.bfloat16)
    gwrite = (0.5 + torch.rand((bh, length), dtype=torch.float64, device=device,
                               generator=generator)).to(torch.bfloat16)
    v_bh = torch.randn((bh, length, VALUE_WIDTH), dtype=torch.float64, device=device,
                       generator=generator).to(torch.bfloat16)
    return pread, pwrite, gwrite, v_bh


def cell_support(*planes):
    """The frozen support word of each plane -- the reference's OWN bit-packing.

    ``[BH, L/32, width]`` int32, bit ``token & 31`` of word ``token // 32`` set where
    the plane's stored value is nonzero. Derived here rather than imported from
    :mod:`rola.ops.intra` for the same reason this module derives its own window: a
    kernel gated against the production packer would only prove the two agree with
    each other. The accumulation is int64 with an explicit two's-complement fold, so
    it shares no arithmetic with production's int32 shifts.
    """
    out = []
    for plane in planes:
        bh, length, width = plane.shape
        live = plane.to(torch.float64) != 0
        words = torch.zeros((bh, length // 32, width), dtype=torch.int64, device=plane.device)
        for bit in range(32):
            words |= live[:, bit::32, :].to(torch.int64) << bit
        out.append(torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32))
    return tuple(out)


def as_token_major(v_bh, heads):
    """``[BH, L, d_v]`` seen as the kernel's token-major ``[B, T, H, d_v]`` value stream."""
    bh, length, d_v = v_bh.shape
    return v_bh.view(bh // heads, heads, length, d_v).permute(0, 2, 1, 3).contiguous()


def slab_skip_rate(pread, pwrite, modes, window=WINDOW, tile=64, slab=16,
                   level_width=LEVEL_WIDTH):
    """The fraction of (tile-pair, level, slab) MMA groups the zero certificate gates.

    A python mirror of the kernel's mask: a level's declared-sparse side is
    summarized as the digit union over a tile's tokens; an undeclared side reads
    all-ones. Returns ``(skipped, total)``.
    """
    bh, length, _ = pread.shape
    tiles = window // tile
    slabs = max(1, level_width // slab)
    skipped = total = 0
    for level, mode in enumerate(modes):
        span = slice(level * level_width, (level + 1) * level_width)

        def union(plane, declared):
            if not declared:
                return torch.ones((bh, length // tile, slabs), dtype=torch.bool,
                                  device=plane.device)
            live = plane[:, :, span].to(torch.float32) != 0
            live = live.view(bh, length // tile, tile, slabs, level_width // slabs)
            return live.any(dim=2).any(dim=-1)
        rows = union(pread, mode in (READ_SPARSE, BOTH_SPARSE))
        cols = union(pwrite, mode in (WRITE_SPARSE, BOTH_SPARSE))
        for start in range(0, length, window):
            w = start // window
            for rt in range(tiles):
                for ct in range(rt + 1):
                    live = rows[:, w * tiles + rt] & cols[:, w * tiles + ct]
                    total += live.numel()
                    skipped += int((~live).sum())
    return skipped, total
