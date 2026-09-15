# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 the cells: the routing draws, the plane crossings, and the build stamp gate.

``k_tok`` (a token's nonzero digits per level, the routing sparsity) is a property
of the DRAW here and of nothing in the kernel -- which is the design's whole claim,
so the fixtures state it explicitly and the kernel is never told.
"""
from __future__ import annotations

import math

import torch

from rola.ops import carry as carry_ops
from rola.ops.paging import from_split_planes


def simplex(shape, k_tok, gen, device, live=None):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense.

    ``live`` truncates the row to its FIRST ``live`` digits -- STRUCTURED support, as
    against ``k_tok``'s unstructured one. The two are different questions: unstructured
    sparsity at long ``T`` still reaches every atom, so only a truncation can produce an
    atom the routing never touches (the idle-resident cell). It is applied after both
    draws so a cell's RNG stream does not depend on it.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    if live is not None and live < shape[-1]:
        x[..., live:] = 0.0
    return x / x.sum(-1, keepdim=True)


class Cell:
    """One realized cell. ``widths`` is the topology, ``k_tok`` the routing draw."""

    def __init__(self, widths, T, k_tok=None, seed=0, d_v=64, B=1, H=1,
                 device="cuda", state_in=False, support=1.0):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.widths, self.T, self.d_v, self.B, self.H = tuple(widths), T, d_v, B, H
        self.N = math.prod(widths)
        shape = (B, T, H)
        self.support = support
        live = tuple(max(1, int(w * support)) for w in widths)
        self.read = tuple(simplex(shape + (w,), k_tok, gen, device, live=lv)
                          for w, lv in zip(widths, live))
        self.write = tuple(simplex(shape + (w,), k_tok, gen, device, live=lv)
                           for w, lv in zip(widths, live))
        self.g_write = torch.rand(*shape, device=device, dtype=torch.float64,
                                  generator=gen) + 0.5
        self.v = torch.randn(*shape, d_v, device=device, dtype=torch.float64, generator=gen)
        #: THE OPERAND PLANES ARE bf16: the bytes a kernel entry reads (`owner_rows` takes them).
        self.read_bf = tuple(t.to(torch.bfloat16) for t in self.read)
        self.write_bf = tuple(t.to(torch.bfloat16) for t in self.write)
        self.state_in = None
        if state_in:
            self.state_in = 0.1 * torch.randn(B * H, self.N, d_v + 1, device=device,
                                              dtype=torch.float64, generator=gen)


def canonical_from_plane(plane):
    """A ``[BH, N/16, 16, cols]`` state plane as ``[BH, N, cols]`` CANONICAL.

    The kernel's plane IS the atom-major view of the canonical one in the STORED
    split-plane form, so this is the hi||lo gather and a reshape. It stays a NAMED step because the assertion it feeds is the family's
    final-state gate, and a leaf-order error must surface at a named seam rather than as
    noise inside a comparison.
    """
    return from_split_planes(plane).reshape(plane.shape[0], -1, plane.shape[-1])


def built_arms() -> set:
    """`{(D, DV, warps_per_cta)}` -- what THIS binary carries.

    `arms()` is the binary's own answer (`rola.ops.carry.arms()`), so this is a fact
    about the `.so` under test and never about the source tree that produced it. EMPTY
    on this line: C0 cleared the carry family, G4 refills the table and C3 builds the
    bodies (card `development/queue/C_CLEAN_SLATE.md`).
    """
    return {tuple(row) for row in carry_ops.arms()}


def assert_fresh_binary():
    """DEVICE-SIDE, because a path or hash check cannot catch an extension trap.

    THERE IS NO SUCH FACT FOR THE CARRY FAMILY ON THIS LINE: it has no device code, so
    the stamp entry refuses rather than returning a stale or invented census, and this
    helper propagates that refusal instead of substituting a weaker check for it
    (`extension-trap-device-side-check`). C3-K0 restores the fact and this assertion
    with it.
    """
    return carry_ops.build_stamp()


def relative(actual, reference):
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference.double()).abs().max()) / scale


def relative_per_token(actual, reference):
    """THE PER-TOKEN RELATIVE ERROR: the worst token's error on ITS OWN scale, the largest entry of its reference row.
    ``relative`` is one ratio on the tensor's largest entry, and a token wrong by half hides under it when its row is
    small -- which is how the box-words hazard of 2026-09-08 passed at two grains. A row that is exactly zero in both is
    a match; a row zero in the reference and not in the kernel is an infinite error. The last axis is the token's row;
    every axis before it indexes tokens."""
    a = actual.double().reshape(-1, actual.shape[-1])
    r = reference.double().reshape(-1, reference.shape[-1])
    err, scale = (a - r).abs().amax(1), r.abs().amax(1)
    zero = scale == 0
    if bool((err[zero] > 0).any()):
        return float("inf")
    return float((err[~zero] / scale[~zero]).max()) if bool((~zero).any()) else 0.0
