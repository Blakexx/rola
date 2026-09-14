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
from rola.ops.paging import from_split_planes, to_split_planes


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
        #: THE OPERAND PLANES ARE bf16 AND THE REFERENCE READS THE SAME BYTES.  The
        #: gate is a conformance band, not a rounding coincidence: comparing an fp64
        #: reference of the fp64 draws against a kernel fed their bf16 images would
        #: measure the CAST, which is not what is under test.
        self.read_bf = tuple(t.to(torch.bfloat16) for t in self.read)
        self.write_bf = tuple(t.to(torch.bfloat16) for t in self.write)
        self.g_bf = self.g_write.to(torch.bfloat16)
        self.v_bf = self.v.to(torch.bfloat16)
        self.state_in = None
        if state_in:
            self.state_in = 0.1 * torch.randn(B * H, self.N, d_v + 1, device=device,
                                              dtype=torch.float64, generator=gen)

    # -- the reference reads the ROUNDED operands ---------------------------
    def ref_levels(self):
        return (tuple(t.double() for t in self.read_bf),
                tuple(t.double() for t in self.write_bf),
                self.g_bf.double(), self.v_bf.double())

    def state_in_plane(self):
        """``state_in`` as the kernel's state plane, ``[BH, N/16, 16, cols]``.

        A RE-BLOCKING AND THE SPLIT-PLANE PACK since K50: the consumer addresses its state by
        CANONICAL atom id in both backings, so the plane is the atom-major view of
        ``[BH, N, cols]`` that `docs/internals/state.md` section 3 defines, and the leaf
        permutation this method used to apply is gone with the lattice keying.
        """
        if self.state_in is None:
            return None
        return to_split_planes(self.state_in.to(torch.float32).reshape(
            self.B * self.H, self.N // 16, 16, self.d_v + 1).contiguous())


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


def declared_arms() -> set:
    """The rows the ARM LIST declares, in the same key `built_arms` reports.

    Read from `tools/gen_shards.py`, which owns the list and generates both the
    dispatch's macro and the per-arm translation units.
    """
    import sys
    from pathlib import Path as _Path

    tools = _Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import gen_shards

    return {tuple(row) for row in gen_shards.CARRY_ARMS}


def require_arm(D: int, DV: int, warps_per_cta: int) -> None:
    """SKIP unless this binary carries the arm, and NAME the one it is missing.

    AN UNDECLARED ARM IS NOT A SKIP. If the key is in no row of the list at all, the
    binary is not missing it -- nothing builds it, in any configuration -- and turning
    that into a skip would report an absent arm as satisfied coverage. Only a DECLARED
    row this build did not compile is skippable; anything else falls through to the
    launch surface, which refuses it and names the reason. On the clean line the list is
    EMPTY, so nothing is skippable and every carry cell reaches the surface's honest
    no-implementation failure -- which is the state card C's TEST-DRIVEN ruling asks for.
    """
    import pytest

    key = (D, DV, warps_per_cta)
    if key in built_arms() or key not in declared_arms():
        return
    pytest.skip(f"this binary does not carry the carry arm {key}; it is a declared TEST "
                f"row. Build the battery's set: ROLA_CARRY_ARMS=all ROLA_CUDA_ARCHS=86 "
                f"pip install -e . --no-build-isolation")


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


def relative_per_token(actual, reference, floor=0.02):
    """THE PER-TOKEN RELATIVE ERROR: the worst token's error on ITS OWN scale, over the
    tokens whose scale is at least ``floor`` of the tensor's. ``relative`` is one ratio on
    the tensor's largest entry, and a few tokens wrong by half hide under it when their
    rows are small -- which is how the box-words hazard of 2026-09-08 passed at two grains.
    The last axis is the token's row; every axis before it indexes tokens."""
    a = actual.double().reshape(-1, actual.shape[-1])
    r = reference.double().reshape(-1, reference.shape[-1])
    scale = r.abs().amax(1)
    keep = scale >= floor * max(1e-30, float(scale.max()))
    if not bool(keep.any()):
        return 0.0
    return float(((a - r).abs().amax(1)[keep] / scale[keep]).max())
