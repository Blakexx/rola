# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LEAF ORDER: the ``(k, m)`` box lattice, as host arithmetic, and the ONE host
authority over it -- ``rola.ops.carry`` reads its box from here.

Symbols, each at first use: ``widths`` = the per-level digit counts (``B_l`` each),
``D = len(widths)``, ``k`` = the uniform sub-box span per level, ``m`` = the capacity
multiplier, ``m_l`` its per-level share, ``s_l = k * m_l`` = an owner's span at level
``l``, ``g_l = B_l / s_l`` = the owner-grid extent, ``BC = prod_l s_l`` = the leaves one
owner holds, ``N = prod_l B_l``, ``ell`` = a leaf's canonical MSB-first mixed-radix index,
``lambda`` = its LATTICE index, ``r_l`` = a leaf's run offset inside its owner at level
``l``, and ``atom`` = the ``MMA_K_QUANTUM``-leaf page granule.

THE INTRA-BOX ORDER IS MIXED RADIX over the owner's run offsets, most significant level
first: ``lambda = O*BC + sum_l r_l * prod_{l' > l} s_l'``. That is the device's order
(the addressing block, ``csrc/rola/src/common/geom.cuh``), it is the ONLY order (the
one-order ruling), and it is
what makes each level's digits CONTIGUOUS in the owner-local index, and each atom an
aligned digit rectangle over the trailing levels (:func:`atom_rectangle`).

``pi`` is a fixed BIT PERMUTATION of the leaf index, so :func:`permutation` is a
convenience for the ORACLE and TEST seams and never a table any kernel consults.
"""
from __future__ import annotations

import math

import torch

from rola.engine.types import MMA_K_QUANTUM

__all__ = ["LATTICE_CAP_BC", "atom_rectangle", "box_shape", "derive_lattice", "permutation",
           "to_canonical", "to_lattice", "walk_unit"]

#: The leaves ONE OWNER holds, and the ceiling this solver searches down from. It is the
#: ONE number here that is a choice rather than a consequence, and it is the carry's --
#: its register budget sizes the box, and the two families write and read one plane. It
#: is stated as ``BC`` rather than as a capacity multiplier because ``BC = k^D * m``
#: grows with DEPTH: a ceiling on ``m`` is a ceiling on the box only at one ``D``.
LATTICE_CAP_BC = 128

#: The owner span the device path reads as a single 32-bit slice, mirrored from
#: `csrc/rola/src/decode/decode.cuh`. A candidate the device would refuse is refused here
#: instead, so a topology's lattice is decided once and never negotiated at a launch.
_MAX_SPAN = 32

#: The uniform sub-box span is the carry family's codegen unit and never a free variable;
#: the search runs over the powers of two at or below it, largest first.
_MAX_SPAN_K = 4


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box on ``widths``.

    The distribution of ``m`` across the levels is R0 spec section 2.5 as the
    atom-rectangle law generalizes it: the INNERMOST level takes as much capacity as it
    can carry -- never less than its balanced share, never less than an atom asks
    (``s_{D-1} >= MMA_K_QUANTUM``), and never more than its own width admits -- and
    the rest of ``j = log2 m`` is spread balanced over the outer levels, innermost first,
    so those exponents differ by at most one.

    THE ATOM FLOOR IS A PREFERENCE, THE WIDTH CEILING IS THE LAW. A level narrower than an
    atom cannot absorb the floor, and the capacity spills outward rather than the plan
    being refused: the atom then spans more than one level (:func:`atom_rectangle`), which
    costs occupancy where a spanned level is sparse but never correctness. What is still
    refused is a ``j`` no span vector can carry -- raised below.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    j = int(math.log2(m))
    if 2 ** j != m:
        raise ValueError(f"the lattice capacity m must be a power of two, got {m}")
    kappa = int(math.log2(k))
    if 2 ** kappa != k:
        raise ValueError(f"the lattice span k must be a power of two, got {k}")
    floor_inner = max(0, int(math.log2(MMA_K_QUANTUM)) - kappa)
    ceil_inner = 0
    while ((k << (ceil_inner + 1)) <= widths[D - 1]
           and widths[D - 1] % (k << (ceil_inner + 1)) == 0):
        ceil_inner += 1
    a_inner = min(max(j // D + (1 if j % D else 0), floor_inner), ceil_inner, j)
    j_outer = j - a_inner
    a = [a_inner if l == D - 1 else
         j_outer // (D - 1) + (1 if l >= (D - 1) - (j_outer % (D - 1)) else 0)
         for l in range(D)]
    if sum(a) != j:
        raise ValueError(
            f"the capacity m = {m} does not fit widths {widths} at k = {k}: the levels can "
            f"carry 2^{sum(a)} (an owner sits inside a level)")
    m_l = [1 << x for x in a]
    s = [k * x for x in m_l]
    for l, (w, t) in enumerate(zip(widths, s, strict=True)):
        if w % t != 0:
            raise ValueError(
                f"level {l} of width {w} is not a whole number of owner spans k*m_l = {t} "
                "(an owner sits inside a level)")
    g = [w // t for w, t in zip(widths, s, strict=True)]
    return m_l, s, g, math.prod(s), math.prod(g)


def atom_rectangle(spans) -> tuple[int, int]:
    """``(whole_levels, partial_digits)``: the atom, as a digit rectangle.

    An atom is ``MMA_K_QUANTUM`` consecutive ``local`` values at a multiple of it, so
    under the mixed-radix order it is the TRAILING levels whose spans multiply into it,
    taken whole, times an aligned sub-run of ``partial_digits`` of the next level out.
    Every admissible span vector gives such a rectangle -- that is a property of the
    order, not a constraint on ``(k, m)``.

    A token FILLS an atom exactly when every level of the rectangle is dense for its
    side, which is why :func:`box_shape` pushes capacity innermost: a one-level rectangle
    asks that of one level only. Where the rectangle spans two levels and one of them is
    sparse, touched atoms are partially occupied -- an occupancy cost, never an error.
    """
    atom = int(MMA_K_QUANTUM)
    whole, prod = 0, 1
    for span in reversed([int(x) for x in spans]):
        if prod * span > atom:
            break
        prod *= span
        whole += 1
    return whole, atom // prod


def walk_unit(spans) -> tuple[int, int, int]:
    """``(unit_base, unit_leaves, unit_atoms)``: the decode walk's enumeration tier.

    THE PAGED BACKING'S ONE LATTICE CONDITION, made total. The walk enumerates
    ``(owner, r_0 .. r_{unit_base-1})`` and the unit's leaves are the trailing spans from
    ``unit_base`` inward -- the SHALLOWEST such suffix holding a whole number of atoms,
    which is what makes the candidate-atom enumeration a PARTITION. ``BC`` is a multiple
    of an atom, so ``unit_base = 0`` always qualifies and the search is total. The
    two clauses the interleaved order needed (``k^D >= 16`` and ``4 % log2 k == 0``) are
    discharged by the mixed-radix order itself.
    """
    atom = int(MMA_K_QUANTUM)
    spans = [int(x) for x in spans]
    prod = 1
    for base in range(len(spans) - 1, -1, -1):
        prod *= spans[base]
        if prod % atom == 0:
            return base, prod, prod // atom
    raise ValueError(f"no suffix of the spans {tuple(spans)} holds a whole atom")


def _admissible(k: int, m: int, widths) -> tuple[int, ...] | None:
    """``(s_l)`` if this ``(k, m)`` sits inside every level and inside the box, else None."""
    try:
        _, s, _, bc, _ = box_shape(widths, k, m)
    except ValueError:
        return None
    if bc > LATTICE_CAP_BC or any(x > _MAX_SPAN for x in s):
        return None
    return tuple(s)


def derive_lattice(widths) -> tuple[int, int]:
    """``(k, m)`` for a topology: the ONE authority, host-side and static.

    The objective is R0 spec section 7.3's, in its order: MAXIMIZE ``BC`` subject to the
    register budget (:data:`LATTICE_CAP_BC`); among the boxes that reach it, take the
    SHALLOWEST atom rectangle (:func:`atom_rectangle`), because an atom spanning two
    levels is only ever filled when both are dense for the side; among those, the largest
    ``k``, since the sub-box span is the carry family's codegen unit.

    That reproduces the built arms -- ``(64, 64)`` and ``(256, 256)`` both give
    ``(4, 8)``, ``BC = 128`` -- and it is stated on ``BC`` rather than on ``m`` because
    ``BC = k^D * m``: at depth four a capacity ceiling of eight is a box of two thousand.

    A topology no box fits takes ``(1, 1)``: every leaf is then its own owner, ``pi`` is
    the identity, and the paged backing -- the only consumer of the atom rectangle --
    refuses it, since an owner box below one atom has no walk unit (:func:`walk_unit`).
    """
    widths = tuple(int(w) for w in widths)
    best = (0, 0, 0, 1, 1)  # (BC, -atom levels, k) ranked; the (1, 1) box is the floor
    kappa = 0
    while (1 << kappa) <= _MAX_SPAN_K:
        k = 1 << kappa
        j = 0
        while (1 << j) <= math.prod(widths):
            m = 1 << j
            s = _admissible(k, m, widths)
            j += 1
            if s is None:
                continue
            whole, partial = atom_rectangle(s)
            rank = (math.prod(s), -(whole + (1 if partial > 1 else 0)), k)
            if rank > best[:3]:
                best = rank + (k, m)
        kappa += 1
    return best[3], best[4]


def permutation(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    THE ORACLE AND TEST SEAM'S VIEW, and nothing else. The oracle's state is canonical
    forever; a plane crosses into lattice order by ``lat.index_copy_(1, pi, canonical)``
    and back by ``canonical = lat.index_select(1, pi)``.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    _, s, g, bc, _owners = box_shape(widths, k, m)
    ell = torch.arange(math.prod(widths), device=device, dtype=torch.int64)
    o_term = torch.zeros_like(ell)
    local = torch.zeros_like(ell)
    for l in range(D):
        d = (ell // math.prod(widths[l + 1:])) % widths[l]
        o_term += (d // s[l]) * math.prod(g[l + 1:])
        #: THE INTRA-BOX ORDER: mixed radix over the owner's run offsets, most
        #: significant level first -- `box.cuh`'s `run_of`.
        local += (d % s[l]) * math.prod(s[l + 1:])
    return o_term * bc + local


def to_lattice(plane: torch.Tensor, widths, k: int, m: int) -> torch.Tensor:
    """A ``[..., N, cols]`` plane in CANONICAL leaf order, re-read in LATTICE order.

    THE ORACLE/TEST SEAM, and the only place a conversion is legal: no shipped path ever
    converts a plane, because every kernel is born in lattice order
    (docs/internals/decode/decode_lattice.md#the-seam).
    """
    n = math.prod(int(w) for w in widths)
    pi = permutation(widths, k, m, device=plane.device)
    flat = plane.reshape(-1, n, plane.shape[-1])
    out = torch.empty_like(flat)
    out.index_copy_(1, pi, flat)
    return out.reshape(plane.shape)


def to_canonical(plane: torch.Tensor, widths, k: int, m: int) -> torch.Tensor:
    """The inverse of :func:`to_lattice` -- a lattice plane as the oracle's view."""
    n = math.prod(int(w) for w in widths)
    pi = permutation(widths, k, m, device=plane.device)
    flat = plane.reshape(-1, n, plane.shape[-1])
    return flat.index_select(1, pi).reshape(plane.shape)
