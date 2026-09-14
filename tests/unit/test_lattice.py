# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE LEAF ORDER's own arithmetic: `rola.ops.lattice`, on the host, with no kernel.

`pi` maps a CANONICAL leaf index `ell` (MSB-first mixed radix, the oracle's forever) to a
LATTICE index `lambda`. Every plane the oracle and a kernel both touch crosses at that
map, so two properties of the map itself are load-bearing before any comparison means
anything: it must be a BIJECTION -- a map that dropped or doubled a leaf would silently
drop or double a row -- and `to_canonical . to_lattice` must be the identity, which is
what lets a test state its claim in whichever order is natural and convert.

The third cell is a PIN rather than a property: the two built carry arms must derive the
SAME box, because the flagship and the carry family write and read one plane.

Run: ``pytest tests/unit/test_lattice.py``
"""
from __future__ import annotations

import math

import torch

from rola.ops.lattice import (
    box_shape,
    derive_lattice,
    permutation,
    to_canonical,
    to_lattice,
)
from rola.ops.paging import bytes_equal

#: Every shape of box the shipped topologies produce: `k = 1` (no admissible span),
#: `m = 1` (no admissible capacity), `D` from 1 to 4, and the flagship's `(4, 8)`.
_TOPOLOGIES = [
    (16,), (33,), (256,), (8, 16), (7, 9), (16, 16), (64, 64), (256, 256),
    (4, 4, 4), (3, 5, 7), (8, 8, 8), (2, 3, 4, 4), (4, 4, 4, 4),
]


def test_the_permutation_is_a_bijection():
    for widths in _TOPOLOGIES:
        k, m = derive_lattice(widths)
        pi = permutation(widths, k, m, device="cpu")
        n = math.prod(widths)
        assert pi.shape == (n,) and pi.dtype == torch.int64
        assert torch.equal(pi.sort().values, torch.arange(n)), (
            f"pi is not a bijection at widths={widths}, (k, m)=({k}, {m})")


def test_the_two_conversions_are_inverse():
    for widths in _TOPOLOGIES:
        k, m = derive_lattice(widths)
        plane = torch.randn(2, 3, math.prod(widths), 5)
        assert bytes_equal(to_canonical(to_lattice(plane, widths, k, m), widths, k, m),
                           plane), f"the seam is not round-trip exact at widths={widths}"


def test_the_box_is_consistent_with_the_widths():
    for widths in _TOPOLOGIES:
        k, m = derive_lattice(widths)
        m_l, s, g, bc, owners = box_shape(widths, k, m)
        assert math.prod(m_l) == m
        assert bc * owners == math.prod(widths)
        for level, (width, span, extent) in enumerate(zip(widths, s, g, strict=True)):
            assert span * extent == width, f"level {level} of {widths} is not tiled by its span"


def test_the_two_built_carry_arms_derive_one_box():
    """`(64, 64)` and `(256, 256)` both give `(k, m) = (4, 8)`.

    This is what makes the flagship rows agree with the carry family's built arms: the
    two families write and read ONE plane, so a solver that answered differently for the
    two would put the same leaf in two places.
    """
    assert derive_lattice((64, 64)) == (4, 8)
    assert derive_lattice((256, 256)) == (4, 8)
