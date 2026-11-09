# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carry's cells: alternating bundles on the even box, shared by its fold and readout gates.

Symbols: ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import torch

from rola.routing.factors import RouteFactors
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values
from tests.oracle.fixtures import with_gain


def levels(widths, L, live, *, B=1, H=2, seed=0):
    """Each level's rows on the simplex: ``live[l]`` digits a token at random (the rest exact zeros), positive."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for width, k in zip(widths, live):
        x = torch.rand((B, L, H, width), generator=gen) + 0.05
        if k < width:
            keep = torch.rand((B, L, H, width), generator=gen).argsort(-1) < k
            x = x * keep
        out.append((x / x.sum(-1, keepdim=True)).cuda())
    return out


def listed_bundle(widths, L, write_sparse, *, same=False, live=4, d_v=64, seed=0):
    """A bundle each side listing one level, ``live`` digits a token: the write side sparse at ``write_sparse``, the read
    side at the other level (the alternating box) or, ``same``, at the same one (the same split); each side dense
    elsewhere."""
    D = len(widths)
    write_flags = tuple(level == write_sparse for level in range(D))
    read_flags = write_flags if same else tuple(not flag for flag in write_flags)
    live_w = [live if flag else width for flag, width in zip(write_flags, widths)]
    live_r = [live if flag else width for flag, width in zip(read_flags, widths)]
    gen = torch.Generator().manual_seed(seed + 1)
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    routes = RouteFactors.from_levels(topology, levels(widths, L, live_r, seed=seed), with_gain(levels(widths, L, live_w, seed=seed + 7), (torch.rand((1, L, 2), generator=gen) + 0.5).cuda()), sparse=(read_flags, write_flags))
    values = Values.from_projection(torch.randn((1, L, 2, d_v), generator=gen).cuda())
    return routes, values


def read_listed_bundle(widths, L, read_sparse, *, live=4, d_v=64, seed=0):
    """A bundle whose read side lists level ``read_sparse`` (``live`` digits a token) and whose write side is dense at
    every level (every digit live): a dense write side's box, its fold's coefficients the read side's list level."""
    D = len(widths)
    read_flags = tuple(level == read_sparse for level in range(D))
    write_flags = (False,) * D
    live_r = [live if flag else width for flag, width in zip(read_flags, widths)]
    gen = torch.Generator().manual_seed(seed + 1)
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    routes = RouteFactors.from_levels(topology, levels(widths, L, live_r, seed=seed), with_gain(levels(widths, L, list(widths), seed=seed + 7), (torch.rand((1, L, 2), generator=gen) + 0.5).cuda()), sparse=(read_flags, write_flags))
    values = Values.from_projection(torch.randn((1, L, 2, d_v), generator=gen).cuda())
    return routes, values


def dense_bundle(widths, L, *, d_v=64, seed=0):
    """A bundle dense on both sides at every level (every digit live): the box both sides' dense levels, the readout's
    digits its coefficients' level (readout_dense.md)."""
    D = len(widths)
    gen = torch.Generator().manual_seed(seed + 1)
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    routes = RouteFactors.from_levels(topology, levels(widths, L, list(widths), seed=seed), with_gain(levels(widths, L, list(widths), seed=seed + 7), (torch.rand((1, L, 2), generator=gen) + 0.5).cuda()), sparse=((False,) * D, (False,) * D))
    values = Values.from_projection(torch.randn((1, L, 2, d_v), generator=gen).cuda())
    return routes, values


#: THE DENSE WRITE SIDE'S 64-CHANNEL BOX: 16 lists (the read side's list level, its digits) x 32 rows x 64 channels, two
#: lists a warp
def box64_dense(read_sparse: int) -> tuple[tuple[int, int], int]:
    """``(extents, channels)`` of the dense write side's 64-channel box over a bundle listing level ``read_sparse``."""
    return tuple(16 if level == read_sparse else 32 for level in range(2)), 64


#: THE SAME SPLIT'S 64-CHANNEL BOX: both sides listing ``level``, 16 of its digits (the fold's lists, the readout's) x 32
#: rows x 64 channels
def box64_same(level: int) -> tuple[tuple[int, int], int]:
    """``(extents, channels)`` of the same split's 64-channel box over a bundle whose sides both list ``level``."""
    return tuple(16 if lv == level else 32 for lv in range(2)), 64


#: THE 64-CHANNEL BOX (docs/internals/carry/ampere/plan.md#box): the alternating box of 32 write lists x 16 read digits x 64
#: channels at depth two, its extents a level by the write side's sparse level
def box64(write_sparse: int) -> tuple[tuple[int, int], int]:
    """``(extents, channels)`` of the 64-channel box over a bundle whose write side lists level ``write_sparse``."""
    return tuple(32 if level == write_sparse else 16 for level in range(2)), 64


def assert_box64_arm(carve) -> None:
    """The carve is a 64-channel box's: its arm a built one at 64 channels."""
    from rola.ops import carry as carry_ops

    key = carry_ops.arm_key(carve)
    assert key in carry_ops.arms() and key[4] == 64, key


#: THE PLAN'S FORMS, each exercised by a cell -- its widths, live digits a token, the form it forces (None: the rule's)
#: -- and named as `tests/oracle/carry_plan.forms` names them: each image mode, alone and over read chunks, the fold in
#: chunks, an entry filled ahead and one filled at its own step, a window's first filled by the window before, and a
#: fill that waits on the retired read's drain (docs/internals/carry/ampere/schedule.md#planner).
FORM_CELLS = (
    ((64, 64), 4, None, {"halves", "halves-chunks", "ahead", "deferred", "reuse"}),
    ((64, 64), 16, None, {"fold-chunks", "next"}),
    ((256, 256), 2, None, {"whole", "next"}),
    ((256, 256), 4, "whole/64/rule", {"whole-chunks"}),
    ((256, 256), 2, "halves/128/128", {"halves-chunks", "fold-chunks"}),
)
#: the forms at the 64-channel box: its read slots twice the panel's bytes, a forced 128-slot read chunk does not fit
#: beside the halves there (the rule's plan stands), so the halves' chunks at 64 slots
FORM_CELLS64 = FORM_CELLS[:4] + (((256, 256), 2, "halves/64/128", {"halves-chunks", "fold-chunks"}),)
