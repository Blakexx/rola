# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ADDRESSING BLOCK, read back from the extension: `geometry` and `sub_boxes` wrap `csrc/rola/src/common/geom.cu`'s
`geometry` and sub-box set (docs/internals/common/geom_api.md), the one reader of that positional vector."""
from __future__ import annotations

from rola.ops._ext import extension

__all__ = ["geometry", "sub_boxes"]

#: THE ADDRESSING BLOCK'S FIELD ORDER, mirrored from `csrc/rola/src/common/geom.cu`'s
#: `geometry` -- the ONE reader of that positional vector.
_GEOM_MAX_LEVELS = 4
_GEOM_HEAD = ("D", "bc")
_GEOM_LEVEL = ("width", "span", "grid", "col_base", "run_shift", "owner_div", "weight")
_GEOM_SCALAR = ("wtot", "owners", "leaves", "local_bits", "min_width", "identity_map")
_SIDE_LEVEL = ("level_at", "rank", "row_span", "grid", "grid_div", "row_g", "row_prefix",
               "sub_bits", "sub_shift")
_SIDE_SCALAR = ("carves", "rows", "leaf", "streams", "assign", "interleave")


def geometry(widths,
             level_modes=0,  #: reserved: the declaration's mode word, set by the per-declaration geometry tests
             bc=128,         #: reserved: the owner box's leaves, set by the tall-owner tests
             nsr=1,          #: reserved: the read side's stream count, set by the exchange-law tests
             nsw=1,          #: reserved: the write side's stream count, set by the exchange-law tests
             d_v=64,         #: reserved: the value width, set once the carry derives through this block
             interleave=0,   #: reserved: the int8 carry derives interleaved once its body ships
             atom_floor=1):  #: reserved: the int8 carry derives without the warp-level page floor
    """THE SHIPPED DERIVATION, read back as a dict of dicts.

    ``{scalars..., "r": {side fields}, "w": {...}, "xch": {"w": [...], "r": [...]}}`` --
    every per-level field is a list of length ``D``, the padding levels the block carries
    past ``D`` dropped here because they are a C-array convenience and not part of the
    law. ``xch`` is each side's transit row map over the whole region, ``[stream][slot]``.
    ``interleave`` = 1 places each carved level's stream index in the digit's LOW bits
    (stream ``w`` owns digits ``w, w + NS, ...``), 0 in its high bits (a run); see
    ``docs/internals/common/geom.md#interleaved-placement``. ``atom_floor`` = 1 keeps the
    shipped body's warp-level page floor on the canonical innermost level; 0 lets a side
    carve it (the tall form, ``geom.md#tall-owner-set``).
    """
    flat = list(extension().carry_geometry(list(widths), int(level_modes), int(bc), int(nsr),
                                           int(nsw), int(d_v), int(interleave), int(atom_floor)))
    it = iter(range(len(flat)))

    def take(n):
        return [flat[next(it)] for _ in range(n)]

    out = {}
    for name in _GEOM_HEAD:
        out[name] = take(1)[0]
    D = out["D"]
    for name in _GEOM_LEVEL:
        out[name] = take(_GEOM_MAX_LEVELS)[:D]
    for name in _GEOM_SCALAR:
        out[name] = take(1)[0]
    for side in ("r", "w"):
        block = {}
        for name in _SIDE_LEVEL:
            block[name] = take(_GEOM_MAX_LEVELS)[:D]
        for name in _SIDE_SCALAR:
            block[name] = take(1)[0]
        out[side] = block
    xch = {}
    for side in ("w", "r"):
        rows = out[side]["streams"], out[side]["leaf"]
        xch[side] = [take(rows[1]) for _ in range(rows[0])]
    out["xch"] = xch
    assert len(list(it)) == 0, "the addressing block's flattened field order drifted"
    return out


def sub_boxes(depth, box_leaves_, workers,
              atom_floor=1):  #: reserved: the int8 carry's set is generated without the page floor
    """The GENERATED set of warp sub-box shapes, as spans -- `geom_api.md#sub-box-set`."""
    return [tuple(row) for row in
            extension().carry_sub_boxes(int(depth), int(box_leaves_), int(workers), int(atom_floor))]
