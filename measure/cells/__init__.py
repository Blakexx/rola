# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S READING OF THE CENTRAL CELLS: what the carry kernel is given for a cell of rola-devtools' registry.

The cells are rola-devtools' (`rola_devtools.cells`, `carry.json`): every package's tests and measurements read one set
of inputs with one draw, and a cell names nothing a package runs it with. This module is rola's side of that line --
the pieces of a carry call that are the KERNEL's, derived from a cell:

- the state format descriptor of its shape (`descriptor`);
- the launch shape (`launch`): `WARPS_PER_CTA`, the design point of one CTA per SM;
- the carry arm the two together key (`arm_key`, `(D, DV, warps_per_cta)`);
- the mode word packed from the sparsity the cell's draw declares (`level_modes`);
- the liveness words its draw implies (`liveness_words`) and the conservative activity bytes (`conservative_activity`);
- the state a call binds from the cell's `state` and `backing` (`state_binding`);
- the whole call (`carry_call`).

`layer.py` is the other half: RoLA's layer constructions (routing widths, alpha, logit gain), each named with the
central layer inputs it was declared for.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit count, ``N = prod_l B_l`` = leaf
capacity, ``DV`` = the padded value width, ``L`` = tokens, ``BH`` = batch times heads.
"""
from __future__ import annotations

from rola_devtools.cells import build, central
from rola_devtools.cells.carry import DEPOSIT_TOKEN, CarryCell, RealizedCell, realize

#: THE LAUNCH SHAPE a carry call takes when its caller names none: one CTA per SM, eight warps (the shipped set's
#: `warps_per_cta`, `tools/manifests/shipped_set.json`).
WARPS_PER_CTA = 8


def by_name(name: str):
    """A central cell's typed record: a `CarryCell`, a `LayerCell` or a `QKVCell`."""
    return build(central().cell(name))


def carry_cells(*, tier: str | None = None) -> tuple[CarryCell, ...]:
    """Every central carry cell; ``tier`` keeps the cells sized for that reader (`oracle`, `probe`; `both` is either)."""
    cells = tuple(c for c in map(build, central().cells.values()) if isinstance(c, CarryCell))
    return cells if tier is None else tuple(c for c in cells if c.tier in (tier, "both"))


def descriptor(cell: CarryCell, BH: int = 1):
    """The cell's state format descriptor -- the one shape every entry reads."""
    from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat

    return StateFormat(D=cell.D, B=tuple(cell.widths), order=CANONICAL_ORDER, page_bits=PAGE_RECTANGLE_BITS, DV=cell.dv,
                       dtype=SPLIT_PLANE_DTYPE, ids=BH * (cell.N >> PAGE_RECTANGLE_BITS))


def launch():
    from rola.ops.carry import LaunchShape

    return LaunchShape(warps_per_cta=WARPS_PER_CTA)


def arm_key(cell: CarryCell) -> tuple[int, int, int]:
    """The carry arm a call on this cell launches: ``(D, DV, warps_per_cta)``, the key a binary's arms are listed by
    (`rola.ops.carry.arms()`) and the shipped set is declared in."""
    return (cell.D, cell.dv, WARPS_PER_CTA)


def level_modes(cell: CarryCell) -> int:
    """The cell's declared sparsity in the kernel's mode word: two bits a level, bit 0 the read side sparse there, bit 1
    the write side. The carve order is derived from it, so a cell that declares nothing runs the canonical carve."""
    from rola.ops.carry import SIDE_SPARSE

    modes = 0
    for level, (read, write) in enumerate(cell.declared_sparsity()):
        modes |= ((SIDE_SPARSE[0] if read else 0) | (SIDE_SPARSE[1] if write else 0)) << (2 * level)
    return modes


def liveness_words(realized: RealizedCell, descriptor):
    """THE CELL'S OWN STATEMENT of its draw's support, as the class-1 words.

    A cell knows its support because it drew it, so the words come from the contract's pure-torch MODEL and never from
    the pass: the device pass reads the amplitudes itself and the two legs are then a cross-check, the only shape in
    which two derivations of one fact may exist. The LAYOUT is the one contract both legs share:
    ``[BH, 2, sum_l B_l, ceil(L / 32)]`` int32.
    """
    from rola.engine.facts import liveness as lv
    from rola.ops.carry import LivenessWords, pack_side

    read, write = pack_side(realized.read), pack_side(realized.write)
    layout = lv.LivenessLayout(D=len(descriptor.B), B=tuple(descriptor.B), L=read.shape[-2])
    statics = lv.side_statics(layout, (False,) * layout.D, layout.B)
    return LivenessWords(words=lv.liveness_words(read, write, layout, statics, statics))


def conservative_activity(descriptor, BH: int, device: str):
    """Every page READ and WRITTEN -- the truth a caller with no facts pass may state. It asserts nothing about the draw
    and therefore skips nothing."""
    import torch

    from rola.ops.carry import ACTIVITY_READ, ACTIVITY_WRITTEN, PAGE_LEAVES

    pages = descriptor.N // PAGE_LEAVES
    return torch.full((BH, pages), ACTIVITY_READ | ACTIVITY_WRITTEN, dtype=torch.uint8, device=device)


def carry_call(cell: CarryCell, bh: int):
    """``(drawn, kwargs)`` -- one cell's WHOLE carry call over ``bh`` sequences, built once from the cell.

    The kwargs are exactly `rola.ops.carry.carry_forward`'s keyword operands, plus the two positional ones under
    ``routes`` and ``v``; the state planes and the page table are the caller's (`state_binding` binds the cell's own).
    THIS IS THE ONE BINDING: the oracle tier and the benches both reach the kernel through it, so a cell's operands are
    the same tensors in a correctness run and in a measured one.
    """
    from rola.ops import carry as carry_ops

    drawn = realize(cell)
    desc = descriptor(cell, BH=bh)
    shape = launch()
    return drawn, dict(
        routes=carry_ops.route_planes(drawn.read, drawn.write, drawn.gain),
        v=drawn.v.permute(0, 2, 1, 3).reshape(bh, cell.tokens, cell.dv).contiguous(),
        descriptor=desc,
        geometry=carry_ops.geometry_block(desc, shape, level_modes=level_modes(cell)),
        liveness=liveness_words(drawn, desc),
        activity=conservative_activity(desc, bh, "cuda"),
        launch=shape)


def state_binding(cell: CarryCell, drawn: RealizedCell, desc, bh: int):
    """``(state_in, state_out, page_table)`` the cell's `state` and `backing` bind, allocated once, outside any timed call:
    `none` binds no plane, `fresh` a plane out, `carried` its drawn entry state in, advanced in place. The `dense`
    backing hands the whole plane; the `paged` backing hands the PAGE POOL and a slot table permuted from the cell's
    seed, which is the shape a paged caller's pre-analysis commits (`docs/internals/state.md`)."""
    import torch

    from rola.ops import carry as carry_ops
    from rola.ops.paging import to_split_planes

    if cell.state == "none":
        return None, None, None
    if cell.state == "carried":
        plane = to_split_planes(drawn.entry.float().reshape(bh, -1, 16, drawn.entry.shape[-1]))
        state_in = plane
    else:
        plane, state_in = carry_ops.state_plane(desc, bh), None
    if cell.backing == "dense":
        return state_in, plane, None
    #: THE PAGED BINDING: the state becomes the POOL a slot table indexes (`rola.ops.carry._refuse_state`), one page an
    #: atom of every stream -- this cell commits every page, because its call states the conservative activity -- and the
    #: table permutes each stream's pages inside its own range, from the cell's seed, so the mapping a paged call pays
    #: for is exercised rather than an identity.
    pages = desc.N // carry_ops.PAGE_LEAVES
    gen = torch.Generator(device="cuda").manual_seed(cell.seed)
    slots = torch.stack([torch.argsort(torch.rand(pages, device="cuda", generator=gen)) + stream * pages
                         for stream in range(bh)])
    pool = plane.reshape(bh * pages, carry_ops.PAGE_LEAVES, desc.cols)
    return (None if state_in is None else pool), pool, slots.to(torch.int32)


__all__ = ["DEPOSIT_TOKEN", "WARPS_PER_CTA", "CarryCell", "RealizedCell", "arm_key", "by_name", "carry_call", "carry_cells",
           "conservative_activity", "descriptor", "launch", "level_modes", "liveness_words", "realize", "state_binding"]
