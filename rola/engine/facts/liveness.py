# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The class-1 liveness words' CONTRACT, and the two class-2 folds.

The facts pass that writes the words is a CUDA kernel this module does not own
(:mod:`rola.ops.facts`). What lives here is the contract that kernel writes and every
consumer reads -- the word layout (:class:`LivenessLayout`, the mirror of
``csrc/rola/src/facts/liveness_contract.cuh``), a reader that turns those words back into
per-token bits, and a pure-torch MODEL of the words and of the folds that is the
executable spec the kernel is graded against (docs/internals/facts/liveness_contract.md).

Vocabulary, defined here and used unqualified below: ``D`` = routing depth;
``B_l`` = the PADDED width of level ``l`` (a power of two at or above 16, the
descriptor's); ``b_l`` = its LOGICAL width, which never leaves the host; ``L`` = tokens
in the call; ``BH`` = batch x heads; ``N = prod_l B_l`` = leaves; atom = 16 consecutive
leaves in canonical order, which is the page granule; a BOX = one warp sub-box, an
aligned rectangle in digit space.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN

#: Tokens are the table's minor axis, 32 to a word: one warp ballot per row.
TOK_BITS = 32

#: A TILE UNION's grain: 64 tokens (two words) of a 32-digit group of one level.
TILE_TOKENS = 64
GROUP_DIGITS = 32

#: The two sides, in table order.
READ, WRITE = 0, 1
SIDES = 2

#: The stored word type. ``uint32_t`` on the device; torch carries it as the
#: same 32 bits signed, exactly as the union table already did.
WORD_DTYPE = torch.int32

#: The page granule, in leaves.
ATOM_LEAVES = 16


def token_words(L: int) -> int:
    return (L + TOK_BITS - 1) // TOK_BITS


def mask_words(bits: int) -> int:
    return (bits + 31) // 32


@dataclass(frozen=True, slots=True)
class LivenessLayout:
    """Where the bit for (side, level, digit, token) lives.

    One row per DIGIT in canonical order: row ``row_base(l) + d`` is level ``l``'s digit
    ``d``, the column a packed plane of the levels would hold it at.
    """

    D: int
    B: tuple[int, ...]
    L: int

    def __post_init__(self) -> None:
        if len(self.B) != self.D:
            raise ValueError(f"a layout names D={self.D} levels and {len(self.B)} widths")
        if self.L < 1:
            raise ValueError(f"a call carries at least one token; got L={self.L}")

    @classmethod
    def of(cls, fmt, L: int) -> LivenessLayout:
        """The layout a state's FORMAT descriptor implies for a call of ``L`` tokens."""
        return cls(D=fmt.D, B=tuple(fmt.B), L=int(L))

    def row_base(self, level: int) -> int:
        return sum(self.B[:level])

    @property
    def rows(self) -> int:
        return sum(self.B)

    def row_of(self, level: int, digit: int) -> int:
        if not 0 <= digit < self.B[level]:
            raise ValueError(f"level {level} carries {self.B[level]} digits; got {digit}")
        return self.row_base(level) + digit

    @property
    def words(self) -> int:
        return token_words(self.L)

    @property
    def side_words(self) -> int:
        return self.rows * self.words

    @property
    def side_bytes(self) -> int:
        return self.side_words * 4

    @property
    def call_words(self) -> int:
        return SIDES * self.side_words

    def word_index(self, bh: int, side: int, row: int, token: int) -> int:
        return (bh * self.call_words + side * self.side_words + row * self.words
                + token // TOK_BITS)

    def token_bit(self, token: int) -> int:
        return 1 << (token % TOK_BITS)

    def group_base(self, level: int) -> int:
        return sum(-(-width // GROUP_DIGITS) for width in self.B[:level])

    @property
    def groups(self) -> int:
        """A side's 32-digit groups: each level's digits in groups of 32, a narrower level one group."""
        return self.group_base(self.D)

    @property
    def tiles(self) -> int:
        return -(-self.L // TILE_TOKENS)

    def union_index(self, bh: int, side: int, group: int, tile: int) -> int:
        return ((bh * SIDES + side) * self.groups + group) * self.tiles + tile

    @property
    def N(self) -> int:
        n = 1
        for width in self.B:
            n *= width
        return n

    @property
    def atoms(self) -> int:
        return self.N // ATOM_LEAVES

    @property
    def radix_strides(self) -> tuple[int, ...]:
        strides, suffix = [], 1
        for width in reversed(self.B):
            strides.append(suffix)
            suffix *= width
        return tuple(reversed(strides))


@dataclass(frozen=True, slots=True)
class SideStatics:
    """One side's static operands: which levels the pass SKIPS, and the digit mask.

    A dense level (an activation with no exact zeros) is not read; its row is the WIDTH
    MASK instead. That is exact up to a sound over-report: only a pad digit's logit is
    -inf, and a digit whose value rounded to a zero digit is reported live, which costs
    work and never a contribution. The mask arrives as bits, never as a logical width: a
    width field in a kernel entry signature is what R13's two contracts forbid, and the
    seam is the only translator.
    """

    dense_levels: tuple[bool, ...]
    digit_mask: tuple[bool, ...]

    def mask_word(self, index: int) -> int:
        word = 0
        for bit in range(32):
            row = index * 32 + bit
            if row < len(self.digit_mask) and self.digit_mask[row]:
                word |= 1 << bit
        return word


def side_statics(layout: LivenessLayout, dense_levels, logical_widths) -> SideStatics:
    """THE SEAM'S TRANSLATION: modes and logical widths in, bits out.

    ``dense_levels[l]`` is true where the level's activation cannot produce an exact
    zero (softmax); ``logical_widths[l]`` is ``b_l``, and the digits at or above it are
    the producer's -inf pad.
    """
    dense = tuple(bool(flag) for flag in dense_levels)
    logical = tuple(int(width) for width in logical_widths)
    if len(dense) != layout.D or len(logical) != layout.D:
        raise ValueError(f"a side names {layout.D} levels; got {len(dense)} modes "
                         f"and {len(logical)} logical widths")
    for level, (b, B) in enumerate(zip(logical, layout.B)):
        if not 1 <= b <= B:
            raise ValueError(f"level {level} pads b_l={b} into B_l={B}")
    mask = []
    for level in range(layout.D):
        mask += [digit < logical[level] for digit in range(layout.B[level])]
    return SideStatics(dense_levels=dense, digit_mask=tuple(mask))


def _live_rows(levels, layout: LivenessLayout) -> torch.Tensor:
    """`[BH, rows, L]` bool: each level's per-digit nonzero test, its ``[B, T, H, w_l]`` rows in any dtype (a u8
    digit or a drawn float); a digit at or past ``w_l`` is dead."""
    if len(levels) != layout.D:
        raise ValueError(f"a side holds {layout.D} levels' rows, got {len(levels)}")
    B, T, H = levels[0].shape[:3]
    if T != layout.L:
        raise ValueError(f"the rows carry {T} tokens; the layout {layout.L}")
    live = torch.zeros((B * H, layout.rows, layout.L), dtype=torch.bool, device=levels[0].device)
    for level, rows in enumerate(levels):
        width = rows.shape[-1]
        if tuple(rows.shape[:3]) != (B, T, H) or width > layout.B[level]:
            raise ValueError(f"level {level}'s rows are {tuple(rows.shape)}; [{B}, {T}, {H}, <= {layout.B[level]}]")
        base = layout.row_base(level)
        live[:, base:base + width] = (rows != 0).permute(0, 2, 3, 1).reshape(B * H, width, T)
    return live


def _pack(bits: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """`[..., L]` bool -> `[..., words]` words, token `t` at bit `t % 32`."""
    pad = layout.words * TOK_BITS - layout.L
    if pad:
        bits = torch.cat([bits, bits.new_zeros((*bits.shape[:-1], pad))], dim=-1)
    lanes = bits.reshape(*bits.shape[:-1], layout.words, TOK_BITS).to(torch.int64)
    weights = (1 << torch.arange(TOK_BITS, device=bits.device, dtype=torch.int64))
    words = (lanes * weights).sum(dim=-1)
    return (words - (words >= 1 << 31).to(torch.int64) * (1 << 32)).to(WORD_DTYPE)


def liveness_words(read_rows, write_rows, layout: LivenessLayout, read_statics: SideStatics,
                   write_statics: SideStatics) -> torch.Tensor:
    """THE MODEL OF THE PASS: `[B H_w, g + 1, rows, words]` words from each side's per-level rows (read
    ``[B, T, H, w_l]``, write ``[B, T, H_w, w_l]``, ``g = H / H_w``; batch-major), the executable spec: slots
    ``0 .. g - 1`` a write head's query heads' read side, slot ``g`` its write side (``[BH, 2, ...]`` ungrouped).

    A sparse level's digit is live where its row is nonzero and the digit mask names it; a
    dense level's row is the static width mask, broadcast over every token and costing no
    read. A token past ``L`` votes dead, so the tail word's high bits are zero and a fold
    may OR whole words without masking.
    """
    sides = []
    for levels, statics in ((read_rows, read_statics), (write_rows, write_statics)):
        live = _live_rows(levels, layout)
        mask = torch.tensor(statics.digit_mask, dtype=torch.bool, device=live.device)
        for level, dense in enumerate(statics.dense_levels):
            if dense:
                base = layout.row_base(level)
                live[:, base:base + layout.B[level], :] = True
        sides.append(_pack(live & mask[None, :, None], layout))
    write = sides[1]
    read = sides[0].reshape(write.shape[0], -1, *write.shape[1:])
    return torch.cat((read, write[:, None]), dim=1)


def tile_unions(words: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """THE MODEL OF THE TILE UNIONS: `[BH, slots, rows, words]` words -> `[BH, slots, groups, tiles]` int32, bit ``i`` of
    (group ``g`` of level ``l``, tile ``t``) set when digit ``32 g + i`` of level ``l`` is live at a token in
    ``[64 t, 64 t + 64)``: the OR of the tile's two words, a row a bit."""
    BH, slots = words.shape[:2]
    pad = 2 * layout.tiles - layout.words
    if pad:
        words = torch.cat([words, words.new_zeros((*words.shape[:-1], pad))], dim=-1)
    live = (words.reshape(BH, slots, layout.rows, layout.tiles, 2) != 0).any(-1)
    groups = []
    for level, width in enumerate(layout.B):
        base = layout.row_base(level)
        rows = live[:, :, base:base + width]
        rows = torch.cat([rows, rows.new_zeros((BH, slots, -width % GROUP_DIGITS, layout.tiles))], dim=2)
        groups.append(rows.reshape(BH, slots, -1, GROUP_DIGITS, layout.tiles))
    bits = torch.cat(groups, dim=2).to(torch.int64)
    weights = (1 << torch.arange(GROUP_DIGITS, device=words.device, dtype=torch.int64))[:, None]
    unions = (bits * weights).sum(dim=3)
    return (unions - (unions >= 1 << 31).to(torch.int64) * (1 << 32)).to(WORD_DTYPE)


def unpack(words: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """THE READER: `[BH, 2, rows, words]` words -> `[BH, 2, rows, L]` bool."""
    shifts = torch.arange(TOK_BITS, device=words.device, dtype=torch.int32)
    bits = (words.unsqueeze(-1) >> shifts) & 1
    flat = bits.reshape(*words.shape[:-1], layout.words * TOK_BITS)
    return flat[..., :layout.L].bool()


@dataclass(frozen=True, slots=True)
class Box:
    """One warp sub-box: an aligned rectangle in digit space, per level a run.

    The geometry block derives these; the folds take them as data, so nothing here is
    a second derivation of the carve.
    """

    start: tuple[int, ...]
    span: tuple[int, ...]

    def check(self, layout: LivenessLayout) -> None:
        if len(self.start) != layout.D or len(self.span) != layout.D:
            raise ValueError(f"a box names {layout.D} runs; got {len(self.start)}/{len(self.span)}")
        for level, (start, span) in enumerate(zip(self.start, self.span)):
            width = layout.B[level]
            if span < 1 or span > width or width % span or start % span or start + span > width:
                raise ValueError(f"level {level}: the run [{start}, {start + span}) is not an "
                                 f"aligned divisor run of B_l={width}")

    def leaves(self, layout: LivenessLayout) -> torch.Tensor:
        """The canonical leaf ids this box owns, as a flat tensor."""
        self.check(layout)
        strides = layout.radix_strides
        leaves = torch.zeros(1, dtype=torch.int64)
        for level in range(layout.D):
            digits = torch.arange(self.start[level], self.start[level] + self.span[level],
                                  dtype=torch.int64)
            leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
        return leaves

    def pages(self, layout: LivenessLayout) -> torch.Tensor:
        """THE BOX -> PAGE MAP: the atoms this box's leaves fall in, deduplicated."""
        return torch.unique(self.leaves(layout) // ATOM_LEAVES)


def _live_token_words(layout: LivenessLayout, device) -> torch.Tensor:
    """`[words]` int32: every live token's bit, the tail word's high bits zero -- a dense level's row in the table."""
    return _pack(torch.ones(layout.L, dtype=torch.bool, device=device), layout)


def box_activity(words: torch.Tensor, layout: LivenessLayout, boxes, side: int, statics: SideStatics) -> torch.Tensor:
    """CLASS 2, THE PRIMARY BITS: `[BH, len(boxes)]` bool, per box over the WHOLE CALL.

    A box is live for a token when EVERY level has a live digit inside that level's
    run -- a token's live leaf set is the product of its per-level live digit sets, so
    the AND over levels is exact and never a relaxation -- and the call-level bit is
    the OR of that over the call's tokens. Nothing is kept per window: a window with
    nothing live for a box is a schedule fact, read off these rows while walking. A
    DENSE level's rows are its width mask over the live tokens, taken from the side's
    ``statics`` and never read from ``words`` (the facts pass does not write them).
    """
    table = words[:, side]
    live = _live_token_words(layout, words.device)
    out = []
    for box in boxes:
        box.check(layout)
        clause = None
        for level in range(layout.D):
            base = layout.row_base(level) + box.start[level]
            if statics.dense_levels[level]:
                named = any(statics.digit_mask[base:base + box.span[level]])
                run = (live if named else torch.zeros_like(live)).expand(table.shape[0], -1)
            else:
                run = _or_rows(table, base, box.span[level])
            clause = run if clause is None else clause & run
        out.append((clause != 0).any(dim=-1))
    if not out:
        return torch.zeros((words.shape[0], 0), dtype=torch.bool, device=words.device)
    return torch.stack(out, dim=1)


def _or_rows(table: torch.Tensor, base: int, span: int) -> torch.Tensor:
    run = table[:, base, :]
    for row in range(base + 1, base + span):
        run = run | table[:, row, :]
    return run


def page_activity(layout: LivenessLayout, read_boxes, read_bits, write_boxes,
                  write_bits) -> torch.Tensor:
    """CLASS 2, THE DERIVED BITS: `[BH, N/16]` uint8, the HOST's OR over boxes.

    One READ and one WRITTEN bit per page, in the byte
    ``rola.engine.facts.planes``'s two constants already name: the page is READ if any
    read-side box that intersects it is read, WRITTEN if any write-side box that
    intersects it is written. ``plan_exact`` allocates on the WRITTEN half, residency
    checks read the READ half, and a decode step consumes this set because its unit is
    the page.
    """
    BH = read_bits.shape[0]
    bits = torch.zeros((BH, layout.atoms), dtype=torch.uint8, device=read_bits.device)
    for boxes, live, flag in ((read_boxes, read_bits, ATOM_READ),
                              (write_boxes, write_bits, ATOM_WRITTEN)):
        for index, box in enumerate(boxes):
            pages = box.pages(layout).to(bits.device)
            hit = live[:, index, None].expand(-1, pages.numel())
            bits[:, pages] |= hit.to(torch.uint8) * flag
    return bits


def atom_spans(layout: LivenessLayout) -> tuple[int, ...]:
    """THE ATOM AS A BOX: the per-level spans of the trailing 16 canonical leaves.

    A page atom is 16 consecutive leaves in canonical order, and canonical order is the
    radix over the levels with the last one minor, so an atom IS an aligned rectangle:
    the last levels whole until their product reaches 16, and an aligned run in the
    level that completes it. That is what makes the exact per-page fold a special case
    of :func:`page_activity` rather than a second derivation.
    """
    spans, remaining = [1] * layout.D, ATOM_LEAVES
    for level in reversed(range(layout.D)):
        span = min(layout.B[level], remaining)
        if remaining % span or layout.B[level] % span:
            raise ValueError(f"level {level}'s width {layout.B[level]} does not tile the "
                             f"{remaining} leaves the page atom still needs")
        spans[level], remaining = span, remaining // span
        if remaining == 1:
            break
    if remaining != 1:
        raise ValueError(f"the topology holds {layout.N} leaves, fewer than one page atom")
    return tuple(spans)


def atom_boxes(layout: LivenessLayout) -> list[Box]:
    """Every atom as a :class:`Box`, in canonical PAGE order -- the finest box set.

    The box->page map is a bijection here, so :func:`page_activity` over these boxes is
    the exact per-page answer rather than a box-grain over-report.
    """
    spans = atom_spans(layout)
    grid = [[]]
    for level in range(layout.D):
        grid = [row + [digit] for row in grid
                for digit in range(0, layout.B[level], spans[level])]
    return [Box(start=tuple(row), span=spans) for row in grid]


def atom_activity(words: torch.Tensor, layout: LivenessLayout, side: int, statics: SideStatics) -> torch.Tensor:
    """`[BH, N/16]` bool: :func:`box_activity` at the ATOM box set, every box at once, of the words' slot ``side``
    (``READ``, ``WRITE`` ungrouped; a query head's or the write side's grouped).

    The same law -- the OR over tokens of the AND over levels of the OR over the box's
    digit run -- folded as one outer product over the levels, which the atom set admits
    because it is the whole aligned partition and its canonical order is exactly the
    order that product builds. A dense level is its width mask from ``statics``.
    """
    spans = atom_spans(layout)
    bits = unpack(words, layout)[:, side]
    BH = bits.shape[0]
    mask = torch.tensor(statics.digit_mask, dtype=torch.bool, device=bits.device)
    clause = None
    for level in range(layout.D):
        base, width, span = layout.row_base(level), layout.B[level], spans[level]
        if statics.dense_levels[level]:
            named = mask[base:base + width].reshape(width // span, span).any(dim=1)
            run = named[None, :, None].expand(BH, -1, layout.L)
        else:
            run = bits[:, base:base + width, :].reshape(BH, width // span, span, layout.L).any(dim=2)
        clause = run if clause is None else (
            clause[:, :, None, :] & run[:, None, :, :]).reshape(BH, -1, layout.L)
    return clause.any(dim=-1)


def page_bits(words: torch.Tensor, layout: LivenessLayout, read_statics: SideStatics,
              write_statics: SideStatics) -> torch.Tensor:
    """`[BH, N/16]` uint8: the EXACT activity byte, `ATOM_READ | ATOM_WRITTEN`, a write head's (its state's): read
    by any of its query heads (slots ``0 .. g - 1``), written by its write side (slot ``g``).

    :func:`page_activity` is this fold at an arbitrary carve, where a box straddling a
    page makes the answer a sound over-report; at the atom box set the map is a
    bijection and the byte is exact, which is what `plan_exact` allocates on and what a
    residency check reads.
    """
    heads = words.shape[1] - 1
    read = atom_activity(words, layout, READ, read_statics)
    for slot in range(1, heads):
        read = read | atom_activity(words, layout, slot, read_statics)
    write = atom_activity(words, layout, heads, write_statics).to(torch.uint8) * ATOM_WRITTEN
    return read.to(torch.uint8) * ATOM_READ | write
