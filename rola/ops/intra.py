# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The intra kernel's call surface — a side kernel, outside the shipped dispatch.

The kernel REDUCES into ``o`` and ``den`` (``fan_in_add`` semantics) rather than
returning them, so the caller owns their allocation and their initial contents;
:func:`intra_forward` allocates zeros when none are handed in, which is what a
standalone conformance run wants and what a concurrent run with the carry kernel
must never do.
"""

from __future__ import annotations

import torch

from rola.ops._ext import extension
from rola.ops.carry import pack_side
from rola.ops.padding import pad_routes, pad_v, slice_y

#: THE PER-LEVEL MODE, two bits per level: bit 0 the READ side is value-sparse there,
#: bit 1 the WRITE side.  ``BOTH_SPARSE`` says BOTH SIDES ARE SPARSE AT THAT LEVEL AND
#: NOTHING MORE: each side votes its own bitlists from
#: its own factors, so the two supports never have to coincide.  Drawing ONE support for
#: both is a modeling choice with a locality bonus, never a kernel requirement.  The
#: class is what the carry's order-by-mode rule puts OUTERMOST, because it is where the
#: two sides' carves agree and the exchange therefore does not move a leaf.
DENSE_BOTH = 0
READ_SPARSE = 1
WRITE_SPARSE = 2
BOTH_SPARSE = 3

#: the built windows -- the carry family's `W` arms, mirrored.  ``WINDOW`` names the
#: flagship because the architecture doc's two
#: constraints (the C* basin and the intra's ``W/4`` flop/byte intensity) settle
#: it there; ``WINDOW_SMALL`` is the conformance cell's.
WINDOW = 384
WINDOW_WIDE = 512
WINDOW_SMALL = 64
#: the built LEVEL WIDTHS -- an arm is a ``(D, B, W)`` triple, the carry family's
#: topology arms mirrored.  A level narrower than the MMA's contraction step is
#: contracted padded, which costs that level's Gram throughput and nothing else.
LEVEL_WIDTH = 256
LEVEL_WIDTH_DEEP3 = 16
LEVEL_WIDTH_DEEP4 = 16
VALUE_WIDTH = 64


def pack_modes(modes: tuple[int, ...]) -> int:
    """Two bits per level, level 0 in the low pair."""
    packed = 0
    for level, mode in enumerate(modes):
        if mode not in (DENSE_BOTH, READ_SPARSE, WRITE_SPARSE, BOTH_SPARSE):
            raise ValueError(f"level {level}: unknown support mode {mode!r}")
        packed |= mode << (2 * level)
    return packed


#: the producer's support word blocks 32 tokens to an ``int32`` -- ``factor.cu:236-239``,
#: and the routing ABI's ``ROUTE_TILE_SIZE``.  It is the certificate's token grain.
WORD_TOKENS = 32


def support_words(plane) -> torch.Tensor:
    """A packed ``[BH, L, D*B]`` plane's frozen support word, ``[BH, L/32, D*B]`` int32.

    THE SAME QUESTION THE PRODUCER ALREADY ANSWERED, for a caller that DREW its
    planes instead of solving them. The test is exact bf16-storage-boundary
    nonzero -- ``rola/routing/entmax/production.py``'s ``bf16_nonzero``, and the
    kernel's own magnitude test -- so the word this returns is the word an entmax
    solve of the same amplitudes would have emitted, bit for bit.

    A producer-emitted word may be CONSERVATIVE (a set bit is allowed to accompany
    a stored zero -- the ``RouteDescriptor`` contract); that costs a gated MMA
    group whose product is exactly zero and can move no number.
    """
    bh, length, width = plane.shape
    if length % WORD_TOKENS:
        raise ValueError(
            f"the support word blocks {WORD_TOKENS} tokens; L = {length} is not a whole "
            f"number of them")
    live = plane.to(torch.bfloat16) != 0
    words = torch.zeros((bh, length // WORD_TOKENS, width), dtype=torch.int32,
                        device=plane.device)
    for bit in range(WORD_TOKENS):
        words |= live[:, bit::WORD_TOKENS, :].to(torch.int32) << bit
    return words


def pack_support(levels) -> torch.Tensor:
    """:func:`support_words` of ``carry.pack_side(levels)``, without the plane copy.

    Same digit order as the plane it certifies, level by level -- the word's column
    IS the plane's digit, which is what lets the kernel index one from the other.
    """
    batch, tokens, heads, _ = levels[0].shape
    parts = [support_words(level.permute(0, 2, 1, 3).reshape(batch * heads, tokens,
                                                             level.shape[-1]))
             for level in levels]
    return torch.cat(parts, dim=-1)


def intra_forward(pread, pwrite, gwrite, v, sread, swrite, modes, *, o=None, den=None,
                  window=WINDOW):
    """RED the within-window causal intra term into ``(o, den)`` and return them.

    ``pread``/``pwrite`` are ``[BH, L, D*B]`` bf16 operand planes, ``gwrite`` is
    ``[BH, L]`` bf16, ``v`` is token-major ``[B, T, H, 64]`` bf16. ``sread``/``swrite``
    are the frozen support words of the plane of the same name (``[BH, L/32, D*B]``
    int32, :func:`support_words`) -- the zero certificate READS the support rather
    than re-deriving it from the amplitudes. ``modes`` is one per-level support mode,
    and ``window`` names the built arm -- the SAME window the carry call is given,
    since the two kernels share one grid.
    """
    bh, length, width = pread.shape
    levels = len(modes)
    if levels == 0 or width % levels:
        raise ValueError(f"the plane is {width} wide and cannot hold {levels} uniform levels")
    level_width = width // levels
    if o is None:
        o = torch.zeros((bh, length, VALUE_WIDTH), dtype=torch.float32, device=pread.device)
    if den is None:
        den = torch.zeros((bh, length), dtype=torch.float32, device=pread.device)
    extension().intra_forward(pread, pwrite, gwrite, v, sread, swrite, o, den, levels,
                              level_width, pack_modes(modes), int(window))
    return o, den


def _shipped_level_width(D: int, window: int) -> int:
    """The uniform, kernel-facing per-level width this binary ships at ``(D, window)``.

    Read off :func:`arms` (the device's own declared matrix), never a literal
    ``{LEVEL_WIDTH, LEVEL_WIDTH_DEEP3, LEVEL_WIDTH_DEEP4}`` lookup: those constants
    name today's declared arms, and the arm table -- not a mirror of it -- is what a
    call is padded against.
    """
    widths_here = sorted({lw for levels, lw, w, _smem in arms() if levels == D and w == window})
    if not widths_here:
        raise ValueError(
            f"no shipped intra arm at D={D}, window={window} -- there is no uniform "
            "level width this call could be padded to")
    return widths_here[0]


def intra_forward_padded(read_levels, write_levels, gwrite, v, modes, *, window=WINDOW,
                         o=None, den=None):
    """:func:`intra_forward`, taking the caller's LOGICAL per-level tensors and ``v``.

    HOST-SIDE PADDING ("padding at the producer" / "d_v padding is also
    HOST-SIDE"): every level here is padded to the ONE uniform width this arm ships
    (:func:`_shipped_level_width` -- intra's own kernel takes a single ``level_width``
    for every level, a stricter uniformity than carry's per-level one), ``v`` to
    ``VALUE_WIDTH``, and the returned ``o`` is sliced back to the caller's ``d_v``
    before it leaves. The kernel never sees a width other than its own.
    """
    D = len(read_levels)
    level_width = _shipped_level_width(D, window)
    read_levels = pad_routes(read_levels, (level_width,) * D)
    write_levels = pad_routes(write_levels, (level_width,) * D)
    d_v = v.shape[-1]
    v = pad_v(v, VALUE_WIDTH)
    pread = pack_side(read_levels)
    pwrite = pack_side(write_levels)
    sread = pack_support(read_levels)
    swrite = pack_support(write_levels)
    o, den = intra_forward(pread, pwrite, gwrite, v, sread, swrite, modes,
                           o=o, den=den, window=window)
    return slice_y(o, d_v), den


def arms():
    """The built ``(levels, level_width, window, smem_bytes)`` rows."""
    return extension().intra_arms()


def build_stamp() -> int:
    """A fact only the loaded binary can produce — the extension-trap check."""
    return extension().intra_build_stamp()


__all__ = [
    "BOTH_SPARSE",
    "DENSE_BOTH",
    "READ_SPARSE",
    "WINDOW",
    "WINDOW_SMALL",
    "WINDOW_WIDE",
    "WRITE_SPARSE",
    "arms",
    "build_stamp",
    "intra_forward",
    "intra_forward_padded",
    "pack_modes",
    "pack_support",
    "support_words",
]

