# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTRA, as a caller reaches it: every pair inside one window, over the bundle, reduced into the output planes.

One launch over the bundle's rows and values as stored (docs/internals/intra/intra_kernel.md): each level's Gram as f16
MMAs -- a sparse level's u8 digits widened in registers, a dense level's f16 rows as they are -- the epilogue by the
facts pass's writer factors ``fm``, the value product in f16 weights against the f16 values, and the rows REDUCED into
the caller's ``num`` ``[BH, L, 64]`` and ``den`` ``[BH, L]`` fp32 -- the planes the carry reduces into beside it, zeroed
by their owner. The reader's per-token factor is not applied: it cancels in ``num / den`` (the output contract, carry
and intra alike). The slab gate reads the facts pass's tile unions (`rola.ops.facts`), and a side's dense levels -- the
same statics that pass took -- are never gated. The MMAs accumulate by the device's rule, the carry readout's
(:func:`rola.ops.carry.readout_accumulate`): fp32 where it runs at the fp16 rate, fp16 on GeForce.

The per-level support modes are the carry's geometry input (its carve order reads them), kept here until the carry's
port (docs/internals/intra/intra_api.md).
"""
from __future__ import annotations

import torch

from rola.engine.facts.liveness import LivenessLayout, SideStatics
from rola.ops._ext import device_index, extension

#: The value width the kernel reads.
VALUE_WIDTH = 64
#: THE ROWS' LEVELS: the built arms serve depth <= 2, each level a whole number of 16-digit granules up to 256.
MAX_LEVELS = 2
WIDTH_GRANULE = 16
MAX_LEVEL_WIDTH = 256
#: A LEVEL'S FORM, the kernel's: bit 0 its read rows f16, bit 1 its write rows f16; the widest footprint is f16 on both.
FORM_F16_F16 = 3
#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `IntraPhase` order, booked over every CTA; a card
#: whose block has fewer warps books its own rows and leaves the rest.
LEDGER_PHASES = ("reader", "gate", "wait", "gram", "epilogue", "value", "deposit")
LEDGER_WARPS = 8
_ACCUMULATE = {"fp32": 0, "fp16": 1}

__all__ = ["FORM_F16_F16", "LEDGER_PHASES", "LEDGER_WARPS", "MAX_LEVELS", "MAX_LEVEL_WIDTH", "VALUE_WIDTH",
           "WIDTH_GRANULE", "admits", "block_rows", "build_stamp", "footprint", "intra_forward", "ledger_bind", "reads"]


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the gate answers all-ones on."""
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def intra_forward(routes, values, facts, layout: LivenessLayout, read_statics: SideStatics,
                  write_statics: SideStatics, num: torch.Tensor, den: torch.Tensor, *,
                  accumulate: str | None = None) -> None:
    """REDUCE the within-window pairs of the bundle (``routes``, ``values``) into ``num`` and ``den``.

    ``facts`` is the facts pass's output over this bundle in ``layout`` (its tile unions and writer factors), the
    statics the ones it was made with. Each level's rows are read at their own width (:func:`reads`), at most
    ``layout``'s, in their stored dtype; a partial last window reduces its pairs up to the call's end. ``accumulate``
    (``"fp32"`` or ``"fp16"``) is reserved: the gates' choice of form, the device's rule by default.
    """
    form = _ACCUMULATE[accumulate] if accumulate else -1
    extension().rola_intra_forward(
        list(routes.amp_read), list(routes.amp_write), values.v, facts.fm, facts.unions, num, den, list(layout.B),
        _dense_bits(read_statics), _dense_bits(write_statics), form)


def footprint(widths) -> int:
    """The kernel's shared-memory bytes at these level widths on the current device's card in the widest form, f16 on
    both sides -- for a gate reading the law (the seam refuses widths it does not read)."""
    forms = [FORM_F16_F16] * len(widths)
    return extension().intra_footprint([int(width) for width in widths], forms, torch.cuda.current_device())


def block_rows(device) -> int:
    """The readers a CTA holds resident on ``device``'s card."""
    return extension().intra_block_rows(device_index(device))


def reads(widths) -> bool:
    """Whether the kernel reads levels of these widths: one or two, each a whole number of `WIDTH_GRANULE`-digit
    granules up to `MAX_LEVEL_WIDTH` (no device asked)."""
    widths = tuple(int(width) for width in widths)
    return 1 <= len(widths) <= MAX_LEVELS and all(
        width % WIDTH_GRANULE == 0 and WIDTH_GRANULE <= width <= MAX_LEVEL_WIDTH for width in widths)


def admits(widths, d_v: int) -> bool:
    """Whether the intra runs a call at these level widths and value width: widths it :func:`reads` and ``d_v`` the
    `VALUE_WIDTH`. Every width it reads fits every card's shared memory in every form: each build's walk is sized for the
    widest rows (`Walk<B>`, asserted at compile time; docs/internals/intra/intra.md#block)."""
    return d_v == VALUE_WIDTH and reads(widths)


def build_stamp() -> int:
    """A fact only the loaded binary can produce -- the extension-trap check."""
    return extension().intra_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/intra/intra_kernel.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[8, 7]`` int64 cycles, or ``None``."""
    extension().rola_intra_ledger_bind(ledger)
