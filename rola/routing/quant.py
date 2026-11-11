# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTERNAL FORMAT: scaled blocks in their storage forms, and the straight-through view.

A SCALED BLOCK is unit-max rows in a storage FORM and each row's max as its fp32 scale (docs/CONSTITUTION.md#r23): a
routing level's rows on either side, a token's values. A :class:`Form` states the storage type and whether an entry may
be negative, and quantizing into it and dequantizing out of it are written once, on it. The two forms: ``ROWS`` (u8,
the row max at 255, a zero a zero -- every routing level, dense or sparse) and ``VALUES`` (bf16 in ``[-1, 1]`` -- the
values). A routing form is nonnegative, because the den, the liveness words and the decay's shares need it, and
:meth:`Form.quantize` refuses an entry its form does not hold. ``straight_through`` hands a consumer the DEQUANTIZED
values with an identity gradient, which is how the format trains (docs/api.md section 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "AMP_LEVELS",
    "ROWS",
    "ROW_ALIGN",
    "ROW_DTYPE",
    "VALUES",
    "Form",
    "aligned_rows",
    "refuse_unheld",
    "row_slot",
    "straight_through",
]

#: u8 coefficient levels: the row max at 255.
AMP_LEVELS = 255.0
#: the stored coefficient form: u8 at every depth.
ROW_DTYPE = torch.uint8
#: THE ROW ALIGNMENT LAW: every level's rows start on, and step batch, token and head by, multiples of this many bytes
#: (docs/internals/facts/liveness_contract.md#alignment).
ROW_ALIGN = 32
_TINY = 1e-30


def row_slot(width: int) -> int:
    """A level's bytes in a packed row: its width in bytes rounded up to the alignment law's grain."""
    return -(-width // ROW_ALIGN) * ROW_ALIGN


def aligned_rows(B: int, T: int, H: int, width: int, device, dtype: torch.dtype = ROW_DTYPE) -> torch.Tensor:
    """``[B, T, H, width]`` zeros of ``dtype`` under the alignment law: a view of storage whose rows are
    ``row_slot`` of their bytes."""
    size = torch.empty((), dtype=dtype).element_size()
    return torch.zeros((B, T, H, row_slot(width * size) // size), dtype=dtype, device=device)[..., :width]


@dataclass(frozen=True, slots=True)
class Form:
    """A SCALED BLOCK'S STORAGE FORM: the stored type, whether an entry may be negative, the stored value of a row's max
    (``top``) and whether its rows lie under the row alignment law (the routing form: the facts pass reads them in
    32-byte grains)."""

    name: str
    dtype: torch.dtype
    signed: bool
    top: float
    aligned: bool

    def unheld(self, x: torch.Tensor) -> torch.Tensor:
        """Whether ``x`` has an entry this form does not hold -- a negative one in a nonnegative form, a non-finite one
        in any -- as a 0-dim bool on ``x``'s device, read by no one yet (:func:`refuse_unheld` reads it)."""
        unfinite = ~torch.isfinite(x).all()
        return unfinite if self.signed else unfinite | (x < 0).any()

    def quantize(self, x: torch.Tensor, *, checked: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows [..., w], scale fp32 [...])``: each row against its own max, rounded once into the form (u8 to the
        nearest digit, bf16 to the nearest value); an all-zero row stays zero. Refuses
        an entry the form does not hold (:func:`refuse_unheld`), unless the caller ``checked`` the block already."""
        exact = x.detach().float()
        if not checked:
            refuse_unheld([(self, exact)])
        scale = exact.abs().amax(dim=-1)
        unit = exact / scale.clamp_min(_TINY).unsqueeze(-1)
        stored = torch.round(unit * self.top) if self.dtype is ROW_DTYPE else unit * self.top
        if not self.aligned:
            return stored.to(self.dtype), scale
        rows = aligned_rows(*exact.shape, exact.device, self.dtype)
        rows.copy_(stored.to(self.dtype))
        return rows, scale

    def dequantize(self, rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """The block's values, fp32: each stored entry over ``top`` times its row's scale."""
        return rows.float() / self.top * scale.unsqueeze(-1)


def refuse_unheld(blocks) -> None:
    """Refuse the first ``(form, x)`` of ``blocks`` with an entry its form does not hold: ONE host read for them all, so
    a factory over many blocks waits on the device once."""
    blocks = list(blocks)
    flags = [form.unheld(x.detach().float()) for form, x in blocks]
    if flags and bool(torch.stack([f.to(flags[0].device) for f in flags]).any()):
        form, _ = next(block for block, flag in zip(blocks, flags) if bool(flag))
        raise ValueError(f"the {form.name} form holds {'finite' if form.signed else 'nonnegative, finite'} entries; "
                         "this block has one it does not")


#: a routing level's rows, dense or sparse: u8, the row max at 255, a zero a zero
ROWS = Form("u8 rows", ROW_DTYPE, signed=False, top=AMP_LEVELS, aligned=True)
#: the values: bf16 in [-1, 1] against the token's largest channel
VALUES = Form("bf16 values", torch.bfloat16, signed=True, top=1.0, aligned=False)


def straight_through(exact: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    """The quantized VALUES, fp32, with ``exact``'s gradient: the rounding is stepped through as identity."""
    wide = exact.float()
    return wide + (dequantized.float() - wide).detach()
