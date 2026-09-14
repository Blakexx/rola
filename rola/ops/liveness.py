# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ONE LIVENESS PASS, as the caller reaches it: one launch, the class-1 words.

The op and nothing else. The LAYOUT those words carry, the reader that unpacks them
and the pure-torch MODEL this launch is graded against all live in
:mod:`rola.engine.facts.liveness`, which owns the contract; the folds every consumer's
grain is built from live there too, because they are host folds. What is here is the
translation from that contract's value types to the extension's argument list --
:class:`~rola.engine.facts.liveness.SideStatics` packed into the bits R13 requires, so
no logical width crosses the seam (docs/internals/facts/liveness_api.md).
"""
from __future__ import annotations

import torch

from rola.engine.facts.liveness import LivenessLayout, SideStatics, mask_words
from rola.ops._ext import extension


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the kernel switches its rows on."""
    bits = 0
    for level, dense in enumerate(statics.dense_levels):
        if dense:
            bits |= 1 << level
    return bits


def liveness_words(read_plane: torch.Tensor, write_plane: torch.Tensor,
                   layout: LivenessLayout, read_statics: SideStatics,
                   write_statics: SideStatics) -> torch.Tensor:
    """`[BH, 2, rows, words]` int32: the class-1 words, in ONE launch over both sides.

    Same arguments and same answer, bit for bit, as
    :func:`rola.engine.facts.liveness.liveness_words` -- that model is the executable
    spec and ``tests/unit/test_liveness_pass.py`` is where the two are held equal. The
    planes are the packed `[..., L, sum_l B_l]` bf16 rows a producer already emits, so
    a row index IS the column it votes from and the pass needs no second layout.

    A SPARSE level is voted from the amplitudes; a DENSE one is not read at all -- its
    rows are the static width mask ``statics`` carries, which is exact because only a
    pad digit's logit is ``-inf``.
    """
    words = mask_words(layout.rows)
    return extension().liveness_words(
        read_plane, write_plane, list(layout.B),
        _dense_bits(read_statics), [read_statics.mask_word(i) for i in range(words)],
        _dense_bits(write_statics), [write_statics.mask_word(i) for i in range(words)])
