# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The plan objects -- one per kernel family, built once at a DAG's terminal node.

A plan holds two kinds of field and no third: a HOST field is some rule's verdict,
and a TENSOR field is a reference to a terminal fact. A field that is neither means
something is still being decided at the launch site, which is the condition the facts
engine exists to remove (docs/internals/engine/engine.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DecodeGeometry:
    """The T=1 seam's CONFIG half: what the topology and `d_v` fix for the sequence.

    It is the config half of :class:`DecodePlan` rather than a field of it because the
    per-shape memo key is over this alone -- the terminal facts beside it are per-step
    and never cached (docs/internals/engine/decode_dag.md).

    ``BC`` IS NOT A FIELD and must not become one. Owner blocking is a property of how
    the tiled consumer amortizes a ``[BC, cols]`` block across many tokens; with one
    token there is nothing to amortize, and the decode kernels are BC-blind by
    construction (``tests/integration/test_decode_bc_blindness.py`` asserts
    byte-identical decode output under prefill ``BC`` in ``{16, 32, 64}``).
    """

    D: int
    widths: tuple[int, ...]
    level_row_offset: tuple[int, ...]
    #: The `(k, m)` BOX the state plane's leaves are ordered by — the leaf order travels
    #: with the plane rather than being assumed (`rola.ops.lattice`).
    lattice_k: int
    lattice_m: int
    d_v: int
    has_decay: bool
    eps: float
    #: THE RESIDENT GRID: ``None`` for the step's own (the binary sizes it from the card,
    #: docs/internals/decode/decode.md#the-split-policy), or a gate's forced CTA count; the CTAs claim work from every
    #: batch-head's queue (docs/internals/decode/decode.md#queue).
    ctas: int | None
    #: THE INNERMOST LEVEL'S WRITE SIDE IS DENSE BY DECLARATION (its activation has no exact zeros), so every write
    #: unit is a whole atom and the step walks it as one (docs/internals/decode/decode.md#whole-atoms).
    write_dense: bool

    @property
    def N(self) -> int:
        return math.prod(self.widths)

    @property
    def cols(self) -> int:
        return self.d_v + 1  # the mass column is unconditional (raw is dead)


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """Everything one decode launch reads, and nothing it does not.

    NO FIELD IS DERIVED AT THE LAUNCH SITE. `pool` is an input and never a mode: the
    arena whose slack slots a growth step admits itself into, present exactly on a paged
    step that has one, and the kernel's own per-batch-head verdict is what makes a step it
    cannot cover a device-side no-op (docs/internals/decode/decode.md#per-bh-verdict).
    """

    config: DecodeGeometry
    heads: int
    #: The bundle's own `[B, 1, H, width_l]` unit-max coefficient rows (u8) and their fp32 `[B, 1, H]`
    #: maxima, per side. The step kernel folds them itself, so no `[BH, width_l]` plane stands between the two
    #: (docs/internals/decode/decode_fold.md).
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    read_scale: tuple[torch.Tensor, ...]
    write_scale: tuple[torch.Tensor, ...]
    #: Per side, the dense levels as a bitmask: their zero digits, in a live row, read as the floor
    #: (docs/internals/facts/liveness_contract.md#floor).
    floor: tuple[int, int]
    dials: torch.Tensor | None
    #: The value, `[B, 1, H, d_v]` f16 against the token's largest channel, and its fp32 `[B, 1, H]` scale.
    v: torch.Tensor
    v_scale: torch.Tensor
    state_plane: torch.Tensor
    page_table: torch.Tensor | None
    scratch: object
    pool: object
