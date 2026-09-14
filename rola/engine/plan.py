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

from rola.engine.types import Arm, CallClass


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Everything one chunk launch reads, and nothing it does not.

    NO FIELD IS `O(N * L)`. The two-pass reverse pass replays the schedule from the
    union table rather than from per-chunk snapshots, and that claim is a property of
    this object's field list (docs/internals/chunk/backward.md).
    """

    call_class: CallClass
    arm: Arm
    widths: tuple[int, ...]
    d_v: int
    v_row_bytes: int
    eps: float
    #: the block-skip truth table over the liveness byte's three bits, one `uint32`
    #: per call class -- RunTableRule's verdict, and a launch argument like every
    #: other one (`rola/engine/rules/run_table.py`).
    run_table: int
    #: the OPERAND's dtype, which the divide narrows `y` back to. `v` below is the
    #: kernel's own bf16-contiguous copy and cannot answer for it.
    y_dtype: torch.dtype
    #: the bundle, for its LAYOUTS alone: the autograd node passes the routing
    #: tensors flat so the graph attaches to them, and the reverse pass needs each
    #: level's width and dtype to un-pack the digit gradients.
    routes: object
    read_plane: torch.Tensor
    write_plane: torch.Tensor
    g_write: torch.Tensor
    read_mass: torch.Tensor | None
    v: torch.Tensor
    union_table: torch.Tensor
    block_bits: torch.Tensor
    page_table: torch.Tensor | None
    #: the facts pass' `[BH, N/16]` uint8 ACTIVITY byte per atom -- bit 0 WRITTEN,
    #: bit 1 READ. The launch gates its state sweeps on it identically under both
    #: backings; `None` on a stateless call, which has no sweep to gate.
    atom_bits: torch.Tensor | None
    state_in: torch.Tensor | None
    state_out: torch.Tensor | None


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
    n_split: int

    @property
    def N(self) -> int:
        return math.prod(self.widths)

    @property
    def cols(self) -> int:
        return self.d_v + 1  # the mass column is unconditional (raw is dead)

    @property
    def total_rows(self) -> int:
        return sum(self.widths)


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
    #: The producer's own `[B, 1, H, width_l]` views, per side. The step kernel folds them
    #: itself, so no `[BH, width_l]` plane stands between the two
    #: (docs/internals/decode/decode_fold.md).
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    #: Per READ level: does it still carry its own mass, so the fold divides it by its row
    #: sum? The write side is never normalized, and the DETACHED clock IS the write
    #: allocation (docs/internals/decode/decode.md#detached-clock).
    normalize: tuple[bool, ...]
    dials: torch.Tensor | None
    g_write: torch.Tensor
    v: torch.Tensor
    state_plane: torch.Tensor
    page_table: torch.Tensor | None
    scratch: object
    pool: object
