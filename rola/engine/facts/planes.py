# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The state plane and the write-atom bitmap: the packing facts, and their bodies."""
from __future__ import annotations

import torch

from rola.engine import MMA_K_QUANTUM


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """A `[BH, N/16, page_bytes]` uint8 state plane for this routing: one int8-format page an atom, in canonical
    atom order (`rola.ops.paging.page_layout`). An all-zero page is the zero state.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    """
    from rola.ops.paging import page_layout

    widths = tuple(level.width for level in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, page_layout(d_v).bytes)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.uint8)


#: THE TWO ORTHOGONAL BITS the words' page fold publishes per atom
#: (`rola.engine.facts.liveness.page_bits`); the mirror of
#: `csrc/rola/src/common/ops.cuh`'s `kAtomWritten`/`kAtomRead`, which is what the
#: kernel derives `load iff resident and (read or written)` / `store iff written` from.
ATOM_WRITTEN = 1
ATOM_READ = 2


def written_atoms(bits: torch.Tensor) -> torch.Tensor:
    """The WRITE set as a bool tensor: the page plan's input, and only ever that.

    Residency is keyed on the atoms a call WRITES -- reads of never-written state are
    inert by the self-normalizing readout -- so a planner that took the whole activity
    byte would admit read-only atoms and turn exact commitment into a superset.
    """
    return (bits & ATOM_WRITTEN).bool()
