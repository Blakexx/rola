# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The operand planes and the write-atom bitmap: the packing facts, and their bodies.

Every chunk entry reads one `[B, H, T, sumW]` bf16 tensor per side and one atom-major
state plane, and the packing happens ONCE per call so every entry reads the same
bytes. On the shipped path the pack is a VIEW: the producer's solve writes the plane
directly and :func:`_side_plane` reconstructs it from the levels' strides
(docs/internals/chunk/chunk.md#the-side-plane). The copy survives for the sides that
are not already that layout -- a hand-built bundle, a foreign producer.
"""
from __future__ import annotations

import torch

from rola.engine import MMA_K_QUANTUM


def packed_copy(view: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """`view` as a contiguous `dtype` tensor, in ONE copy.

    `.to(dtype)` preserves a permuted view's strides, so `.to(dtype).contiguous()`
    walks the tensor twice; an already-contiguous destination fuses the transpose
    and the narrowing into the one copy that is actually needed.
    """
    return torch.empty(view.shape, dtype=dtype, device=view.device).copy_(view)


def _side_plane(levels) -> torch.Tensor | None:
    """The `[B, H, T, sumW]` bf16 plane these levels are already views OF, or None.

    A STRIDE fact, not a provenance one: the levels are the plane iff each is the
    window of one bf16 allocation whose layout is exactly this side's, which is
    checkable and is what a hand-built bundle either satisfies or does not
    (docs/internals/chunk/chunk.md#the-side-plane).
    """
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    total = sum(widths)
    want = (heads * tokens * total, total, tokens * total, 1)
    base = levels[0]
    offset = 0
    for level, width in zip(levels, widths):
        if (level.dtype is not torch.bfloat16
                or level.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
                or level.stride() != want
                or level.storage_offset() != base.storage_offset() + offset):
            return None
        offset += width
    span = base.storage_offset() + batch * heads * tokens * total
    if span * base.element_size() > base.untyped_storage().nbytes():
        return None
    return base.as_strided((batch, heads, tokens, total),
                           (heads * tokens * total, tokens * total, total, 1),
                           base.storage_offset())


def pack_side(levels) -> torch.Tensor:
    """The per-level `[B, T, H, width_l]` tuple as one `[B, H, T, sumW]` bf16 tensor.

    ONE destination and one pass per level: concatenating in the public layout
    materializes an intermediate no kernel reads, and the transpose and the
    narrowing are then a second pass over it. A side that is already the plane skips
    all of it -- the producer emits this layout.
    """
    plane = _side_plane(levels)
    if plane is not None:
        return plane
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    packed = torch.empty((batch, heads, tokens, sum(widths)),
                         dtype=torch.bfloat16, device=levels[0].device)
    offset = 0
    for level, width in zip(levels, widths):
        packed[:, :, :, offset:offset + width].copy_(level.permute(0, 2, 1, 3))
        offset += width
    return packed


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """An empty `[BH, N/16, 16, d_v+1]` fp32 state plane for this routing.

    The ATOM-MAJOR view of `[BH, N, d_v+1]`, byte for byte: leaves are stored in
    canonical order, so the reshape between the two is free and the kernel's
    per-atom addressing and a reader's per-leaf one describe the same bytes.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    A FRESH bind is the one call that allocates, and it is uninitialized because that
    plane is written WHOLE: the owner blocks tile the leaf space exactly, so every
    atom is stored by exactly one CTA.
    """
    widths = tuple(level.width for level in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, MMA_K_QUANTUM, d_v + 1)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.float32)


#: THE TWO ORTHOGONAL BITS the liveness pass publishes per atom; the mirror of
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


def read_atoms(bits: torch.Tensor) -> torch.Tensor:
    """The READ set as a bool tensor: the half of the activity fact residency ignores."""
    return (bits & ATOM_READ).bool()


def atom_bits(write_plane: torch.Tensor, widths,
              read_plane: torch.Tensor | None = None) -> torch.Tensor:
    """`[BH, N/16]` uint8: the per-atom ACTIVITY byte, `WRITTEN | READ`.

    Bit 0 is set for an atom this routing WRITES, bit 1 for one it READS; the two are
    orthogonal and are never pre-ORed, because the read-only distinction is what the
    touched-atom census and the continuation accounting are about. THE PAGE PLAN TAKES
    THE WRITE SET ALONE -- :func:`written_atoms` -- and the kernel takes the byte.

    A CONVENIENCE ENTRY: with no ``read_plane`` the write plane stands in for both
    sides, which answers the write bits exactly and makes the read bits their mirror --
    all a caller that only wants residency needs. A caller that gates a state sweep
    passes BOTH planes.

    IT IS A FOLD, NOT A PASS. The one liveness pass emits the class-1 words and this
    reduces them at the atom box set, where the box->page map is a bijection, so the
    byte is exact (docs/internals/facts/liveness_contract.md#folds). It is keyed to the
    16-leaf quantum and to nothing else, which is what lets residency be an OUTPUT of
    the plan rather than a consequence of it.

    EVERY LEVEL IS VOTED FROM ITS AMPLITUDES here, which is what the static width mask
    reduces to on an UNPADDED topology: a softmax amplitude is never zero, so a dense
    level votes all ones either way. A padded topology's mask is
    :func:`~rola.engine.facts.liveness.side_statics`'s, over a logical width this entry
    is not given and does not derive.
    """
    from rola.engine.facts import liveness as lv
    from rola.ops.liveness import liveness_words

    widths = tuple(int(width) for width in widths)
    layout = lv.LivenessLayout(D=len(widths), B=widths, L=int(write_plane.shape[-2]))
    statics = lv.side_statics(layout, (False,) * layout.D, widths)
    words = liveness_words(read_plane if read_plane is not None else write_plane,
                           write_plane, layout, statics, statics)
    return lv.page_bits(words, layout)
