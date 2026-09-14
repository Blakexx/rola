# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The primitive library: one declared node each, typed on the FACTS.

Nodes are typed on facts and not on any one family's context, which is what lets a
DAG share them. These are BAND 1 -- a node may launch and may allocate -- and each
body is the smallest thing that produces its fact, over the packing and manifest
primitives beside it (docs/internals/engine/engine.md).
"""
from __future__ import annotations

import torch

from rola.engine import CallClass, Mask, node
from rola.engine.facts import manifest as census
from rola.engine.facts import planes
from rola.engine.facts.operands import operands_of, requires_backward
from rola.ops._ext import extension


@node(gives=("manifest",))
def manifest(f):
    """What this binary was BUILT for -- the arm decision's only non-call input."""
    return census.chunk_arms()


@node(gives=("widths", "d_v"), needs=("routes", "v"))
def topology(f):
    return (tuple(level.width for level in f["routes"].topology.levels),
            f["v"].shape[-1])


@node(gives=("call_class",),
      needs=("routes", "v", "decay", "state", "given_state_in", "given_arena",
             "want_state", "paging"))
def classify_call(f):
    state, given_arena = f["state"], f["given_arena"]
    return CallClass(
        state_in=f["given_state_in"] is not None or state is not None,
        state_out=bool(f["want_state"]),
        paged=given_arena is not None or (state is not None
                                          and state._wants_pages(f["paging"])),
        grad=requires_backward(*operands_of(f["routes"], f["v"], f["decay"])))


@node(gives=("read_plane", "write_plane", "g_write", "read_mass"), needs=("routes",),
      stream="main")
def pack_planes(f):
    routes = f["routes"]
    mass = routes.read_mass
    return (planes.pack_side(routes.read),
            planes.pack_side(routes.write),
            planes.packed_copy(routes.g_write.permute(0, 2, 1)),
            #: fp32, because it scales the readout's floor and `eps` is an fp32
            #: definition constant; bf16 here would quantize the floor.
            None if mass is None else planes.packed_copy(mass.permute(0, 2, 1),
                                                         torch.float32))


@node(gives=("union_table", "atom_bits"),
      needs=("read_plane", "write_plane", "arm", "widths", "mask"),
      stream="main")
def facts_tables(f):
    """BOTH ROW SETS IN ONE PASS over the amplitude planes, which is why the atom
    bitmap is a SECOND OUTPUT here and not a second launch.

    The published bitmap stays keyed to the 16-leaf quantum whatever arm this pass
    carries -- its reduction is instantiated `(D, B)` alone -- so residency is still
    an OUTPUT of the plan rather than a consequence of it.

    IT IS ASKED FOR BY EVERY STATEFUL CALL, not only a paged one: the byte is
    the ACTIVITY fact both backings gate their state sweeps on, and only its WRITE
    half is residency's input. A dense-backed call that skipped it would touch a
    different atom set from the paged one, and the equality gate would be certifying
    two different predicates.
    """
    atoms = int(f["mask"]) == int(Mask.M2)
    return extension().chunk_facts(f["read_plane"], f["write_plane"], *f["arm"],
                                   list(f["widths"]), int(f["mask"]), atoms)


@node(gives=("state_in", "arena"),
      needs=("state", "given_state_in", "given_arena", "routes", "d_v", "atom_bits",
             "paging", "arm"),
      stream="side", masks=(Mask.M2,), host_sync=True)
def residency(f):
    state = f["state"]
    if state is None:
        return f["given_state_in"], f["given_arena"]
    return state._kernel_entry(f["routes"], f["atom_bits"], d_v=f["d_v"],
                               paging=f["paging"], BC=f["arm"].bc)


@node(gives=("v_packed",), needs=("v",), stream="main")
def pack_v(f):
    """THE KERNEL TAKES V AS THE CALLER HOLDS IT: token-major, and bf16 because that
    is the operand dtype the consumer reads."""
    v = f["v"]
    return v if v.dtype is torch.bfloat16 and v.is_contiguous() else planes.packed_copy(v)


@node(needs=("arena",), stream="main", masks=(Mask.M2,), waits=("residency",))
def join(f):
    """The admission's table write and slot zeroing ran on the arena's side stream;
    this wait puts them before the BLOCK-BITS pass, which READS the page table, and
    not merely before the consumer launch."""
    arena = f["arena"]
    if arena is not None:
        arena.wait()


@node(gives=("page_table", "state_out"),
      needs=("arena", "state_in", "want_state", "routes", "d_v"))
def backing(f):
    """The two state planes a launch is given -- and for a CONTINUATION they are ONE.

    A FRESH DENSE PLANE IS ZEROED, and that is the price of the store
    predicate: the exit sweep writes the atoms this call WRITES and no others, so the
    atoms it skips must already hold the zero an absent paged atom reads as. It is one
    fill per SEQUENCE on the expert-override backing, never per call.

    A continued sequence advances its state IN PLACE, in either backing: the arena has
    always handed the same plane in and out, and the dense one does now
    (docs/internals/state.md#the-dense-continuation). Only a FRESH dense bind allocates,
    and that plane is uninitialized because it is written whole.
    """
    arena, want_state = f["arena"], f["want_state"]
    if arena is not None:
        return arena.page_table, (arena.state if want_state else None)
    if not want_state:
        return None, None
    state_in = f["state_in"]
    return None, (planes.state_plane(f["routes"], d_v=f["d_v"], zeros=True)
                  if state_in is None else state_in)


@node(gives=("block_bits",),
      needs=("union_table", "arm", "widths", "routes", "page_table"), stream="main")
def block_bits(f):
    return extension().chunk_block_bits(f["union_table"], f["arm"].bc, list(f["widths"]),
                                        f["routes"].tokens, f["page_table"])
