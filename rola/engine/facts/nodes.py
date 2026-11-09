# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The primitive library: one declared node each, typed on the FACTS.

Nodes are typed on facts and not on any one family's context, which is what lets a
DAG share them. These are BAND 1 -- a node may launch and may allocate -- and each
body is the smallest thing that produces its fact, over the packing and manifest
primitives beside it (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from rola.engine import CallClass, Mask, node
from rola.engine.facts import planes
from rola.engine.facts.operands import operands_of, requires_backward


@node(gives=("widths", "d_v"), needs=("routes", "values"))
def topology(f):
    return (tuple(level.width for level in f["routes"].topology.levels),
            f["values"].d_v)


@node(gives=("call_class",),
      needs=("routes", "values", "decay", "state", "given_state_in", "given_arena",
             "want_state", "paging"))
def classify_call(f):
    state, given_arena = f["state"], f["given_arena"]
    return CallClass(
        state_in=f["given_state_in"] is not None or state is not None,
        state_out=bool(f["want_state"]),
        paged=given_arena is not None or (state is not None
                                          and state._wants_pages(f["paging"])),
        grad=requires_backward(*operands_of(f["routes"], f["values"], f["decay"])))


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


