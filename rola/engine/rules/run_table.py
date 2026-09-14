# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RunTableRule -- the block-skip truth table over the liveness byte's three bits.

The rule is written once, in readable form, and the mask is DERIVED by evaluating it
over all eight bit values; the kernel's whole predicate is then one shift and one AND
(docs/internals/engine/rules.md#run-table). This fold is the ONE derivation and the
`uint32` is a launch argument: the table a call PLANS is the table it RUNS, which a
second derivation behind the extension boundary could not promise.

THE TABLE IS BACKING-INVISIBLE at the store clause: the exit sweep stores
the atoms this call WRITES in either backing, so a block with no write has nothing to
store in either, and the fresh-dense "written whole" exception is gone with the zeroed
fresh plane (`rola/engine/facts/nodes.py`'s backing node).

The three inputs are PRESENCE facts -- the entry-state plane, the exit-state plane and
the page table, each a tensor or None -- and never the call class's engagement bits.
The two differ: a fresh paged sequence engages a state and reaches the launch with no
entry plane at all, and it is the launch argument that decides what a skipped block
would have cost.
"""
from __future__ import annotations

from typing import Any

from rola.engine import rule


@rule(gives=("run_table",), needs=("state_in", "state_out", "page_table"))
def run_table(state_in: Any, state_out: Any, page_table: Any) -> int:
    """The `uint32` whose bit `v` says a block with liveness byte `v` must run."""
    carries_in, stores_out, paged = (state_in is not None, state_out is not None,
                                     page_table is not None)
    table = 0
    for v in range(8):
        R, W, S = bool(v & 1), bool(v & 2), bool(v & 4)
        #: `y` needs a read entry AND something for it to read: this call's own
        #: write, or a carried state the block may hold. Without a residency map the
        #: dense backing cannot witness the latter and must assume it.
        run = R and (W or (carries_in and (S if paged else True)))
        #: the state epilogue stores this block's atoms. An unwritten block must end
        #: holding what it carried in, which is what skipping leaves: a paged slot
        #: unchanged, and a dense plane too, since a continuation is updated in place
        #: and a FRESH dense plane is ZEROED at bind rather than written whole.
        if stores_out:
            run = run or W
        if run:
            table |= 1 << v
    return table
