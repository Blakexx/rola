# `handoff_alone`

Mirrors `measure/lab/experiments/handoff_alone.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does production's hand-off cost a window alone on sm_86 -- the digits' keys and the den's maxima
(`post_keys`, `den_tables_max`), their exponents and the den's tables (`post_exps`, `den_tables_out`), the image
stored in halves (`store_batches`), with the kernel's rendezvous between -- against the [`handoff`](handoff.md) lab's
lean forms (copies of earlier forms, not production's code) and against the composed kernel's maxima and store phases?

THE METHOD. One CTA an SM with arm 0's plan tenants; the box's state in the fold's registers (values spanning
octaves by list and digit, a digit zero in some lists, made opaque each window), each list's unit in the book (one list
never live), the keys cleared and the scales' bits set as the kernel sets them, the den's maxima cleared each window.
64 windows; a window's span two barriers past its setup (as [`read_alone`](read_alone.md)); the exponents and the image
read into a sink after it. The image and exponents are the oracle's (`test_carry_readout`): the harness checks none.

THE MEASUREMENT (sm_86, cycles a warp-window, median of 7):

| span | cycles |
|---|---|
| the hand-off as sm_86 runs it a read chunk (keys, a rendezvous, exponents and den tables, half 0's stores, a rendezvous, half 1's) | 2,920 |
| the keys and the den's maxima | 836 |
| the keys, a rendezvous, the exponents and the den's tables | 1,345 |
| the stores, both halves, a rendezvous between (compile-time batch bounds) | 1,632 |
| the stores at the kernel's call (one call site, the halves' bounds at run time) | 1,772 |
| the empty span | 2 |

WHAT IT SETTLES. Alone, production's hand-off runs near the `handoff` lab's lean 4-byte form (2,252 a window in halves,
without the den's tables): the slice layout's 4-byte stores (a writer lane holds 2 of a slice row's 8 channels) cost
what that lab measured for 4-byte stores against its 16-byte image (1,705). The kernel's one call site for both
halves costs ~0.14K a window. In the composed kernel the same work reads ~4.0K (maxima 1,676, store 2,330 at
nl64k-alt-k4, 005643fa): ~1.1K the composition adds, not yet located; the untested lead is the read chunk's copies
landing in shared memory under the stores (a copy load with a distinct source a CTA, which the bench does not provide
yet: the same 5 KB read by every CTA measures an L2 hotspot instead).
