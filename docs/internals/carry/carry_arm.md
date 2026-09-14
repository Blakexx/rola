# `carry/carry_arm.cuh` — one arm's launch and census

Mirrors `csrc/rola/src/carry/carry_arm.cuh`.

`carry_launch` is the whole launch: the grid the box plan implies, the block the arm's warp
count fixes, the dynamic shared block the ledger sizes, and the one-time opt-in above the
48 KB default. `carry_census` pairs what the compiler did — registers, local size, shared
bytes, maximum threads, binary version — with what the box plan says it should have done.
Both are read from the DEVICE side; see [`carry.md#launch`](carry.md#launch) and
[`carry.md#census`](carry.md#census).

`ROLA_CARRY_ARM_DEFINE` is what a generated per-arm translation unit expands. This header is
included ONLY by those units, never by the dispatch, so a subset build cannot instantiate an
arm locally and leave its shard doing nothing.
