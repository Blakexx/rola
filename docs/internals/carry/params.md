# `carry/params.cuh` — the parameter block

Mirrors `csrc/rola/src/carry/params.cuh`.

One `struct` by value: the addressing block the host derived (`CarryGeomRT`), each side's
layout against the box (`SideLayout`, read then write), the operands, the problem scalars,
the order dial and the phase ledger. It reaches the kernel as a GRID CONSTANT, so every
inlined stage reads its fields as constant-bank operands — which is why the layouts live
here rather than being derived in the kernel (32 registers a side otherwise).

`order` is the ORDER POLICY: `kOrderFirstBox`, the read side's counting sort by first live
box with the dead last; `kOrderIdentity`, token order and every tile — the sort's A/B. The
write side has no order: its fold compacts as it walks.

`ledger` is the phase ledger's device row, `[cta][warp][kPhases]` int64, or null; the
phases are `CarryPhase`. See [`carry_kernel.md#phase-ledger`](carry_kernel.md#phase-ledger).
