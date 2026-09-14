# `carry/box.cuh` — the CTA box's compile-time plan

Mirrors `csrc/rola/src/carry/box.cuh`.

`BoxPlan<D, DV, warps_per_cta>` holds every DERIVED quantity of one arm as a `constexpr`:
the box's leaves, pages and boxes, the warp's owned boxes (`kDealt`) and n-tiles (`kNT`),
the liveness words, the head's products, the pool's geometry, the snapshot, the readout's
private blocks, and the shared-memory ledger's offsets and total. Nothing in it is a
literal, and every relation that could silently round refuses as a `static_assert`:
the boxes deal evenly to the warps, a warp's state fits 128 registers a lane, the readout's
blocks alias the pool's region, the whole fits the architecture's maximum.

The byte offsets and the per-moment aliasing are in [`smem_ledger.md`](smem_ledger.md);
what each tenant is for is in [`carry_kernel.md`](carry_kernel.md). `Edge` names the head's
two named barriers.
