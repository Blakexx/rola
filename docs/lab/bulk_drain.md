# `bulk_drain`

Mirrors `measure/lab/experiments/bulk_drain.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A CTA's fp32 output panel leaves shared memory for a global sum that other CTAs add into too. What does a token cost
by one asynchronous bulk reduction a panel (`cp.reduce.async.bulk`, fp32 add, shared to global), by one a token's
span, and by the warps' own reductions a value straight from registers?

THE FORM. A CTA an SM, 4 warps. A round writes a panel of 64 tokens × 64 channels fp32 (16 KB) and drains it: one
bulk add of the panel, waited before the next round; the same over two panels in turn with one drain in flight; one
bulk add a token's 256 bytes, each of 64 threads issuing its own; or no panel and a `red.global.add.f32` a value.
The panel's stores alone, with the round's barriers, are the base (one neighbour's word read a round keeps its stores
live). Every CTA adds into ONE global panel, the contended sum, or into its own; every sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a bulk add is one `UBLKRED.G.S.ADD.F32.RN` whatever its
size, issued by one thread; the panel is 32 `STS` a thread, a warp's lanes on 32 banks, and the writers' proxy fence before the round's barrier; the warps' own drain is 32
`REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` a thread a round.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60164819, driver 595.91.07; 2026-11-11, one session, 7
repetitions, every sum exact; cycles a token -- 64 fp32 -- a CTA, the median, the spread under 0.2%):

| the drain | one sum for every CTA | each CTA its own sum |
|---|---|---|
| none: the panel's stores and the round's barriers | 3.2 | 3.2 |
| one bulk add a panel, waited | 46.1 | 19.9 |
| one bulk add a panel, one in flight | 46.0 | 19.8 |
| one bulk add a token's span | 46.0 | 30.4 |
| the warps' own reduction a value, no panel | 45.9 | 19.8 |

WHAT IT SHOWS. The bulk reduction drains no faster than the warps' own reductions: 19.8 cycles a token either way
into a CTA's own sum, 46 either way into the one sum every CTA adds into. The rate belongs to what receives the
adds, not to the engine that issues them, and contention between CTAs more than doubles it. Keeping a second panel's
drain in flight changes nothing; a bulk add a token is slower than one a panel when uncontended. What the bulk form
does change is the warps' issue: one instruction a round from one thread where the warps' own drain is 32 a thread --
its worth beside tensor work is not measured here (the round waits for its drain in every form).

Record: `measure/lab/results/bulk_drain/sm_90a.json`; its counters are the written exception (no profiler on that
box).
