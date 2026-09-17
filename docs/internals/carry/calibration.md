# The calibrations: what one operation costs on this card

`measure/harness/bench_carry_calib.py` runs the kernels in `measure/harness/carry_parts/carry_calib.cuh`
(built into the part harness's module, `ROLA_BUILD_PARTS=1`). Each kernel isolates one cost the
carry kernel's parts are made of and runs it back to back on every warp of 80 CTAs, one per SM. The
host times it under the locked clock. A row is the median launch's seconds, times the clock's cycles
per second, divided by the operations a warp issued: **cycles an operation a warp**. The rows are
reference data for this card (`<store>/<hw-profile>/calibration-<utc>.json` in the measurements store). The composer's and
the tracer's readings use them in place of assumed constants (KERNEL_STANDARDS §22 (10)).

## RTX 3080 Ti (sm_86), 2026-09-12, 1.665 GHz locked

| calibration | cycles an op a warp | reading |
|---|---|---|
| HMMA, the kernel's atom (`aligned m16n8k16 row.col`, bf16 A and B, fp32 accumulate), two warps a scheduler | 65.0 | 32.5 a scheduler: the pipe's rate, shared by the scheduler's warps |
| the same HMMA, one warp a scheduler | 32.8 | one warp saturates its scheduler |
| scalar shared loads, bursts of 4 / 16 / 64 | 6.9 / 3.0 / 2.3 | cheap, and a burst's loads overlap |
| shared stores, bursts of 4 / 16 / 64 | 17.4 / 16.3 / 16.3 | serialized at ~16 cycles |
| two-n-tile `ldmatrix.trans` loads, bursts of 4 / 16 | 32.7 / 32.5 | serialized at ~32 cycles: as costly as the HMMA it feeds |
| a global f32 reduction (a 1.3 MB output) | 85.6 | serialized at ~86 cycles |
| a CTA barrier | 50.8 | |
| a shared-memory barrier, arrive and wait | 123.9 | |
| a warp sync between one lane's store and every lane's load | 57.5 | |

### The settling rows (2026-09-16): an HMMA burst with the parts' loads or reductions interleaved

Nine HMMAs a unit (the atom), read as cycles an HMMA a warp against `hmma_2w`'s 64.8.

| calibration | cycles an HMMA a warp | reading |
|---|---|---|
| one `ldmatrix.trans` load ahead of the nine, feeding their B, two warps a scheduler | 64.9 | free |
| four loads ahead of the nine, feeding their B (the readout's box) | 65.0 | free: the loads issue under the previous burst's pipe time |
| four loads, one warp a scheduler (`hmma_1w` is 32.4) | 32.7 | free WITHOUT a partner warp: in-order issue runs ahead of the pipe |
| four loads beside the nine, their results a sink | 66.0 | the pipes do not share: +1 cycle an HMMA |
| one / two / four / eight global f32 reductions between the nine, each a warp's 128-byte line | 65.7 / 67.6 / 68.9 / 76.3 | ~10 cycles of issue a reduction |
| four / eight DIVERGENT reductions, a lane at its accumulator's row and column (eight rows' sectors a red) | 122.6 / 242.8 | ~200 cycles a reduction: the accumulator layout cannot reduce straight to global |
| the fold fragment's burst, eighteen HMMAs into eighteen accumulators, one warp / two | = `hmma_1w` / = `hmma_2w` | a wide burst issues at the pipe's rate alone (ratios; taken under a foreign GPU consumer, so the absolute row waits) |
| the same burst behind the fragment's gather chain (a shuffle, an `ldmatrix`, two packed multiplies), one warp / two | +18% / +0% | the chain is exposed only on a warp with no partner issuing |

**Not yet valid:** the asynchronous-copy rows (347.3 bank-free, 92.7 "aliased" a 16-byte run). The aliased
variant writes every lane's run to the same destination, so it measures overwriting, not bank conflicts.
The time also includes each group's landed wait. The kernel is being redesigned: distinct destinations
sharing one bank group, group sizes swept, and the wait timed apart.

## What the rows already explain

- **The readout's drain.** Not reduction-bound: the reduction row times the global path's throughput
  until every add has landed, and the kernel depends on none of them (corrected 2026-09-12); the
  settling row prices a reduction's issue at ~1.4 cycles. The drain's 2.8K cycles a tile is its stage
  round trip -- the stores, the syncs and the loads of four passes through one stage.
- **The two loads parts.** An `ldmatrix` load costs half an HMMA's pipe time a scheduler when timed
  alone, and NOTHING under a burst: in-order issue runs a warp's loads ahead while its last burst drains
  the pipe, with or without a partner warp. The loads parts' short-scoreboard signature is the loads
  waiting on what precedes them (a wait, a sync, a ballot), not the pipe.
- **The HMMA floor.** The kernel's own atom confirms the 32-cycle constant: 32.5 a scheduler.

The reduction row uses a small output. The kernel's output at N = L alt-k4 is 16.7 MB, so a large-output
row is needed before the drain's number is read at scale.
