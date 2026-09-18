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
| two-n-tile `ldmatrix.trans` loads, bursts of 4 / 16 | 32.7 / 32.5 | serialized at ~32 cycles -- BUT a broadcast (every lane one address, one wavefront a load; found 2026-09-18): the load's issue and latency, not the kernel's four-wavefront loads; `matrix_load_rows` is the kernel's pattern |
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

**The asynchronous-copy rows (2026-09-18, six):** bank-free contiguous 347.3, aliased 92.7 (every lane's run
to one of four destinations: overwriting, not a conflict row), a 128-byte line a lane 502.0, four-byte runs
48.8, THE FILL'S V PATTERN (sixteen token rows, two adjacent chunks a run, the channel-row swizzle) 262.4,
THE SAME ROWS A WHOLE LINE AT A TIME (eight lanes a row, four rows a run) 259.7 cycles a run a warp; each
row includes its group's landed wait. Under `ncu`'s source counters the same six read shared wavefronts
4 of ideal 4, 4 of 1, 32 of 4, 1 of 1, 16 of 4 and 4 of 4: an asynchronous copy's wavefront count is
the global lines it touches, and the rows and the lines forms copy at one rate because they fetch the
same sectors. The census's `LDGSTS` lines are therefore line counts, not landing conflicts.

## The rows re-taken with the card idle (2026-09-18, 1.665 GHz locked, idle utilization 0-2%)

The 2026-09-17 03:xx rows were taken under a Windows-side GPU consumer and read ~20% slow; re-taken:
`hmma_1w` 32.6, `hmma_2w` 65.4, `hmma_load_4_1w` 32.9, `hmma_frag_1w` 32.4, `hmma_frag_2w` 65.2,
`hmma_frag_chain_1w` 38.3, `hmma_frag_chain_2w` 65.3, `shared_store_16` 16.4, `cta_barrier` 50.6,
`matrix_load_4` (a broadcast) 32.7. THE KERNEL-PATTERN LOADS (`matrix_load_rows`, a lane its own
row, four wavefronts a load): 4 / 16 a unit with eight warps 32.3 / 32.3, 4 a unit with four warps 16.4
-- the SM's shared-memory pipe at 128 bytes a cycle, shared by the SM's warps: eight warps' four-wavefront
loads cost 32 cycles a load a warp, four warps' 16. These rows are the memory pipe of `tools/pipe_sim.py`.

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
