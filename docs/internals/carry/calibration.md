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
THE BURST WITH ALU WORK BESIDE IT (`hmma_alu`, eighteen HMMAs and 32 / 64 fp32 adds in four chains): one warp
38.0 / 43.6 (3 cycles an add exposed), two warps 71.3 / 78.9 -- the partner covers none of it, where it covers
the gather chain's exposure (`hmma_frag_chain_2w` 65.3): a warp's ALU stretch beside its burst is paid in full
by the pair, the number behind the mass-on-FMA form's loss.
THE FORK ROWS (2026-09-19): `hmma_frag_chain_hooked` (the chain row's gather in four pieces, each hooked into a
later HMMA's B) 36.3 alone; `hooked2` (each piece's input ALSO pinned after an earlier HMMA's accumulator, so
ptxas can neither hoist it nor compute its hook while its loads fly) 33.0 alone, 66.1 paired -- a lone warp at the
pipe's rate through its own gather, the pair paying the pins' instructions. `hmma_latency` (eighteen HMMAs into one
accumulator, each dependent on the last) 33.3: the completion latency is the issue interval. `hmma_operands` (the
fragment's burst with a fresh B register pair every HMMA, two A sets, no loads) 32.6 / 64.9: operand reuse does not
move the rate.
THE DEAD-LANE ROWS (2026-09-19; `async_copy_*`, eight warps). At FOUR copies a group a row is the group's landing
latency over four (a zero-size copy 70, a live sixteen-row copy 263, a sixteen-lane dead four-byte copy 213): read
the SIXTEEN-a-group rows for a copy's own cost. There, with eight warps issuing: a live sixteen-row copy
(`async_copy_rows_16`) 80.9; a sparse round's copy, four lanes live and twenty-eight zero-size to the one zero-row
slot (`mixed_sink_16`) 55.3, the same with the dead lanes predicated OFF (`lanes_16`, `ops::stage_run_lanes`) 56.9
-- the census counts 36 wavefronts a copy against 5, and the time does not move: a sixteen-byte copy is ~55 cycles
of the shared pipe by its instruction and its live lines, and a dead lane's zero fill is free. The four-byte
`.ca` copies differ: sixteen lanes live and contiguous (`async_copy_4_16`) 36.5, sixteen lanes zero-size to one
word (`async_copy4_zfill_sink_16`, a round's dead gain pairs) 137.3, two lanes live and fourteen off
(`async_copy4_lanes_16`) 76.3. The fill's cost at sparse is therefore its instruction count under the pair's
lockstep (twelve sixteen-byte and two four-byte copies a round for four live tokens: ~930 cycles a round on this
ledger against the trace's 1,320 a fill), not its dead lanes; predicating them would save ~120 cycles a round
(the gain copies), under the A/B's drift, so the fill stays on `stage_run_if`.
THE LATENCY, BRANCH AND CODE-SIZE ROWS (2026-09-19, read with their SASS). DEPENDENT CHAINS, one warp a scheduler:
a shared load whose address is the last load's result (`lds_chain_1w`) 29.0 cycles, a one-matrix `ldmatrix` the same
way (`ldsm_chain_1w`) 29.1, both with an address add in the chain, so the loads' own latency is ~25; a shuffle chain
(`shfl_chain_1w`, sixteen dependent `SHFL.BFLY`) 24.4. A COPY GROUP'S ROUND TRIP, one 16-byte `cp.async.cg` a lane
from L2, committed and waited on (`async_copy_latency_1w` / `_2w`): 324.9 / 358.6. A TAKEN UNIFORM BRANCH
(`branch_taken_1w` / `_2w`; ptxas hoisted the compare, so the row is a bare `@P0 BRA` over a skipped block): 10.9
cycles branch to branch, 14.0 a warp with two warps a scheduler -- the scheduler's branch path is busy ~7 a branch.
A divergent branch and its reconvergence (`branch_divergent_1w`) 20.3 a unit. THE INSTRUCTION CACHE
(`icache_*`, a loop body of independent `IMAD`s, 16 bytes an instruction): 4 KB to 64 KB bodies 4.04-4.07 cycles an
instruction with two warps a scheduler (IMAD issues every 2 cycles a scheduler: the FMA pipe's rate), 128 KB 4.55;
one warp 2.04 at 16 KB and 3.06 at 128 KB -- the cache holds 64 KB and a loop past it pays up to a cycle an
instruction. In place, not a row: the census's scoreboard wait at the uniform instruction after an `R2UR` is 36.6
and 39.4 cycles an execution on two cells, so `R2UR`'s result latency is ~39 (a probe row is owed: ptxas emits it only
where it proves a value warp-uniform).
THE QUEUE ROWS (`hmma_queue`, eighteen HMMAs and one dependent fma chain of 16 / 40 / 80 links, one warp):
34.7 / 35.5 / 43.5; the chain of forty with two warps 69.4. Read with the SASS (ptxas interleaves part of the
chain among the HMMAs, `asm volatile` notwithstanding) they put the tensor pipe's queue at zero to one on this
card: a warp posts no HMMA more than about one ahead of the unit; the A100's three in flight is its 8-cycle
issue interval, not this card's 32.5 (`docs/internals/tools/pipe_sim.md#calibrate`).

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
