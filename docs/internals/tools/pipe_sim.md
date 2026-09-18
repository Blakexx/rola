# `tools/pipe_sim.py` — a burst loop's pipe occupancy, from its SASS alone

Every SASS instruction on Volta, Turing and Ampere carries a control word ptxas wrote: a stall
count before the next issue, a yield bit, the scoreboard barrier it sets on completion and the
barriers it waits on, and reuse flags. Decoded (bits 105 to 125 of the 128-bit word, validated
on this card by the reuse bits matching the `.reuse` annotations of the same operands), these
give one warp's issue timeline for a loop whose memory is shared memory and registers, once the
scoreboard classes have a latency each and the machine's shared resources are modelled.

`--arm N` compiles the carry arm with line information; `--source F --function G` names the
loop: the SASS whose inline chain touches the function's lines, from its first instruction to
its last backward branch, one iteration. `--warps` is the warps a scheduler (one or two),
`--sm-warps` the warps on the SM running the loop (eight in the kernel's fold and readout, where
every warp is in the same phase within ~150 cycles). The report: the period and the tensor
pipe's occupancy.

<a id="model"></a>
## The model (2026-09-18, fitted)

A warp issues in order. Instruction i issues at the latest of: the previous issue plus the previous
instruction's stall count; the completion of every scoreboard barrier its wait mask names (a
barrier completes at its producer's issue plus the class latency, or at the memory pipe's take
plus the latency for a memory instruction); and, for an HMMA, the tensor pipe's next free slot.
Each scheduler has an issue port (one instruction a cycle, a global access `issue` cycles) and a
tensor pipe (one HMMA every `hmma_pipe` cycles, NO queue: a warp waits at an HMMA until the pipe
is free -- the chain row refuted the two-deep queue the first model assumed). The SM has ONE
memory pipe: every shared access holds it a cycle a 128-byte wavefront (`sass_control.mem_cost`:
an `ldmatrix` of k matrices k cycles, a 128-bit access four), a global access `mem_sector` cycles
a sector; it takes instructions in issue order behind a queue of `mem_queue` cycles, and a memory
instruction does not issue while the queue is longer (the profiler's `mio_throttle`). Every warp
starts at once. Occupancy = HMMAs x hmma_pipe x warps a scheduler over the steady-state period.

<a id="machine"></a>
## The machine is an architecture

`tools/sass_control.py`'s `MACHINES` holds one row an architecture: the pipe cost of an HMMA (sm_86 32.5 cycles, bf16 with
fp32 accumulate at half rate; sm_80 full rate), the latency by opcode class, the schedulers, the
memory pipe's wavefront and sector costs and queue, and the issue cost of a global access -- from
`carry/calibration.md`'s rows. The cubin's `.target` selects the row and an architecture without
one is refused, never approximated. Hopper and after are a different model, not a row: `wgmma`
issues asynchronously and its operands come from shared memory by descriptor, so a warp is never
stalled at an MMA and the question this tool answers is moot there; the two-tier rule
(KERNEL_STANDARDS §23) is what carries over, the simulator is Ampere's.

<a id="calibrate"></a>
## `--calibrate`: the model against the probe

`--calibrate rola_cu13/_C_parts.abi3.so` finds every `calib_kernel<W, M, B>` instantiation in the
part harness's module, simulates its loop as the probe runs it (W warps on the SM, W/4 a
scheduler) and prints the cycles an op beside the row's latest clocked measurement in the results
store (KERNEL_STANDARDS §23 addendum 2). The reading at the fitted parameters (`hmma_queue` 0,
`mem_queue` 4, the literature's latencies), rows measured on this card 2026-09-17/18 with the clock
locked and the card idle:

| row | model | measured | miss |
|---|---|---|---|
| hmma, 1 warp / 2 warps a scheduler | 32.5 / 65.0 | 32.6 / 65.4 | <1% |
| hmma + 1 / 4 `ldmatrix` ahead, 2 warps; 4 loads, 1 warp; 4 loads sunk | 65.0 / 65.0 / 32.7 / 65.0 | 64.9 / 65.0 / 32.9 / 66.0 | <2% |
| the fragment's wide burst, 1 / 2 warps | 32.5 / 65.0 | 32.4 / 65.2 | <1% |
| the burst behind its gather chain, 1 / 2 warps | 37.6 / 65.0 | 38.3 / 65.3 | 2% |
| `ldmatrix` as the kernel issues it, 4 / 16 a unit, 8 warps; 4 a unit, 4 warps | 32.6 / 32.0 / 19.0 | 32.3 / 32.3 / 16.4 | 1% / 1% / **+16%** |
| hmma + 1 / 2 / 4 / 8 coalesced global reductions | 65.0 flat | 65.6 / 67.6 / 70.0 / 76.4 | to **−15%** |
| hmma + 4 / 8 divergent reductions | 65.0 flat | 122.6 / 242.8 | **−47 / −73%** |

What the misses are. The four-warp `ldmatrix` row: with fewer warps the model leaves the memory pipe
idle between a warp's landings and its next issue where the hardware does not (the queue or the
landing latency is finer than modelled). The reductions: the per-line census puts the loss at the
instructions after each `RED` waiting on the LONG scoreboard -- a reduction's operand registers are
read late, through the memory pipe, and the instruction that next writes them waits (a write-after-
read on the long scoreboard) -- and the model charges a read barrier at most 20 cycles; not fitted,
because the kernel's drain issues its reductions once a tile, not between HMMAs. Divergent
reductions are eight sectors a lane and are not read off the text. An asynchronous copy's lines are
not read off the text either (the fill's runs touch sixteen lines an instruction, modelled as four).

What the fit changed, and why. The first model had no memory pipe and read the kernel-pattern
`ldmatrix` row at 5.7 cycles a load against 32.3 measured (the SM's 128 bytes a cycle over eight
warps); with a pipe and an unbounded queue it read 44, because every warp's loads queued behind
every other warp's and the landing came late -- the profiler's `mio_throttle` at 72% of the row's
load samples says the hardware throttles at ISSUE, and a four-cycle queue (Huerta et al.'s four)
reads 32.6. The two-deep HMMA queue hid the gather chain the lone-warp row exposes (33.9 against
38.3); no queue reads 37.6. The pair's half-period offset changed no row; every warp starts at once.

<a id="reading"></a>
## Readings of the kernel's loops (2026-09-18, the fitted model, the arm at 7468b883)

| loop | instructions | HMMAs | alone (4 on the SM) | paired (8 on the SM) | the trace |
|---|---|---|---|---|---|
| the fold's pair (`burst.cuh:run`) | 155 | 36 | 1,351: 86.6% | 2,400: 97.5% | a chunk's four fragments 2,159 paired |
| the same, the mass on the FMA pipe (tombstoned) | 220 | 32 | 1,254: 82.9% | 2,133: 97.5% | 2,223 paired: the model's second point MISSES |
| the readout's box loop (`readout_tile`) | 68 | 8 | 381: 68.2% | 520: 100% | a box 559 paired |

**The model's standing.** It reproduces the probe within 2% on every row that exercises what the
kernel's loops do -- the pipe, the gather chain, the kernel-pattern loads under eight warps -- and it
still reads the fold's second form 11% fast where the trace measured it 3% slow. So the fold's
fragment is bound by something no probe row exercises yet, and the per-line census of the two forms
(KERNEL_STANDARDS §23 addendum 2, rule 5) is the next measurement, not a reading of the model. Until
that closes, the floors are model-relative ratchets on the ALONE reading (`burst_gate.md`), the
paired reading is advisory, and no form is budgeted from the model.
