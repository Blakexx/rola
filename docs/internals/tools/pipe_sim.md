# `tools/pipe_sim.py` — a burst loop's pipe occupancy, from its SASS alone

Every SASS instruction on Volta, Turing and Ampere carries a control word ptxas wrote: a stall
count before the next issue, a yield bit, the scoreboard barrier it sets on completion and the
barriers it waits on, and reuse flags. Decoded (bits 105 to 125 of the 128-bit word, validated
on this card by the reuse bits matching the `.reuse` annotations of the same operands), these
give one warp's issue timeline exactly for a loop whose memory is shared memory and registers,
once the scoreboard classes have a latency each. The tensor pipe is a resource: an HMMA takes a
slot every `hmma_pipe` cycles a scheduler and a warp cannot issue past an HMMA without one.

`--arm N` compiles the carry arm with line information; `--source F --function G` names the
loop: the SASS whose inline chain touches the function's lines, from its first instruction to
its last backward branch, one iteration. The report: the period and pipe occupancy for one warp
alone and for two sharing the scheduler, offset by half a period.

<a id="machine"></a>
## The machine is an architecture

`MACHINES` holds one row an architecture: the pipe cost of an HMMA (sm_86 32.5 cycles, bf16 with
fp32 accumulate at half rate; sm_80 full rate) and the latency by opcode class, from
`carry/calibration.md` where a row exists. The cubin's `.target` selects the row and an
architecture without one is refused, never approximated. Hopper and after are a different
model, not a row: `wgmma` issues asynchronously and its operands come from shared memory by
descriptor, so a warp is never stalled at an MMA and the question this tool answers is moot
there; the two-tier rule (KERNEL_STANDARDS §23) is what carries over, the simulator is Ampere's.

<a id="reading"></a>
## First readings (2026-09-18, the arm at f497bc6)

| loop | instructions | HMMAs | one warp | two warps | the trace |
|---|---|---|---|---|---|
| the fold's pair (`burst.cuh:run`) | 155 | 36 | period 1,282, occupancy 91% | 2,328, 100% | a pair 2,147 paired |
| the readout's box loop (`readout_tile`) | 68 | 8 | 377, 69% | 512, 102% | a box 559 paired; ~49% alone by the drain overlap |

The model runs 8 to 9% fast against the trace and reads the lone box loop 20 points high: its
latencies are the literature's, not yet this card's, and the pair offset is an assumption. The
`--calibrate` step (the trace's per-unit numbers beside the model's) fixes the table before a
floor is ratcheted. What it already says without calibration: the fold's interleave holds a lone
warp near the pipe's rate, and the readout's rolled loop alone does not, which is the loss the
drain overlap exposes -- the readout's next form is measured here first.
