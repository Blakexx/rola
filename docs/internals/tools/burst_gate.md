# `tools/burst_gate.py` — what a burst loop's SASS must satisfy, deterministically

Mirrors `tools/burst_gate.py` and `tools/sass_control.py`; the `burst` instrument of every suite
run. KERNEL_STANDARDS §23 states the rule; this is the gate that holds it on the compiled code.

<a id="control"></a>
## The control words

Every instruction on Volta, Turing and Ampere carries, in bits 105 to 125 of its encoding, what
ptxas decided about it: the stall count before the next issue, a yield bit, the scoreboard barrier
it sets on completion and the one guarding its source registers, the mask of barriers it waits
on, and the reuse flags (Jia et al.; validated on this card by the reuse bits matching the
disassembly's `.reuse` operands). ptxas sets a stall to the producer's latency minus the
instructions it placed between producer and consumer (Huerta et al.), so the words record its own
accounting of every dependence. `sass_control.py` decodes them off `cuobjdump -sass` and finds
the loop a function compiles to: the instructions any inline frame of which lies in the function's
lines, from the target of the last backward branch to that branch.

<a id="purity"></a>
## Purity

A `//: @burst` function's instructions hold no `VOTE`, `BAR`, `BSSY`/`BSYNC`, `CALL`, `WARPSYNC`,
`MEMBAR`, `BRA`, `EXIT` or `RET`: the constructs the compiler inserts, which the source lint
(`tools/lint/burst_tier.py`) cannot see — a collective after a value ptxas cannot prove uniform
gets a convergence bracket or a subroutine, each a serialization point between two bursts. A row
a function: instructions, HMMAs, shuffles (allowed), the forbidden opcodes; red on any.

<a id="coverage"></a>
## Coverage

A burst loop covers itself on ONE warp when the pipe, not the warp, is its bound alone: the loop's
period for a lone warp, simulated from the control words with the tensor pipe as a resource
(`pipe_sim.simulate`), against its HMMAs' pipe time — the occupancy. Why the lone warp: two warps
sharing a scheduler can only improve on a loop that covers itself alone, and the lone case is what
the partner's drain, fill and walk expose. Why not simply "issue time under pipe time": the
readout's rolled box loop issues 210 cycles of work against 260 of pipe time and still runs at
69% alone, because its non-HMMA work sits after the burst and the pipe empties while it issues;
only a model with the pipe as a resource sees that.

Beside the occupancy, the EXPOSED LATENCY: the cycles a consumer waits on a barrier whose
producer's latency the placed distance did not cover, each dependence named
(`SHFL@52->HFMA2@55: 17`). It is the load-gap reading: the fold's pair loop admits 58 cycles of it
in 1,170 of pipe time and is pipe-bound anyway.

The floors (`tools/budgets/burst_floors.json`) are a ratchet: `--write-floors` records every
loop's occupancy, and a later build that reads lower is red. The loops: every `Burst::run`, and
`--loop FILE:FUNCTION` for a rolled loop kept under the same floor (the readout's box loop).

<a id="registers"></a>
## Registers

The arm's allocation (`cuobjdump -res-usage`), rounded to the eight-register grain, against the
partition's 16,384 registers: the warps a scheduler it leaves room for, and the threshold that
buys one more (248 buys two, 168 three — GA102 whitepaper, Jia et al.).

<a id="readings"></a>
## Readings at 7468b883 (2026-09-18), the fitted model

| loop | instructions | HMMAs | alone | paired | exposed | floor |
|---|---|---|---|---|---|---|
| the fold's pair (`carry_kernel.cuh:step`, the `Burst::run` user) | 155 | 36 | 1,351 of 1,170: 86.6% | 2,400: 97.5% | 58 (three shuffle-to-multiply chains) | 86.6 |
| the readout's box loop (`readout_tile`) | 68 | 8 | 381 of 260: 68.2% | 520: 100% | 24 (the outer-factor load to its multiply) | 68.2 |

A loop is named by its burst function (`--loop FILE:FUNCTION`, the fold's `frag_mma`, the readout's
`readout_tile`), never by `burst.cuh:run`: every user inlines the same lines of the primitive. The loop found
is the backward-branch loop of the whole kernel that is DENSEST in the function's HMMAs -- the primitive's
control carries the primitive's frames, not its user's, and a phase loop around the burst holds more HMMAs
but fewer per instruction. Purity attributes a branch by its innermost frame (a burst function's frames sit
under the primitive's when the branch is the primitive's exit or latch) and does not count a vote on the
constant predicate (the uniform datapath materializing a warp-uniform value).
Registers 244, allocated 248: two warps a scheduler; a third at 168. The floors were re-seeded when the
model was fitted (`pipe_sim.md#calibrate`): the alone reading lost the two-deep HMMA queue and gained the
SM's memory pipe, so the same loops read lower than under the first model (94.8 and 82.3).

<a id="calibration"></a>
## What the model is and is not

The model is read against the calibration probe before it is read against the kernel
(`tools/pipe_sim.py --calibrate`, KERNEL_STANDARDS §23 addendum 2): it reproduces every tensor-pipe,
gather-chain and kernel-pattern load row within 2%, and misses the reduction rows and the four-warp
load row by known amounts, stated with the table in `pipe_sim.md#calibrate`. Its first version was
called trusted on one point (a fragment pair 2,147 measured against 2,340 modelled) and budgeted the
fold's mass on the FMA pipe at -11% paired; the form measured +3% on the fragment and lost the A/B.
That second point still misses under the fitted model (2,133 modelled against 2,223 measured for the
form, 2,400 against 2,159 for the reference), so the fold's fragment is bound by something the probe
does not exercise, and the per-line stall census of the two forms is the measurement that names it.
Until it does: the floors are ratchets on the alone reading (a form that reads lower than the last is
refused), the paired reading is advisory, and no form is budgeted from the model.

Forms read this way (2026-09-18): the two-box readout on the burst primitive, alone 73% then 81% under
the first model after the gate named its two exposed dependences (a list-byte load and a four-load XOR
chain), paired 95 to 98% against the rolled loop's 100% -- the measured +7% paired loss, reproduced
without a launch; the rotated rolled loop, alone 80%, no better than the plain rolled loop. The plain
rolled loop stays.
