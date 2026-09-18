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
## Readings at 20bfa2a4 (2026-09-18), the pipe two deep

| loop | instructions | HMMAs | alone | paired | exposed | floor |
|---|---|---|---|---|---|---|
| the fold's pair (`burst.cuh:run`) | 155 | 36 | 1,234 of 1,170: 94.8% | 2,340: 100% | 58 (three shuffle-to-multiply chains) | 94.8 |
| the readout's box loop (`readout_tile`) | 68 | 8 | 316 of 260: 82.3% | 520: 100% | 24 (the outer-factor load to its multiply) | 82.3 |

Registers 244, allocated 248: two warps a scheduler; a third at 168.

<a id="calibration"></a>
## What the model is and is not, so far

The pipe queue is two deep (the row with four loads after each burst hid 35 of their 42 cycles: one
slot's worth). Paired, the model puts both loops at the pipe's rate, which the trace confirms
(a fragment pair 2,147 measured against 2,340 modelled; a box 559 against 520: the model is 5 to
8% fast). ALONE the model is optimistic, and by a known amount: the fragment-chain calibration
row, eighteen HMMAs behind their gather with one warp a scheduler, measures 830 cycles where the
model says about 620; and the trace's "lone" tile, during the partner's drain, runs at 44% where
the model says 82% -- because a partner in its drain is not absent, it is issuing stores, loads
and reductions through the same memory pipe the tile's `ldmatrix` uses. The model has no memory
pipe. So the floors are model-relative ratchets (a form that reads lower than the last is
refused), the paired reading is trustworthy, and the lone reading ranks forms without predicting
their cycles. `--calibrate`, the step that fits the latency table and adds the memory pipe against
the two calibration rows and the trace, is the next thing the tool needs; until then the trace
remains the measurement of a form's lone rate.

Forms read this way today (2026-09-18): the two-box readout on the burst primitive, alone 73%
then 81% after the gate named its two exposed dependences (a list-byte load and a four-load XOR
chain), paired 95 to 98% against the rolled loop's 100% -- the measured +7% paired loss, reproduced
without a launch; the rotated rolled loop (the next box's loads at the bottom of the body), alone
80%, no better than the plain rolled loop's 82%. The plain rolled loop stays.
