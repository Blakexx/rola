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
## Readings at 9bb3a14 (2026-09-18)

| loop | instructions | HMMAs | alone | exposed | floor |
|---|---|---|---|---|---|
| the fold's pair (`burst.cuh:run`) | 155 | 36 | 1,287 of 1,170: 90.9% | 58 (shuffle to multiply, three chains) | 90.9 |
| the readout's box loop (`readout_tile`) | 68 | 8 | 378 of 260: 68.8% | 24 (the outer-factor load to its multiply) | 68.8 |

Registers 244, allocated 248: two warps a scheduler; a third at 168. Against the trace: the model
runs 8 to 9% fast paired, and the lone box loop's 69% sits between the 49% the drain overlap
implied and the shared rate; the latency table is the literature's until `pipe_sim --calibrate`
sets it from the trace. What the two rows say together: the fold's interleave holds a lone warp
near the pipe's rate; the readout's rolled loop does not, and the gate names the dependence.
