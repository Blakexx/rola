# `issue_latency`

Mirrors `measure/lab/experiments/issue_latency.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

How many cycles after an instruction issues a dependent one can issue: a shared load to its use, a tensor MMA to the
next MMA on its accumulator, and an FFMA to its use, the instrument's probe (documented at 4 cycles on compute
capability 8.x). One warp a scheduler; each repetition reads the SM clock, runs a chain of K links, each link's
operand the one before's result (a shared row holding its own address, so a load returns the next link's address;
the first link tied to the clock read by a run-time zero), and reads the clock again -- in-order issue: after the
K-th link has issued. The latency is the slope between K = 32 and K = 64, the clock reads' own cost cancelled.
sm_86 (RTX 3080 Ti), cycles a link:

| chain | latency |
|---|---|
| FFMA (the probe) | 4.0 |
| `LDS.32`, its address the load's value | 23.0 |
| `LDSM.x4`, its address the load's value | 29.0 |
| fp16-accumulate `HMMA.16816` on one accumulator | 24.0 |
| fp32-accumulate `HMMA.16816` on one accumulator | 32.9 |

The FFMA row reads the documented 4, so the method holds. The fp32-accumulate chain runs at its pipe time on this
card (32.1, [`tensor_tax`](tensor_tax.md)), so its latency is at most that; the fp16-accumulate chain, at 16.1 a
pipe slot, shows the MMA's latency, 24. An `ldmatrix` of four matrices lands 6 cycles after a 32-bit load. These are
`common/hardware.json`'s `latency` rates on sm_86 ([`hardware.md`](../internals/common/hardware.md)). Record:
`measure/lab/results/issue_latency/sm_86.json`.
