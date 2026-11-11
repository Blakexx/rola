# `wgmma_rate`

Mirrors `measure/lab/experiments/wgmma_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What is one `m64n64k16` wgmma's own time on the tensor pipe, apart from its commit group's round trip, and do two
warpgroups of a CTA share that pipe?

THE FORM. One or two warpgroups a CTA, a CTA an SM. A step issues K wgmmas into K accumulator sets in one commit group
and waits for it, K = 1, 2, 4 and 6, with A in registers or in shared memory. Every operand descriptor and register
A's words are held in registers, so the step loop is the fence, the K `HGMMA`, the wait and the branch -- read in the
SASS (sm_90a): no descriptor arithmetic inside it. The slope of a step's cycles over K is a wgmma's own time and the
intercept the group's round trip; the K = 1 rows are [`wgmma_regs`](wgmma_regs.md)'s readout-alone forms with the
descriptors held. Every accumulator is checked against the host's.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60165936, driver 595.91.07; 2026-11-11, one session, 7
repetitions, every accumulator exact against the host's; cycles a step, the median, the spread under 0.1%):

| wgmmas a group | A registers, one warpgroup | A shared, one | A registers, two warpgroups | A shared, two |
|---|---|---|---|---|
| 1 | 96.0 | 105.0 | 98.1 | 103.1 |
| 2 | 123.0 | 132.0 | 128.1 | 140.1 |
| 4 | 182.1 | 191.1 | 256.1 | 256.1 |
| 6 | 246.1 | 255.1 | 384.1 | 384.1 |

WHAT IT SHOWS.
- A WGMMA'S OWN TIME is 32 cycles of the SM's tensor pipe (`m64n64k16`, bf16 into fp32: 65,536 products). Two
  warpgroups at four and six a group run at exactly 64 cycles a wgmma each, the two sharing one pipe; one warpgroup's
  step grows by 27 to 32 a wgmma.
- A GROUP'S ROUND TRIP is about 64 cycles: one wgmma a group is 96, so a step that commits and waits a single wgmma
  spends two thirds of itself not on the pipe, and a lone warpgroup reaches the pipe's rate only as its groups deepen
  (41 a wgmma at six).
- With one warpgroup, A in registers is 9 cycles a step UNDER A in shared memory at every depth; with two the gap is
  5 and 12 at one and two a group and nothing once the pipe binds. [`wgmma_regs`](wgmma_regs.md)'s slow fixed-register
  row (136) reads 96 here, in a kernel that differs from that one in three ways (descriptors held, A's words held,
  launch bounds): the register operand is not what was slow, and which difference was is not isolated.

Record: `measure/lab/results/wgmma_rate/sm_90a.json`; its counters are the written exception (no profiler on that
box).
