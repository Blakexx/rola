# `mma_overlap`

Mirrors `measure/lab/experiments/mma_overlap.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A step of 512 tensor cycles a warp (fp16-acc, fp32-acc or int8 MMAs) plus the fold's operand chain, 1-4 warps a
scheduler, inline or staged, free-running, a CTA barrier every pass of 8 steps, or a two-stage mbarrier (arrive at a
pass's end, wait for the pass before) polled or sleeping 64 ns between polls; the block's span a step over the floor.
sm_86 (fp16-acc / fp32-acc / int8):

| warps a scheduler | inline, free | staged, free | inline, barrier | inline, two-stage polled | inline, two-stage sleeping |
|---|---|---|---|---|---|
| 1 | 1.42 / 1.23 / 1.48 | 1.15 / 1.14 / 1.17 | 1.44 / 1.28 / 1.49 | 1.47 / 1.28 / 1.52 | 1.47 / 1.27 / 1.52 |
| 2 | 1.05 / 1.03 / 1.07 | 1.04 / 1.05 / 1.06 | 1.11 / 1.08 / 1.12 | 1.05 / 1.04 / 1.09 | 1.06 / 1.05 / 1.08 |
| 3 | 1.05 / 1.02 / 1.06 | 1.04 / 1.04 / 1.05 | 1.08 / 1.06 / 1.09 | 1.04 / 1.06 / 1.07 | 1.04 / 1.04 / 1.06 |
| 4 | 1.04 / 1.04 / 1.06 | 1.04 / 1.04 / 1.06 | 1.08 / 1.07 / 1.09 | 1.08 / 1.07 / 1.10 | 1.05 / 1.05 / 1.07 |

One warp a scheduler cannot hide its own chain; a stage recovers most of it. Two or more free-running warps hide each
other's, and a stage adds nothing. A barrier a pass costs 3-6%; the two-stage mbarrier recovers it, polled at two warps
a scheduler, sleeping at three or four (a polling warp takes issue slots from the working ones). Record:
`measure/lab/results/mma_overlap/sm_86.json`.
