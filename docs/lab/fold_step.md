# `fold_step`

Mirrors `measure/lab/experiments/fold_step.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The fp16 fold's dense step transplanted from `owner_fold.cuh` (two warps a scheduler, the arm's accumulators live --
254 registers on the fp16 arm, as in the owner part), over shared rows, one variant a removal or a stage. sm_86, cycles
a step a warp over the floor (1056 fp16 arm, 2112 fp32 arm): as built 1.074 / 1.022; the MMAs alone 1.033 / 1.004; no
coefficient 1.055 / 1.016; no operand loads 1.067 / 1.026; a CTA barrier every pass of 8 steps 1.171 / 1.026. A class
replaced by a `LOP3` reads slower than the class (shuffles 1.117, HMUL2 1.090): the replacement is not free, so those
two rows bound nothing. Staged forms on the fp32 arm (all, loads alone, coefficients alone) 1.024-1.027, no gain; on
the fp16 arm every staged form spills (12-36 LDL/STL) and is not run. The step alone is near its floor and the barrier
realigns the warps. The timed region holds the steps only: the fp16 arm's partial join (every 256 tokens) falls between
regions, and with it the arm runs 1.25x free-running (`pass_pipeline`). Record:
`measure/lab/results/fold_step/sm_86.json`.
