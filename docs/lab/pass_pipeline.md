# `pass_pipeline`

Mirrors `measure/lab/experiments/pass_pipeline.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A whole pass: the owner part's fill (a quad a token, cp.async from a 33 MB source), its restage (two threads a row, the
coefficient run included, since the pool slot refills two passes ahead) and the fold's step at the arm's registers, two
warps a scheduler. Serial = fill, barrier, restage, barrier, steps, barrier (the owner part). Pipelined = two stages
under mbarriers: fill two passes ahead, restage one ahead -- in a burst at the pass's start, by warp w after its step w,
or by every warp after its step 3 (mid-pass) -- each restage waiting for its tile slot's readers just before it. The
fp16 arm's partial join every two passes: a burst, one list a step (a switch on the step: unrolling the step loop
spills the arm), or the burst staggered (a scheduler's two warps on alternate passes). sm_86, the block's span a pass
over the steps' floor (2112 / 1056 a step):

| pass | fp32 arm | fp16, join burst | fp16, join a list a step | fp16, join staggered |
|---|---|---|---|---|
| serial | 1.20 | 1.54 | 1.65 | 1.55 |
| pipelined, restage burst | 1.11 | 1.34 | 1.48 | 1.36 |
| pipelined, restage a warp a step | 1.11 | 1.35 | 1.45 | 1.35 |
| pipelined, restage mid-pass | 1.05 | 1.27 | 1.32 | 1.25 |
| steps alone | 1.03 | 1.25 | 1.32 | 1.16 |

Mid-pass is the pipeline's best: half a pass of slack each way between the restage's wait (the slot's last readers)
and the next pass's wait (the restage). A burst at the pass's start makes every warp wait at the boundary for the
slowest one's previous pass -- a barrier by another name. The join in a burst costs the fp16 arm 12 points, because both
warps of a scheduler leave the tensor pipe at once and the pair falls into lockstep; staggered it costs ~3; spread a
list a step it costs more (the switch, and FP32 adds beside every step's MMAs). Record:
`measure/lab/results/pass_pipeline/sm_86.json`.
