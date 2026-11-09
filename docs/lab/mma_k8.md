# `mma_k8`

Mirrors `measure/lab/experiments/mma_k8.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The tensor pipe's cost of an m16n8k8 MMA against an m16n8k16, two warps a scheduler, 32 independent accumulators.
Measured (sm_86, cycles an MMA a warp): fp32 accumulate 64.0 (k16) and 31.2 (k8); fp16 accumulate 32.0 and 15.6 --
k8 costs exactly half. Record: `measure/lab/results/mma_k8/sm_86.json`.
