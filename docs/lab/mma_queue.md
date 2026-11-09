# `mma_queue`

Mirrors `measure/lab/experiments/mma_queue.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Whether a warp can issue MMAs ahead of the tensor pipe -- fill a queue, then go on to other work while it drains.
One warp a scheduler; each repetition reads the SM clock, issues K independent MMAs into fresh accumulators (their A
operand tied to the clock read by a run-time zero, so none issues before it; the SASS keeps clock, MMAs, clock in
order), reads the clock again -- in-order issue: after the K-th MMA has issued -- then waits for the outputs, so the
next repetition starts on an empty pipe. sm_86, cycles from clock to clock:

| K back to back | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| fp16-accumulate HMMA | 60.0 | 76.0 | 92.0 | 108.0 | 172.0 | 236.0 | 300.0 |
| fp32-accumulate HMMA | 60.0 | 92.8 | 124.0 | 155.5 | 282.2 | 411.5 | 539.8 |

Linear from the second MMA: each issues one MMA's pipe time after the one before (16.0 and 32.1 cycles), so there
is no queue -- the pipe takes the next MMA only when it has taken the last, and a warp issuing a burst of N is held
at it for N times that. (The 60 is the clock read's latency, the operand's and the first MMA's.) An MMA therefore
needs an issue slot at the cycle the pipe frees, and any other warp issuing then costs the pipe that cycle; Hopper's
`wgmma` is asynchronous, which is what lets a warpgroup issue its MMAs and leave. Record:
`measure/lab/results/mma_queue/sm_86.json`.
