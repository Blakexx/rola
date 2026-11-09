# `op_timeline.py`

Mirrors [`tools/op_timeline.py`](../../../tools/op_timeline.py).

THE OP'S TIMELINE: what a call spends outside its kernels. A registered subject (`measure.subjects`, the bench's own
build of it; `prefill` by default, the op as `rola_op` runs it) is called `--warmup` times untimed, then `--calls` times
under `torch.profiler` (CPU and CUDA activity), each call synchronized before and after. It prints, a call:

- the wall time, the kernels' busy time (the union of their intervals) and the idle remainder -- the GPU waiting on
  the host;
- each kernel's launches and time;
- every gap between consecutive kernels of `--gap` microseconds or longer, the kernels on either side and the host
  ops that ran across it, by how much of the gap each covers;
- the host ops by their own time.

`--python` profiles one more call's host Python with `cProfile` (by cumulative and by own time), which names the
Python behind a gap the profiler's ops only frame (`_PrefillNode`) -- how the carve planner's cost was found
([`../engine/carve.md#planned-once`](../engine/carve.md#planned-once)).

A synchronized call exposes all of its host time; back to back, a call's host work overlaps the previous call's
kernels wherever it is shorter than them. Measured work: takes the GPU lock exclusive.
