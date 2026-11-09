# `carry/carry_arm.cuh` — one arm's launch

Mirrors `csrc/rola/src/carry/carry_arm.cuh`, included only by the generated per-arm units
(`csrc/rola/src/instantiations/`, `tools/gen_shards.py`).

<a id="plan"></a>
## The plan's size

An arm's kernels are templated on the BUILD (`Plan<Arm, B>`, `ampere/plan.md#plan`), and the host reaches the current
device's build through the selection layer (`build_switch` over the builds the binary carries:
[`../common/select.md`](../common/select.md)): the host and the device read one plan, and a build's kernels exist in
its image alone (`in_image`). The kernel still traps at entry when the launched dynamic size is not its plan's
(`%dynamic_smem_size`): a host sized for another build never runs it.

<a id="launch"></a>
## The launch

`carry_launch<Arm>` takes the current card's build and launches, at it (`carry_launch_on<B, Arm>`), the plan pass
(`window_plan_kernel<BuildTag<B>, Arm>`: a block a box class and batch-head, a thread a window of each `kPlanThreads`-window tile, `ampere/schedule.md#pass`)
and then `carry_kernel<BuildTag<B>, Arm>` on `(boxes, batch-heads)` with 256 threads and the plan's
dynamic shared memory, opting in once an arm behind a function-local flag. The launch bound is `(256, 1)`: one CTA an
SM on every architecture. `carry_census<Arm>` reads the current build's function attributes back (registers, local bytes,
its maximum block) beside its plan's form (`carry_arm_abi.md#form`). `ROLA_CARRY_ARM_DEFINE` stamps both for one row of the shipped
set, keyed by its six fields.
