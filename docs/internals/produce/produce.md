# `csrc/rola/src/produce/produce.cuh` + `produce.cu` — the producer, one launch

RULED 2026-09-29: the producer is a declared configuration compiled into ONE kernel launch -- from a stretch of
tokens to the bundle the ops take: every routing level's projection, solve and unit-max rows, the write gain, and the
f16 values. The layer declares its routing (`rola.Routing`), owns the weights, and compiles a producer plan once
(`rola/ops/produce.py`); `job_table` turns the plan into the rows the kernel reads, so every width, kind and offset
is a runtime field and the family is ONE instantiation.

`produce.cuh` holds what the host and the kernel share: the launch's parameters (`ProduceParams`), the job table's
fields and the shape word. `produce.cu` is the host entry: it validates the operands (`check_bf16`, `check_f32`), binds
the phase ledger and launches the kernel ([`produce_api.md`](produce_api.md)); the kernel itself, as it runs on ampere,
is [`ampere/produce_kernel.md`](ampere/produce_kernel.md).
