# The lab's experiments

One page per experiment, mirroring `measure/lab/experiments/<name>.cu`: the question it asks of the machine, its
method, its record (`measure/lab/results/<name>/<sm_XX>.json`) and what it settled. How an experiment is written, run
and recorded, and what guards a measurement, is [`docs/internals/tools/lab.md`](../internals/tools/lab.md); a law
about the machine cites the experiment it rests on ([`docs/KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26)).

| experiment | kind |
|---|---|
| [`tensor_tax`](tensor_tax.md) | experiment |
| [`collective_branch`](collective_branch.md) | experiment |
| [`smem_rates`](smem_rates.md) | experiment |
| [`row_span`](row_span.md) | experiment |
| [`vote_stash`](vote_stash.md) | experiment |
| [`facts_block`](facts_block.md) | experiment |
| [`facts_two`](facts_two.md) | experiment |
| [`issue_latency`](issue_latency.md) | experiment |
| [`wgmma_regs`](wgmma_regs.md) | experiment |
| [`bulk_drain`](bulk_drain.md) | experiment |
| [`reg_trade`](reg_trade.md) | experiment |
| [`row_ring`](row_ring.md) | experiment |
| [`wgmma_rate`](wgmma_rate.md) | experiment |
| [`byte_expand`](byte_expand.md) | experiment |
| [`drain`](drain.md) | experiment |
