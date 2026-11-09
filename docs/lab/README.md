# The lab's experiments

One page per experiment, mirroring `measure/lab/experiments/<name>.cu`: the question it asks of the machine, its
method, its record (`measure/lab/results/<name>/<sm_XX>.json`) and what it settled. How an experiment is written, run
and recorded, and what guards a measurement, is [`docs/internals/tools/lab.md`](../internals/tools/lab.md); a law
about the machine cites the experiment it rests on ([`docs/KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26)).

| experiment | kind |
|---|---|
| [`tensor_tax`](tensor_tax.md) | experiment |
| [`mma_queue`](mma_queue.md) | experiment |
| [`mma_pairing`](mma_pairing.md) | experiment |
| [`collective_branch`](collective_branch.md) | experiment |
| [`fold_step`](fold_step.md) | experiment |
| [`mma_overlap`](mma_overlap.md) | experiment |
| [`pass_pipeline`](pass_pipeline.md) | experiment |
| [`pool_layout`](pool_layout.md) | experiment |
| [`sparse_pass`](sparse_pass.md) | experiment |
| [`state_publish`](state_publish.md) | experiment |
| [`pass_fixed`](pass_fixed.md) | experiment |
| [`list_fold`](list_fold.md) | experiment |
| [`read_step`](read_step.md) | experiment |
| [`publish_rounds`](publish_rounds.md) | experiment |
| [`smem_rates`](smem_rates.md) | experiment |
| [`warp_hiding`](warp_hiding.md) | experiment |
| [`row_span`](row_span.md) | experiment |
| [`vote_stash`](vote_stash.md) | experiment |
| [`facts_block`](facts_block.md) | experiment |
| [`facts_two`](facts_two.md) | experiment |
| [`fold_alone`](fold_alone.md) | part harness |
| [`read_image`](read_image.md) | part harness |
| [`read_w1024`](read_w1024.md) | part harness |
| [`image_ring`](image_ring.md) | part harness |
| [`fold_publish`](fold_publish.md) | part harness |
| [`digit_box`, `digit_box32`, `list_box`](digit_box.md) | part harness |
| [`color_union`](color_union.md) | part harness |
| [`mma_k8`](mma_k8.md) | part harness |
| [`issue_latency`](issue_latency.md) | experiment |
| [`row_stream`](row_stream.md) | experiment |
| [`read_stream`](read_stream.md) | experiment |
| [`read_red`](read_red.md) | experiment |
| [`read_value`](read_value.md) | experiment |
| [`read_alone`](read_alone.md) | part harness |
| [`read_dense`](read_dense.md) | part harness |
| [`handoff_alone`](handoff_alone.md) | part harness |
| [`read_group`](read_group.md) | experiment |
| [`drain`](drain.md) | experiment |
| [`handoff`](handoff.md) | experiment |
| [`intra_stream`](intra_stream.md) | experiment |
