# `tools/overlap_gate.py` — a kernel's time against the largest of its floors

KERNEL_STANDARDS §28. A ROW is a kernel on a perf cell (`ROWS`: the kernel, the launch that calls it, the cell, its
heads, its MMA kind if it issues any). One `ncu` launch of the row, after `warmup` launches of the same call, reads:
the elapsed time and SM clock, instructions executed, shared wavefronts, HMMA and IMMA instructions, and DRAM's share
of its sustained peak. The FLOORS, each a time:
- issue: instructions over the SMs' schedulers at one a clock (`common/hardware.json`: `schedulers`, `issue`);
- tensor: the row's MMA kind's instructions at the product's calibrated cycles each, over the SMs' schedulers --
  never the profiler's pipe ratio (§22: GeForce runs fp32-accumulate at half the rate the profiler assumes); a row of
  several shapes (`mma="shapes"`) prices each executed MMA by its SASS shape, read from a second
  `ncu` launch's source counters: its accumulate type's rate, an m16n8k8 (or an int8 m16n8k16) half its larger shape;
- shared: wavefronts at the product's `wavefront` cycles each, over the SMs;
- DRAM: the bytes at the profiler's sustained DRAM peak, `dram__throughput` x elapsed.

The ratio is elapsed / max(floors), red above `LINE` = 1.15. A kernel at its binding floor reads near 1.0; a kernel
whose phases take turns, one idling a stream while another runs, reads near the sum of its floors over their max.

    bazel run //tools:overlap_gate                 # every row; exit 1 when any is red
    bazel run //tools:overlap_gate -- --row facts-alt-k4

THE INSTRUMENT'S CHECK (2026-10-07, RTX 3080 Ti): a pure read stream -- a torch reduction over 512 MB -- reaches 96.3%
of the sustained DRAM peak, 1.04x, so the DRAM floor is reachable. The tensor floor's probe row (2026-10-16,
`--probe probe-gemm-f16`): torch's cuBLAS fp16 GEMM at 8192 reads 1.02x its tensor floor, priced by shape, so the
tensor floor is reachable and its arithmetic holds.

THE ROWS' FIRST READING (2026-10-07, ncu's locked clock): `facts-alt-k4` 300.4 us, 1.48x its DRAM floor (202.7 us;
issue 166.5, shared 89.3) -- the list build's serial phase; `facts-dense` 185.0 us, 1.15x its DRAM floor.

`tools/lint/coverage.py` reads `ROWS` by AST: every kernel's `overlap` obligation in `tools/coverage.json` names a row
gating it, or is exempt or owed with its reason.
