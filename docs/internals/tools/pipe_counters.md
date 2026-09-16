# `tools/pipe_counters.py` — the profiler's pipe and resource counters for one launch

KERNEL_STANDARDS §22. One untimed launch of a cell through rola's runner (`measure.provider`'s `oneshot_argv`, the arm a comparison times), under
`ncu --metrics` with the counters in `COUNTERS`:
- HMMAs and instructions executed;
- scheduler cycles, elapsed and active;
- issue activity, warps active and warps eligible;
- shared loads and stores, ldsm, global reductions and global loads;
- shared-memory wavefronts and bank conflicts on loads and stores;
- LSU writeback activity, global reduction requests and sectors;
- DRAM bytes read and written.

    python tools/pipe_counters.py flagship-dense --json counters.json [--schedule first]

The JSON holds every counter's LAUNCH value (`launch`), the counters the profiler did not report (`missing`), and the
launch's CTA-windows. Nothing derived is stored. A reader divides a sum by `cta_windows` for a unit, and computes
tensor utilization as HMMA count × the calibrated cycles an HMMA holds the pipe, over SMs × schedulers × active cycles
(§22: never the profiler's pipe-active ratio). The build ledger, the composer and the suite read this JSON; before
2026-09-12 each carried its own copy of the capture.
