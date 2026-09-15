# ESSENTIALS — injected after every compaction / resume. Read before acting.

Everything here is IN THIS REPO. A site's own notes (scratch, memory, the queue) are appended by
the hook only when the environment names them (`ROLA_SCRATCH`); the repo never references out.

## The constitution and the laws
`docs/KERNEL_STANDARDS.md` — two tiers: the CONSTITUTION (standing rulings) and the LAWS (dated,
incident-derived). Read §19–§21 before touching a kernel:
- §19 THE COMPONENT BUDGET GATE: SKELETON FIRST (stub components composed in the final form, compiling,
  ledger and edges final); then ONE COMPONENT AT A TIME, each with a BUDGET LINE before code (analytic
  floors, instruction budget from the design's own operation count — `benchmarks/bench/carry_model.py`,
  `tools/budgets/<kernel>.json` — and WHO executes it HOW MANY TIMES: per-window work once per CTA, never
  on every warp); gated in the part harness (`benchmarks/unit/bench_carry_parts.py`) at <= 1.3x its
  binding floor, then in the composed kernel's phase ledger. No design verdict from an ungated build.
- §20 ptxas SIGNATURES: `tools/sass_gate.py <.so>` on every iteration build — local memory (runtime index
  into a register struct / constexpr helper with a runtime arg / non-inlined capturing lambda),
  convergence subroutines (a collective under a branch on a loaded value), predicated HMMA / if-converted
  warp-uniform `if`s (a one-trip loop forces a branch), predicated chains where a straight-line block
  (all live) or a jump table (one live) belongs, copy tasks not contiguous across a warp.
- §21 THE MEASUREMENT PROTOCOL: quiet GPU (`nvidia-smi --query-compute-apps` empty), the baseline's OWN
  schedule for the cell, same grid, `smsp__inst_executed` + `sm__cycles_active/elapsed` beside the time,
  clock rows accepted (a REFUSED row voids the run); "issue active %" never stands in for the count.
- Efficiency is correctness; build the whole design or stop and name the gap; no fallbacks, no dead code;
  comments are decl blocks, prose goes to `docs/internals/`; oracle = truth (`tests/oracle/`).

## The instruments
- Perf: `tools/compare.py` (locked clock, interleaved arms on a point's cells); attribution:
  `tools/region_ledger.py --csv <ncu SourceCounters> --so <.so> --source <kernel.cuh> --budget tools/budgets/<k>.json`.
- Correctness: `tests/oracle/test_carry_vs_oracle.py` (scoped runs only; never a bare `tests/`).
- Build: `python -m pip install -e . --no-build-isolation` with `ROLA_CUDA_ARCHS`; the machine is `tools/dev.py check`
  (`docs/build.md`); the device stamp proves the binary is the source.
- Kernel map: `docs/internals/carry/`; the carry body is `csrc/rola/src/carry/carry_kernel.cuh`, its plan
  `box.cuh`, its host entry `carry.cu`.
