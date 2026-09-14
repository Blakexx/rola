# `tools/build_ledger.py` — every measurement a build is judged by, in one run

KERNEL_STANDARDS §22's suite as one command. It drives the tools the gate already has and reads
their output back, so no number is typed twice: `tools/sass_gate.py` (§20 signatures),
`tools/life_ranges.py` (peak live registers per region beside the allocation), the scoped fp64
oracle, `tools/phase_ledger.py` per cell (cycles a warp a window per phase, per-warp rows),
`ncu` SourceCounters and pipe counters with `tools/region_ledger.py` per component (the launch's
CTA-window divisor, `cta_windows`: owners × windows, the phase ledger's own), tensor utilization as
HMMA count × the measured cycles an HMMA holds the pipe over scheduler cycles, the phase census
(§22 (7): HMMAs a phase counted from the dump, utilization a phase against the phase ledger's
cycles, a warp's cycles at HMMA against its other work, instructions against budget), the
wavefront census (§22 (8): shared wavefronts above ideal by source line, the bank conflicts), and
the pipe timeline per cell (§22 (11), `tools/pipe_timeline.py`: the tensor pipe's true utilization over a launch on
silicon), and `tools/probe_cells.py` against the baseline run with its own schedule (`box` dense, `sparse-g32`
sparse; this tree `first`).

The report lands in `<store>/ledgers/<utc>-<sha7>-<tag>.json` and `.md` (the measurements store), and the
markdown carries the per-phase diff against the newest earlier report for the same cells. A step
that fails is a row with its output, never a stop. `--skip` names steps to leave out (a CPU-only
dry run is `--skip oracle,phases,profile,ab`). After the report is written the ledger renders the
dashboard (`tools/dashboard.py`, [dashboard.md](dashboard.md)).

The tree grows: a number computed by hand during a debug session is added as a step the same day
(§22 addendum); the phase and wavefront censuses were the 2026-09-12 debug session's hand
computations. Candidates on the card: the edge barriers' wait split by site; the profiler's
wavefront accounting for `cp.async` (6-8x the address model's ideal on the pool fill, unexplained).
