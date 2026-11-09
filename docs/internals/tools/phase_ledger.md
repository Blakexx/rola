# `tools/phase_ledger.py` — the kernel's phase clock, read on a cell

One carry launch (by default three, averaged, after a 500 ms warm-up) on a registry cell with `PhaseClock` bound
(`rola.ops.carry.ledger_bind`; `docs/internals/carry/ampere/carry_kernel.md#phase-ledger`), printed as cycles a
warp a window per phase (`rola.ops.carry.LEDGER_PHASES`, the kernel's order: `carry_kernel.md#phases`), averaged over
the CTAs and warps, then the readout's and the fold's cycles per warp so an owner imbalance shows.

The operands are `measure.subjects.carry_operands`, the same the timing session's `carry_forward` entry times. The
ledger has one row a CTA, a CTA a box of the cell's carve (`ctas(cell)`).

Measured work: the GPU lock exclusive (`rola_devtools.locks.gpu`). Under `ncu` it is the one launch to
profile with `--launches 1`.

`--kernel produce` runs the producer (`rola.ops.produce.ledger_bind`, `docs/internals/produce/ampere/produce_kernel.md#phase-ledger`)
as the layer calls it, its argument a layer construction (`measure/cells/layer.py`) on its first cell: its ledger is
`[jobs][warps][phases]` booked over the token tiles, read as cycles a warp a tile, the route jobs and the value jobs
apart.

`--kernel intra` runs the intra (`rola.ops.intra.ledger_bind`, `docs/internals/intra/ampere/intra_kernel.md#phase-ledger`): its ledger is `[warps][phases]` booked over every CTA, so its reading is a warp slot's cycles a window summed over the window's CTAs. `--per-warp` names the phases whose per-warp rows are printed (default `readout,fold`; the den's `den,writes`, the intra's `gram,wait`). A single launch's totals move
by up to ~5% run to run; time is the probe's.

The clock is not free: bound, the carry runs ~16% longer (nl64k-alt-k4's first 32 windows on sm_86: 902 us against
1,045, the window lab's record `measure/lab/results/carry_window`), since each lap's clock read is a point ptxas does not schedule across. The ledger
attributes a launch's time to phases; the time itself is read with it unbound. Launches run before the ledger is bound, at least `--warmup` (default 1)
and for at least `--warmup-ms` (default 500): a fresh binary's first launch is never a reading, and a GPU just locked
reads ~11% slow for its first launches (nl64k-alt-k4 25,460 cycles a warp-window against 22,657, 2026-10-19). `--forced` names the plan's form each window takes where it fits
(`rola.ops.carry.forced_word`, `docs/internals/carry/ampere/schedule.md#planner`), so forms are read side by side.

<a id="budget"></a>
`--budget tools/budgets/<kernel>.json` judges each phase against the cycles a warp a window the design predicted for it
(the cell's `phase_cycles` in the budget, KERNEL_STANDARDS §32.2): red over `MISS` = 1.3 times its prediction, or with
none. The carry's predictions are owed by the window's design, so its check is red until they exist.

`--per-cta` also prints the phases of the grid's first wave of CTAs (one an SM, the first `SMs` in launch order) and
of the rest apart: a phase bound by a resource the SMs share (the L2) runs faster in a partial last wave, one bound
inside the SM does not (the carry's fill, 2026-11-01: 3,302 cycles a warp-window with 48 SMs busy against 3,340 with
80).

`--json PATH` also writes the reading: the cell, launches, warm-up, the launch's CTA-windows, every phase's cycles a
warp a window and the total, and every phase's per-warp row. `cta_windows(cell)` (CTAs times windows) is the one divisor
every per-unit number in the instruments uses (`tools/pipe_counters.py`, `tools/stall_census.py`).
