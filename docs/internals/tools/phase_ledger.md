# `tools/phase_ledger.py` — the kernel's phase clock, read on a cell

One carry launch (by default three, averaged) on a registry cell with `PhaseClock` bound
(`carry_ledger_bind`; `docs/internals/carry/carry_kernel.md#phase-ledger`), printed as cycles a
warp a window per phase — head, readout, fold, snapshot, edges, sweep, and the head's own two edges
(head_words up to the counts edge, head_scans up to the second one; `head` is then the scatter, the
tile masks and the next window's words) — averaged over the CTAs and
warps, then the readout's and the fold's cycles per warp so an owner imbalance shows.

The launch carries a state plane out (the `fresh` arm, what a timing session's `carry_forward` entry times)
unless `--state-arm null`. The first ledger runs passed no plane, so the kernel skipped its exit
sweep and the ledger never saw the phase that was a third of a sparse kernel's time (the scattered
2-byte stores, `carry_kernel.md#state-io`). An instrument that measures a different launch than
the A/B is not an instrument.

Measured work: the GPU lock exclusive (`rola_devtools.locks.gpu`). Under `ncu` it is the one launch to
profile with `--launches 1`.

`--per-warp` names the phases whose per-warp rows are printed (default `readout,fold`). Beside a CTA
barrier those rows carry the barrier's release wait on whichever side issues first (every barrier
here defers its block, `carry_kernel.md#phase-ledger`), so they show that an imbalance exists; a wait
is read from arrival stamps taken before the barrier, not from the rows. A single launch's totals move
by up to ~5% run to run (nl64k-alt-k4 21,971 / 22,008 / 23,252 on one source); time is the probe's.

`--warmup` (default 1) runs launches before the ledger is bound: a fresh binary's first launch read
the fold at 126,797 cycles a warp a window and its second at 35,736 (2026-09-12), so the first launch is
never a reading.

`--json PATH` also writes the reading: the cell, launches, warm-up, state arm, the launch's CTA-windows, every phase's
cycles a warp a window and the total, and every phase's per-warp row. `cta_windows(cell)` (owners times windows) is the
one divisor every per-unit number in the instruments uses (`tools/pipe_counters.py`, `tools/stall_census.py`).
