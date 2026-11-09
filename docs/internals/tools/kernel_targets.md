# `tools/kernel_targets.py` — the instruments' kernels

What a per-launch instrument needs to read one kernel family on a central cell, in one table (`TARGETS`): the
profiler's kernel filter (`symbol`, which `ncu -k regex:` matches against the plain function name), the provider arm
whose one untimed launch it captures (`arm`, `measure.provider.oneshot_argv`), the cubin member and the sources the
region ledger attributes SASS lines against (`member`, `sources`), the family's budget (`tools/budgets/<family>.json`)
and the budget's unit.

| family | arm | unit |
|---|---|---|
| `carry` | `carry_forward` | CTA-windows (the carve's boxes x windows) |
| `intra` | `intra_forward` | warp tile pairs (a warp's 16 readers against a writer tile at or before theirs) |
| `facts` | `facts_pass` | windows |
| `produce` | `produce@layer=C` | job tiles (the token tiles times the jobs) |

`units(target, cell)` is the launch's count of the unit from the cell's record alone (one batch row, the cell's write
heads, each read by its group's query heads); `function(target, cell)` the instantiation the cell launches, as the region ledger's regex
over the cubin's mangled names: the intra's depth and the card's
accumulation (`rola.ops.carry.readout_accumulate`).

The producer's arms are a layer construction's: its instruments take the construction `C` for the cell argument, and
`launch(target, name)` gives the cell (the construction's first) and the arm; `produce_units(C)` its unit count.

`tools/stall_census.py`, `tools/pipe_counters.py` and `tools/phase_ledger.py` take `--kernel FAMILY` from this table,
the carry by default; `declare.py` declares the intra's on the cells of 16K tokens and more
(`census-intra`, `counters-intra`). `bazel run //tools:kernel_targets -- CELL`
prints every family's target on a cell. The pipe timeline reads the carry only (its calibration plateau is the carry's).
