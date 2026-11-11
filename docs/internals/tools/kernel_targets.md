# `tools/kernel_targets.py` — the instruments' kernels

What a per-launch instrument needs to read one kernel family on a central cell, in one table (`TARGETS`): the
profiler's kernel filter (`symbol`, which `ncu -k regex:` matches against the plain function name), the provider arm
whose one untimed launch it captures (`arm`, `measure.provider.oneshot_argv`), the cubin member and the sources the
region ledger attributes SASS lines against (its member and sources), the family's budget file where it has one, and the
budget's unit.

| family | arm | unit |
|---|---|---|
| `facts` | `facts_pass` | windows |

The facts pass is the table's one family on this line: a kernel with a schedule adds its row with the schedule.
`units(target, cell)` is the launch's count of the unit from the cell's record alone (one batch row, the cell's write
heads); `launch(target, name)` the cell and the provider arm an instrument launches.

`tools/stall_census.py` and `tools/pipe_counters.py` take `--kernel FAMILY` from this table.
`bazel run //tools:kernel_targets -- CELL` prints every family's target on a cell.
