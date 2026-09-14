# `tools/stall_census.py` — every warp-stall sample of one launch, attributed

KERNEL_STANDARDS §22 (7) and (8). One untimed launch of a cell through the probe worker, under
`ncu --section SourceCounters` with the warp stall-reason group and the kernel's source imported. The capture is
exported per SASS line, then attributed by `tools/region_ledger.py` against the installed extension and
`tools/budgets/carry.json`'s components.

    python tools/stall_census.py flagship-dense --json census.json [--schedule first]

The JSON is region_ledger's `--json`, plus the cell, the schedule and whether a component exceeded its budget. It holds:
- every component's instructions per CTA-window and its stall samples by reason (spin loops as their own
  `[spin]` components);
- the phase census: HMMAs and other instructions per unit, and the samples at each;
- the wavefront census: shared-memory wavefronts above ideal per unit, in total and by source line with the line's
  text.

Samples are counts. Turning them into cycles (a component's share of the samples times a phase ledger's total) is the
reader's, as the composer does for its part steps. The raw `.ncu-rep` capture is scratch and is not kept.
