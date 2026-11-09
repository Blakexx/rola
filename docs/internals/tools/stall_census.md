# `tools/stall_census.py` — every warp-stall sample of one launch, attributed

KERNEL_STANDARDS §22 (7) and (8). One untimed launch of a cell through the probe worker, under
`ncu --section SourceCounters` with the warp stall-reason group and the kernel's source imported. The capture is
exported per SASS line, then attributed by `tools/region_ledger.py` against the installed extension and the family's
budget's components, over the family's files (the kernel's, then its components'): an instruction is the innermost
component's on its inline chain walked outward no further than the kernel's own body (`region_ledger.owner_of`: beyond
the body the line table splices other call sites' frames in, so an outermost rule booked readout work to the store and
the read units to the readout); work no component owns is booked under its own function's name (`helper_of`), the
kernel's own body (`glue`) only where no function below it is. `region_ledger.py --component NAME` lists one component's instructions by their innermost
frame in any file, its helpers' and the headers' included. `--kernel FAMILY` names the family
(`tools/kernel_targets.py`: the carry by default, the den, the intra, the facts pass); the units a total is divided by
are the family's.

    bazel run //tools:stall_census -- flagship-dense --json census.json

The JSON is region_ledger's `--json`, plus the cell, the schedule and whether a component exceeded its budget. It holds:
- every component's instructions per CTA-window and its stall samples by reason (spin loops as their own
  `[spin]` components);
- the phase census: HMMAs and other instructions per unit, and the samples at each;
- the wavefront census: shared-memory wavefronts above ideal per unit, in total and by source line with the line's
  text.

Samples are counts. Turning them into cycles (a component's share of the samples times a phase ledger's total) is the
reader's, as the composer does for its part steps. The raw `.ncu-rep` capture is scratch and is not kept.

<a id="member"></a>The image it attributes is the cell's own (`kernel_targets.member`): a family's one image, or for the
carry, whose arms are an image each when the build carries several (every build but a `--//:carry_arms` subset), the image of the arm the
cell's carve selects (`rola.ops.carry.arm_key` against the declared arm list).
