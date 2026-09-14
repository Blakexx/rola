#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PIPE COUNTERS (KERNEL_STANDARDS §22): the profiler's pipe and resource counters for one carry launch of a cell,
as launch totals, raw.

    python tools/pipe_counters.py flagship-dense --json counters.json

One oneshot launch through the probe worker under `ncu --metrics` (the toolchain's `ncu`): HMMAs and instructions
executed, scheduler cycles elapsed and active, issue activity, warps active and eligible, shared loads and stores, ldsm
and global reductions and loads, shared-memory wavefronts and bank conflicts, LSU writeback, global reduction requests
and sectors, DRAM bytes. The JSON carries every counter's launch value and the launch's CTA-windows, so a reader divides
a sum into a unit and derives utilization; nothing derived is stored. Docs: docs/internals/tools/pipe_counters.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import phase_ledger  # noqa: E402
import probe_cells  # noqa: E402
from rola_devtools import process  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = (
    "gpu__time_duration.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "smsp__inst_executed.sum",
    "sm__cycles_elapsed.avg", "smsp__cycles_active.avg",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.per_cycle_active", "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__sass_inst_executed_op_shared_ld.sum", "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_ldsm.sum", "smsp__sass_inst_executed_op_global_red.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "lts__t_sectors_op_red.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
)


def measure(cell: str, schedule: str) -> dict:
    cmd = [dev_config.get("toolchain.ncu"), "--target-processes", "all", "-k", "regex:carry_kernel", "-c", "1",
           "--metrics", ",".join(COUNTERS), "--csv", *probe_cells.oneshot_argv(cell, schedule=schedule)]
    done = process.run(cmd, cwd=ROOT, timeout=3600)
    values = {}
    for line in done.stdout.splitlines():
        parts = [x.strip().strip('"') for x in line.split('","')]
        if len(parts) > 12 and parts[0].isdigit() and parts[-3] in COUNTERS:
            with contextlib.suppress(ValueError):
                values[parts[-3]] = float(parts[-1].replace(",", ""))
    if not values:
        raise SystemExit(f"pipe_counters: no counters read (ncu rc {done.returncode}):\n{(done.stdout + done.stderr)[-1500:]}")
    return {"cell": cell, "schedule": schedule, "cta_windows": phase_ledger.cta_windows(cell), "launch": values,
            "missing": sorted(set(COUNTERS) - set(values))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--schedule", default="first", help="the carry family's order policy the launch runs")
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    doc = measure(a.cell, a.schedule)
    a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    print(f"{a.cell}: {len(doc['launch'])} counters, HMMAs {doc['launch'].get('sm__inst_executed_pipe_tensor_op_hmma.sum')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
