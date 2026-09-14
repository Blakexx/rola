#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STALL CENSUS (KERNEL_STANDARDS §22 (7), (8)): every warp-stall sample of one carry launch of a cell, by source
line's component and stall reason; the phase census and the wavefront census -- raw counts.

    python tools/stall_census.py flagship-dense --json census.json

One oneshot launch through the probe worker under `ncu --section SourceCounters` with the stall-reason group and the
kernel's source imported, exported per SASS line, then attributed by `tools/region_ledger.py` against the installed
extension and `tools/budgets/carry.json`'s components. The JSON is region_ledger's: every component's instructions a
CTA-window and stall samples by reason, the phase census (HMMAs and other instructions, samples at each), and the
wavefronts above ideal by source line, with the launch's CTA-windows. Scaling samples into cycles (a phase ledger's
total times a share) is the reader's. Docs: docs/internals/tools/stall_census.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import phase_ledger  # noqa: E402
import probe_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def measure(cell: str, schedule: str, out: Path) -> None:
    ncu = dev_config.get("toolchain.ncu")
    with tempfile.TemporaryDirectory(prefix="stall_census_", dir=dev_config.scratch("stall_census")) as tmp:
        rep = Path(tmp) / "rep"
        done = subprocess.run([ncu, "--target-processes", "all", "-k", "regex:carry_kernel", "-c", "1", "--section",
                               "SourceCounters", "--metrics", "group:smsp__pcsamp_warp_stall_reasons", "--import-source",
                               "yes", "-f", "-o", str(rep), *probe_cells.oneshot_argv(cell, schedule=schedule)],
                              cwd=ROOT, capture_output=True, text=True, timeout=3600)
        exported = subprocess.run([ncu, "--import", f"{rep}.ncu-rep", "--page", "source", "--print-source", "sass", "--csv"],
                                  cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if exported.returncode:
            raise SystemExit(f"stall_census: the capture or export failed:\n{(done.stdout + done.stderr + exported.stderr)[-1500:]}")
        csv = Path(tmp) / "source.csv"
        csv.write_text(exported.stdout)
        so = next((ROOT / "rola").glob("_C*.so"))
        ledger = subprocess.run([PY, "tools/region_ledger.py", "--csv", str(csv), "--so", str(so), "--source",
                                 "csrc/rola/src/carry/carry_kernel.cuh", "--budget", "tools/budgets/carry.json", "--cell",
                                 cell, "--per", str(phase_ledger.cta_windows(cell)), "--json", str(out)],
                                cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if not out.exists():
        raise SystemExit(f"stall_census: region_ledger wrote nothing (rc {ledger.returncode}):\n{ledger.stdout[-1500:]}")
    doc = json.loads(out.read_text())
    doc.update({"cell": cell, "schedule": schedule, "budget_red": ledger.returncode == 1})
    out.write_text(json.dumps(doc, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--schedule", default="first", help="the carry family's order policy the launch runs")
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    measure(a.cell, a.schedule, a.json)
    doc = json.loads(a.json.read_text())
    print(f"{a.cell}: {doc['samples']} stall samples over {len(doc['components'])} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
