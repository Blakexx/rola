#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STALL CENSUS (KERNEL_STANDARDS §22 (7), (8)): every warp-stall sample of one launch of a kernel family on a
cell (the facts pass by default), by source line's component and stall reason; the phase census and the wavefront census --
raw counts.

    bazel run //tools:stall_census -- flagship-dense --json census.json
    bazel run //tools:stall_census -- nl64k-alt-k4 --kernel facts --json census.json   # any family of kernel_targets.py

One oneshot launch through the probe worker under `ncu --section SourceCounters` with the stall-reason group and the
kernel's source imported, exported per SASS line, then attributed by `tools/region_ledger.py` against the installed
extension and the family's budget's components (`tools/kernel_targets.py`). The JSON is region_ledger's: every component's instructions a
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
import kernel_targets  # noqa: E402
import toolchains  # noqa: E402
from rola_devtools import process  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measure"))
from measure.provider import oneshot_argv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def measure(cell: str, out: Path, keep_csv: Path | None, family: str) -> None:
    from measure.cells import by_name

    ncu = dev_config.ncu()
    t = kernel_targets.target(family)
    launch_cell, arm = kernel_targets.launch(t, cell)
    fn, per, mem = t.symbol, kernel_targets.units(t, by_name(cell)), t.member
    with tempfile.TemporaryDirectory(prefix="stall_census_", dir=dev_config.scratch("stall_census")) as tmp:
        rep = Path(tmp) / "rep"
        done = process.run([ncu, "--target-processes", "all", "-k", f"regex:{t.symbol}", "-c", "1", "--section",
                            "SourceCounters", "--metrics", "group:smsp__pcsamp_warp_stall_reasons", "--import-source",
                            "yes", "-f", "-o", str(rep), *oneshot_argv(launch_cell, arm)],
                           cwd=ROOT, timeout=3600)
        exported = process.run([ncu, "--import", f"{rep}.ncu-rep", "--page", "source", "--print-source", "sass", "--csv"],
                               cwd=ROOT, timeout=1800)
        if exported.returncode:
            raise SystemExit(f"stall_census: the capture or export failed:\n{(done.stdout + done.stderr + exported.stderr)[-1500:]}")
        csv = Path(tmp) / "source.csv"
        csv.write_text(exported.stdout)
        if keep_csv is not None:
            keep_csv.write_text(exported.stdout)
        so = toolchains.built_extension(ROOT)
        budget = ["--budget", t.budget] if t.budget else []
        ledger = subprocess.run([PY, "tools/region_ledger.py", "--csv", str(csv), "--so", str(so), "--member",
                                 mem, "--kernel", fn, "--source", *t.sources, *budget, "--cell", cell,
                                 "--per", str(per), "--json", str(out)],
                                cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if not out.exists():
        raise SystemExit(f"stall_census: region_ledger wrote nothing (rc {ledger.returncode}):\n"
                         f"{(ledger.stdout + ledger.stderr)[-1500:]}")
    doc = json.loads(out.read_text())
    doc.update({"cell": cell, "kernel": family, "unit": t.unit, "budget_red": ledger.returncode == 1})
    out.write_text(json.dumps(doc, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--kernel", default="facts", choices=sorted(kernel_targets.TARGETS),
                    help="the kernel family launched and attributed (tools/kernel_targets.py)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="keep ncu's per-SASS-line export here (the warp timeline joins it); "
                         "default the tool's scratch, `<scratch>/stall_census/<cell>.csv`")
    a = ap.parse_args()
    measure(a.cell, a.json, a.csv or dev_config.scratch("stall_census") / f"{a.cell}.{a.kernel}.csv", a.kernel)
    doc = json.loads(a.json.read_text())
    print(f"{a.cell}: {doc['samples']} stall samples over {len(doc['components'])} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
