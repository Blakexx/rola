#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OVERLAP GATE (KERNEL_STANDARDS §28): a kernel's time against the largest of its floors, on its perf cells.

    bazel run //tools:overlap_gate      # every row; exit 1 when any is red
    bazel run //tools:overlap_gate -- --row facts-alt-k4

A ROW is a kernel on a perf cell. One `ncu` launch of it (after warm-up launches of the same call) reads the
elapsed time and clock, the instructions, the shared wavefronts, the tensor instructions and DRAM's share of its
sustained peak. Its FLOORS, each a time:
- issue: instructions over the SMs' schedulers at one a clock (`common/hardware.json`'s `schedulers`, `issue`);
- tensor: the row's MMA kind's instructions at the product's calibrated cycles each, over the SMs' schedulers
  (never the profiler's pipe ratio: §22); a row of several shapes (`mma="shapes"`) prices each executed MMA by its
  own (ncu's source counters: its accumulate type's rate, an m16n8k8 or an int8 k16 half its k16 or k32);
- shared: wavefronts at the product's `wavefront` cycles each, over the SMs;
- DRAM: the bytes at the profiler's sustained DRAM peak (`dram__throughput` x elapsed).
The ratio is elapsed / max(floors): 1.0 is a kernel at its binding floor. A kernel whose phases take turns -- one
idling a stream while another runs -- lands near the SUM of its floors, and is red above `LINE`.
Docs: docs/internals/tools/overlap_gate.md
"""
from __future__ import annotations

import argparse
import contextlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import dev_config  # noqa: E402 -- path insert must precede this import

#: the largest elapsed / max(floors) a row passes at (§28)
LINE = 1.15
COUNTERS = (
    "gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second", "smsp__inst_executed.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "sm__inst_executed_pipe_tensor_op_imma.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


@dataclass(frozen=True)
class Row:
    kernel: str
    launch: str
    cell: str
    heads: int
    mma: str | None = None
    warmup: int = 3


#: THE ROWS: every kernel's gate row, named in `tools/coverage.json`'s `overlap` obligation
ROWS = {
    "facts-alt-k4": Row(kernel="facts_pass", launch="facts", cell="nl64k-alt-k4", heads=4),
}
#: THE PROBE ROWS: the instrument's own check, a kernel known to run at one floor (`--probe`)
PROBES = {
    "probe-gemm-f16": Row(kernel="gemm|cutlass|ampere", launch="gemm", cell="8192", heads=1, mma="shapes"),
}


def _launch_facts(cell: str, heads: int, repeat: int) -> None:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(cell)
    routes, values, layout, statics = bundle_of(spec, realize(spec, H=heads))
    form = facts_ops.WHOLE_GROUPS
    for _ in range(repeat):
        facts_ops.facts(routes, values, layout, *statics, form)
    torch.cuda.synchronize()


def _launch_gemm(cell: str, heads: int, repeat: int) -> None:
    import torch

    n = int(cell)
    a, b = (torch.randn(n, n, device="cuda", dtype=torch.float16) for _ in range(2))
    for _ in range(repeat):
        a @ b
    torch.cuda.synchronize()


LAUNCHES = {"facts": _launch_facts, "gemm": _launch_gemm}


def _mma_cycles(opcode: str, rates: dict) -> float:
    """An executed MMA's cycles on a scheduler's tensor pipe, by its SASS shape (`HMMA.16816.F32`, `.F32.BF16`,
    `.F16`, `HMMA.1688.*`, `IMMA.16832.*`)."""
    parts = opcode.split(".")
    if parts[0] == "IMMA":
        return rates["mma.imma_u8_s8_k32"].cycles / (1 if parts[1] == "16832" else 2)
    kind = "hmma_f16_f16" if parts[2] == "F16" else "hmma_bf16_f32" if "BF16" in parts else "hmma_f16_f32"
    return rates[f"mma.{kind}"].cycles / (1 if parts[1] == "16816" else 2)


def shapes(row: Row) -> dict[str, int]:
    """Executed MMAs by SASS shape: one `ncu` launch's source counters (KERNEL_STANDARDS §22's executed counts)."""
    import region_ledger
    from rola_devtools import process

    out_csv = ROOT / "build" / f"overlap_{re.sub(r'[^A-Za-z0-9]+', '_', row.kernel)}_{row.cell}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    cmd = [dev_config.ncu(), "--target-processes", "all", "-k", f"regex:{row.kernel}",
           "--launch-skip", str(row.warmup), "-c", "1", "--section", "SourceCounters", "--page", "source",
           "--print-source", "sass", "--csv", sys.executable, str(Path(__file__).resolve()), "--launch", row.launch,
           "--cell", row.cell, "--heads", str(row.heads), "--repeat", str(row.warmup + 1)]
    done = process.run(cmd, cwd=ROOT, timeout=3600)
    out_csv.write_text(done.stdout)
    executed = {a: n for a, n, _, _ in region_ledger.read_source_counters(out_csv)}
    found: dict[str, int] = {}
    for a, (op, _, _) in region_ledger.read_source_detail(out_csv).items():
        if op.startswith(("HMMA", "IMMA")):
            found[op] = found.get(op, 0) + executed.get(a, 0)
    if not found:
        raise SystemExit(f"overlap_gate: no MMA read in {row.kernel}'s source counters (ncu rc {done.returncode})")
    return found


def counters(row: Row) -> dict[str, float]:
    from rola_devtools import process

    cmd = [dev_config.ncu(), "--target-processes", "all", "-k", f"regex:{row.kernel}",
           "--launch-skip", str(row.warmup), "-c", "1", "--metrics", ",".join(COUNTERS), "--csv",
           sys.executable, str(Path(__file__).resolve()), "--launch", row.launch, "--cell", row.cell,
           "--heads", str(row.heads), "--repeat", str(row.warmup + 1)]
    done = process.run(cmd, cwd=ROOT, timeout=3600)
    values = {}
    for line in done.stdout.splitlines():
        parts = [x.strip().strip('"') for x in line.split('","')]
        if len(parts) > 12 and parts[0].isdigit() and parts[-3] in COUNTERS:
            with contextlib.suppress(ValueError):
                values[parts[-3]] = float(parts[-1].replace(",", ""))
    missing = set(COUNTERS) - set(values)
    if missing:
        raise SystemExit(f"overlap_gate: {sorted(missing)} not read (ncu rc {done.returncode}):\n"
                         f"{(done.stdout + done.stderr)[-1500:]}")
    return values


def floors(row: Row, c: dict[str, float], sms: int, product: str | None, arch: str) -> dict[str, float]:
    """The row's floors in seconds from its counters, the device's SM count and its product's rates."""
    from measure import hardware

    rates = hardware.resolve(arch=arch, product=product)
    clock = c["sm__cycles_elapsed.avg.per_second"]
    slots = sms * rates["schedulers"].cycles / rates["issue"].cycles
    out = {"issue": c["smsp__inst_executed.sum"] / slots / clock,
           "shared": c["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"] * rates["wavefront"].cycles / sms / clock,
           "dram": c["dram__throughput.avg.pct_of_peak_sustained_elapsed"] / 100 * c["gpu__time_duration.sum"] * 1e-9}
    tensor = c["sm__inst_executed_pipe_tensor_op_hmma.sum"] + c["sm__inst_executed_pipe_tensor_op_imma.sum"]
    if row.mma == "shapes":
        busy = sum(n * _mma_cycles(op, rates) for op, n in c["shapes"].items())
        out["tensor"] = busy / (sms * rates["schedulers"].cycles) / clock
    elif row.mma is not None:
        out["tensor"] = tensor * rates[f"mma.{row.mma}"].cycles / (sms * rates["schedulers"].cycles) / clock
    elif tensor:
        raise SystemExit(f"overlap_gate: {row.kernel} issued {tensor:.0f} MMAs but its row declares no MMA kind")
    return out


def judge(name: str) -> bool:
    import torch

    from measure import hardware

    row = {**ROWS, **PROBES}[name]
    props = torch.cuda.get_device_properties(0)
    arch = hardware.device_arch(props.major, props.minor)
    c = counters(row)
    if row.mma == "shapes":
        c["shapes"] = shapes(row)
    f = floors(row, c, props.multi_processor_count, hardware.product_of(props.name), arch)
    elapsed = c["gpu__time_duration.sum"] * 1e-9
    binding = max(f, key=f.get)
    ratio = elapsed / f[binding]
    red = ratio > LINE
    shown = "  ".join(f"{k} {v * 1e6:.1f}" for k, v in sorted(f.items(), key=lambda kv: -kv[1]))
    print(f"{'RED ' if red else 'ok  '} {name}: {elapsed * 1e6:.1f} us = {ratio:.2f}x its {binding} floor "
          f"(floors us: {shown})")
    return not red


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", choices=sorted(ROWS), help="a row to judge (default: every row)")
    ap.add_argument("--probe", action="append", choices=sorted(PROBES), help="the instrument's probe rows")
    ap.add_argument("--launch", choices=sorted(LAUNCHES), help=argparse.SUPPRESS)
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    ap.add_argument("--heads", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.launch:
        LAUNCHES[a.launch](a.cell, a.heads, a.repeat)
        return 0
    from rola_devtools.locks.gpu import gpu_lock

    with gpu_lock(mode="exclusive"):
        passed = [judge(name) for name in (a.probe or a.row or sorted(ROWS))]
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
