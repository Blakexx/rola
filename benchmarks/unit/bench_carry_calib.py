#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CALIBRATIONS (KERNEL_STANDARDS §22 (10)): each cost the carry kernel's parts are made of, alone,
on every warp of every CTA, timed by the host under the locked clock, as cycles an operation a warp.

    python benchmarks/unit/bench_carry_calib.py [--only hmma,shared_load,...] [--owners 80]

The kernels are `benchmarks/unit/carry_parts/carry_calib.cuh`, built into the part harness's module
(`ROLA_BUILD_PARTS=1`). A calibration is sized from a short launch to take `--seconds` a launch, run
once to warm, then `--launches` times; the median launch's seconds times the locked clock's cycles a
second, over the operations a warp issues in it, is the row. The rows are stored
through `rola_results` at `calibration`: reference rows for this card, the composer's and the tracer's
pipe replay read them instead of a constant.
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch  # noqa: E402

#: (name, warps, mode, burst, operations a unit a warp, what it stands for in the kernel)
CALIBRATIONS = (
    ("hmma_2w", 8, 0, 1, 9, "the kernel's HMMA atom, two warps a scheduler (the kernel's regime)"),
    ("hmma_1w", 4, 0, 1, 9, "the same atom, one warp a scheduler"),
    ("shared_load_4", 8, 1, 4, 4, "four scalar shared loads a unit, each its own bank line"),
    ("shared_load_16", 8, 1, 16, 16, "sixteen"),
    ("shared_load_64", 8, 1, 64, 64, "sixty-four"),
    ("shared_store_4", 8, 2, 4, 4, "four shared stores a unit"),
    ("shared_store_16", 8, 2, 16, 16, "sixteen"),
    ("shared_store_64", 8, 2, 64, 64, "sixty-four"),
    ("matrix_load_4", 8, 9, 4, 4, "four two-n-tile ldmatrix.trans loads a unit (the readout's B, the fold's operands)"),
    ("matrix_load_16", 8, 9, 16, 16, "sixteen"),
    ("async_copy", 8, 3, 4, 4, "four 16-byte asynchronous runs a group, bank-free, landed each group"),
    ("async_copy_aliased", 8, 4, 4, 4, "the same runs aliased to one bank group"),
    ("global_reduce", 8, 5, 1, 1, "a global f32 reduction into the output's pages"),
    ("cta_barrier", 8, 6, 1, 1, "a CTA barrier over every lane"),
    ("shm_barrier", 8, 7, 1, 1, "a shared-memory barrier: arrive and wait"),
    ("warp_sync", 8, 8, 1, 1, "a warp sync between one lane's store and every lane's load"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of calibration names")
    ap.add_argument("--owners", type=int, default=80, help="CTAs a launch (one a SM on this card)")
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--launches", type=int, default=3)
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}

    import clock_lock
    from gpu_lock import GPU_LOCK_DEFAULT, gpu_lock

    from rola.ops import carry as carry_ops
    try:
        from rola_cu13 import _C_parts  # noqa: F401 -- loading the library registers torch.ops.rola_parts
    except ImportError as ex:
        raise SystemExit("the part harness's module is not built (ROLA_BUILD_PARTS=1)") from ex
    mod = torch.ops.rola_parts

    dev = torch.device("cuda")
    out = torch.zeros((a.owners, 16, 256), dtype=torch.float32, device=dev)
    src = torch.randint(0, 255, (a.owners, 256, 16), dtype=torch.uint8, device=dev)
    rows = []
    with gpu_lock(GPU_LOCK_DEFAULT, mode="exclusive"):
        clock = clock_lock.engage(carry_ops.sm_clock_ghz)
        ghz = carry_ops.sm_clock_ghz()

        def launch(w, m, b, iters) -> float:
            torch.cuda.synchronize()
            t = time.perf_counter()
            mod.calibrate(w, m, b, iters, a.owners, out, src)
            torch.cuda.synchronize()
            return time.perf_counter() - t

        for name, w, m, b, ops_a_unit, what in CALIBRATIONS:
            if only and name not in only:
                continue
            probe = 2000
            launch(w, m, b, probe)
            dt_probe = launch(w, m, b, probe)
            iters = max(probe, int(probe * a.seconds / max(dt_probe, 1e-4)))
            launch(w, m, b, iters)
            secs = [launch(w, m, b, iters) for _ in range(a.launches)]
            med = statistics.median(secs)
            cyc = (ghz or 0.0) * 1e9 * med / (iters * ops_a_unit)
            row = {"name": name, "warps": w, "mode": m, "burst": b, "iters": iters, "ops_a_unit": ops_a_unit,
                   "seconds": secs, "ghz": ghz, "cycles_an_op_a_warp": cyc, "what": what}
            rows.append(row)
            print(f"{name:20s} {cyc:9.2f} cycles an op a warp   ({iters} units, median {med:.3f} s)  {what}",
                  flush=True)
        read = carry_ops.sm_clock_ghz()
    from rola_results import Store, checkout, digest

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    semantics = {"binary": digest(mod.__file__), "device": torch.cuda.get_device_name(0), "owners": a.owners,
                 "seconds": a.seconds, "launches": a.launches, "only": sorted(only),
                 "clock_ghz": clock["ghz"] if clock else None}
    sample = Store("calibration").put(semantics, output={"utc": stamp, "owners": a.owners, "clock": clock,
                                                         "ghz_before": ghz, "ghz_after": read, "rows": rows},
                                      provenance=checkout(ROOT))
    print(f"report: calibration sample {sample['n']}")
    return 0 if clock_lock.within(read, clock) else 1


if __name__ == "__main__":
    sys.exit(main())
