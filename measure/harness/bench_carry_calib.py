#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CALIBRATIONS (KERNEL_STANDARDS §22 (10)): each cost the carry kernel's parts are made of, alone,
on every warp of every CTA, timed by the host under the locked clock, as cycles an operation a warp.

    python measure/harness/bench_carry_calib.py [--only hmma,shared_load,...] [--owners 80]

The kernels are `measure/harness/carry_parts/carry_calib.cuh`, built into the part harness's module
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
sys.path.insert(0, str(ROOT / "measure"))

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
    ("matrix_load_rows_4", 8, 20, 4, 4, "four ldmatrix.trans loads a unit as the kernel issues them: a lane its own row, four wavefronts a load"),
    ("matrix_load_rows_16", 8, 20, 16, 16, "sixteen"),
    ("matrix_load_rows_4_1w", 4, 20, 4, 4, "four, one warp a scheduler"),
    ("async_copy", 8, 3, 4, 4, "four 16-byte asynchronous runs a group, bank-free, landed each group"),
    ("async_copy_aliased", 8, 4, 4, 4, "the same runs aliased to one bank group"),
    ("async_copy_strided", 8, 16, 4, 4, "the bank-free runs fed from a 128-byte line a lane (sixteen lines a run)"),
    ("async_copy_4", 8, 17, 4, 4, "four-byte runs a lane, one contiguous line in and out"),
    ("async_copy_rows", 8, 18, 4, 4, "the pool fill's V pattern: sixteen token rows, two adjacent chunks a run, swizzled"),
    ("async_copy_lines", 8, 19, 4, 4, "the same rows landed a whole line at a time: eight lanes a row, four rows a run"),
    ("async_copy_zfill_sink", 8, 27, 4, 4, "zero-size 16-byte copies, every lane's landing one slot: the fill's dead lanes on the zero row"),
    ("async_copy_zfill_spread", 8, 28, 4, 4, "zero-size copies, a slot a lane"),
    ("async_copy_mixed_sink", 8, 29, 4, 4, "four lanes live from their own rows, twenty-eight zero-size to the one slot: a sparse round"),
    ("async_copy4_zfill_sink", 8, 30, 4, 4, "sixteen lanes' zero-size four-byte copies to one word: a dead gain pair"),
    ("async_copy_lanes", 8, 31, 4, 4, "the sparse round with its twenty-eight dead lanes predicated off, not zero-size"),
    ("async_copy4_lanes", 8, 32, 4, 4, "two of sixteen lanes' four-byte copies live, the rest predicated off"),
    ("async_copy_rows_16", 8, 18, 16, 16, "the V pattern, sixteen copies a group: the copy's own cost, the group's landing amortized"),
    ("async_copy_mixed_sink_16", 8, 29, 16, 16, "the sparse round zero-size, sixteen a group"),
    ("async_copy_lanes_16", 8, 31, 16, 16, "the sparse round predicated, sixteen a group"),
    ("async_copy_4_16", 8, 17, 16, 16, "four-byte runs, sixteen a group"),
    ("async_copy4_zfill_sink_16", 8, 30, 16, 16, "the dead gain pair zero-size, sixteen a group"),
    ("async_copy4_lanes_16", 8, 32, 16, 16, "the gain pair with two lanes live, sixteen a group"),
    ("global_reduce", 8, 5, 1, 1, "a global f32 reduction into the output's pages"),
    ("cta_barrier", 8, 6, 1, 1, "a CTA barrier over every lane"),
    ("shm_barrier", 8, 7, 1, 1, "a shared-memory barrier: arrive and wait"),
    ("warp_sync", 8, 8, 1, 1, "a warp sync between one lane's store and every lane's load"),
    #: the settling rows: an HMMA burst (nine, the atom) with the parts' loads or reductions interleaved,
    #: read as cycles an HMMA against `hmma_2w`'s 65 (32.5 a scheduler)
    ("hmma_load_1", 8, 10, 1, 9, "nine HMMAs a unit fed by one ldmatrix.trans load issued ahead of them, two warps a scheduler"),
    ("hmma_load_4", 8, 10, 4, 9, "nine HMMAs fed by four loads ahead of them (the readout's box)"),
    ("hmma_load_4_1w", 4, 10, 4, 9, "the same, one warp a scheduler: no partner warp to hide the loads"),
    ("hmma_load_4_free", 8, 11, 4, 9, "four loads beside nine HMMAs, the loads' results a sink: the pipes' sharing alone"),
    ("hmma_reduce_1", 8, 12, 1, 9, "nine HMMAs with one global f32 reduction issued between them"),
    ("hmma_reduce_2", 8, 12, 2, 9, "two reductions between them"),
    ("hmma_reduce_4", 8, 12, 4, 9, "four"),
    ("hmma_reduce_8", 8, 12, 8, 9, "eight, one an HMMA"),
    ("hmma_reduce_div_4", 8, 13, 4, 9, "four DIVERGENT reductions (a lane at its accumulator's row and column: eight sectors a red)"),
    ("hmma_reduce_div_8", 8, 13, 8, 9, "eight divergent"),
    ("hmma_frag_1w", 4, 14, 18, 18, "the fold fragment's burst: eighteen HMMAs into eighteen accumulators, one warp a scheduler"),
    ("hmma_frag_2w", 8, 14, 18, 18, "the same, two warps a scheduler"),
    ("hmma_frag_chain_1w", 4, 15, 18, 18, "the burst behind the fragment's gather chain (shuffle, ldmatrix, two multiplies), one warp"),
    ("hmma_frag_chain_2w", 8, 15, 18, 18, "the same, two warps"),
    ("hmma_alu_32_1w", 4, 21, 32, 18, "eighteen HMMAs with thirty-two fp32 adds after them (four chains), one warp a scheduler"),
    ("hmma_alu_32_2w", 8, 21, 32, 18, "the same, two warps"),
    ("hmma_alu_64_1w", 4, 21, 64, 18, "sixty-four adds, one warp"),
    ("hmma_alu_64_2w", 8, 21, 64, 18, "sixty-four, two warps"),
    ("hmma_queue_16_1w", 4, 22, 16, 18, "eighteen HMMAs then one dependent chain of sixteen fp32 fmas, one warp: the pipe's queue depth"),
    ("hmma_queue_40_1w", 4, 22, 40, 18, "a chain of forty"),
    ("hmma_queue_80_1w", 4, 22, 80, 18, "a chain of eighty"),
    ("hmma_queue_40_2w", 8, 22, 40, 18, "a chain of forty, two warps"),
    ("hmma_frag_chain_hooked_1w", 4, 23, 18, 18, "the burst with the next gather's chain in four pieces, each pinned into a three-HMMA window (the granular fork), one warp"),
    ("hmma_frag_chain_hooked_2w", 8, 23, 18, 18, "the same, two warps"),
    ("hmma_frag_chain_hooked2_1w", 4, 24, 18, 18, "the fork with the hook values pinned after an earlier HMMA's result too, one warp"),
    ("hmma_frag_chain_hooked2_2w", 8, 24, 18, 18, "the same, two warps"),
    ("hmma_latency", 4, 25, 18, 18, "eighteen HMMAs into one accumulator, each dependent on the last: the completion latency"),
    ("hmma_operands_1w", 4, 26, 18, 18, "the fragment's burst with its operand pattern: a different B register pair every HMMA, two A sets, no loads, one warp"),
    ("hmma_operands_2w", 8, 26, 18, 18, "the same, two warps"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of calibration names")
    ap.add_argument("--owners", type=int, default=80, help="CTAs a launch (one a SM on this card)")
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--launches", type=int, default=3)
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}

    from rola_devtools.locks import clock as clock_lock
    from rola_devtools.locks.gpu import gpu_lock

    from rola.ops import carry as carry_ops
    try:
        from rola_cu13 import _C_parts  # loading the library registers torch.ops.rola_parts
    except ImportError as ex:
        raise SystemExit("the part harness's module is not built (ROLA_BUILD_PARTS=1)") from ex
    mod = torch.ops.rola_parts

    dev = torch.device("cuda")
    out = torch.zeros((a.owners, 128, 256), dtype=torch.float32, device=dev)
    src = torch.randint(0, 255, (a.owners, 256, 128), dtype=torch.uint8, device=dev)
    rows = []
    with gpu_lock(mode="exclusive"):
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
    semantics = {"binary": digest(_C_parts.__file__), "device": torch.cuda.get_device_name(0), "owners": a.owners,
                 "seconds": a.seconds, "launches": a.launches, "only": sorted(only),
                 "clock_ghz": clock["ghz"] if clock else None}
    sample = Store("calibration").put(semantics, output={"utc": stamp, "owners": a.owners, "clock": clock,
                                                         "ghz_before": ghz, "ghz_after": read, "rows": rows},
                                      provenance=checkout(ROOT))
    print(f"report: calibration sample {sample['n']}")
    return 0 if clock_lock.within(read, clock) else 1


if __name__ == "__main__":
    sys.exit(main())
