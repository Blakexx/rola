# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTRA KERNEL'S ROOFLINE: wall time beside the device's own f16 mma.sync ceiling.

The other benches answer "how long"; this one answers "how much of the machine". It
reports, per cell, the wall time, the tile-pairs per millisecond, and the achieved
fraction of the f16 HMMA ceiling (fp16 accumulate) -- the number that says whether a window's
cost is the arithmetic or everything around it.

THE CELLS AND THE STEP ARE THE REGISTRY'S. This file defines neither: the cells are
`measure/cells` records and the timed callable is the registered `intra_forward`
bench, so a number here is commensurable with the same cell's number from a timing
session of `declare.py`. What is added is arithmetic ON TOP of that
measurement -- the issued MAC count of a dense cell is EXACT (the window's lower-triangular
tile grid is `n(n+1)/2` of its `n = W/64` squared tiles, a pair's Gram over every level's
digits and its value product over the value channels), so the fraction
is a fraction and not an estimate; a sparse cell's gate skips pairs, and its fraction is of
the dense count.

Symbols at first use: ``BH`` = batch times heads, ``L`` = tokens, ``W`` = the window the
carry and the intra share, ``D`` = routing levels, ``DV`` = the padded value width.

    bazel run //tools:roofline -- --cells flagship-dense-L16384,flagship-alt-k4-L16384
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

TILE = 64
#: f16 MACs per clock per SM at fp16 accumulation (`common/hardware.json`'s `hmma_f16_f16`: an m16n8k16 a scheduler
#: every 16 cycles on sm_86, every 8 on sm_80) -- the ceiling this kernel's fraction is quoted against
MACS_PER_CLOCK_PER_SM = {(8, 0): 1024, (8, 6): 512}


def peak_flops():
    """The ceiling, taken from the device's own advertised boost clock and SM count.

    THE CURRENT DEVICE, and no dial for another one: a roofline is a fraction of the machine this launch ran on, so a
    ceiling read off a second device would be a fraction of something else."""
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    rate = MACS_PER_CLOCK_PER_SM[(properties.major, properties.minor)]
    clock = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0]) * 1e6
    return 2.0 * rate * properties.multi_processor_count * clock, clock


def tile_pairs(window: int) -> int:
    tiles = window // TILE
    return tiles * (tiles + 1) // 2


def macs(bh: int, length: int, widths, window: int, dv: int) -> int:
    per_tile = TILE * TILE * (sum(widths) + dv)
    return bh * (length // window) * tile_pairs(window) * per_tile


def median_ms(call, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        call()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "measure"))
    sys.path.insert(0, str(root))
    from measure.cells import by_name, carry_cells
    from measure.subjects import SUBJECTS, applicable
    from rola.ops.window import window_of

    WINDOW = window_of()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default=None,
                        help="comma-separated registry cells; default is every probe-tier "
                             "cell the intra bench applies to")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--json", type=Path, default=None, help="also write every cell's row here")
    args = parser.parse_args()

    if args.cells:
        specs = [by_name(name) for name in args.cells.split(",")]
    else:
        specs = [c for c in carry_cells(tier="probe") if "intra_forward" in applicable(c, "carry")]
    if not specs:
        raise SystemExit("no registry cell applies to the intra bench")

    flops, clock = peak_flops()
    print(f"device {torch.cuda.get_device_name()}  clock {clock / 1e9:.3f} GHz  "
          f"f16 HMMA ceiling {flops / 1e12:.1f} TFLOP/s  window {WINDOW}")
    print(f"{'cell':<28}{'ms':>9}{'tiles/ms':>12}{'TOP/s':>10}{'peak':>8}")
    rows = []
    for spec in specs:
        if spec.tokens % WINDOW:
            raise SystemExit(f"{spec.name}: L = {spec.tokens} is not a whole number of "
                             f"{WINDOW}-token windows")
        arm = SUBJECTS["intra_forward"].build({"cell": spec})
        arm.call()
        torch.cuda.synchronize()
        ms = median_ms(arm.call, args.repeats)
        windows = spec.tokens // WINDOW
        tiles = windows * tile_pairs(WINDOW)
        achieved = 2.0 * macs(1, spec.tokens, spec.widths, WINDOW, spec.dv) / (ms * 1e-3)
        print(f"{spec.name:<28}{ms:>9.3f}{tiles / ms:>12.0f}{achieved / 1e12:>10.2f}"
              f"{achieved / flops * 100:>7.1f}%")
        rows.append({"cell": spec.name, "ms": ms, "tiles_per_ms": tiles / ms, "top_per_s": achieved,
                     "ceiling_top_per_s": flops, "fraction": achieved / flops, "clock_hz": clock})
        del arm
        torch.cuda.empty_cache()
    if args.json:
        args.json.write_text(json.dumps({"window": WINDOW, "rows": rows}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
