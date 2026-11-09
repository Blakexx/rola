#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OP'S TIMELINE: a registered subject's calls on a cell under torch's profiler, each call split into the time the
GPU runs kernels and the time it waits -- the gaps between kernels and what the host did in them.

    bazel run //tools:op_timeline -- nl64k-alt-k4-h4            # the op's prefill (`prefill`)
    bazel run //tools:op_timeline -- nl64k-alt-k4-h4 --subject carry_intra --calls 5 --top 15
    bazel run //tools:op_timeline -- nl64k-alt-k4-h4 --python   # and one call's host Python (cProfile)

Per call: the wall time (synchronized), the kernels' busy time (the union of their intervals), the idle remainder;
per kernel name its launches and time; the gaps longer than `--gap` microseconds with the host ops that ran across
them; and the host ops by their own time. The subject is built as the bench builds it (`measure.subjects`), its first
call untimed. Docs: docs/internals/tools/op_timeline.md. Measured work: takes the GPU lock exclusive.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def busy_union(spans: list[tuple[float, float]]) -> float:
    """The union of `(start, end)` intervals' length."""
    total, end = 0.0, float("-inf")
    for s, e in sorted(spans):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--subject", default="prefill")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gap", type=float, default=50.0, help="the gaps reported, microseconds and longer")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--python", action="store_true",
                    help="also profile one call's host Python by cumulative time (cProfile, after the timed calls)")
    args = ap.parse_args()

    import torch
    from rola_devtools.locks.gpu import gpu_lock
    from torch.profiler import ProfilerActivity, profile

    from measure.cells import by_name
    from measure.subjects import SUBJECTS

    launch = SUBJECTS[args.subject].build({"cell": by_name(args.cell)})
    with gpu_lock(mode="exclusive"), torch.no_grad():
        for _ in range(args.warmup):
            if launch.reset:
                launch.reset()
            launch.call()
        torch.cuda.synchronize()
        walls = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.calls):
                if launch.reset:
                    launch.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                launch.call()
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - t0) * 1e6)

    events = prof.events()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA and e.time_range.elapsed_us() > 0]
    host = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU]
    spans = [(e.time_range.start, e.time_range.end) for e in kernels]
    busy = busy_union(spans) / args.calls
    wall = sum(walls) / len(walls)
    print(f"{args.subject} on {args.cell}, {args.calls} calls on {torch.cuda.get_device_name(0)}")
    print(f"  wall {wall / 1e3:8.3f} ms a call   kernels busy {busy / 1e3:8.3f} ms   idle {(wall - busy) / 1e3:8.3f} ms "
          f"({100 * (wall - busy) / wall:.0f}%)")

    by_name_t: dict[str, list[float]] = defaultdict(list)
    for e in kernels:
        by_name_t[e.name].append(e.time_range.elapsed_us())
    print(f"  kernels a call ({len(kernels) / args.calls:.0f} launches):")
    for name, ts in sorted(by_name_t.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
        print(f"    {sum(ts) / args.calls / 1e3:8.3f} ms  x{len(ts) / args.calls:4.0f}  {name[:110]}")

    ordered = sorted(kernels, key=lambda e: e.time_range.start)
    gaps = []
    for k0, k1 in zip(ordered, ordered[1:]):
        a, b = k0.time_range.end, k1.time_range.start
        if b - a >= args.gap:
            gaps.append((a, b, k0.name, k1.name))
    print(f"  gaps of {args.gap:.0f} us and longer between kernels: {len(gaps) / args.calls:.1f} a call, "
          f"{sum(b - a for a, b, _, _ in gaps) / args.calls / 1e3:.3f} ms a call")
    for a, b, before, after in sorted(gaps, key=lambda g: g[0] - g[1])[: args.top]:
        print(f"    {(b - a) / 1e3:7.3f} ms  after {before[:60]}  before {after[:60]}")
        #: the host ops that ran inside the gap, by how much of it each covers (the call's outer frames excluded)
        inside = defaultdict(float)
        for e in host:
            lo, hi = max(e.time_range.start, a), min(e.time_range.end, b)
            if hi > lo and e.time_range.elapsed_us() < 2 * (b - a):
                inside[e.name] += hi - lo
        for name, t in sorted(inside.items(), key=lambda kv: -kv[1])[:8]:
            print(f"        {t / 1e3:7.3f} ms  {name[:100]}")

    if args.python:
        import cProfile
        import pstats

        with gpu_lock(mode="exclusive"), torch.no_grad():
            if launch.reset:
                launch.reset()
            torch.cuda.synchronize()
            pr = cProfile.Profile()
            pr.enable()
            launch.call()
            pr.disable()
            torch.cuda.synchronize()
        print("  host Python of one call, by cumulative time:")
        pstats.Stats(pr).sort_stats("cumulative").print_stats(args.top * 3)
        print("  host Python of one call, by own time:")
        pstats.Stats(pr).sort_stats("tottime").print_stats(args.top)

    self_t: dict[str, float] = defaultdict(float)
    for e in host:
        self_t[e.name] += e.self_cpu_time_total
    print("  host ops by their own time a call:")
    for name, t in sorted(self_t.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"    {t / args.calls / 1e3:8.3f} ms  {name[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
