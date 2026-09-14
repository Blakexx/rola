#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""`.ninja_log` -> a Chrome/Perfetto trace, the compile ledger's visual form.

`build/temp/.ninja_log` (torch's `CUDAExtension` build drives ninja underneath
`setup.py`) already carries, per compiled TU, the wall-clock start/end this
script needs -- KERNEL_STANDARDS §16 addendum 3's compile ledger records the
same facts (compile wall, `cicc` peak RSS) as one number per arm TU; this is
that ledger's TIMELINE, one row per output, viewable in `chrome://tracing` or
https://ui.perfetto.dev.

`.ninja_log` FORMAT (v5, ninja's own documented schema, tab-separated after the
`# ninja log vN` header): `start_ms  end_ms  restat_mtime  output  cmdhash`. A
later line for the same `output` supersedes an earlier one (ninja appends,
never rewrites in place) -- only the LAST line per output is real.

TRACK ASSIGNMENT mirrors the reference `ninjatracing` tool (Nico Weber): each
build step is placed on the lowest-numbered "thread" (parallel job slot) whose
previous step already ended at or before this one's start -- a greedy interval
scheduling that reconstructs which of ninja's `-j` slots a given compile
likely occupied, since the log itself does not record that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_ninja_log(path: Path) -> list[tuple[int, int, str]]:
    """[(start_ms, end_ms, output), ...] -- one entry per output, LAST write wins."""
    by_output: dict[str, tuple[int, int]] = {}
    order: list[str] = []
    with path.open() as f:
        first = f.readline()
        if not first.startswith("# ninja log v"):
            raise ValueError(f"{path}: not a ninja log (missing '# ninja log vN' header)")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start_ms, end_ms, _restat, output = parts[0], parts[1], parts[2], parts[3]
            if output not in by_output:
                order.append(output)
            by_output[output] = (int(start_ms), int(end_ms))
    return [(by_output[o][0], by_output[o][1], o) for o in order]


def assign_tracks(entries: list[tuple[int, int, str]]) -> list[tuple[int, int, str, int]]:
    """Greedy interval scheduling onto the fewest parallel tracks."""
    track_free_at: list[int] = []  # track index -> time it becomes free
    out = []
    for start, end, output in sorted(entries, key=lambda e: e[0]):
        track = next((i for i, free_at in enumerate(track_free_at) if free_at <= start), None)
        if track is None:
            track = len(track_free_at)
            track_free_at.append(0)
        track_free_at[track] = end
        out.append((start, end, output, track))
    return out


def to_chrome_trace(entries: list[tuple[int, int, str, int]], pid: int = 1) -> dict:
    events = [
        {
            "name": output, "cat": "build", "ph": "X", "pid": pid, "tid": track,
            "ts": start * 1000, "dur": max(end - start, 0) * 1000,
        }
        for start, end, output, track in entries
    ]
    n_tracks = 1 + max((t for *_, t in entries), default=-1)
    for t in range(n_tracks):
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": t,
                        "args": {"name": f"slot {t}"}})
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ninja_log", type=Path, help="path to a .ninja_log file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                     help="output trace JSON (default: <ninja_log>.trace.json)")
    args = ap.parse_args()

    entries = parse_ninja_log(args.ninja_log)
    if not entries:
        print(f"{args.ninja_log}: no build steps recorded", file=sys.stderr)
        return 1
    tracked = assign_tracks(entries)
    trace = to_chrome_trace(tracked)

    out = args.out or args.ninja_log.with_suffix(args.ninja_log.suffix + ".trace.json")
    out.write_text(json.dumps(trace))
    n_tracks = 1 + max(t for *_, t in tracked)
    span_ms = max(e for _, e, *_ in tracked) - min(s for s, *_ in tracked)
    print(f"{out}: {len(tracked)} steps, {n_tracks} parallel slots, "
          f"{span_ms / 1000:.1f}s span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
