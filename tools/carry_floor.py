#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HOPPER CARRY'S FLOOR, COUNTED: what one token costs an SM, term by term, for a declared routing shape, a box
and a cluster -- before any hopper carry exists, so a design is priced by one model and not by arithmetic redone a
conversation.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \\
        --box 16 16 --cluster 1 16 [--dv 64] [--tokens 65536] [--json floor.json]

THE SHAPE. `--widths` a level; `--read-sparse` / `--write-sparse` a flag a level (a sparse level holds `--k` live
digits a token, drawn uniformly; a dense one all of them). A BOX is `--box` digits of each level; a CLUSTER is
`--cluster` boxes along each level. A token's digits are independent draws, so every count below is an expectation.

THE TERMS, SM-cycles a token (each line of the report says which rate it used and whether that rate is MEASURED --
read from an H100 lab record under `measure/lab/results/` at run time -- or ESTIMATED, a constant of this file with
its reason):
- tensor: the leaves a token reads and writes, times the channels, over the pipe's products a cycle (`wgmma_rate`);
  a value width under 64 runs wgmma's 64 rows (the idle half);
- the readout's operand: its (token, leaf) coefficients expanded into shared memory, at the SM's shared access rate
  (`byte_expand`, its bank-spread form);
- the output's sum, in a cluster by one of two forms (`--sum`), then one fp32 drain a (token, cluster read) to global
  memory (`bulk_drain`, a CTA's own sum to one sum for every CTA); with no cluster (`--cluster` all ones) every box
  drains. `s64`: a box's partial a (token, box read) added as 64-bit integers into its cluster's collector at the
  shared rate, the conversion a value and the state's scan a window at the SM's issue rate. `slots`: a box's fp32
  partial stored into its OWN slot of the collector's shared memory, the collector loading a token's slots and
  adding them in registers -- the fp32 sum itself, no scale: the bytes in, the same bytes read back, an add a value;
- the fold's operands: its (token, leaf) coefficients into shared memory, and its values loaded a (token, list) of
  each cluster it writes into (one load serves the cluster's boxes; with no cluster, each box loads);
- the state in and out, a call (24-bit state, at a memory bandwidth).

NOT COUNTED: the hand-over between producer and consumer warps, the den's readout, the divergent carve's register
swap, paging's atom loads, the facts pass, the intra and the producer. FlashAttention's tensor floor at the same
tokens and width is printed beside the total.

Docs: docs/internals/tools/carry_floor.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "measure" / "lab" / "results"

#: wgmma's rows: the channels a warpgroup's MMA carries
MMA_ROWS = 64
#: ESTIMATED, each with its reason: nothing on an H100 has measured these
ESTIMATES = {
    "issue": (4.0, "warp-instructions a cycle an SM: one a scheduler, four schedulers"),
    "convert": (2.0, "instructions a value: fp32 to a 64-bit integer at a scale"),
    "scan": (140.0, "instructions a thread a box a window (a warpgroup's 4 warps): the state's largest magnitude"),
    "window": (1024.0, "tokens a window (the hopper row's provisional W)"),
    "load_low": (5.0, "cycles a 128-byte load request: the ampere fill's floor"),
    "load_high": (10.0, "cycles a 128-byte load request: twice the ampere fill's floor"),
    "clock": (1.8e9, "SM cycles a second: inferred from the card's published tensor rate, not measured"),
    "memory_low": (2.0e12, "bytes a second across memory, the low guess"),
    "memory_high": (3.0e12, "bytes a second across memory, the high guess"),
}
#: an SM's count on the card the records are from
SMS = 132


def median(lab: str, row: str) -> float:
    """A lab record's row on the H100: its median (refused if the record or the row is gone)."""
    path = RESULTS / lab / "sm_90a.json"
    if not path.exists():
        raise SystemExit(f"carry_floor: no record {path.relative_to(ROOT)}")
    for result in json.loads(path.read_text())["results"]:
        if result["name"] == row:
            return float(result["median"])
    raise SystemExit(f"carry_floor: {path.relative_to(ROOT)} has no row `{row}`")


def measured() -> dict[str, tuple[float, str]]:
    """THE MEASURED RATES, each with the record row it is read from."""
    pipe_row = "6 wgmmas a group, A registers, two warpgroups"
    spread_row = "the expansion, a warp's lanes on eight bank groups, 8 warps a CTA"
    #: twelve m64n64k16 wgmmas an SM a step; a lane's 6 stores and 1 load of 16 bytes, 32 lanes, 8 warps a tile time
    products = 12 * 64 * 64 * 16 / median("wgmma_rate", pipe_row)
    shared = 8 * 7 * 16 * 32 / median("byte_expand", spread_row)
    return {
        "products": (products, f"wgmma_rate / {pipe_row}"),
        "shared": (shared, f"byte_expand / {spread_row} (7 16-byte accesses a lane)"),
        "drain_own": (median("bulk_drain", "one bulk add a panel, each CTA its own sum") / 256.0,
                      "bulk_drain / one bulk add a panel, each CTA its own sum (a byte)"),
        "drain_shared": (median("bulk_drain", "one bulk add a panel, one sum for every CTA") / 256.0,
                         "bulk_drain / one bulk add a panel, one sum for every CTA (a byte)"),
    }


@dataclass(frozen=True)
class Side:
    """One side's expected counts a token."""
    leaves: float
    boxes: float
    clusters: float
    lists: float


def distinct(ranges: float, draws: int) -> float:
    """The expected distinct ranges `draws` uniform digits fall in, of `ranges` equal ones."""
    return ranges * (1.0 - (1.0 - 1.0 / ranges) ** draws)


def side(widths, sparse, k: int, box, cluster) -> Side:
    """A side's counts: the leaves a token touches, the boxes and clusters those fall in, and the lists it sits in a
    box -- the box's leaves it touches over its digits on the level the MMA runs along (a dense level of the side
    inside the box where there is one, else its level of most digits)."""
    leaves = boxes = clusters = 1.0
    in_box = []
    for width, flag, b, c in zip(widths, sparse, box, cluster):
        digits = min(k, width) if flag else width
        ranges = width / b
        touched = distinct(ranges, digits) if flag else ranges
        leaves *= digits
        boxes *= touched
        clusters *= distinct(ranges / c, digits) if flag else ranges / c
        in_box.append(digits / touched)
    dense = [n for n, flag in zip(in_box, sparse) if not flag]
    along = max(dense) if dense else max(in_box)
    return Side(leaves, boxes, clusters, (leaves / boxes) / along)


def floor(widths, read_sparse, write_sparse, k: int, box, cluster, dv: int, tokens: int, form: str = "s64") -> dict:
    """The report: each term's SM-cycles a token as `(low, high)`, with the rate it used."""
    rates = measured()
    est = {name: value for name, (value, _) in ESTIMATES.items()}
    read = side(widths, read_sparse, k, box, cluster)
    write = side(widths, write_sparse, k, box, cluster)
    rows = max(dv, MMA_ROWS)
    clustered = math.prod(cluster) > 1
    products, shared = rates["products"][0], rates["shared"][0]
    drain = (rates["drain_own"][0], rates["drain_shared"][0])
    value_requests = math.ceil(dv * 2 / 128)
    leaves_a_box = math.prod(box)
    #: every box scans once a window, its four warps each issuing the scan
    scan = est["scan"] * 4 / est["issue"] * (math.prod(widths) / leaves_a_box) / est["window"]

    def both(x: float) -> tuple[float, float]:
        return (x, x)

    partial = dv * (8 if form == "s64" else 4)
    #: `slots`: the collector reads back what the boxes stored, and adds each value once (an instruction a value)
    passes = 1 if form == "s64" else 2
    work = (read.boxes * dv * est["convert"] / 32 / est["issue"] + scan if form == "s64"
            else read.boxes * dv / 32 / est["issue"])
    drains = read.clusters if clustered else read.boxes
    loads = (write.clusters if clustered else write.boxes) * write.lists * value_requests
    terms = {
        "tensor, readout": (both(read.leaves * rows / products), "products"),
        "tensor, fold": (both(write.leaves * rows / products), "products"),
        "readout operand, expanded": (both(read.leaves * 2 / shared), "shared"),
        "sum: partials into the cluster": (both(passes * read.boxes * partial / shared if clustered else 0.0), "shared"),
        "sum: drains to global memory": (tuple(drains * (dv * 4 + 4) * d for d in drain), "drain_own .. drain_shared"),
        "sum: its instructions": ((0.0, work if clustered else 0.0), "ESTIMATED issue (and convert, scan); hidden .. not"),
        "fold operand, expanded": (both(write.leaves * 2 / shared), "shared"),
        "fold values, loaded": ((loads * est["load_low"], loads * est["load_high"]), "ESTIMATED load_low .. load_high"),
    }
    total = tuple(sum(v[0][i] for v in terms.values()) for i in (0, 1))
    seconds = tuple(t * tokens / SMS / est["clock"] for t in total)
    state = math.prod(widths) * (dv + 1) * 3 * 2
    flash = tokens / 2 * dv * 2 / products
    return {
        "shape": {"widths": list(widths), "read_sparse": list(read_sparse), "write_sparse": list(write_sparse), "k": k,
                  "box": list(box), "cluster": list(cluster), "dv": dv, "tokens": tokens, "sum": form},
        "counts": {"read": vars(read), "write": vars(write)},
        "terms": {name: {"cycles": list(cycles), "rate": rate} for name, (cycles, rate) in terms.items()},
        "total_cycles_a_token": list(total),
        "call_seconds": list(seconds),
        "state_seconds_a_call": [state / est["memory_high"], state / est["memory_low"]],
        "flash_cycles_a_token": flash,
        "flash_call_seconds": flash * tokens / SMS / est["clock"],
        "measured": {name: {"value": value, "from": where} for name, (value, where) in rates.items()},
        "estimated": {name: {"value": value, "why": why} for name, (value, why) in ESTIMATES.items()},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", type=int, nargs="+", required=True)
    ap.add_argument("--read-sparse", type=int, nargs="+", required=True, help="a flag a level: 1 sparse, 0 dense")
    ap.add_argument("--write-sparse", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True, help="a sparse level's live digits a token")
    ap.add_argument("--box", type=int, nargs="+", required=True, help="a box's digits of each level")
    ap.add_argument("--cluster", type=int, nargs="+", required=True, help="a cluster's boxes along each level")
    ap.add_argument("--sum", choices=("s64", "slots"), default="s64", help="the cluster's sum: integers, or fp32 slots")
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    depth = len(a.widths)
    for name in ("read_sparse", "write_sparse", "box", "cluster"):
        if len(getattr(a, name)) != depth:
            raise SystemExit(f"carry_floor: --{name.replace('_', '-')} takes one value a level ({depth})")
    for width, b, c in zip(a.widths, a.box, a.cluster):
        if width % b or (width // b) % c:
            raise SystemExit(f"carry_floor: a level of {width} digits is not whole boxes of {b} in whole clusters of {c}")

    report = floor(a.widths, a.read_sparse, a.write_sparse, a.k, a.box, a.cluster, a.dv, a.tokens, a.sum)
    read, write = report["counts"]["read"], report["counts"]["write"]
    print(f"a token reads {read['leaves']:.0f} leaves in {read['boxes']:.1f} boxes of {read['clusters']:.2f} clusters; "
          f"writes {write['leaves']:.0f} in {write['boxes']:.1f} boxes of {write['clusters']:.2f} clusters, "
          f"{write['lists']:.2f} lists a box")
    for name, term in report["terms"].items():
        low, high = term["cycles"]
        span = f"{low:8.1f}" if low == high else f"{low:8.1f} .. {high:.1f}"
        print(f"  {name:32s} {span:22s} {term['rate']}")
    low, high = report["total_cycles_a_token"]
    print(f"  {'TOTAL, SM-cycles a token':32s} {low:8.1f} .. {high:.1f}")
    s_low, s_high = report["call_seconds"]
    print(f"a call of {a.tokens} tokens: {1e6 * s_low:.0f} .. {1e6 * s_high:.0f} us (ESTIMATED clock, {SMS} SMs); "
          f"its state in and out {1e6 * report['state_seconds_a_call'][0]:.1f} .. "
          f"{1e6 * report['state_seconds_a_call'][1]:.1f} us more (ESTIMATED memory)")
    print(f"FlashAttention's tensor floor: {report['flash_cycles_a_token']:.0f} SM-cycles a token, "
          f"{1e6 * report['flash_call_seconds']:.0f} us the call")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
