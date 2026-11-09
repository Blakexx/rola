#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OUTPUT'S QUALITY: a stateless prefill's per-token error against the fp64 oracle, at each readout accumulation.

    bazel run //tools:quality -- flagship-alt-k4 [--json quality.json]

A carry cell's bundle (the quantized operands the kernels read) runs the prefill -- the facts pass, the scale pass,
the carry and the intra, then `y = num / den` -- once per pairing of the two legs' MMA accumulations (the carry's
readout, the intra's: fp32 and fp32, fp16 and fp32, fp32 and fp16, fp16 and fp16), and the fp64 recurrence
(`tests.oracle.fixtures.oracle_run`) on the same operands gives `y64`. A token's error is `|y - y64| / |y64|` over its
channels (L2), over the tokens whose `y64` is not zero; the record holds the median, the 99th percentile and the
largest, and which read side the cell's carve gives (listed or dense: a dense read side's readout accumulates fp32
whatever it is told, so its record holds the intra's pairings alone). `declare.py` runs it on the cells sized for both
the oracle and a measurement.

Docs: docs/internals/tools/quality.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def token_error(y, want) -> dict:
    """The per-token relative error's median, 99th percentile and largest, over the tokens that read something."""
    import torch

    num = (y.double() - want).flatten(0, -2).norm(dim=-1)
    den = want.flatten(0, -2).norm(dim=-1)
    live = den > 0
    rel = (num[live] / den[live])
    q = torch.quantile(rel[:: max(1, rel.numel() // (1 << 24))], torch.tensor([0.5, 0.99], dtype=torch.float64,
                                                                               device=rel.device))
    return {"median": float(q[0]), "p99": float(q[1]), "max": float(rel.max()), "tokens": int(live.sum())}


#: the legs' accumulations a run turns, the carry's readout first: both fp32, one leg at a time, both fp16
PAIRS = (("fp32", "fp32"), ("fp16", "fp32"), ("fp32", "fp16"), ("fp16", "fp16"))


def measure(name: str) -> dict:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.engine.carve import plan_carve
    from rola.ops import carry as carry_ops
    from rola.ops import facts as facts_ops
    from rola.ops import prefill as prefill_ops
    from tests.oracle.fixtures import oracle_run

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.lists_form(carry_ops.arm_key(carve)))
    read, write = routes.dequantized()
    want = oracle_run(values.dequantized().double(), tuple(x.double() for x in read),
                      tuple(x.double() for x in write), routes.topology).y
    key = carry_ops.arm_key(carve)
    #: a dense read side's operand is its two dense levels (`arm_key`'s read_factors), a listed one's its one level
    out = {"cell": name, "arm": list(key), "read_side": "dense" if key[3] == 2 else "listed",
           "device_form": carry_ops.readout_accumulate(), "forms": {}}
    for carry_form, intra_form in PAIRS:
        #: the entry refuses an fp16 readout on a dense read side, which accumulates fp32 alone
        if out["read_side"] == "dense" and carry_form == "fp16":
            continue
        num, den = prefill_ops.prefill(routes, values, facts, carve, layout, *statics, accumulate=carry_form,
                                       intra_accumulate=intra_form)
        torch.cuda.synchronize()
        out["forms"][f"carry {carry_form}, intra {intra_form}"] = token_error(prefill_ops.readout(num, den, routes),
                                                                              want)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--json", type=Path, help="write the record here")
    a = ap.parse_args(argv)

    from rola_devtools.locks.gpu import gpu_lock

    with gpu_lock(mode="shared"):
        record = measure(a.cell)
    for form, e in record["forms"].items():
        print(f"{record['cell']} ({record['read_side']} read side) {form}: median {100 * e['median']:.4f}%  "
              f"p99 {100 * e['p99']:.4f}%  max {100 * e['max']:.4f}%  over {e['tokens']} tokens")
    if a.json:
        a.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
