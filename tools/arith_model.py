#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE bf16 FAMILY'S ARITHMETIC, MODELLED against the fp64 recurrence on a cell's own draw: u8 rows at every level,
bf16 values and operands, fp32 accumulation and an fp32 sum.

    bazel run //tools:arith_model -- flagship-alt-k4 nl64k-dense [--window 1024] [--json model.json]

The cell's bundle is put in the format first -- each level's rows u8 unit-max with its fp32 scale (a dense level
too: no floor, a zero a zero), the values bf16 unit-max with an fp32 scale a token -- and the fp64 recurrence
on THOSE operands is the truth: summed a window at a time (what a 64K run can hold), and proven the oracle's every
run (`rola.ops.naive.naive_rola` over the first two windows, refused on a difference). The model rounds where a kernel of the family
rounds, a window `W` at a time:
- THE FOLD: a (token, leaf)'s coefficient, the levels' bytes over 255 times the token's `m s` (the value column) or
  `m` (the mass column), rounded to bf16 once; times the bf16 value; summed into an fp32 state.
- THE READOUT: the state rounded to bf16 (the accumulators as the operand), the read coefficient (the levels' bytes
  over 255, the reader's scale left out) rounded to bf16, an fp32 sum.
- THE INTRA: a window's token pairs' products exact (integers), times `m s` or `m`, rounded to bf16 once, times the
  bf16 value, an fp32 sum.
Each token's error is `|y - y64| / |y64|` over its channels; the record holds the median, the 99th percentile and the
largest with every rounding on, and with each one alone (the others exact), so the family's error has its sources.
One write head a run (`--head`); a grouped bundle is refused.

Docs: docs/internals/tools/arith_model.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: the roundings a run may turn on: the fold's coefficient, the state as the readout's operand, the read coefficient,
#: the intra's weight
SOURCES = ("fold", "image", "read", "intra")


def _bf16(x):
    import torch

    return x.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def _fp32(x):
    import torch

    return x.to(torch.float32).to(torch.float64)


def _format(levels):
    """Each level's rows as ``(unit, scale)`` in the u8 scaled-block form (`rola.routing.quant.ROWS`: against the
    row's max at 255, no floor, a zero a zero), ``[T, width]`` fp64 and ``[T]``."""
    from rola.routing import quant

    out = []
    for x in levels:
        rows, scale = quant.ROWS.quantize(x.float()[None, :, None])
        out.append((rows[0, :, 0].double() / quant.ROWS.top, scale[0, :, 0].double()))
    return out


def _prove_truth(reads, writes, u, s, topology, y64, tokens: int) -> None:
    """THE TRUTH IS THE ORACLE'S: `rola.ops.naive.naive_rola` on the formatted operands over the run's first
    ``tokens``, against this tool's windowed fp64 sums there -- refused where they differ, so the windowed form (what a
    64K run can hold) is the recurrence, proven every run."""

    from rola.ops.naive import naive_rola

    level = lambda side: tuple((unit * scale[:, None])[None, :tokens, None] for unit, scale in side)  # noqa: E731
    want, _ = naive_rola((u * s[:, None])[None, :tokens, None], level(reads), level(writes), topology, None)
    worst = float((y64[:tokens] - want[0, :, 0]).abs().amax() / want.abs().amax().clamp_min(1e-300))
    if not worst < 1e-9:
        raise SystemExit(f"the windowed truth is not the oracle's over the first {tokens} tokens: {worst:.3e}")


def _errors(y, want) -> dict:
    import torch

    num, den = (y - want).norm(dim=-1), want.norm(dim=-1)
    live = den > 0
    rel = num[live] / den[live]
    q = torch.quantile(rel, torch.tensor([0.5, 0.99], dtype=torch.float64, device=rel.device))
    return {"median": float(q[0]), "p99": float(q[1]), "max": float(rel.max()), "tokens": int(live.sum())}


def model(name: str, window: int, head: int) -> dict:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops.constants import READOUT_EPS
    from rola.ops.naive import _leaf_capacity, _leaf_product, _radix_strides

    spec = by_name(name)
    routes, values, _layout, _statics = bundle_of(spec, realize(spec))
    if routes.group != 1:
        raise SystemExit(f"{name}: a grouped bundle is not modelled")
    read, write = routes.dequantized()
    widths = routes.topology.widths
    N, strides = _leaf_capacity(widths), _radix_strides(widths)
    reads = _format([x[0, :, head].double() for x in read])
    writes = _format([x[0, :, head].double() for x in write])
    v = values.dequantized()[0, :, head].double()                           # [T, d_v]
    s = v.abs().amax(dim=-1)
    u = _bf16(torch.where(s[:, None] > 0, v / s[:, None].clamp_min(1e-300), torch.zeros_like(v)))
    T, d_v = v.shape
    r = torch.stack([scale for _, scale in reads]).prod(dim=0)              # the reader's scale, [T]
    m = torch.stack([scale for _, scale in writes]).prod(dim=0)             # the write scale, [T]
    variants = {"all": set(SOURCES), **{src: {src} for src in SOURCES}, "none": set()}
    truth = torch.zeros(N, d_v + 1, dtype=torch.float64, device=v.device)
    states = {k: torch.zeros_like(truth) for k in variants}
    want, got = [], {k: [] for k in variants}
    ones = torch.ones(1, dtype=torch.float64, device=v.device)

    for lo in range(0, T, window):
        sl = slice(lo, min(lo + window, T))
        leaves = lambda side: _leaf_product(tuple(unit[None, sl, None] for unit, _ in side), strides, N)[0, :, 0]  # noqa: E731
        Rq, Wq = leaves(reads), leaves(writes)                              # [t, N] unit products
        gram = torch.ones(Rq.shape[0], Rq.shape[0], dtype=torch.float64, device=v.device)
        for (xr, _), (xw, _) in zip(reads, writes):
            gram = gram * (xr[sl] @ xw[sl].T)
        gram = torch.tril(gram)                                             # a token reads its own write
        uw, mw, fw, rw = u[sl], m[sl], (m * s)[sl], r[sl]
        eps = READOUT_EPS / rw.clamp_min(1e-300)
        # the truth, the reader's scale left out of num and den alike
        deposit = torch.cat((uw * fw[:, None], mw[:, None] * ones), dim=-1)  # [t, d_v + 1]
        out = Rq @ truth + gram @ deposit
        want.append(out[:, :d_v] / (out[:, d_v:] + eps[:, None]))
        truth = truth + Wq.T @ deposit
        for key, on in variants.items():
            rnd = (lambda x, src: _bf16(x) if src in on else x)             # noqa: E731
            state = states[key]
            inter = rnd(Rq, "read") @ rnd(state, "image")
            intra = torch.cat((rnd(gram * fw[None, :], "intra") @ uw,
                               rnd(gram * mw[None, :], "intra").sum(dim=1, keepdim=True)), dim=-1)
            o = _fp32(inter + intra) if on else inter + intra
            got[key].append(o[:, :d_v] / (o[:, d_v:] + eps[:, None]))
            fold = torch.cat((rnd(Wq * fw[:, None], "fold").T @ uw,
                              rnd(Wq * mw[:, None], "fold").sum(dim=0)[:, None]), dim=-1)
            states[key] = _fp32(state + fold) if on else state + fold
            del inter, intra, fold
        del Rq, Wq
    y64 = torch.cat(want)
    _prove_truth(reads, writes, u, s, routes.topology, y64, min(T, 2 * window))
    return {"cell": name, "tokens": T, "window": window, "head": head,
            "errors": {k: _errors(torch.cat(ys), y64) for k, ys in got.items()}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--window", type=int, default=1024, help="the window W the model folds and reads at")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the records here")
    a = ap.parse_args(argv)

    from rola_devtools.locks.gpu import gpu_lock

    records = []
    with gpu_lock(mode="shared"):
        for cell in a.cells:
            record = model(cell, a.window, a.head)
            records.append(record)
            for key, e in record["errors"].items():
                print(f"{cell} [{key}]: median {100 * e['median']:.4f}%  p99 {100 * e['p99']:.4f}%  "
                      f"max {100 * e['max']:.4f}%  over {e['tokens']} tokens")
    if a.json:
        a.json.write_text(json.dumps(records, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
