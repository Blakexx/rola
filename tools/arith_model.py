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
- THE CROSS-BOX SUM, alone (nothing else rounded; `--box FIRST SECOND` digits a level a box, a cluster the boxes sharing one
  range of level `--along`): the readout's
  partial of each box is an fp32 accumulator, and the partials are summed box after box in fp32 (`sum fp32`) or as
  INTEGERS of 32 or 64 bits at a scale `2^q`, `q = bits - 2 - ceil(log2(bound))` (a partial times `2^q` rounded,
  the integers summed exactly). The bound is one of:
  `call` -- `N sum(g)` over the whole call, `g` a token's write factor (`m s` for num, `m` for den);
  `coarse` -- for a window, the largest `g` times its row mass so far, times the tokens so far;
  `running` -- for a window, the sum of `g` times its row mass so far;
  `state` -- for a window and a CLUSTER of boxes, the largest channel's sum of the cluster's state magnitudes;
  `planned` -- for a window and a cluster, the sum so far of `g` times the token's write mass inside the cluster.
  The first three sum every box as integers; the last two sum a cluster's boxes as integers and the clusters in fp32.
  `norm` -- no bound from the state at all: a box's num over its CLUSTER'S den for the token is a share of a weighted
  average of values, under the call's largest value scale; the cluster's shares are summed as integers at that one
  scale, turned back and multiplied by the cluster's den, and the clusters summed in fp32.
  `--scale-spread B` raises the tokens' write scales by B binades over the call, `--value-spread B` their value
  scales, and `--leaf-spread B` spreads the write scales over the leaves of a cluster, to stress the scales.
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


#: the integer sums modelled: (bits, the bound's form)
INT_SUMS = tuple((bits, form) for bits in (32, 64) for form in ("call", "coarse", "running", "state", "planned"))


def _as_integers(partials, bound, bits: int):
    """`partials` `[..., t, d_v + 1]` as integers at the scale the bound gives a column kind -- `bound` `[..., 2]` (num,
    den), broadcast over the leading axes -- and that scale `[..., 1, d_v + 1]`. A zero bound holds zeros."""
    import torch

    d_v = partials.shape[-1] - 1
    q = (bits - 2) - torch.ceil(torch.log2(bound.clamp_min(1e-300)))
    scale = torch.where(bound > 0, 2.0 ** q, torch.zeros_like(bound))
    scale = torch.cat((scale[..., :1].expand(*scale.shape[:-1], d_v), scale[..., 1:]), dim=-1)[..., None, :]
    return torch.round(partials * scale).to(torch.int64), scale


def _sums(partials, cluster: int, bounds: dict) -> dict:
    """Every modelled sum of one window's box partials `[boxes, t, d_v + 1]` (each already an fp32 accumulator):
    `fp32` box after box, and each integer form of `INT_SUMS` under its bound (`bounds[form]`: `[2]` for the forms
    that sum every box, `[clusters, 2]` for the cluster forms, whose clusters are then summed in fp32)."""
    import torch

    out = {}
    total = torch.zeros_like(partials[0])
    for partial in partials:
        total = _fp32(total + partial)
    out["sum fp32"] = total
    grouped = partials.view(-1, cluster, *partials.shape[1:])                # [clusters, cluster, t, d_v + 1]
    for bits, form in INT_SUMS:
        bound = bounds[form]
        if bound.dim() == 1:
            ints, scale = _as_integers(partials, bound, bits)
            summed = ints.sum(dim=0).double()
            out[f"sum s{bits} {form}"] = torch.where(scale > 0, summed / scale.clamp_min(1e-300), torch.zeros_like(summed))
        else:
            ints, scale = _as_integers(grouped, bound[:, None, :], bits)     # scale [clusters, 1, 1, d_v + 1]
            summed = ints.sum(dim=1).double()                                # [clusters, t, d_v + 1]
            parts = _fp32(torch.where(scale[:, 0] > 0, summed / scale[:, 0].clamp_min(1e-300), torch.zeros_like(summed)))
            total = torch.zeros_like(partials[0])
            for part in parts:
                total = _fp32(total + part)
            out[f"sum s{bits} {form}"] = total
    #: NORMALIZED IN THE CLUSTER: each box's num over the cluster's den for the token (the boxes' den partials summed
    #: in fp32) is a share of a weighted average of values, so it sits under the call's largest value scale -- the one
    #: scale, `bounds["norm"]`, known before the launch. The cluster's integers summed, back to fp32, times its den;
    #: the clusters' num and den then summed in fp32
    d_v = partials.shape[-1] - 1
    den = torch.zeros_like(grouped[:, 0, :, d_v])
    for k in range(cluster):
        den = _fp32(den + grouped[:, k, :, d_v])                             # [clusters, t]
    live = den > 0
    share = _fp32(torch.where(live[:, None, :, None], grouped[..., :d_v] / den.clamp_min(1e-300)[:, None, :, None],
                              torch.zeros_like(grouped[..., :d_v])))
    for bits in (32, 64):
        scale = 2.0 ** ((bits - 2) - torch.ceil(torch.log2(bounds["norm"])))
        num = _fp32(torch.round(share * scale).to(torch.int64).sum(dim=1).double() / scale * den[..., None])
        total = torch.zeros_like(partials[0])
        for c in range(num.shape[0]):
            total[:, :d_v] = _fp32(total[:, :d_v] + num[c])
            total[:, d_v] = _fp32(total[:, d_v] + den[c])
        out[f"sum s{bits} norm"] = total
    return out


def _errors(y, want) -> dict:
    import torch

    num, den = (y - want).norm(dim=-1), want.norm(dim=-1)
    live = den > 0
    rel = num[live] / den[live]
    q = torch.quantile(rel, torch.tensor([0.5, 0.99], dtype=torch.float64, device=rel.device))
    return {"median": float(q[0]), "p99": float(q[1]), "max": float(rel.max()), "tokens": int(live.sum())}


def model(name: str, window: int, head: int, shape: tuple[int, int], along: int, spread: int,
          value_spread: int, leaf_spread: int) -> dict:
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
    T = v.shape[0]
    u = _bf16(torch.where(s[:, None] > 0, v / s[:, None].clamp_min(1e-300), torch.zeros_like(v)))
    T, d_v = v.shape
    if value_spread:
        #: THE VALUES' STRESS: each token's value scale rises over the call by `value_spread` binades, so the early
        #: tokens' values sit that far under the call's largest -- the one thing a normalized sum's scale depends on
        s = s * 2.0 ** torch.round(-value_spread * (1.0 - torch.arange(T, dtype=torch.float64, device=v.device)
                                                         / max(T - 1, 1)))
    r = torch.stack([scale for _, scale in reads]).prod(dim=0)              # the reader's scale, [T]
    if spread:
        #: THE STRESS: each token's write scale (its first level's) times a power of two that RISES over the call by
        #: `spread` binades, so the early tokens' sums sit that far under the bound the late ones set -- the worst
        #: case for one scale a call
        ramp = torch.arange(T, dtype=torch.float64, device=v.device) / max(T - 1, 1)
        factor = 2.0 ** torch.round(-spread * (1.0 - ramp))
        writes[0] = (writes[0][0], writes[0][1] * factor)
    if leaf_spread:
        #: THE LEAVES' STRESS: each token's write scale falls by up to `leaf_spread` binades with its largest write
        #: digit on the level a cluster spans whole, so one cluster's leaves hold magnitudes that far apart -- the
        #: worst case for a scale taken from the cluster's own state
        unit = writes[1 - along][0]
        digit = unit.argmax(dim=-1).double() / max(unit.shape[-1] - 1, 1)
        writes[0] = (writes[0][0], writes[0][1] * 2.0 ** torch.round(-leaf_spread * digit))
    m = torch.stack([scale for _, scale in writes]).prod(dim=0)             # the write scale, [T]
    variants = {"all": set(SOURCES), **{src: {src} for src in SOURCES}, "none": set()}
    sum_keys = ("sum fp32", *(f"sum s{bits} {form}" for bits, form in INT_SUMS), "sum s32 norm", "sum s64 norm")
    if len(widths) != 2:
        raise SystemExit(f"{name}: the box sums are modelled at two levels")
    #: THE BOXES: `shape[l]` digits of level `l` each; a CLUSTER is the boxes sharing one range of level `along`,
    #: across the whole other level. `order` lists the leaves box by box, a cluster's boxes together
    digits = [torch.arange(w, device=v.device) for w in widths]
    grid = digits[0][:, None] * strides[0] + digits[1][None, :] * strides[1]         # [w0, w1] leaf of (d0, d1)
    blocks = grid.view(widths[0] // shape[0], shape[0], widths[1] // shape[1], shape[1]).permute(0, 2, 1, 3)
    if along == 1:
        blocks = blocks.transpose(0, 1)
    order = blocks.reshape(-1)
    box, cluster = shape[0] * shape[1], widths[1 - along] // shape[1 - along]
    clusters = N // (box * cluster)
    #: the bounds' running terms: a token's `g` (num's `m s`, den's `m`), times its row mass over every leaf (coarse,
    #: running) or inside each cluster (planned)
    g_all = torch.stack((m * s, m), dim=-1)                                  # [T, 2]
    call_bound = N * g_all.sum(dim=0)
    coarse_max = torch.zeros(2, dtype=torch.float64, device=v.device)
    running = torch.zeros(2, dtype=torch.float64, device=v.device)
    planned = torch.zeros(clusters, 2, dtype=torch.float64, device=v.device)
    read_clusters = []
    truth = torch.zeros(N, d_v + 1, dtype=torch.float64, device=v.device)
    states = {k: torch.zeros_like(truth) for k in variants}
    want, got = [], {k: [] for k in (*variants, *sum_keys)}
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
        #: THE SUMS, alone: the truth's state, each box's partial an fp32 accumulator, every other term exact
        Ro, So = Rq[:, order], truth[order]
        partials = torch.stack([_fp32(Ro[:, at:at + box] @ So[at:at + box]) for at in range(0, N, box)])
        mags = So.abs().view(clusters, box * cluster, d_v + 1).sum(dim=1)     # [clusters, d_v + 1]
        del Ro
        bounds = {"call": call_bound, "coarse": coarse_max * lo, "running": running, "planned": planned,
                  "state": torch.stack((mags[:, :d_v].amax(dim=1), mags[:, d_v]), dim=-1), "norm": s.max()}
        exact_intra = gram @ deposit
        for key, inter in _sums(partials, cluster, bounds).items():
            o = _fp32(inter + exact_intra)
            got[key].append(o[:, :d_v] / (o[:, d_v:] + eps[:, None]))
        read_clusters.append((partials.view(clusters, cluster, *partials.shape[1:])[..., :d_v].abs().amax(dim=(1, 3)) > 0)
                             .sum(dim=0).double())
        del partials
        mass = Wq[:, order].view(Wq.shape[0], clusters, box * cluster).sum(dim=-1)     # [t, clusters]
        g = g_all[sl]
        planned = planned + torch.einsum("tc,tk->ck", mass, g)
        whole = g * mass.sum(dim=-1, keepdim=True)
        running = running + whole.sum(dim=0)
        coarse_max = torch.maximum(coarse_max, whole.amax(dim=0))
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
    return {"cell": name, "tokens": T, "window": window, "head": head, "box": list(shape), "along": along,
            "cluster": cluster,
            "scale_spread": spread, "value_spread": value_spread, "leaf_spread": leaf_spread, "clusters_read_a_token": float(torch.cat(read_clusters[1:] or read_clusters).mean()),
            "errors": {k: _errors(torch.cat(ys), y64) for k, ys in got.items()}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--window", type=int, default=1024, help="the window W the model folds and reads at")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--box", type=int, nargs=2, default=(16, 16), metavar=("FIRST", "SECOND"),
                    help="a box's digits of level 0 and of level 1: the cross-box sum's partials")
    ap.add_argument("--value-spread", type=int, default=0,
                    help="stress: each token's value scale times 2^-k, k falling from this many binades to none")
    ap.add_argument("--leaf-spread", type=int, default=0,
                    help="stress: a token's write scale times 2^-k, k up to this many binades by its largest write digit")
    ap.add_argument("--along", type=int, choices=(0, 1), default=0,
                    help="the level a cluster holds one range of (its boxes span the other): the cluster forms' scope")
    ap.add_argument("--scale-spread", type=int, default=0,
                    help="stress: each token's write scale times 2^-k, k falling from this many binades to none over the call")
    ap.add_argument("--json", type=Path, help="write the records here")
    a = ap.parse_args(argv)

    from rola_devtools.locks.gpu import gpu_lock

    records = []
    with gpu_lock(mode="shared"):
        for cell in a.cells:
            record = model(cell, a.window, a.head, tuple(a.box), a.along, a.scale_spread, a.value_spread, a.leaf_spread)
            print(f"{cell}: a token reads {record['clusters_read_a_token']:.2f} of its {record['cluster']}-box clusters")
            records.append(record)
            for key, e in record["errors"].items():
                print(f"{cell} [{key}]: median {100 * e['median']:.3g}%  p99 {100 * e['p99']:.3g}%  "
                      f"max {100 * e['max']:.3g}%  over {e['tokens']} tokens")
    if a.json:
        a.json.write_text(json.dumps(records, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
