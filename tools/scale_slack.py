#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALE'S SLACK: whether a bound looser than the exact read product can stand as the scale of the carry's f16 sum
across CTAs. The carry runs on one bundle at the exact bound's scale bytes twice and once at each candidate's -- the
scale pass's own bytes among them -- its f16-pair ``num`` taken back by each token's power of two; the sum's order
across CTAs is the atomics' arrival, so two runs at one scale differ by f16's rounding already -- a candidate is
harmless where its spread from the exact-bound run is that run-to-run spread. Beside it, each token's slack: log2 of
the bound over the exact read product.

    bazel run //tools:scale_slack -- nl64k-alt-k4 flagship-cohort-k16-L16384 [--json slack.json]   # carry cells
    bazel run //tools:scale_slack -- nl64k-alt-h4 nl64k-alt-h4-g16                                 # layer constructions

THE BOUNDS on ``|num(t)|`` (``x`` a read row, ``w`` a write row, ``g`` the write gain, ``s`` a writing token's value
scale, each sum over the writing tokens of the windows before t's), in the recurrence's units:
  exact  sum_a x_S(t, a) sum_r x_R(t, r) M(a, r),  M(a, r) = sum w_S(a) w_R(r) g s
  write  (sum_r x_R(t, r)) sum_a x_S(t, a) G(a),     G(a) = sum w_S(a) max_r w_R(r) g s
  facts  (sum_a x_S(t, a)) (sum_r x_R(t, r)) sum max_a w_S(a) max_r w_R(r) g s
  window (sum_a x_S(t, a)) (sum_r x_R(t, r)) max_a G(a)                          (a window's one write mass)
the scale bytes take each over the reader's per-token row scales and the carry's factor (`p_of`), by the byte rule.

Docs: docs/internals/tools/scale_slack.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bounds(routes, values) -> dict:
    """Each bound ``[B H, T]`` fp64, a query head's (its write head's mass), window by window at the device's W."""
    import torch

    from rola.ops.window import window_of

    W = window_of(routes.device)
    ls, lr = routes.level_s, 1 - routes.level_s
    read, write = routes.dequantized()
    x_s, x_r = read[ls].double(), read[lr].double()                               # [B, T, H, w]
    w_s, w_r = write[ls].double(), write[lr].double()                             # [B, T, Hw, w]
    s = values.scale.double()                                                     # [B, T, Hw]: a producer's gain is in the write levels
    B, T, H, _ = x_s.shape
    g = routes.group
    out = {k: torch.zeros(B, T, H, dtype=torch.float64, device=x_s.device)
           for k in ("exact", "write", "facts", "window")}
    M = torch.zeros(B, w_s.shape[2], w_s.shape[-1], w_r.shape[-1], dtype=torch.float64, device=x_s.device)
    G = torch.zeros(M.shape[:-1], dtype=torch.float64, device=x_s.device)
    F = torch.zeros(M.shape[:2], dtype=torch.float64, device=x_s.device)

    def heads(m):
        return m.repeat_interleave(g, 1)                                          # a query head its write head's

    for t0 in range(0, T, W):
        #: the window's tokens read the mass of the windows before, then its writes join it
        sl = slice(t0, min(t0 + W, T))
        xs, xr = x_s[:, sl], x_r[:, sl]
        out["exact"][:, sl] = torch.einsum("btha,bthr,bhar->bth", xs, xr, heads(M))
        out["write"][:, sl] = xr.sum(-1) * torch.einsum("btha,bha->bth", xs, heads(G))
        out["facts"][:, sl] = xs.sum(-1) * xr.sum(-1) * heads(F)[:, None]
        out["window"][:, sl] = xs.sum(-1) * xr.sum(-1) * heads(G).amax(-1)[:, None]
        ws, wr, f = w_s[:, sl], w_r[:, sl], s[:, sl]
        M += torch.einsum("btha,bthr,bth->bhar", ws, wr, f)
        G += torch.einsum("btha,bth->bha", ws, wr.amax(-1) * f)
        F += (ws.amax(-1) * wr.amax(-1) * f).sum(1)
    return {k: v.permute(0, 2, 1).reshape(B * H, T).contiguous() for k, v in out.items()}


def p_of(routes, bound):
    """``[B H, T]`` fp64: a bound in the recurrence's units over the reader's factor and the carry's (the byte rule's
    ``P``)."""
    from rola.ops.prefill import carry_factor, reader_factor

    return bound / (reader_factor(routes).double() * carry_factor(routes))


def bytes_of(p):
    """``[B H, windows W]`` int8: the byte rule on ``P`` over the window grid (`csrc/rola/src/carry/carry.cuh`'s
    `unit_byte`), a partial window's tail reading nothing."""
    import torch

    from rola.ops.prefill import UNIT_HEADROOM, UNIT_LEVELS, UNIT_NONE
    from rola.ops.window import window_of

    s = torch.ceil(torch.log2(UNIT_HEADROOM * p.clamp_min(1e-300))) - UNIT_LEVELS
    s = torch.where(p > 0, s, torch.full_like(s, UNIT_NONE)).clamp(-128, 127).to(torch.int8)
    W = window_of(p.device)
    grid = torch.full((p.shape[0], -(-p.shape[1] // W) * W), UNIT_NONE, dtype=torch.int8, device=p.device)
    grid[:, :p.shape[1]] = s
    return grid


def operands(name: str):
    """``(routes, values, facts, carve)``: a carry cell's (`measure.subjects.carry_operands`), or a layer
    construction's on its own cell, the producer run once (`measure.subjects._layer_facts`)."""
    from measure.cells import by_name
    from measure.cells.layer import CONSTRUCTIONS, build

    if name in CONSTRUCTIONS:
        from measure.subjects import _layer_facts

        construction = CONSTRUCTIONS[name]
        routes, values, _layout, _statics, carve, facts = _layer_facts(build(by_name(construction.cells[0]),
                                                                             construction))
        return routes, values, facts, carve
    from measure.subjects import carry_operands

    routes, values, facts, carve, _zero = carry_operands(by_name(name))
    return routes, values, facts, carve


def reading(name: str) -> dict:
    """One cell: the scale pass's bytes against the write bound's, each candidate's slack, and each candidate's spread
    -- the pass's own bytes among them -- from the exact-bound run against two exact-bound runs' own."""
    import torch

    from rola.ops import carry as carry_ops
    from rola.ops import scale as scale_ops
    from rola.ops.prefill import carry_factor, unpack_num

    routes, values, facts, carve = operands(name)
    true = bounds(routes, values)
    exact = true["exact"]
    live = exact > 0

    def num_at(scale):
        num, _den, _, _ = carry_ops.carry(routes, values, facts, carve, scale, reduce_dtype=torch.float16)
        got = unpack_num(num).double()
        return got * torch.exp2(scale[:, :got.shape[1]].double() + 1)[..., None]

    def spread(got, ref):
        r = ((got - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-300))[live]
        return {"median": float(r.median()), "max": float(r.max()), "inf": int((~torch.isfinite(got)).sum())}

    def slack(b):
        r = torch.log2(b[live] / exact[live])
        q = torch.quantile(r[:: max(1, r.numel() // (1 << 24))],
                           torch.tensor([0.5, 0.99], dtype=torch.float64, device=r.device))
        return {"min": round(float(r.min()), 3), "median": round(float(q[0]), 2), "p99": round(float(q[1]), 2),
                "max": round(float(r.max()), 2)}

    ref_scale = bytes_of(p_of(routes, exact))
    ref = num_at(ref_scale)
    pass_scale = scale_ops.scale(routes, facts, carry_factor(routes))
    T = exact.shape[1]
    out = {"cell": name, "tokens": int(live.sum()),
           "pass_vs_write": int((pass_scale[:, :T] != bytes_of(p_of(routes, true["write"]))[:, :T]).sum()),
           "run_to_run": spread(num_at(ref_scale), ref), "candidates": {}}
    for key in ("write", "facts", "window"):
        got = num_at(bytes_of(p_of(routes, true[key])))
        out["candidates"][key] = {"slack": slack(true[key]), "spread": spread(got, ref),
                                  "below_exact": int((true[key][live] < exact[live] * (1 - 1e-9)).sum())}
    out["candidates"]["pass"] = {"slack": out["candidates"]["write"]["slack"], "spread": spread(num_at(pass_scale), ref),
                                 "below_exact": out["candidates"]["write"]["below_exact"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for c in a.cells:
        r = reading(c)
        rows.append(r)
        rr = r["run_to_run"]
        print(f"{r['cell']}: {r['tokens']} tokens; the scale pass's bytes off the write bound's: {r['pass_vs_write']};"
              f" run to run median {rr['median']:.2e} max {rr['max']:.2e}")
        for key, c_ in r["candidates"].items():
            s, sp = c_["slack"], c_["spread"]
            print(f"  {key:5s} slack log2 median {s['median']} p99 {s['p99']} max {s['max']} (min {s['min']},"
                  f" {c_['below_exact']} below exact); spread median {sp['median']:.2e} max {sp['max']:.2e}"
                  f" inf {sp['inf']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
