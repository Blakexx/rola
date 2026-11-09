#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READOUT'S TWO SCALINGS against the fp64 oracle, on a cell's real state: a box's readout from the f16 image at
each digit's power of two, either OUTPUT SCALING (each digit's sum over the lists in fp32, times its coefficient, the
digits summed in fp32) or OPERAND SCALING (each digit's coefficient on the f16 read rows at a per-token exponent, ONE
fp32 accumulator over every leaf of the box) -- the value-carve readout's two candidates for a dense side (event
2026-11-06 "the dense specialization weighs one shared accumulator a tile").

    bazel run //tools:readout_scaling -- flagship-dense
    bazel run //tools:readout_scaling -- nl64k-dense --windows 1 64 127 --box 32x16x64

Each slot's error is over its magnitude ``sum |x| |a| |S|`` (the readout reference's): printed max, 99.9th and 50th
percentiles for both forms, and the share of operand products under f16's normal range and flushed to zero. The
cross-CTA sum is not in either (a box's in-CTA sum only). Docs: docs/internals/tools/readout_scaling.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_TOP = 10
#: f16's least normal
F16_NORMAL = 2.0 ** -14


def _ceil_exp(x: torch.Tensor) -> torch.Tensor:
    m, e = torch.frexp(x)
    return torch.where(m == 0.5, e - 1, e).double()


def box_forms(x: torch.Tensor, a: torch.Tensor, S: torch.Tensor, box: tuple[int, int, int]) -> dict:
    """One window, one batch-head: ``x`` ``[t, B_S]`` level S's coefficients, ``a`` ``[t, B_R]`` level R's rows (both
    fp64 as stored), ``S`` ``[B_S, B_R, C]`` the state over kappa before the window; ``box`` (lists, digits,
    channels) -- lists along level R, digits along level S. Returns the truth, the magnitude and both forms
    ``[t, C]`` fp64, and the operand products' range counts."""
    lists, digits, cb = box
    t, B_S = x.shape
    B_R, C = a.shape[1], S.shape[-1]
    KS, KR, KC = B_S // digits, B_R // lists, C // cb
    Sb = S.reshape(KS, digits, KR, lists, KC, cb)
    xb, ab = x.reshape(t, KS, digits), a.reshape(t, KR, lists)
    truth = torch.einsum("tsd,tkr,sdkrjc->tskjc", xb, ab, Sb)
    mag = torch.einsum("tsd,tkr,sdkrjc->tskjc", xb.abs(), ab.abs(), Sb.abs())
    #: the image: each digit of each box at its power of two (its largest over the box's lists and channels <= 2^10)
    top = Sb.abs().amax(dim=(3, 5))                                                    # [KS, d, KR, KC]
    e_d = torch.where(top > 0, _ceil_exp(top.clamp_min(1e-300)), torch.full_like(top, IMAGE_TOP - 127.0))
    e_d = e_d.clamp_min(IMAGE_TOP - 127.0)
    img = (Sb * torch.exp2(IMAGE_TOP - e_d)[:, :, :, None, :, None]).half().float()
    a16 = ab.half().float()
    #: OUTPUT SCALING: a digit's sum over the lists in fp32, times x 2^(e_d - top) in fp32, the digits summed in fp32
    q = torch.einsum("tkr,sdkrjc->tsdkjc", a16, img)
    f = (xb.float()[:, :, :, None, None] * torch.exp2(e_d - IMAGE_TOP).float()[None])  # [t, KS, d, KR, KC]
    out = torch.einsum("tsdkjc,tsdkj->tskjc", q, f).double()
    #: OPERAND SCALING: e_t the token's largest digit exponent in the box, y = f16(a x 2^(e_d - e_t)), one accumulator
    ex = torch.where(xb > 0, _ceil_exp(xb.clamp_min(1e-300)), torch.full_like(xb, -1e9))
    e_tok = (ex[:, :, :, None, None] + e_d[None]).amax(dim=2)                          # [t, KS, KR, KC]
    s = xb[:, :, :, None, None] * torch.exp2(e_d[None] - e_tok[:, :, None])            # [t, KS, d, KR, KC]
    yr = ab[:, None, None, :, None, :] * s[..., None]                                 # [t, KS, d, KR, KC, r]
    y = yr.half()
    live = yr != 0
    sub = (live & (yr.abs() < F16_NORMAL)).sum().item()
    flushed = (live & (y == 0)).sum().item()
    acc = torch.einsum("tsdkjr,sdkrjc->tskjc", y.float(), img)
    oper = acc.double() * torch.exp2(e_tok - IMAGE_TOP)[..., None]
    return {"truth": truth.sum(dim=(1, 2)).reshape(t, C), "mag": mag.sum(dim=(1, 2)).reshape(t, C),
            "output": out.sum(dim=(1, 2)).reshape(t, C), "operand": oper.sum(dim=(1, 2)).reshape(t, C),
            "products": live.sum().item(), "subnormal": sub, "flushed": flushed}


def run(name: str, windows: list[int] | None, box: tuple[int, int, int]) -> list[str]:
    from measure.cells import by_name
    from measure.subjects import carry_operands
    from tests.oracle.fold_reference import fold_state, unit_scale
    from tests.oracle.readout_reference import _read_rows

    torch.set_float32_matmul_precision("highest")
    routes, values, facts, _carve, _u = carry_operands(by_name(name))
    level_s = 1
    window = 512
    x, a = _read_rows(routes, level_s)
    kappa = unit_scale(routes, level_s)
    want = set(windows) if windows else None
    errs = {"output": [], "operand": []}
    counts = {"products": 0, "subnormal": 0, "flushed": 0}

    def at_window(w, state, _env):
        if want is not None and w not in want:
            return
        sl = slice(w * window, min((w + 1) * window, routes.tokens))
        st = state / kappa
        for b in range(st.shape[0]):
            for h in range(st.shape[1]):
                r = box_forms(x[b, sl, h], a[b, sl, h], st[b, h], box)
                ok = r["mag"] > 0
                for k in errs:
                    errs[k].append(((r[k] - r["truth"]).abs()[ok] / r["mag"][ok]).cpu())
                for k in counts:
                    counts[k] += r[k]

    fold_state(routes, values, facts.list_exp, level_s, window, at_window=at_window)
    lines = [f"{name} box {box[0]}x{box[1]}x{box[2]} windows {sorted(want) if want else 'all'}"]
    for k, v in errs.items():
        e = torch.cat(v).double()
        q = torch.quantile(e[torch.randperm(e.numel())[:1_000_000]], torch.tensor([0.5, 0.999], dtype=torch.float64))
        lines.append(f"  {k:8s} error / magnitude: max {e.max().item():.3e}  p99.9 {q[1].item():.3e}  "
                     f"p50 {q[0].item():.3e}  ({e.numel():,} slots)")
    p = max(1, counts["products"])
    lines.append(f"  operand products: {counts['products']:,}; under f16's normal range {counts['subnormal'] / p:.2e}, "
                 f"flushed to zero {counts['flushed'] / p:.2e}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--windows", type=int, nargs="+", default=None, help="the windows to price (default every one)")
    ap.add_argument("--box", default="32x16x64", help="lists x digits x channels")
    a = ap.parse_args()
    box = tuple(int(v) for v in a.box.lower().split("x"))
    for name in a.cells:
        print("\n".join(run(name, a.windows, box)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
