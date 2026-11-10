#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALED OPERANDS' f16 RANGE: where a readout's read rows land when each (token, digit)'s factor scales them before
the MMA, at the token's panel scale.

    bazel run //tools:factor_census -- flagship-alt-k4 [--json census.json]

A token `t` reading digit `d` scales its rows `a[t, r]` by `f = c[t, d] 2^(e_d - kImageTop - s_t - 1)` -- its read
coefficient, the digit's image exponent (the hand-off's: the power of two over the box's largest |state| of the
digit), the token's scale byte (the scale pass's) -- so the MMA against the digit's f16 image sums at the panel's
scale. The census reads, over every window of the cell's own draw (`tests/oracle/readout_reference.py`'s states, the
same arithmetic): the exponents of `f` and of the scaled operand `a f` over the nonzero entries -- the largest, the
smallest, and the shares under f16's normal floor 2^-14 (subnormal, fewer bits) and under its least subnormal 2^-24
(flushed) -- and the same for `a f 2^(s_t + 1)`, the operand scaled by everything but the token's byte. It is the f16
range check of the fp16 readouts' operand scaling (the plan's section 3.1).

Docs: docs/internals/tools/factor_census.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: f16's least normal and least subnormal powers of two
NORMAL_FLOOR, SUBNORMAL_FLOOR = -14, -24


def _shares(exps) -> dict:
    """The largest and smallest exponent and the shares under f16's normal floor and its least subnormal."""
    import torch

    if exps.numel() == 0:
        return {"count": 0}
    return {"count": int(exps.numel()), "max": float(exps.max()), "min": float(exps.min()),
            "median": float(torch.quantile(exps[:: max(1, exps.numel() // (1 << 24))], 0.5)),
            "subnormal": float((exps < NORMAL_FLOOR).double().mean()),
            "flushed": float((exps < SUBNORMAL_FLOOR).double().mean())}


def census(name: str) -> dict:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.engine.carve import plan_carve
    from rola.ops import carry as carry_ops
    from rola.ops import facts as facts_ops
    from rola.ops import prefill as prefill_ops
    from rola.ops import scale as scale_ops
    from rola.ops.window import window_of
    from tests.oracle import readout_reference as ref

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    key = carry_ops.arm_key(carve)
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.lists_form(key))
    scale = scale_ops.scale(routes, facts, prefill_ops.carry_factor(routes))
    #: level S the readout's digits: a listed read side's list level, a dense one's operand's second level
    level_s = carve.readout.lists[0] if carve.readout.lists else carve.readout.operand[1]
    window = window_of()
    x, a = ref._read_rows(routes, level_s)                                   # [B, T, H, B_S], [B, T, H, B_R]
    B, T, H, B_S = x.shape
    B_R = a.shape[-1]
    ES, ER = carve.extents[level_s], carve.extents[1 - level_s]
    KS, KR = B_S // ES, B_R // ER
    s_bytes = scale.double().reshape(B, H, -1).permute(0, 2, 1)[:, :T]       # [B, T, H]
    f_exps, op_exps, op_frame_exps = [], [], []
    for w, (state, _env) in enumerate(ref._windows(routes, values, facts.list_exp, level_s, window)):
        sl = slice(w * window, (w + 1) * window)
        top = state.abs().reshape(B, H, KS, ES, KR, ER, -1).amax(dim=(5, 6))  # [B, H, KS, ES, KR]
        live = top > 0
        e_d = torch.where(live, ref._ceil_exp(top.clamp_min(1e-300)).double(), torch.zeros_like(top))
        xs = x[:, sl].reshape(B, -1, H, KS, ES)
        s = s_bytes[:, sl]                                                   # [B, t, H]
        # the factor's exponent, a (token, digit, list box): log2|c| + e_d - kImageTop - s - 1
        fe = (torch.log2(xs.abs().clamp_min(1e-300))[..., None] + e_d[:, None]
              - ref.IMAGE_TOP - s[..., None, None, None] - 1)                 # [B, t, H, KS, ES, KR]
        reads = (xs != 0)[..., None] & live[:, None]
        f_exps.append(fe[reads].flatten())
        # the scaled operand: each nonzero row of the list box times the factor
        ar = torch.log2(a[:, sl].abs().clamp_min(1e-300)).reshape(B, -1, H, KR, ER)
        rows = (a[:, sl] != 0).reshape(B, -1, H, KR, ER)
        op = fe[..., None] + ar[:, :, :, None, None, :, :]                   # [B, t, H, KS, ES, KR, ER]
        mask = reads[..., None] & rows[:, :, :, None, None, :, :]
        op_exps.append(op[mask].flatten())
        op_frame_exps.append((op + (s + 1)[..., None, None, None, None])[mask].flatten())

    def cat(parts):
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

    return {"cell": name, "arm": list(key), "read_side": "dense" if key[3] == 2 else "listed",
            "factor": _shares(cat(f_exps)), "scaled_operand": _shares(cat(op_exps)),
            "scaled_operand_without_the_byte": _shares(cat(op_frame_exps))}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--json", type=Path, help="write the record here")
    a = ap.parse_args(argv)

    from rola_devtools.locks.gpu import gpu_lock

    with gpu_lock(mode="shared"):
        record = census(a.cell)
    for what in ("factor", "scaled_operand", "scaled_operand_without_the_byte"):
        e = record[what]
        if e["count"]:
            print(f"{record['cell']} ({record['read_side']}) {what}: exponents {e['min']:.1f} .. {e['max']:.1f} "
                  f"(median {e['median']:.1f}); under 2^-14 {100 * e['subnormal']:.3f}%, under 2^-24 "
                  f"{100 * e['flushed']:.4f}% of {e['count']}")
    if a.json:
        a.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
