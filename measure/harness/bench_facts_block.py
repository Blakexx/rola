# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS ON THE LAB'S `facts_block` CELL, against its line (docs/internals/facts/facts.md#budget): rows of four
level-sides `[65,536 tokens, 4 heads, 256 digits]` u8 (268 MB) -- each side's plane head-major, its two levels' slots
side by side, as the producer writes them -- each token-head live in `k` random digits of each level-side; the write
side lists level 0 and the read side level 1, the other level of each voted (the lab cell's two listed and two voted
level-sides). One launch of the production kernel in the sm_86 even arm's form (the read side's list group 32 or 16,
docs/internals/facts/facts.md#list-groups), timed by CUDA events; the required bytes (the rows, every output the pass
writes, the columns it reads) priced at the lab's recorded flat read.

    python measure/harness/bench_facts_block.py [--k 4 16] [--reps 20] [--read-list-group 32 16] [--ledger]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

TOKENS, HEADS, WIDTH = 65536, 4, 256
#: the GPU spin a timed launch is queued behind, in cycles (about a millisecond): the stream stays busy while the host
#: allocates the launch's outputs, so the events bracket the kernel alone
SPIN_CYCLES = 2_000_000
#: the line (ruled 2026-10-13): 1.15x the pass's required-bytes floor at the colored lists' places, by k
LINE_US = {4: 516.0, 16: 665.0}
#: the lab's recorded flat read of the cell's 268 MB of rows (measure/lab/results/facts_two/sm_86.json)
FLAT_RECORD = ROOT / "measure/lab/results/facts_two/sm_86.json"


def flat_rate() -> float:
    """Bytes a microsecond, the lab's flat read of every row."""
    rows = json.loads(FLAT_RECORD.read_text())["results"]
    us = next(r["median"] for r in rows if r["name"] == "flat read of all the rows")
    return 4 * TOKENS * HEADS * WIDTH / us


#: the cell's draw
SEED = 0


def cell(k: int):
    """``(read, write, scales, vs)``: each side's two levels as ``[1, T, H, 256]`` views of its head-major plane,
    each token-head live in ``k`` random digits of each (u8 1..255); the write levels' scales, a drawn write gain folded
    into level 0's as a producer folds its own (docs/CONSTITUTION.md#r23), and the value scale ``[1, T, H]`` fp32."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    sides = []
    for _ in range(2):
        plane = torch.zeros((1, HEADS, TOKENS, 2 * WIDTH), dtype=torch.uint8, device="cuda")
        for level in range(2):
            idx = torch.randint(0, WIDTH, (1, HEADS, TOKENS, k), device="cuda", generator=gen) + level * WIDTH
            val = torch.randint(1, 256, (1, HEADS, TOKENS, k), device="cuda", generator=gen).to(torch.uint8)
            plane.scatter_(-1, idx, val)
        sides.append([plane[..., level * WIDTH:(level + 1) * WIDTH].transpose(1, 2) for level in range(2)])

    def column():
        return (torch.rand((1, HEADS, TOKENS), device="cuda", generator=gen) + 0.1).transpose(1, 2)

    scales = [column(), column()]
    scales[0] = scales[0] * column()
    return sides[0], sides[1], scales, column()


def launch(read, write, scales, vs, form):
    """One launch in the lists' ``form``; its outputs, each side's lists a ``Lists``."""
    from rola.engine.facts import liveness as lv
    from rola.engine.facts.lists import Lists
    from rola.ops import facts as facts_ops

    masks = [0xFFFFFFFF] * lv.mask_words(2 * WIDTH)
    outs = facts_ops.extension().rola_facts_forward(
        read, write, scales, vs, [WIDTH, WIDTH], 0, masks, 0, masks, 0, 1, *form)
    return (*outs[:6], Lists(*outs[6:11]), Lists(*outs[11:]))


def required_bytes(outs) -> float:
    """The rows, the columns read, and every byte the pass writes: the words and tile unions, each list's entries,
    boundaries, totals, union and extent, the exponents, the factor products and gains."""
    words, unions, list_exp, fm, gains16, gains_exp, wl, rl = outs
    rows = 4 * TOKENS * HEADS * WIDTH
    columns = 4 * TOKENS * HEADS * 4
    entries = sum(int(l.totals[..., -1].long().sum()) for l in (wl, rl)) * 8
    lists = sum(l.pre.numel() * 2 + l.totals.numel() * 4 + l.union.numel() * 2 + l.count.numel() * 4 for l in (wl, rl))
    rest = (words.numel() + unions.numel()) * 4 + list_exp.numel() * 2 + fm.numel() * 4 + gains16.numel() * 2
    return rows + columns + entries + lists + rest + gains_exp.numel() * 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--read-list-group", type=int, nargs="+", choices=(16, 32), default=[32],
                    help="the read side's list groups (the write side's 32), timed interleaved launch by launch")
    ap.add_argument("--ledger", action="store_true", help="also book the phase ledger (a separate, untimed run)")
    args = ap.parse_args()
    from rola_devtools.locks.gpu import gpu_lock

    from rola.ops import facts as facts_ops
    forms = [facts_ops.ListsForm(read_row=64, read_list_group=group, write_list_group=32)
             for group in args.read_list_group]
    rate = flat_rate()
    with gpu_lock():
        for k in args.k:
            read, write, scales, vs = cell(k)
            for _ in range(3):
                outs = [launch(read, write, scales, vs, form) for form in forms]
            torch.cuda.synchronize()
            times = [[] for _ in forms]
            for _ in range(args.reps):
                for form, booked in zip(forms, times):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(SPIN_CYCLES)
                    a.record()
                    launch(read, write, scales, vs, form)
                    b.record()
                    b.synchronize()
                    booked.append(1000.0 * a.elapsed_time(b))
            first = statistics.median(times[0])
            for form, out, booked in zip(forms, outs, times):
                us = statistics.median(booked)
                floor = required_bytes(out) / rate
                line = LINE_US.get(k)
                ratio = statistics.median(x / y for x, y in zip(booked, times[0]))
                print(f"k {k} read list group {form.read_list_group}: {us:.1f} us (min {min(booked):.1f}), "
                      f"required-bytes floor {floor:.1f} us at {rate / 1000:.0f} GB/s: {us / floor:.3f}x"
                      + (f"; the line {line:.0f} us" if line else "")
                      + (f"; {ratio:.3f}x the first group's {first:.1f} us, launch by launch" if booked is not times[0]
                         else ""))
            if args.ledger:
                for form in forms:
                    ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64,
                                         device="cuda")
                    facts_ops.ledger_bind(ledger)
                    try:
                        launch(read, write, scales, vs, form)
                        torch.cuda.synchronize()
                    finally:
                        facts_ops.ledger_bind(None)
                    total = ledger.sum(0).double()
                    print(f"  read list group {form.read_list_group}, phase share (cycles summed over warps): "
                          + ", ".join(f"{name} {v / total.sum():.2f}"
                                      for name, v in zip(facts_ops.LEDGER_PHASES, total.tolist())))
            del read, write, outs
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
