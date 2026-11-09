# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COLORING LAB'S INPUTS for `color_union.cu` (docs/internals/tools/lab.md#color-union): each box-window's read
digits as the facts pass holds them -- a bit TILE, 32 digit rows over the window's 512 tokens, 16 words a row -- and
the slot each union token gets from the coloring, by `color`, the exact mirror of the kernel.

THE COLORING (KERNEL_STANDARDS section 30): a slot's residue mod 8 is its pool row's bank group, so each digit's
tokens should spread over the residues. Tokens read by two or more of the box's digits go first, in token order, each to
the residue its digits use least (summed over them), then the least loaded, then the lowest. Each digit's single
tokens then take the residues its digit uses least (water-filling), the digits in order, a level's ties going to the
least loaded residues. A token's slot is its residue plus 8 x its rank among the residue's tokens; the extent is 8 x
the largest residue's count.

Three cells, 32 windows each, eight 32-digit groups a window: `alt-k4/` nl64k-alt-k4 (~40 multi-digit tokens a
box-window), `alt-k16/` flagship-alt-k16-L16384 (~317), `thin/` nl64k-alt-k4 with each (digit, token) kept with
probability 1/16 (~0.1). Each holds `tiles.bin` (a header of four int32 -- 0x524F4C43, box-windows, windows, groups --
then `[box-windows][32][16]` u32), `expected.bin` (`[box-windows][512]` u16 slots, 0xFFFF off the union, then
`[box-windows]` u16 extents), `banks.json` (the readout's wavefronts an MMA under this coloring, the bank model
through `read_image.inputs.emit`) and `banks_batched.json` (the same under `batched`, claims in lane order)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = 32
CELLS = (("alt-k4", "nl64k-alt-k4", 1.0), ("alt-k16", "flagship-alt-k16-L16384", 1.0), ("thin", "nl64k-alt-k4", 1 / 16))
SEED = 16


def tiles(cell: str, keep: float):
    """`[windows, groups, 32, 16]` u32: each 32-digit group's read rows over each window's tokens, the facts pass's
    words at the read list level, each (digit, token) kept with probability `keep`."""
    import numpy as np

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, lists_form_of
    from rola.engine.facts import liveness as lv
    from rola.ops import facts as facts_ops

    spec = by_name(cell)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, 512 * WINDOWS))
    words = facts_ops.facts(routes, values, layout, *statics, lists_form_of(spec)).words
    level = facts_ops.list_level(statics[0])
    base, width = layout.row_base(level), layout.B[level]
    bits = lv.unpack(words, layout)[0, lv.READ, base:base + width].cpu().numpy()
    windows = layout.L // 512
    out = np.zeros((windows, width // 32, 32, 16), np.uint32)
    rng = np.random.default_rng(SEED)
    for w in range(windows):
        for d in range(width):
            for t in np.flatnonzero(bits[d, 512 * w:512 * w + 512]):
                if keep >= 1.0 or rng.random() < keep:
                    out[w, d // 32, d % 32, t >> 5] |= np.uint32(1 << (t & 31))
    return out


def plan(c: list[int], k: int) -> tuple[list[int], int, int]:
    """A digit's water-fill of `k` single tokens over its counts `c`: the level T the residues below it are filled to
    (`forced`), the singles left for T's residues (`rem`) and those residues (`active`, a bit each)."""
    level, placed = min(c), 0
    while True:
        a = sum(x <= level for x in c)
        above = [x for x in c if x > level]
        if above and placed + a * (min(above) - level) <= k:
            placed += a * (min(above) - level)
            level = min(above)
            continue
        q = (k - placed) // a
        level, placed = level + q, placed + q * a
        break
    return [max(0, level - x) for x in c], k - placed, sum(1 << r for r in range(8) if c[r] <= level)


def color(rows: list[int]) -> tuple[list[int], int, int]:
    """A box-window's slots (0xFFFF off the union), extent and multi-digit tokens from its 32 digit rows (512-bit
    ints), as the kernel computes them."""
    union = 0
    for x in rows:
        union |= x
    member = {t: [d for d in range(32) if rows[d] >> t & 1] for t in range(512) if union >> t & 1}
    cnt = [[0] * 8 for _ in range(32)]
    load = [0] * 8
    res = {}
    for t in sorted(t for t in member if len(member[t]) > 1):
        f = [sum(cnt[d][r] for d in member[t]) for r in range(8)]
        r = min(range(8), key=lambda r: (f[r], load[r], r))
        res[t] = r
        load[r] += 1
        for d in member[t]:
            cnt[d][r] += 1
    singles = [[t for t in range(512) if rows[d] >> t & 1 and len(member[t]) == 1] for d in range(32)]
    plans = [plan(cnt[d], len(singles[d])) for d in range(32)]
    take = []
    for forced, rem, active in plans:
        key = {r: (load[r], r) for r in range(8) if active >> r & 1}
        n = [forced[r] + (r in key and sum(key[q] < key[r] for q in key) < rem) for r in range(8)]
        load = [a + b for a, b in zip(load, n, strict=True)]
        take.append(n)
    for d in range(32):
        bounds = [sum(take[d][:r]) for r in range(1, 8)]
        for j, t in enumerate(singles[d]):
            res[t] = sum(b <= j for b in bounds)
    base = [0] * 8
    slots = [0xFFFF] * 512
    for t in sorted(res):
        slots[t] = res[t] + 8 * base[res[t]]
        base[res[t]] += 1
    assert base == load
    return slots, 8 * max(base), sum(len(m) > 1 for m in member.values())


def batched(rows: list[int]) -> tuple[dict[int, int], int]:
    """THE BATCHED FORM's colors (residue a token) and blocked tokens, claims resolved in lane order: the multi-digit
    tokens queued (three or more digits first, then two, each in token order) and taken 32 at a time, each picking the
    color least used over its digits among those under every digit's capacity (ceil(n / 8)), ties by (c - t) & 7; a
    claim over a capacity is withdrawn and requeued; then each digit's singles fill its room first-fit from an even
    start, and the digits whose odd start moves the parity count the needed way flip, in order, until half the
    imbalance."""
    union = 0
    for x in rows:
        union |= x
    member = {t: [d for d in range(32) if rows[d] >> t & 1] for t in range(512) if union >> t & 1}
    cap = [-(-bin(x).count("1") // 8) for x in rows]
    cnt = [[0] * 8 for _ in range(32)]
    queue = [t for t in sorted(member) if len(member[t]) > 2] + [t for t in sorted(member) if len(member[t]) == 2]
    col, blocked, solo = {}, 0, False

    while queue:
        batch, queue = queue[:1 if solo else 32], queue[1 if solo else 32:]
        claims, lost = {}, []
        for t in batch:
            full = 0
            for d in member[t]:
                full |= sum(1 << c for c in range(8) if cnt[d][c] >= cap[d])
            allowed = ~full & 0xFF
            pool = allowed or 0xFF
            c = min((sum(cnt[d][c] for d in member[t]), (c - t) & 7, c) for c in range(8) if pool >> c & 1)[2]
            over = any(cnt[d][c] + claims.get((d, c), 0) >= cap[d] for d in member[t]) and allowed != 0
            for d in member[t]:
                claims[(d, c)] = claims.get((d, c), 0) + 1
            if over:
                lost.append(t)
            else:
                col[t] = c
                blocked += allowed == 0
        for t in batch:
            if t in col:
                for d in member[t]:
                    cnt[d][col[t]] += 1
        solo = len(lost) == len(batch)
        queue += lost

    even = sum(1 for c in col.values() if c & 1 == 0)
    singles = [[t for t in range(512) if rows[d] >> t & 1 and len(member[t]) == 1] for d in range(32)]
    fills = []
    for d in range(32):
        room = [max(0, cap[d] - cnt[d][c]) for c in range(8)]
        fill = []
        for start in (0, 1):
            n, before = [0] * 8, 0
            for i in range(8):
                c = (start + i) & 7
                n[c] = min(room[c], max(0, len(singles[d]) - before))
                before += room[c]
            fill.append(n)
        fills.append(fill)

    total = len(col) + sum(map(len, singles))
    need = even + sum(sum(f[0][0::2]) for f in fills) - (total + 1) // 2
    upto = 0
    for d in range(32):
        delta = sum(fills[d][1][0::2]) - sum(fills[d][0][0::2])
        gain = max(0, -delta) if need > 0 else max(0, delta)
        upto += gain
        n = fills[d][1] if gain > 0 and upto <= abs(need) else fills[d][0]
        dealt = [c for c in range(8) for _ in range(n[c])]
        for t, c in zip(singles[d], dealt, strict=True):
            col[t] = c
    return col, blocked


def _banks(t, slots) -> dict:
    """The readout's wavefronts an MMA under this coloring: each digit's colored slots through
    `read_image.inputs.emit` (its own coloring off), the pass sized to hold the largest extent."""
    import tempfile

    import numpy as np

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    windows, groups = t.shape[:2]
    lists = np.zeros((windows, groups * 32, 512), np.int64)
    at = np.zeros((windows, groups * 32, 8), np.int64)
    extent = np.zeros((windows, groups), np.int64)
    for w in range(windows):
        for g in range(groups):
            s = slots[w * groups + g]
            extent[w, g] = max((x for x in s if x != 0xFFFF), default=-1) + 1
            for d in range(32):
                row = sorted(s[32 * j + b] for j in range(16) for b in range(32) if int(t[w, g, d, j]) >> b & 1)
                lists[w, g * 32 + d, :len(row)] = row
                at[w, g * 32 + d, 1:] = len(row)
    base.color = lambda digits, count: ({x: x for d in digits for x in d}, count)
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        base.emit(Path(d), lists, at, extent, max(512, -(-int(extent.max()) // 8) * 8), 16, 16, rounds=True)
        return json.loads((Path(d) / "banks.json").read_text())


def write(directory: Path) -> list[str]:
    import numpy as np
    import torch

    files = []
    for name, cell, keep in CELLS:
        (directory / name).mkdir(parents=True, exist_ok=True)
        t = tiles(cell, keep)
        windows, groups = t.shape[:2]
        flat = t.reshape(-1, 32, 16)
        out = [color([sum(int(r[j]) << (32 * j) for j in range(16)) for r in box]) for box in flat]
        header = struct.pack("<4i", 0x524F4C43, len(flat), windows, groups)
        (directory / name / "tiles.bin").write_bytes(header + flat.astype(np.uint32).tobytes())
        body = np.array([o[0] for o in out], np.uint16).tobytes() + np.array([o[1] for o in out], np.uint16).tobytes()
        (directory / name / "expected.bin").write_bytes(body)
        banks = _banks(t, [o[0] for o in out])
        union = [sum(s != 0xFFFF for s in o[0]) for o in out]
        over = [o[1] - 8 * -(-u // 8) for o, u in zip(out, union, strict=True)]
        banks["cell"] = {"union a box-window": float(np.mean(union)),
                         "multi-digit tokens": float(np.mean([o[2] for o in out])),
                         "extent over 8 ceil(union / 8)": float(np.mean(over)), "extent over, max": int(max(over))}
        (directory / name / "banks.json").write_text(json.dumps(banks, indent=1) + "\n")
        print(name, "coloring:", json.dumps(banks["n"]), json.dumps(banks["cell"]))
        mirrored = [batched([sum(int(r[j]) << (32 * j) for j in range(16)) for r in box]) for box in flat]
        slots = []
        for colors, _ in mirrored:
            rank = [0] * 8
            row = [0xFFFF] * 512
            for tok in sorted(colors):
                row[tok] = colors[tok] + 8 * rank[colors[tok]]
                rank[colors[tok]] += 1
            slots.append(row)
        model = _banks(t, slots)
        model["cell"] = {"blocked a box-window": float(np.mean([b for _, b in mirrored]))}
        (directory / name / "banks_batched.json").write_text(json.dumps(model, indent=1) + "\n")
        print(name, "batched:", json.dumps(model["n"]), json.dumps(model["cell"]))
        files += [f"{name}/tiles.bin", f"{name}/expected.bin", f"{name}/banks.json", f"{name}/banks_batched.json"]
    torch.cuda.empty_cache()
    return files
