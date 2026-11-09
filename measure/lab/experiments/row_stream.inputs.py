# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ROW STREAM'S INPUTS: real write lists for `row_stream.cu` (docs/internals/tools/lab.md#row-stream), the rows
they read, shared between the boxes as the carry's are, the warps' stage records, and the exact state the fold over
them must reach -- each of `DATASETS` (nl64k-alt-k4's 64 windows; flagship-alt-k16-L16384's 32 in `k16/` and
flagship-cohort-k16-L16384's in `c16/`).

CTA `b` is box `b` of one head (the carry's order, its channel tile fastest): channel tile `b % 2`, write group
`(b // 2) % 8` (its 32 lists), level-1 slice group `b // 16`; every CTA folds its dataset's windows, so a token's
slice is read by each box of its slice group and its values by each box of its channel tile, as the carry reads them.
Files (little-endian):
- `lists.bin`: int32 header (magic 0x4D525453, ctas, windows, groups); then per window, per write group: u16 `n[32]`
  (a list's entries), u16 `tok[32][512]` (an entry's token in its window), u16 `count` (the union's slots) and u16
  `union[512]` (a slot's token).
- `slices.bin`: f16 `[windows][512][8][16]`: a token's level-1 slices, 32 bytes a slice group (the carry's u8 slice
  bytes), 16 digits of 0 or 1 -- the fold reads a 64-byte slice slot whose last 32 bytes are zero-filled.
- `values.bin`: f16 `[windows][512][64]`: a token's 64 channels' values (0, 1 or 2) -- small integers, so every
  product and sum the MMAs form is exact.
- `records.bin`: int32 header (magic 0x4D525453, a record's bytes); then u16 records `[windows][groups][8 warps]`: a
  warp's stage plan for the window, as a pre-pass would emit it -- its stages `S` (in a 16-byte header), then each
  stage's 8 tokens and their 8 coefficients (f16; a pad row the stage's first, the same line, its coefficient zero),
  its lists `w + 8 j` in order, each list's entries 8 a stage.
- `state.bin`: f32 `[ctas][32 lists][32 digits][32 channels]`: the sum over the windows and each list's entries of
  slice x values, digits 16 .. 31 zero -- the state every form must reach bit for bit.
- `staged.bin`: int32 records a stream, then `staged_records`'s u32 array: the staged form's one record stream a warp.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CTAS, GROUPS, TILES, WARPS = 80, 8, 2, 8
#: an entry's coefficient (f16 1.0; a pad's 0)
ONE = 0x3C00
#: the staged records: a stage's copy carries the tokens of the stage `STAGED_AHEAD` on (the ring's depth);
#: `STAGED_TAIL` pad stages past a stream's last, which the producer copies running ahead; a pad's token `PAD`, token 0
#: (a valid address) with its top bit set, copied as zeros from nothing
STAGED_AHEAD, STAGED_TAIL, PAD = 4, 8, 0x80000000
W = 512
#: THE DATASETS: (subdirectory, cell, windows) -- the calibration cell, a cell of longer lists, and a clustered cell
DATASETS = (("", "nl64k-alt-k4", 64), ("k16", "flagship-alt-k16-L16384", 32), ("c16", "flagship-cohort-k16-L16384", 32))


def box(b: int) -> tuple[int, int, int]:
    """CTA `b`'s (channel tile, write group, slice group)."""
    return b % TILES, (b // TILES) % GROUPS, b // (TILES * GROUPS)


def stage(tok: np.ndarray, n: int, k: int) -> np.ndarray:
    """A list's stage from entry `k` (`n` entries): its 8 tokens (a pad the stage's first) and their coefficients."""
    live = min(8, n - k)
    tokens = np.concatenate([tok[k:k + live], np.full(8 - live, tok[k], np.uint16)])
    return np.concatenate([tokens, np.where(np.arange(8) < live, ONE, 0).astype(np.uint16)])


def write(directory: Path) -> list[str]:
    out = []
    for sub, cell, windows in DATASETS:
        (directory / sub).mkdir(exist_ok=True)
        out += [str(Path(sub) / f) for f in dataset(directory / sub, cell, windows)]
    return out


def dataset(directory: Path, cell: str, windows: int) -> list[str]:
    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import facts as facts_ops

    _routes, _values, facts, _carve, _unit = carry_operands(by_name(cell))
    lists = facts_ops.side_lists(facts, "write")
    entries = lists.entries.cpu().numpy()
    pre = lists.pre.long().cpu().numpy() & 0xFFFF
    union = lists.union.cpu().numpy().astype(np.int64) & 0xFFFF
    count = lists.count.cpu().numpy()
    rng = np.random.default_rng(20261017)
    slices = rng.integers(0, 2, size=(windows, W, 8, 16)).astype(np.float16)
    values = rng.integers(0, 3, size=(windows, W, 64)).astype(np.float16)
    state = np.zeros((CTAS, 32, 32, 32), dtype=np.float64)
    boxes = [box(b) for b in range(CTAS)]
    plans: dict[tuple[int, int, int], list[np.ndarray]] = {}
    steps: dict[tuple[int, int], list[list[int]]] = {(g, wp): [] for g in range(GROUPS) for wp in range(WARPS)}
    out = bytearray(struct.pack("<4i", 0x4D525453, CTAS, windows, GROUPS))
    for w in range(windows):
        for g in range(GROUPS):
            n = ((pre[0, w, g][1:33] - pre[0, w, g][:32]) & 0xFFFF)[:, -1].astype(np.int64)
            tok = np.zeros((32, W), dtype=np.uint16)
            for l in range(32):
                slots = entries[0, w, 32 * g + l, :n[l], 0].astype(np.int64) & 0xFFFF
                tok[l, :n[l]] = union[0, w, g, slots] & 0x1FF
                t = tok[l, :n[l]].astype(np.int64)
                prod = np.einsum("nsd,nc->sdc", slices[w, t].astype(np.float64), values[w, t].astype(np.float64))
                for b, (ct, gb, sg) in enumerate(boxes):
                    if gb == g:
                        state[b, l, :16] += prod[sg, :, 32 * ct:32 * ct + 32]
            for wp in range(WARPS):
                plans[w, g, wp] = [stage(tok[wp + 8 * j], int(n[wp + 8 * j]), k)
                                   for j in range(4) for k in range(0, n[wp + 8 * j], 8)]
                for j in range(4):
                    l = wp + 8 * j
                    steps[g, wp] += step_stages(512 * w + tok[l, :n[l]].astype(np.int64))
            un = np.zeros(W, dtype=np.uint16)
            u = int(count[0, w, g])
            un[:u] = union[0, w, g, :u] & 0x1FF
            out += n.astype(np.uint16).tobytes() + tok.tobytes() + struct.pack("<H", u) + un.tobytes()
    (directory / "lists.bin").write_bytes(bytes(out))
    most = max(len(x) for x in plans.values())
    rec = np.zeros((windows, GROUPS, WARPS, 8 + 16 * most), dtype=np.uint16)
    for (w, g, wp), stages in plans.items():
        rec[w, g, wp, 0] = len(stages)
        if stages:
            rec[w, g, wp, 8:8 + 16 * len(stages)] = np.concatenate(stages)
    (directory / "records.bin").write_bytes(struct.pack("<2i", 0x4D525453, rec.shape[-1] * 2) + rec.tobytes())
    (directory / "slices.bin").write_bytes(slices.tobytes())
    (directory / "values.bin").write_bytes(values.tobytes())
    assert state.max() < 2 ** 24, "the state must stay exact in fp32"
    (directory / "state.bin").write_bytes(state.astype(np.float32).tobytes())
    flat = staged_records(steps)
    (directory / "staged.bin").write_bytes(struct.pack("<i", flat.shape[2]) + flat.tobytes())
    return ["lists.bin", "slices.bin", "values.bin", "records.bin", "state.bin", "staged.bin"]


def step_stages(tokens: np.ndarray) -> list[list[int]]:
    """A digit's steps (the fold's: 16 tokens while more than 8 remain, then 8), a stage each: its 16 tokens (global,
    `512 w + t`), a pad past the digit's entries and in a tail's last 8."""
    out, n, k = [], len(tokens), 0
    while n - k > 8:
        out.append([int(t) for t in tokens[k:k + 16]] + [-1] * (16 - len(tokens[k:k + 16])))
        k += 16
    if n - k > 0:
        out.append([int(t) for t in tokens[k:k + 8]] + [-1] * (16 - len(tokens[k:k + 8])))
    return out


def staged_records(steps: dict) -> np.ndarray:
    """THE STAGED RECORDS: u32 `[groups][8 warps][STAGED_AHEAD + stages + STAGED_TAIL][24]`, one stream a warp across
    its windows and digits in the consumer's order. Record `r` is 96 bytes: stage `r - STAGED_AHEAD`'s 16 coefficients
    (f16, lane `t`'s c(2t), c(2t + 1), c(2t + 8), c(2t + 9) at 8 t; 1 an entry, 0 a pad), then stage `r`'s 16 tokens
    as pairs (q, q + 8) at 8 q."""
    stages = max(len(x) for x in steps.values())
    flat = np.zeros((GROUPS, WARPS, STAGED_AHEAD + stages + STAGED_TAIL, 24), np.uint32)
    flat[..., 8:] = PAD
    for (g, wp), stream in steps.items():
        for k, toks in enumerate(stream):
            c = [ONE if t >= 0 else 0 for t in toks]
            halves = [c[2 * t + i] for t in range(4) for i in (0, 1, 8, 9)]
            flat[g, wp, k + STAGED_AHEAD, :8] = [halves[2 * i] | halves[2 * i + 1] << 16 for i in range(8)]
            flat[g, wp, k, 8:] = [PAD if t < 0 else t for q in range(8) for t in (toks[q], toks[q + 8])]
    return flat


if __name__ == "__main__":
    write(Path(sys.argv[1]))
