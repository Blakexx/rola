# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GROUPED READOUT'S INPUTS: `read_stream.inputs.py`'s three cells (the same coloring, rows and panels) with each
warp's 4 read digits in groups of M (1: each digit; 2: digits w, w + 8 and w + 16, w + 24; 4: all four), a group's
tiles the 8-token tiles of its digits' union in a pass -- each union token's coefficient for each of the group's digits,
zero where the digit does not read it -- for `read_group.cu` (docs/lab/read_group.md).

A group-tile's record, the staged stream's (`read_stream.inputs.py`'s `staged_records`) with its entries widened:
- EXIT (64 bytes): position e's (scale, panel field) at `8 e`, the lane with t = e // 2 reading its two at `16 t` --
  OUTPUT scaling: the scale the group's first digit's coefficient; OPERAND scaling: 1.0.
- COEFFICIENTS: OUTPUT, `8 (M - 1)` bytes a t: the coefficients of tokens 2t, 2t + 1 for digits 1 .. M - 1, fp32;
  OPERAND, `4 M` bytes a position g: each digit's coefficient as bf16x2 (both halves), the B fragment's column g.
- TOKENS (32 bytes): the 8 tokens of the stage `STAGED_AHEAD` on, a pad `PAD`.
THE UNIFIED FORM (ruled 2026-10-18: one form for every M, output scaling) has its own record: each position's panel
field and its token's unit (`field << 16 | unit`, 32 bytes), then each t's two tokens' M coefficients as f16 pairs
(`4 M` bytes a t: digit d's word `c(2t) | c(2t + 1) << 16`), then the tokens. A coefficient is stored as its target
(256 or 512) times `2^(unit - sx)`, the token's unit `(slot >> 1) % 3` and the digit's exponent `sx = (digit >> 3) & 1`,
so the kernel's factor `c x 2^sx x 2^-unit` is the target exactly.
Files (little-endian), each dataset's directory as `read_stream.inputs.py`'s:
- `head.bin`: int32 (magic 0x50524752, ctas, windows).
- `passes.bin`: u16 `[windows][8 read groups]`.
- `tiles_m{M}.bin`: u16 `[windows][2 passes][8 read groups][8 warps][4]`: a group's tiles in a pass (its first `4 / M`).
- `staged_m{M}_{mode}.bin`: int32 records a stream, then u32 `[8 read groups][8 warps][records][words]`.
- `rows.bin`, `panel.bin`: `read_stream.inputs.py`'s.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stream = _load("read_stream.inputs")
base = _load("read_image.inputs")

CTAS, SLOTS, PASSES, LISTS = stream.CTAS, stream.SLOTS, stream.PASSES, stream.LISTS
STAGED_AHEAD, STAGED_TAIL, PAD = stream.STAGED_AHEAD, stream.STAGED_TAIL, stream.PAD
#: the group sizes and, for M > 1, the two scalings (M = 1 is the readout as `read_stream` runs it: "output")
FORMS = ((1, "output"), (2, "output"), (2, "operand"), (4, "output"), (4, "operand"), (1, "unified"), (2, "unified"),
         (4, "unified"))
ONE = 0x3F800000


def coefficient(s: int, d: int) -> float:
    """A live entry's coefficient (`read_image.inputs.py`'s bf16 bits): 256 or 512 by `(slot ^ digit) & 1`."""
    return float(np.array([base.COEFFICIENT[(s ^ d) & 1] << 16], np.uint32).view(np.float32)[0])


def f32(x: float) -> int:
    return int(np.array([x], np.float32).view(np.uint32)[0])


def bf16x2(x: float) -> int:
    h = f32(x) >> 16
    return h << 16 | h


def group_tiles(spans: list[list[int]], digits: list[int], p: int) -> list[tuple[int, int, list[float]]]:
    """A group's tiles in pass `p`: its digits' union placed as a digit's are (a pad tile where it has none); each
    position (token or -1, panel field, each digit's coefficient)."""
    mine = {d: {s for s in spans[d] if s // SLOTS == p} for d in digits}
    union = sorted(set().union(*mine.values()))
    placed = base.place(union) or [(r, True) for r in range(8)]
    panel_rows = base.DUMMY_ROWS + 8 * base.PANEL_PITCH
    out = []
    for e, (s, pad) in enumerate(placed):
        field = (base.DUMMY_ROWS if pad else panel_rows) + base.PANEL_PITCH * (s % SLOTS) - base.FOLD["n"](e)
        coefs = [0.0 if pad or s not in mine[d] else coefficient(s, d) for d in digits]
        out.append((-1, field, coefs) if pad else (s, field, coefs))
    return out


def unit_of(s: int) -> int:
    """A live token's unit in the unified form (module docstring)."""
    return (s >> 1) % 3


def f16(x: float) -> int:
    return int(np.array([x], np.float16).view(np.uint16)[0])


def entries(tile: list[tuple[int, int, list[float]]], m: int, mode: str, digits: list[int]) -> list[int]:
    """A group-tile's record entries as u32 words: the exit part, then the coefficients (module docstring)."""
    if mode == "unified":
        units = [0 if s < 0 else unit_of(s) for s, _, _ in tile]
        out = [field << 16 | u for (_, field, _), u in zip(tile, units, strict=True)]
        for t in range(4):
            for k, d in enumerate(digits):
                c = [tile[e][2][k] * 2.0 ** (units[e] - ((d >> 3) & 1)) for e in (2 * t, 2 * t + 1)]
                out.append(f16(c[0]) | f16(c[1]) << 16)
        return out
    out = []
    for _, field, coefs in tile:
        out += [f32(coefs[0]) if mode == "output" else ONE, field]
    if mode == "output":
        for t in range(4):
            for d in range(1, m):
                out += [f32(tile[2 * t][2][d]), f32(tile[2 * t + 1][2][d])]
    else:
        for g in range(8):
            out += [bf16x2(c) for c in tile[g][2]]
    return out


def write(directory: Path) -> list[str]:
    out = []
    for sub, cell, windows in stream.DATASETS:
        (directory / sub).mkdir(exist_ok=True)
        out += [str(Path(sub) / f) for f in dataset(directory / sub, cell, windows)]
    return out


def dataset(directory: Path, cell: str, windows: int) -> list[str]:
    slots, at, count, union = base.lists(windows, with_tokens=True, cell=cell)
    width = slots.shape[1]
    groups = width // 32
    rng = np.random.default_rng(20261018)
    high = rng.integers(0, 2, size=(windows * 512, stream.SLICES))
    rows = np.where(high[..., None] == 1, stream.VALUE[1], stream.VALUE[0]).astype(np.uint16).repeat(LISTS, axis=-1)

    passes = np.zeros((windows, groups), np.uint16)
    tiles = {m: np.zeros((windows, PASSES, groups, 8, 4), np.uint16) for m in (1, 2, 4)}
    streams = {(m, mode): {} for m, mode in FORMS}
    acc = np.zeros((CTAS, PASSES, SLOTS), np.int64)
    for w in range(windows):
        for g in range(groups):
            spans = [[int(x) for x in slots[w, 32 * g + d, at[w, 32 * g + d, 0]:at[w, 32 * g + d, -1]]]
                     for d in range(32)]
            new, extent = base.color(spans, int(count[w, g]))
            spans = [[new[s] for s in span] for span in spans]
            token_of = {s_new: int(union[w, g, s_old]) for s_old, s_new in new.items()}
            npass = -(-extent // SLOTS)
            passes[w, g] = npass
            for warp in range(8):
                for m, mode in FORMS:
                    sets = [[warp + 8 * i for i in range(m * k, m * k + m)] for k in range(4 // m)]
                    plan = streams[m, mode].setdefault((g, warp), [])
                    for p in range(npass):
                        by_group = [group_tiles(spans, ds, p) for ds in sets]
                        if mode == "output":
                            for k, t in enumerate(by_group):
                                tiles[m][w, p, g, warp, k] = len(t) // 8
                        ordered = [(t[:8], ds) for t, ds in zip(by_group, sets, strict=True)]
                        ordered += [(t[k:k + 8], ds) for t, ds in zip(by_group, sets, strict=True)
                                    for k in range(8, len(t), 8)]
                        for tile, ds in ordered:
                            plan.append(([PAD if s < 0 else 512 * w + token_of[s] for s, _, _ in tile],
                                         entries(tile, m, mode, ds)))
            live = [(s // SLOTS, s % SLOTS, 1 + ((s ^ d) & 1), 512 * w + token_of[s]) for d in range(32)
                    for s in spans[d]]
            if live:
                pa, sl, mult, tok = (np.array(x) for x in zip(*live, strict=True))
                for b in range(CTAS):
                    sg, rg = stream.box(b)
                    if rg == g:
                        np.add.at(acc[b], (pa, sl), mult * (1 + high[tok, sg]))

    one = np.array([stream.one_add(k) for k in range(stream.WORDS)], np.uint64)
    panel = ((acc.astype(np.uint64)[..., None] * one) & 0xFFFFFFFF).astype(np.uint32)

    (directory / "head.bin").write_bytes(struct.pack("<3i", 0x50524752, CTAS, windows))
    (directory / "passes.bin").write_bytes(passes.tobytes())
    (directory / "rows.bin").write_bytes(rows.tobytes())
    (directory / "panel.bin").write_bytes(panel.tobytes())
    files = ["head.bin", "passes.bin", "rows.bin", "panel.bin"]
    for m in (1, 2, 4):
        (directory / f"tiles_m{m}.bin").write_bytes(tiles[m].tobytes())
        files.append(f"tiles_m{m}.bin")
    for (m, mode), plans in streams.items():
        flat = records(plans, groups, m, mode)
        name = f"staged_m{m}_{mode}.bin"
        (directory / name).write_bytes(struct.pack("<i", flat.shape[2]) + flat.tobytes())
        files.append(name)

    per = {m: tiles[m].sum() / (windows * groups) for m in (1, 2, 4)}
    print(f"read_group {cell}: group-tiles a box-window M 1 {per[1]:.1f}, M 2 {per[2]:.1f}, M 4 {per[4]:.1f}; "
          f"MMAs a box-window {4 * per[1]:.0f}, {8 * per[2]:.0f}, {16 * per[4]:.0f}")
    return files


def records(plans: dict, groups: int, m: int, mode: str) -> np.ndarray:
    """THE STAGED RECORDS: u32 `[groups][8 warps][STAGED_AHEAD + stages + STAGED_TAIL][words]`, one stream a warp in the
    consumer's order; record `r` stage `r - STAGED_AHEAD`'s entries then stage `r`'s 8 tokens (past the last, pads)."""
    width = {"output": 16 + 8 * (m - 1), "operand": 16 + 8 * m, "unified": 8 + 4 * m}[mode]
    stages = max(len(x) for x in plans.values())
    flat = np.zeros((groups, 8, STAGED_AHEAD + stages + STAGED_TAIL, width + 8), np.uint32)
    flat[..., width:] = PAD
    for (g, warp), plan in plans.items():
        for k, (tokens, words) in enumerate(plan):
            flat[g, warp, k + STAGED_AHEAD, :width] = words
            flat[g, warp, k, width:] = tokens
    return flat


if __name__ == "__main__":
    write(Path(sys.argv[1]))
