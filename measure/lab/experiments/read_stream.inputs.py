# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READ STREAM'S INPUTS: read lists for `read_stream.cu` (docs/internals/tools/lab.md#read-stream) in the readout's
tokens-on-N form (`read_image.inputs.py`'s coloring and placement), each warp's stage records, the read rows shared
between the boxes as the carry's are, and each CTA's exact panel -- for each of `DATASETS` (nl64k-alt-k4's 64 windows;
flagship-alt-k16-L16384's 32 in `k16/` and flagship-cohort-k16-L16384's in `c16/`).

CTA `b` is box `b` of one head (the carry's order, its channel tile fastest): channel tile `b % 2`, level-0 slice group
`(b // 2) % 8` (its read rows' 32 lists), read group `b // 16` (its 32 read digits); every CTA reads its dataset's
windows, so a token's read row is fetched by each box of its slice group. A box-window's colored slots run in passes of
256 (a panel's rows), each digit's entries placed within its pass. Files (little-endian):
- `head.bin`: int32 (magic 0x4D525453, ctas, windows, a token record's bytes, a word record's bytes).
- `passes.bin`: u16 `[windows][8 read groups]`: a box-window's passes.
- `tiles.bin`: u16 `[windows][2 passes][256]`: a read digit's 8-token tiles in a pass, at least one where the pass runs
  (an empty digit's all pads, as the readout steps every digit's first tile).
- `records.bin`: u16 `[windows][8 read groups][8 warps]` token records: a warp's stages for the window in a 16-byte
  header, then each stage's 8 tokens -- the stages in the readout's order, each pass's every digit's first tile, then
  each digit's rest; a pad the tile's first live token (or 0 in an all-pad tile: the same line, its coefficient zero).
- `words.bin`: u32 `[windows][8 read groups][8 warps]` word records: each stage's 8 entries lane-ready, 8 bytes each --
  the coefficient times the unit as fp32 and the panel row's offset past the lane's base (`read_image.inputs.py`'s
  entry word, split) -- in the same order.
- `rows.bin`: bf16 `[windows x 512][8 slice groups][32 lists]`: a token's read row, its 32 lists one value, 0.0625 or
  0.125 -- so every product the MMAs form is exact and a misplaced row changes the sum.
- `panel.bin`: u32 `[ctas][2 passes][256 slots][16 words]`: each CTA's drained panel summed over the windows.
- `staged.bin`: int32 records a stream, then `staged_records`'s u32 array: the staged form's one record stream a warp.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

CTAS, TILES, SLICES, LISTS, PASSES = 80, 2, 8, 32, 2
#: the staged records: a stage's copy carries the tokens `STAGED_AHEAD` stages on; `STAGED_TAIL` pad stages past a
#: stream's last, which the producer copies running ahead
STAGED_AHEAD, STAGED_TAIL = 8, 16
#: a pad's token in the staged records: token 0 (a valid address) with its top bit set, copied as zeros from nothing
PAD = 0x80000000
#: the panel's slots and words, and the bf16 bits of a read row's two values
SLOTS, WORDS, VALUE = 256, 16, (0x3D80, 0x3E00)
#: THE DATASETS: (subdirectory, cell, windows) -- the calibration cell, a cell of longer lists, and a clustered cell
DATASETS = (("", "nl64k-alt-k4", 64), ("k16", "flagship-alt-k16-L16384", 32), ("c16", "flagship-cohort-k16-L16384", 32))


def box(b: int) -> tuple[int, int]:
    """CTA `b`'s (slice group, read group)."""
    return (b // TILES) % SLICES, b // (TILES * SLICES)


def one_add(k: int) -> int:
    """A panel word's one add at the unit (`read_image.cuh`'s, tokens on N): channels lo = 16 (k / 8) + k % 8 and
    hi = lo + 8, each -/+ the lists' count, packed hi x 2^16 + lo mod 2^32."""
    lo = 16 * (k >> 3) + (k & 7)
    v = lambda ch: LISTS if ch & 1 else -LISTS  # noqa: E731
    return (v(lo + 8) * 65536 + v(lo)) & 0xFFFFFFFF


def write(directory: Path) -> list[str]:
    out = []
    for sub, cell, windows in DATASETS:
        (directory / sub).mkdir(exist_ok=True)
        out += [str(Path(sub) / f) for f in dataset(directory / sub, cell, windows)]
    return out


def placed_tiles(base, spans: list[list[int]], d: int, p: int, token_of: dict[int, int]) -> list[tuple[int, int]]:
    """Digit `d`'s entries in pass `p` placed (a pad tile where it has none): (token, word) each, a pad's token -1."""
    mine = [s for s in spans[d] if s // SLOTS == p]
    placed = base.place(mine) or [(res, True) for res in range(8)]
    panel_rows = base.DUMMY_ROWS + 8 * base.PANEL_PITCH
    entries = []
    for e, (s, pad) in enumerate(placed):
        field = (base.DUMMY_ROWS if pad else panel_rows) + base.PANEL_PITCH * (s % SLOTS) - base.FOLD["n"](e)
        entries.append((-1 if pad else token_of[s], field << 16 | (0 if pad else base.COEFFICIENT[(s ^ d) & 1])))

    return entries


def dataset(directory: Path, cell: str, windows: int) -> list[str]:
    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count, union = base.lists(windows, with_tokens=True, cell=cell)
    width = slots.shape[1]
    groups = width // 32
    assert (count <= PASSES * SLOTS).all(), "a box-window's union needs more passes than the lab's panel holds"
    rng = np.random.default_rng(20261018)
    high = rng.integers(0, 2, size=(windows * 512, SLICES))
    rows = np.where(high[..., None] == 1, VALUE[1], VALUE[0]).astype(np.uint16).repeat(LISTS, axis=-1)

    tiles = np.zeros((windows, PASSES, width), np.uint16)
    passes = np.zeros((windows, groups), np.uint16)
    plans: dict[tuple[int, int, int], list[list[tuple[int, int]]]] = {}
    m = np.zeros((CTAS, PASSES, SLOTS), np.int64)
    for w in range(windows):
        for g in range(groups):
            spans = [[int(x) for x in slots[w, 32 * g + d, at[w, 32 * g + d, 0]:at[w, 32 * g + d, -1]]]
                     for d in range(32)]
            new, extent = base.color(spans, int(count[w, g]))
            spans = [[new[s] for s in span] for span in spans]
            token_of = {s_new: int(union[w, g, s_old]) for s_old, s_new in new.items()}
            npass = -(-extent // SLOTS)
            passes[w, g] = npass
            plan: dict[int, list[list[tuple[int, int]]]] = {warp: [] for warp in range(8)}
            for p in range(npass):
                entries = {d: placed_tiles(base, spans, d, p, token_of) for d in range(32)}
                for d in range(32):
                    tiles[w, p, 32 * g + d] = len(entries[d]) // 8
                for warp in range(8):
                    ds = [warp + 8 * i for i in range(4)]
                    plan[warp].append([x for d in ds for x in entries[d][:8]] + [x for d in ds for x in entries[d][8:]])
            for warp in range(8):
                plans[w, g, warp] = plan[warp]

            live = [(s // SLOTS, s % SLOTS, 1 + ((s ^ d) & 1), 512 * w + token_of[s]) for d in range(32)
                    for s in spans[d]]
            if live:
                pa, sl, mult, tok = (np.array(x) for x in zip(*live, strict=True))
                for b in range(CTAS):
                    sg, rg = box(b)
                    if rg == g:
                        np.add.at(m[b], (pa, sl), mult * (1 + high[tok, sg]))

    stages = max(sum(len(p) for p in x) // 8 for x in plans.values())
    token_record, word_record = 16 + 16 * stages, 64 * stages
    rec = np.zeros((windows, groups, 8, token_record // 2), np.uint16)
    words = np.zeros((windows, groups, 8, word_record // 4), np.uint32)
    for (w, g, warp), by_pass in plans.items():
        plan = [x for p in by_pass for x in p]
        rec[w, g, warp, 0] = len(plan) // 8
        if plan:
            rec[w, g, warp, 8:8 + len(plan)] = [t for t, _ in pads_first(plan)]
            words[w, g, warp, :2 * len(plan)] = [v for _, x in plan for v in ((x & 0xFFFF) << 16, x >> 16)]
    flat = staged_records(plans, windows, groups)

    one = np.array([one_add(k) for k in range(WORDS)], np.uint64)
    panel = ((m.astype(np.uint64)[..., None] * one) & 0xFFFFFFFF).astype(np.uint32)

    (directory / "head.bin").write_bytes(struct.pack("<5i", 0x4D525453, CTAS, windows, token_record, word_record))
    (directory / "passes.bin").write_bytes(passes.tobytes())
    (directory / "tiles.bin").write_bytes(tiles.tobytes())
    (directory / "records.bin").write_bytes(rec.tobytes())
    (directory / "words.bin").write_bytes(words.tobytes())
    (directory / "rows.bin").write_bytes(rows.tobytes())
    (directory / "panel.bin").write_bytes(panel.tobytes())
    (directory / "staged.bin").write_bytes(struct.pack("<i", flat.shape[2]) + flat.tobytes())
    print(f"read_stream {cell}: records {token_record} + {word_record} bytes ({stages} stages at most), "
          f"{tiles.sum() / (windows * groups):.1f} tiles and {passes.mean():.2f} passes a box-window, "
          f"{flat.shape[2] - STAGED_AHEAD - STAGED_TAIL} staged stages a warp at most")
    return ["head.bin", "passes.bin", "tiles.bin", "records.bin", "words.bin", "rows.bin", "panel.bin", "staged.bin"]


def pads_first(plan: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A plan's pads at their tile's first live token (0 in an all-pad tile): the same line, its coefficient zero."""
    out = []
    for k in range(0, len(plan), 8):
        first = next((t for t, _ in plan[k:k + 8] if t >= 0), 0)
        out += [(first if t < 0 else t, word) for t, word in plan[k:k + 8]]
    return out


def staged_records(plans: dict, windows: int, groups: int) -> np.ndarray:
    """THE STAGED RECORDS: u32 `[groups][8 warps][STAGED_AHEAD + stages + STAGED_TAIL][24]`, one stream a warp across
    its windows and passes in the consumer's order. Record `r` is 96 bytes: stage `r - STAGED_AHEAD`'s 8 entries
    lane-ready (`words.bin`'s, zero before the first stage), then stage `r`'s 8 tokens -- `512 w + t`, a pad `PAD`
    (token 0 with its top bit set) -- so a stage's copy brings its entries and the tokens of the stage `STAGED_AHEAD`
    on; past the last stage, pads."""
    streams = {}
    for g in range(groups):
        for warp in range(8):
            stream = []
            for w in range(windows):
                for p in plans.get((w, g, warp), []):
                    stream += [(-1 if t < 0 else 512 * w + t, x) for t, x in p]
            streams[g, warp] = stream
    stages = max(len(x) for x in streams.values()) // 8
    flat = np.zeros((groups, 8, STAGED_AHEAD + stages + STAGED_TAIL, 24), np.uint32)
    flat[..., 16:] = PAD
    for (g, warp), stream in streams.items():
        for k in range(len(stream) // 8):
            tile = stream[8 * k:8 * k + 8]
            flat[g, warp, k + STAGED_AHEAD, :16] = [v for _, x in tile for v in ((x & 0xFFFF) << 16, x >> 16)]
            flat[g, warp, k, 16:] = [PAD if t < 0 else t for t, _ in tile]
    return flat


if __name__ == "__main__":
    write(Path(sys.argv[1]))
