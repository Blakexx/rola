# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READOUT LAB'S INPUTS: one head's read lists as the facts pass would hand them to the carry, for `read_image.cu`
(docs/internals/tools/lab.md#read-image), in the final form a readout step consumes and conflict-free by construction
(KERNEL_STANDARDS section 30).

The facts pass's choices, made here: a box-window's union slots COLORED -- each token a residue mod 8 (its row's bank
group in the pool and the panel), greedily so a digit's tokens differ, at most ceil(count / 8) a residue, its slot the
residue plus 8 x its rank among them; a digit's tokens DEALT one a residue at a time, and each 8-token group PLACED at
the positions of its residues, the positions left PADS (a free residue's row, coefficient 0, its add on that residue's
dummy panel row). A lane's add then lands on distinct banks: a step's 8 tokens on M, a tile's 4 even or 4 odd ones on N.

`read.bin` is a header of four int32 -- the magic word 0x44414552, windows, the list level's width, the 32-digit groups
-- then the colored slots `[windows][width][512]` u16, the pass starts `[windows][width][8]` u16 and each box-window's
slot extent (8 x the largest residue's count) `[windows][groups]` u16: the floors, the check and the drain read it. For
each step form (`m`, tokens on M; `n`, on N), `entries_<form>.bin` is each digit's placed list `[windows][width][512]`
of 8-byte entries `(pool, word)`: the pool row's byte offset from the CTA's shared base with its swizzle folded in (its
high half the digit's steps -- of 16 positions on M, tiles of 8 on N -- bit 24 when they run past the HEAD staged
entries, bits 25-29 the digit), and the coefficient times the unit (bf16, `256 x (1 + (slot ^ digit) & 1)`) low with the
panel field high: the token's row (a pad's dummy row) past the lane bases, less the lane offset of the entry's position
in its step (16 x (e % 8) on M, 32 x (e % 8 / 2) on N). `heads_<form>.bin` is each warp's window block
`[windows][boxes][8][4 x HEAD x 8]`: its four digits' first HEAD entries; `heads_bal_<form>.bin` (when asked) the same
for a balanced deal of each box-window's digits to warps, longest first, each to the least loaded warp with room, 4 a
warp. `banks.json` is the bank model's count (`tools/bank_model.py`) of the wavefronts a CTA-window's steps cost per
MMA, by part, a form.
"""
from __future__ import annotations

import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

#: the probe cell, its first 32 windows
CELL, TOKENS = "nl64k-alt-k4", 16384
#: `read_image.cu`'s layout: 2-KB image planes before the pool, then the lane bases (128 bytes), the 8 dummy panel rows
#: (one a residue) and the panel, 80-byte rows
PLANE_BYTES, DUMMY_ROWS, PANEL_PITCH = 2048, 128, 80
#: the coefficients' bf16 bits by `(slot ^ digit) & 1`: 256 and 512
COEFFICIENT = (0x4380, 0x4400)
#: the lane offset of an entry's position in its step, and the positions a step holds
FOLD = {"m": lambda e: 16 * (e % 16 % 8), "n": lambda e: 32 * (e % 8 // 2)}
PER = {"m": 16, "n": 8}


def lists(windows: int, with_tokens: bool = False, cell: str = CELL) -> tuple:
    """`cell`'s read lists over `windows` windows of 512 tokens, from the facts pass's words: (slots, at, count)
    as int64 arrays -- each digit's tokens as their ranks in its 32-digit group's union, in token order (`at[0]` 0, the
    rest its count), and each union's count; `with_tokens` adds each union's tokens `[windows][groups][512]` (a rank's
    token in its window)."""
    import numpy as np

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, lists_form_of
    from rola.engine.facts import liveness as lv
    from rola.ops import facts as facts_ops

    spec = by_name(cell)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, 512 * windows))
    words = facts_ops.facts(routes, values, layout, *statics, lists_form_of(spec)).words
    level = facts_ops.list_level(statics[0])
    base, width = layout.row_base(level), layout.B[level]
    bits = lv.unpack(words, layout)[0, lv.READ, base:base + width].cpu().numpy()
    slots = np.zeros((windows, width, 512), np.int64)
    at = np.zeros((windows, width, 8), np.int64)
    count = np.zeros((windows, width // 32), np.int64)
    tokens = np.zeros((windows, width // 32, 512), np.int64)
    for w in range(windows):
        for g in range(width // 32):
            group = bits[32 * g:32 * g + 32, 512 * w:512 * w + 512]
            union = group.any(0)
            rank = np.cumsum(union) - 1
            count[w, g] = union.sum()
            tokens[w, g, :count[w, g]] = np.flatnonzero(union)
            for d in range(32):
                mine = rank[group[d]]
                slots[w, 32 * g + d, :len(mine)] = mine
                at[w, 32 * g + d, 1:] = len(mine)
    return (slots, at, count, tokens) if with_tokens else (slots, at, count)


def _row(slot: int, row_bytes: int = 64) -> int:
    """A pool row's byte offset, its 16-byte chunks swizzled so a phase's 8 rows on distinct residues are 8 bank
    groups: 64-byte rows by `(slot >> 1) & 3`, 32-byte rows by `(slot >> 2) & 1`."""
    if row_bytes == 64:
        return slot * 64 + 16 * ((slot >> 1) & 3)
    return slot * 32 + 16 * ((slot >> 2) & 1)


def select(slots, at, digits: int, keep: float = 1.0, seed: int = 0) -> tuple:
    """Read lists for boxes of `digits` digits, each entry kept with probability `keep` (a box of the same state with
    fewer writes, as at more leaves for the same fan-out): each box-window's union re-ranked compactly; returns
    (slots, at, count) in `emit`'s form."""
    import numpy as np

    rng = np.random.default_rng(seed)
    windows, width, cap = slots.shape
    out = np.zeros_like(slots)
    out_at = np.zeros_like(at)
    count = np.zeros((windows, width // digits), np.int64)
    for w in range(windows):
        for b in range(width // digits):
            rows = range(digits * b, digits * b + digits)
            kept = {}
            for d in rows:
                span = [int(x) for x in slots[w, d, at[w, d, 0]:at[w, d, -1]]]
                kept[d] = [x for x in span if keep >= 1.0 or rng.random() < keep]
            rank = {x: i for i, x in enumerate(sorted({x for d in rows for x in kept[d]}))}
            count[w, b] = len(rank)
            for d in rows:
                out[w, d, :len(kept[d])] = [rank[x] for x in kept[d]]
                out_at[w, d, 1:] = len(kept[d])
    return out, out_at, count


def color(digits: list[list[int]], count: int) -> tuple[dict[int, int], int]:
    """A box-window's slots recolored: tokens in the most and longest lists first, each the residue its lists use least
    (then the least loaded) with room left; returns the new slot of each old one and the extent."""
    room = -(-count // 8)
    member: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(digits):
        for s in d:
            member[s].append(i)
    residue: dict[int, int] = {}
    load = [0] * 8
    for s in sorted(member, key=lambda s: -sum(len(digits[i]) for i in member[s])):
        r = min((sum(residue.get(t) == r for i in member[s] for t in digits[i]), load[r], r)
                for r in range(8) if load[r] < room)[2]
        residue[s] = r
        load[r] += 1
    rank = [0] * 8
    new = {}
    for s in sorted(residue):
        new[s] = residue[s] + 8 * rank[residue[s]]
        rank[residue[s]] += 1
    return new, 8 * room


def place(span: list[int]) -> list[tuple[int, bool]]:
    """A digit's positions: its tokens dealt one a residue at a time, each 8-token group at its residues' positions
    (a repeat on a position left free), the rest pads on their position's residue: (slot or residue, pad)."""
    buckets = [[x for x in span if x & 7 == r] for r in range(8)]
    dealt = [b[j] for j in range(max(map(len, buckets), default=0)) for b in buckets if len(b) > j]
    out: list[tuple[int, bool]] = []
    for g in range(0, len(dealt), 8):
        group: list[tuple[int, bool] | None] = [None] * 8
        rest = []
        for s in dealt[g:g + 8]:
            if group[s & 7] is None:
                group[s & 7] = (s, False)
            else:
                rest.append((s, False))
        out += [x if x is not None else (rest.pop() if rest else (r, True)) for r, x in enumerate(group)]
    return out


def place_rounds(span: list[int]) -> list[tuple[int, bool]]:
    """A digit's positions by ROUNDS: step i holds each residue's i-th token at its residue's position and pads on the
    rest, so a step never repeats a residue and the steps are the largest residue's count."""
    buckets = [[x for x in span if x & 7 == r] for r in range(8)]
    return [(b[i], False) if len(b) > i else (r, True) for i in range(max(map(len, buckets), default=0))
            for r, b in enumerate(buckets)]


def emit(directory: Path, slots, at, count, pass_slots: int, planes: int, head: int, plane_bytes: int = PLANE_BYTES,
         pitch: int = PANEL_PITCH, channels: int = 32, balanced: bool = False, digits: int = 32,
         row_bytes: int = 64, rounds: bool = False) -> list[str]:
    """Write the files for a `pass_slots`-slot readout pass over `planes` image planes of `plane_bytes` with HEAD
    entries a digit staged, `channels` a box (panel rows `pitch` bytes) and pool rows of `row_bytes` past the planes,
    from read lists whose every box's union fits one pass (a digit's entries `slots[w, d, at[w, d, 0]:at[w, d, -1]]`)."""
    import bank_model
    import numpy as np

    windows, width, cap = slots.shape
    groups = count.shape[-1]
    assert (count <= pass_slots).all(), "a box's union exceeds one readout pass"
    colored, extent = slots.copy(), count.copy()
    clean = small = 0
    for w in range(windows):
        for box in range(groups):
            rows = range(box * digits, box * digits + digits)
            spans = [[int(x) for x in slots[w, d, at[w, d, 0]:at[w, d, -1]]] for d in rows]
            new, extent[w, box] = color(spans, int(count[w, box]))
            for d, span in zip(rows, spans, strict=True):
                colored[w, d, at[w, d, 0]:at[w, d, -1]] = [new[s] for s in span]
                if len(span) <= 8:
                    small += 1
                    clean += len({new[s] & 7 for s in span}) == len(span)
    header = struct.pack("<4i", 0x44414552, windows, width, groups)
    body = colored.astype(np.uint16).tobytes() + at.astype(np.uint16).tobytes() + extent.astype(np.uint16).tobytes()
    (directory / "read.bin").write_bytes(header + body)
    pool = planes * plane_bytes
    lane_base = pool + pass_slots * row_bytes
    panel_rows = DUMMY_ROWS + 8 * pitch
    model: dict[str, dict[str, float]] = {"colored": {"digits of at most 8 tokens on distinct residues":
                                                      clean / small if small else None}}
    files = ["read.bin"]
    for form in ("m", "n"):
        per = PER[form]
        entries = np.zeros((windows, width, cap, 2), np.uint32)
        heads = np.zeros((windows, width // digits, 8, digits // 8 * head * 2), np.uint32)
        bal = np.zeros_like(heads)
        cost = defaultdict(float)
        mmas = 0
        for w in range(windows):
            for d in range(width):
                placed = (place_rounds if rounds else place)([int(x) for x in colored[w, d, at[w, d, 0]:at[w, d, -1]]])
                steps = -(-len(placed) // per)
                tag = (steps | (steps * per > head) << 8 | d % digits << 9) << 16
                placed += [(r, True) for _ in range(steps * per // 8 + 2) for r in range(8)]
                for e in range(max(steps * per, head)):
                    s, pad = placed[e]
                    field = (DUMMY_ROWS if pad else panel_rows) + pitch * s - FOLD[form](e)
                    word = field << 16 | (0 if pad else COEFFICIENT[(s ^ d) & 1])
                    entries[w, d, e] = (pool + _row(s, row_bytes) | tag, word)
                block = heads[w, d // digits, d % 8]
                i = d % digits // 8
                block[2 * head * i:2 * head * (i + 1)] = entries[w, d, :head].reshape(-1)
                mmas += steps * (channels // 8 if form == "m" else channels // 16) * (row_bytes // 32)
                cost["state"] += channels // 2 * row_bytes // 64
                for k in range(steps):
                    step = [(int(entries[w, d, per * k + e, 0]) & 0xFFFF, int(entries[w, d, per * k + e, 1]) >> 16)
                            for e in range(per)]
                    cost["entries"] += bank_model.wavefronts([8 * (lane % per) for lane in range(32)], 8)
                    if form == "m":
                        for kb in range(row_bytes // 32):
                            cost["gathers"] += bank_model.ldmatrix(
                                [[step[8 * (q & 1) + r][0] ^ 16 * (2 * kb + (q >> 1)) for r in range(8)]
                                 for q in range(4)])
                        for half in range(2):
                            for nt in range(channels // 8):
                                cost["adds"] += bank_model.wavefronts(
                                    [lane_base + 4 * lane + step[8 * half + (lane >> 2)][1] + 16 * nt
                                     for lane in range(32)], 4)
                    else:
                        cost["gathers"] += bank_model.ldmatrix([[step[r][0] ^ 16 * q for r in range(8)]
                                                                for q in range(row_bytes // 16)])
                        for par in range(2):
                            for m in range(channels // 16):
                                cost["adds"] += bank_model.wavefronts(
                                    [lane_base + 4 * (lane >> 2) + 32 * (lane & 3) + step[2 * (lane & 3) + par][1]
                                     + 32 * m for lane in range(32)], 4)
        model[form] = {"MMAs a CTA-window": mmas / (windows * groups)}
        model[form].update({f"{k} a MMA": v / mmas for k, v in cost.items()})
        model[form]["wavefronts a MMA"] = sum(cost.values()) / mmas
        (directory / f"entries_{form}.bin").write_bytes(entries.tobytes())
        (directory / f"heads_{form}.bin").write_bytes(heads.tobytes())
        files += [f"entries_{form}.bin", f"heads_{form}.bin"]
        if balanced:
            for w in range(windows):
                for box in range(width // digits):
                    load = [0] * 8
                    held: list[list[int]] = [[] for _ in range(8)]
                    tiles = [int(entries[w, box * digits + d, 0, 0]) >> 16 & 0xFF for d in range(digits)]
                    for d in sorted(range(digits), key=lambda d: (-tiles[d], d)):
                        k = min((x for x in range(8) if len(held[x]) < digits // 8), key=lambda x: (load[x], x))
                        held[k].append(d)
                        load[k] += tiles[d]
                    for k in range(8):
                        for i, d in enumerate(held[k]):
                            row = entries[w, box * digits + d, :head].reshape(-1)
                            bal[w, box, k, 2 * head * i:2 * head * (i + 1)] = row
            (directory / f"heads_bal_{form}.bin").write_bytes(bal.tobytes())
            files.append(f"heads_bal_{form}.bin")
    (directory / "banks.json").write_text(json.dumps(model, indent=1) + "\n")
    print("bank model:", json.dumps(model))
    return files + ["banks.json"]


def write(directory: Path) -> list[str]:
    import torch

    slots, at, count = lists(TOKENS // 512)
    out = emit(directory, slots, at, count, 256, 16, 16)
    torch.cuda.empty_cache()
    return out
