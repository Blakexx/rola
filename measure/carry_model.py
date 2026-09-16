# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY MODEL: what the design says each component of the carry kernel must cost on a
cell, from its liveness alone -- the analytic floors (tensor, bandwidth, latency) and the
instruction budget derived from the design's own operation count (never from a baseline).
KERNEL_STANDARDS §19: these numbers are written BEFORE a component's code and the part
harness and the region ledger are gated against them.

Units: one CTA box (BC leaves) over one window (512 tokens); instructions are warp
instructions summed over the CTA's warps; cycles are SM cycles at the locked clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TILE = 16
WARPS = 8
SCHEDULERS = 4
HMMA_CYCLES = 32          #: bf16 m16n8k16 with fp32 accumulate, one scheduler (measured 2026-09-04)


@dataclass
class Liveness:
    """[512, boxes] bool a side: token x box live cells, from the box words."""
    read: np.ndarray
    write: np.ndarray

    @property
    def boxes(self) -> int:
        return self.read.shape[1]


def _sorted_tiles(live: np.ndarray, wlen: int) -> tuple[int, int, int]:
    """(tiles, live (tile, box) pairs, live tokens) of a side under the first-live-box sort:
    the live tokens first, by first live box; a tile is sixteen ranks; a pair is live when
    any of the tile's tokens is live in the box."""
    boxes = live.shape[1]
    key = np.where(live.any(1), live.argmax(1), boxes)
    order = np.lexsort((np.arange(wlen), key))
    L = live[order]
    n_live = int(live.any(1).sum())
    tiles = math.ceil(n_live / TILE)
    pairs = int(L[: tiles * TILE].reshape(-1, TILE, boxes).any(1).sum()) if tiles else 0
    return tiles, pairs, n_live


def _fold_kept(write: np.ndarray, wlen: int, warps: int = WARPS) -> tuple[int, int, int]:
    """(union-live tokens, fragments over the warps, HMMAs): a warp keeps a token live in
    either of its boxes (`w`, `w + warps`); fragments are its kept count in sixteens, the
    chunk tails padded."""
    boxes = write.shape[1]
    union = int(write.any(1).sum())
    chunks = math.ceil(union / POOL_TOK) if union else 0
    frags = 0
    for w in range(warps):
        kept = write[:, [w + i * warps for i in range(boxes // warps)]].any(1)
        ranks = np.cumsum(write.any(1)) - 1
        per_chunk = [int(kept[(ranks // POOL_TOK) == c].sum()) for c in range(chunks)]
        frags += sum(math.ceil(k / TILE) for k in per_chunk)
    return union, frags, frags * (boxes // warps) * (DV_NT + 1)


DV_NT = 8                 #: n-tiles of a box at DV = 64
POOL_TOK = 64             #: tokens a pool slot at DV = 64 (box.cuh: kPoolTok)


def _chunk_rounds(write: np.ndarray, wlen: int) -> list[tuple[int, int]]:
    """Per pool chunk: the rounds it SPANS (the fold's walk visits them all) and the rounds
    that are LIVE (a union token in them: the fill copies only those; a dead round is the
    skip's ~30 instructions). At N = L most spanned rounds are dead for a CTA."""
    union = write.any(1)
    ranks = np.cumsum(union) - 1
    total = int(union.sum())
    out = []
    for c in range(math.ceil(total / POOL_TOK) if total else 0):
        rounds = np.nonzero((ranks // POOL_TOK == c) & union)[0] // 32
        out.append((int(rounds.max() - rounds.min() + 1), int(len(np.unique(rounds)))))
    return out


def budget(lv: Liveness, wlen: int = 512, dv: int = 64, D: int = 2) -> dict:
    """Per component: the floors and the instruction budget. Every formula names the design
    operation it counts; a component is never budgeted by what an implementation happened
    to cost. The form: state-carve fold over the union pool, sorted position-carve readout."""
    boxes = lv.boxes
    nt = dv // 8                                # n-tiles of a box (8 at DV = 64)
    dealt = boxes // WARPS
    tiles_r, pairs_r, live_r = _sorted_tiles(lv.read, wlen)
    union, frags, fold_hmma = _fold_kept(lv.write, wlen)
    chunks = math.ceil(union / POOL_TOK) if union else 0

    # --- the head ------------------------------------------------------------------------
    #: a round group (two a warp): the read side's box words (~63), the mask (32), the key
    #: and the counts by match (~22); the write side's box words (~63), the warps' words
    #: and the union (~24). The scans (~100 on two warps), the scatter (~10 a group), the
    #: tile masks (~12 a pass, four passes).
    head = 2 * WARPS * (63 + 32 + 22 + 63 + 24 + 10) + 200 + WARPS * 4 * 12

    # --- the fill: a chunk's rounds dealt to warps, two passes of sixteen tokens a round --
    #: a pass: rank by popcount, in-chunk test, row (~8); seven copies with their addresses
    #: and ptxas's scoreboard padding (~14 each); a round's setup (~10). A warp without a
    #: round: the ballots for the chunk's rounds and the skip (~25); the arrive (~8).
    rounds_per_chunk = _chunk_rounds(lv.write, wlen)
    fill = sum(2 * 110 * live + WARPS * 33 for _span, live in rounds_per_chunk)

    # --- the fold: the walk a round, the fragments --------------------------------------
    #: a round of the walk: the words, the rank, the keep, the ring store with its gain, the
    #: prefix bookkeeping (~20). A fragment: the ring read and the loop (~10), two entry
    #: shuffles, six ldmatrix with their addresses (~20), the gain pairs (6), then a box:
    #: two shuffles, four packed multiplies, nt + 1 HMMAs, the loop (~3); ~102 + 18.
    fold = WARPS * sum(span for span, _live in rounds_per_chunk) * 20 + frags * (36 + dealt * (9 + nt + 1))

    # --- the readout: a warp its tiles, a pair a live box --------------------------------
    #: a tile: the ring issue and wait (~8), A by ldmatrix (1), y zero (4 nt), the mask loop
    #: (~6); the drain, sixteen rows out through the four-row stage as whole lines by
    #: reduction: 4 nt stores, 4 nt line loads, 4 nt reductions (16 rows x dv / 32 lines), a
    #: row's token and rank test (2 x 16), four passes' syncs (8), the row bases (16), den
    #: (~8): 12 nt + 64 (counted from the kernel 2026-09-12; the earlier 40 + 4 nt missed the
    #: stage reload and half the lines). A pair: the outer scalars (4), A scaled (4), B by
    #: ldmatrix (nt / 2 x4), nt HMMAs, den's fragment and HMMA (5), the loop (2).
    readout = tiles_r * (14 + 4 * nt + 12 * nt + 64) + pairs_r * (15 + nt // 2 + nt)
    read_hmma = pairs_r * (nt + 1)

    # --- the snapshot and the edges ---------------------------------------------------
    snapshot = WARPS * (dealt * (2 * nt * 3 + 6) + 10)
    edges = 4 * WARPS * 10
    total = head + fill + fold + readout + snapshot + edges
    hmma = fold_hmma + read_hmma
    tensor_floor = hmma * HMMA_CYCLES / SCHEDULERS
    issue_floor = total / SCHEDULERS
    return {
        "head": head, "fill": fill, "fold": fold, "readout": readout, "snapshot": snapshot,
        "edges": edges, "total": total, "hmma": hmma,
        "tensor_floor_cycles": tensor_floor, "issue_floor_cycles": issue_floor,
        "pool_bytes": union * (dv * 2 + 64), "binding_floor_cycles": max(tensor_floor, issue_floor),
        "per_hmma": total / max(1, hmma),
        "live_tokens": (live_r, union), "tiles": (tiles_r, chunks), "pairs": (pairs_r, frags),
    }


def cell_liveness(spec, side: str, window: int = 1) -> np.ndarray:
    """[512, 16] bool: token x box liveness of the FIRST CTA box of a two-level cell on one
    side, window `window` -- the budget's input. A box is sixteen consecutive leaves of the
    read order: the outer read-order digit; a token is live in box `b` iff its outer digit
    `b` is nonzero and any inner digit is. Two-level cells only: the general box words are
    the kernel's head, and this model is gated before that head exists."""
    from measure.cells import level_modes, realize
    from rola.ops import carry as c

    if len(spec.widths) != 2:
        raise ValueError("cell_liveness models two-level cells")
    drawn = realize(spec)
    levels = drawn.read if side == "read" else drawn.write
    g = c.geometry(list(spec.widths), level_modes=level_modes(spec), bc=256, nsr=WARPS, nsw=WARPS)
    rank = g["r"]["rank"] if side == "read" else g["w"]["rank"]
    outer = rank.index(1)
    inner = 1 - outer
    t0 = window * 512
    amp = [x[0, t0:t0 + 512, 0, :16].float().cpu().numpy() != 0 for x in levels]
    return amp[outer] & amp[inner].any(axis=1, keepdims=True)


BUDGET_CELLS = ("flagship-dense", "flagship-alt-k4", "flagship-alt-k16", "flagship-cohort-k4",
                "flagship-both-k4")
COMPONENTS = {"head": "head", "fill": "pool_fill", "fold": "FoldStream", "readout": "ReadoutStream",
              "snapshot": "snapshot_publish"}


def main(argv=None) -> int:
    """Write `tools/budgets/carry.json` from the model: every budget cell's components."""
    import argparse
    import json
    from pathlib import Path

    from measure.cells import by_name

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "tools/budgets/carry.json"))
    args = ap.parse_args(argv)
    cells = {}
    for name in BUDGET_CELLS:
        spec = by_name(name)
        lv = Liveness(read=cell_liveness(spec, "read"), write=cell_liveness(spec, "write"))
        b = budget(lv)
        cells[name] = {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in b.items()}
        print(f"{name:20s} total {b['total']:7.0f} head {b['head']:6.0f} fill {b['fill']:6.0f} "
              f"fold {b['fold']:6.0f} read {b['readout']:6.0f} | hmma {b['hmma']:5d} "
              f"per-hmma {b['per_hmma']:5.1f} tensor {b['tensor_floor_cycles']:7.0f} "
              f"issue-floor {b['issue_floor_cycles']:7.0f}")
    Path(args.out).write_text(json.dumps(
        {"unit": "instructions per CTA-window (8 warps), cells at window 1", "cells": cells,
         "components": COMPONENTS}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
