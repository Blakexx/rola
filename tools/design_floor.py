#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DESIGN'S FLOORS: the value-carve carry (ruled 2026-11-06) counted a CTA-window from a cell's real lists at
any box, and priced on each card -- each phase's warp instructions, shared-memory cycles, tensor cycles and global
path, the window's floors (every phase overlapped; phases in turn) and the box's spread across CTAs.

    bazel run //tools:design_floor -- nl64k-alt-k4 flagship-alt-k16 --box 32x16x64 --box 32x32x32
    bazel run //tools:design_floor -- nl64k-dense --card sm_80 --transit 8 --batch 128

A BOX is `lists x digits x channels`: lists the fold's list level and the readout's dense level (its K), digits the
readout's list level and the fold's dense level. ESTIMATED: there is no kernel of this form; the counts are exact over
the lists, the instruction budgets and the card constants are stated below and in the docs, and the readout's
latency is not modelled (the lab measures it). Docs: docs/internals/tools/design_floor.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: the warps a CTA, its schedulers, the box's state (8 warps x 32 lanes x 128 fp32 registers)
WARPS, SCHEDULERS, BOX_VALUES = 8, 4, 32768
#: the facts pass's window: the lists come at 512 tokens and are merged to the card's window
FACTS_WINDOW = 512
#: each card's window, an m16n8k16 / m16n8k8 at fp32 accumulate in cycles a scheduler, and its global path (an L2
#: line, an atomic request; measure/common hardware.json's `global`, sm_80's announced from sm_86 -- `l2_bracket`
#: prices the A100's line at half as well)
CARDS = {"sm_80": {"window": 1024, "mma16": 8.0, "mma8": 4.0},
         "sm_86": {"window": 512, "mma16": 32.132, "mma8": 16.066}}
#: THE INSTRUCTION BUDGETS, warp instructions (stated, not measured):
#: a readout tile a warp (row loads, MMAs, the coefficient, the panel's load-add-store, the entry, the loop)
READ_TILE = 23
#: a digit a warp with entries (its slice's loads, its run's bounds, the loop)
READ_DIGIT = 8
#: a den tile (its MMAs, the coefficient FFMAs, the den word's read-add-write)
DEN_TILE = 9
#: a fold step's work beyond its MMAs and operand loads (the entries, the c pairs on B, the loop)
FOLD_STEP = 5
FOLD_LIST = 10
#: a hand-off's maxima a warp (the lane's 128 values, the row joins)
MAXIMA = 150
#: today's carry on the A100 at W = 1,024: its phase ledger's cycles a warp-window, measured 2026-11-06
TODAY_SM80 = {"nl64k-alt-k4": 24241, "flagship-alt-k16": 76813, "flagship-cohort-k16": 71345, "nl64k-dense": 173029}
#: the record's scaling at alt-k4: 24.2K cycles a warp-window <-> 5.09 ms carry_forward (L 64K x 4 heads)
MS_PER_CYCLE = 5.09 / 24241


@dataclass
class Box:
    lists: int
    digits: int
    channels: int

    @classmethod
    def parse(cls, s: str) -> Box:
        a, b, c = (int(x) for x in s.lower().split("x"))
        if a * b * c != BOX_VALUES:
            raise ValueError(f"a box holds {BOX_VALUES} values, {s} holds {a * b * c}")
        return cls(a, b, c)

    @property
    def channel_groups(self) -> int:
        return min(WARPS, self.channels // 8)

    @property
    def token_groups(self) -> int:
        return WARPS // self.channel_groups

    @property
    def line_tokens(self) -> int:
        """The tokens one 128-byte line of num holds (the token groups' interleave)."""
        return max(1, 128 // (2 * self.channels))

    def __str__(self) -> str:
        return f"{self.lists}x{self.digits}x{self.channels}"


@dataclass
class Phase:
    issue: float = 0.0    #: warp instructions
    shared: float = 0.0   #: shared-pipe cycles (a 128-byte load one, a store two)
    tensor: float = 0.0   #: MMA cycles summed over the schedulers' pipes
    lines: float = 0.0    #: 128-byte lines from L2
    requests: float = 0.0  #: atomic requests

    def add(self, o: Phase) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))


@dataclass
class Window:
    phases: dict = field(default_factory=dict)

    def ph(self, name: str) -> Phase:
        return self.phases.setdefault(name, Phase())


def membership(lists, union_of_slots: bool, windows: int, merge: int):
    """``M[bh, window, digit, token]`` bool: the digit's list holds the token, the facts pass's windows merged
    ``merge`` at a time (their tokens disjoint)."""
    import torch

    pre = lists.pre.long() & 0xFFFF
    BH, nw, digits, W = lists.entries.shape[0], lists.entries.shape[1], lists.entries.shape[2], lists.entries.shape[3]
    list_group = pre.shape[3] - 1  #: a list group's prefix rows: one a list, then its totals
    lengths = ((pre[..., 1:, -1] - pre[..., :-1, -1]) & 0xFFFF).flatten(2)[..., :digits]  #: [BH, windows, digits]
    ent = lists.entries.long()
    if union_of_slots:
        slots = ent[..., 0] & 0xFFFF
        group = torch.arange(digits, device=ent.device) // list_group
        un = lists.union.long()  #: [BH, windows, groups, W]
        idx = group.view(1, 1, digits, 1).expand(BH, nw, digits, W)
        tokens = torch.gather(un, 2, idx.clamp(max=un.shape[2] - 1)).gather(3, slots.clamp(max=W - 1))
    else:
        tokens = ent[..., 1]
    live = torch.arange(W, device=ent.device).view(1, 1, 1, W) < lengths.unsqueeze(-1)
    tokens = torch.where(live & (tokens >= 0) & (tokens < W), tokens, torch.full_like(tokens, W))
    m = torch.zeros(BH, nw, digits, W + 1, dtype=torch.bool, device=ent.device)
    m.scatter_(3, tokens, True)
    m = m[..., :W]
    pad = (-nw) % merge
    if pad:
        m = torch.cat([m, torch.zeros(BH, pad, digits, W, dtype=torch.bool, device=m.device)], 1)
    m = m.view(BH, -1, merge, digits, W).permute(0, 1, 3, 2, 4).reshape(BH, -1, digits, merge * W)
    return m.cpu().numpy()[:, :windows]


def dense_membership(BH: int, tokens: int, digits: int, window: int) -> np.ndarray:
    nw = -(-tokens // window)
    m = np.zeros((BH, nw, digits, window), dtype=bool)
    for w in range(nw):
        m[:, w, :, : min(window, tokens - w * window)] = True
    return m


def side_boxes(m: np.ndarray, extent: int) -> np.ndarray:
    """``[BH, windows, boxes, extent, W]`` the side's membership split into the boxes at its list level."""
    BH, nw, digits, W = m.shape
    return m.reshape(BH, nw, digits // extent, extent, W)


def _dense_tokens(win: Window, U: int, digits: int, box: Box, transit: int, batch: int, mma16: float) -> None:
    """A dense side's token group of ``U`` tokens: the nest (token batches, transit groups), the den over 8 digits a
    tile, the exit staged to full lines."""
    rd, den, drain = win.ph("readout"), win.ph("den"), win.ph("drain")
    cg, K = box.channel_groups, box.lists
    tiles = -(-U // 16)
    rd.tensor += cg * tiles * digits * (K // 16) * mma16
    rd.shared += cg * tiles * (digits / transit) * 16 * K * 2 / 128          # rows, once a transit group
    rd.shared += cg * (-(-U // batch)) * digits * K * 8 * 2 / 128           # slices, once a token batch
    rd.issue += cg * tiles * digits * (K // 16 + K // 4 + 2)                 # MMAs, the HMUL2 rows, coefficient
    rd.issue += cg * tiles * (digits / transit) * (2 * (K // 16) + 2)

    den.tensor += tiles * (-(-digits // 8)) * (K // 16) * mma16
    den.issue += tiles * (-(-digits // 8)) * (K // 16 + 4)
    # each warp's 8-channel slices stored, a token's row loaded whole, a full-line RED
    drain.shared += U * box.channels * 2 * 3 / 128
    drain.issue += U * box.channels / 2 / 32 * 3 + U / 32 * 2
    drain.requests += U * box.channels * 2 / 128 + U / 16


def _listed_tokens(win: Window, n: np.ndarray, U: int, box: Box, pb: int, mma16: float) -> None:
    """A listed side's token group: ``n`` its entries a digit, ``U`` its tokens; tiles of the digit's entries into
    the panel, the den's tile a tile, the drain."""
    rd, den, drain = win.ph("readout"), win.ph("den"), win.ph("drain")
    cg, K = box.channel_groups, box.lists
    tiles = -(-n // 16)
    rows8 = -(-n // 8) * 8
    live = (n > 0).astype(float)
    rd.tensor += cg * tiles.sum() * (K // 16) * mma16
    rd.shared += cg * (rows8.sum() * K * 2 + live.sum() * K * 8 * 2 + n.sum() * 8) / 128
    rd.shared += cg * n.sum() * 8 * pb * 3 / 128                                # the panel: load, store x2
    rd.issue += cg * (tiles.sum() * READ_TILE + live.sum() * READ_DIGIT)

    den.tensor += tiles.sum() * (K // 16) * mma16
    den.shared += (np.minimum(tiles, cg) * K * 8 * 2).sum() / 128 + n.sum() * 4 * 3 / 128
    den.issue += tiles.sum() * DEN_TILE
    # a token's channels and its warps' den words read, cleared by stores, a full-line RED a line
    drain.shared += U * (box.channels * pb * 3 + cg * 4 * 3) / 128
    drain.issue += U * box.channels / 2 / 32 * 4 + U * cg / 32 * 3
    drain.requests += U * box.channels * 2 / 128 + U / 16


def readout(rb: np.ndarray, box: Box, dense: bool, pb: int, transit: int, batch: int, mma16: float) -> Window:
    """THE READOUT a box-window (value carve, token x channel groups): ``rb`` ``[extent, W]`` its digits' lists."""
    win = Window()
    W = rb.shape[1]
    group_of = (np.arange(W) // box.line_tokens) % box.token_groups
    in_union = rb.any(0)
    for tau in range(box.token_groups):
        mine = group_of == tau
        U = int((in_union & mine).sum())
        if dense:
            _dense_tokens(win, U, rb.shape[0], box, transit, batch, mma16)
        else:
            _listed_tokens(win, rb[:, mine].sum(1), U, box, pb, mma16)
    return win


def fold(wb: np.ndarray, box: Box, dense: bool, mma16: float, mma8: float) -> Window:
    """THE FOLD a box-window (the state carve: warps own lists, the den a fifth n-tile): ``wb`` ``[lists, W]``."""
    win = Window()
    f = win.ph("fold")
    D, C = box.digits, box.channels
    per16 = (D // 16) * (C // 8 + 1)  #: MMAs a 16-token step (the den's n-tile one more)
    n = wb.sum(1)
    a, b = n // 16, ((n % 16) > 0).astype(int)
    f.tensor += (a.sum() * per16) * mma16 + (b.sum() * per16) * mma8
    steps = a + b
    a_bytes, b_bytes = D * 16 * 2, 16 * C * 2
    if dense:  # the warp's lists share each step's value operand
        f.shared += (steps.sum() * a_bytes + steps.sum() / (box.lists / WARPS) * b_bytes) / 128
    else:
        f.shared += (steps.sum() * (a_bytes + b_bytes) + n.sum() * 8) / 128
    f.issue += steps.sum() * (per16 + a_bytes / 512 + b_bytes / 512 + C / 4 + FOLD_STEP) + len(n) * FOLD_LIST
    return win


def handoff(box: Box) -> Window:
    win = Window()
    h = win.ph("handoff")
    nbytes = BOX_VALUES * 2 + box.lists * box.digits * 2
    h.shared += nbytes * 2 / 128
    h.issue += WARPS * (BOX_VALUES / (WARPS * 32) / 2 + BOX_VALUES * 2 / (WARPS * 32 * 16) + MAXIMA)
    return win


def fills(rb: np.ndarray, wb: np.ndarray, box: Box, read_dense: bool, write_dense: bool) -> Window:
    win = Window()
    f = win.ph("fills")
    read_row = box.lists * 2 + (box.digits * 2 if read_dense else 0)
    write_row = box.digits * 2 + box.channels * 2 + (box.lists * 2 if write_dense else 0)
    ur, uw = int(rb.any(0).sum()), int(wb.any(0).sum())
    entries = (0 if read_dense else int(rb.sum())) + (0 if write_dense else int(wb.sum()))
    f.lines += (ur * read_row + uw * write_row + entries * 8) / 128
    f.shared += f.lines * 0.5
    f.issue += f.lines / 4 * 3
    return win


def price(win: Window, card: str, l2_scale: float) -> dict:
    from measure import hardware

    rates = hardware.resolve(arch=card, product="rtx3080ti" if card == "sm_86" else None)
    line, req = rates["global.l2_line"].cycles * l2_scale, rates["global.red_request"].cycles
    tot = Phase()
    in_turn = 0.0
    out = {}
    for name, p in win.phases.items():
        tot.add(p)
        fl = max(p.issue / SCHEDULERS, p.shared, p.tensor / SCHEDULERS)
        out[name] = (p.issue / SCHEDULERS, p.shared, p.tensor / SCHEDULERS)
        in_turn += fl
    glob = tot.lines * line + tot.requests * req
    out["_totals"] = {"issue": tot.issue / SCHEDULERS, "shared": tot.shared, "tensor": tot.tensor / SCHEDULERS,
                      "global": glob, "lines": tot.lines, "requests": tot.requests}
    out["_overlapped"] = max(out["_totals"][k] for k in ("issue", "shared", "tensor", "global"))
    out["_in_turn"] = max(in_turn, glob)
    return out


def cell_windows(name: str, box: Box, card: str, pb: int, transit: int, batch: int):
    """Every CTA-window of the cell at the box on the card: a list of `Window`s, each one box's one window."""
    from measure.cells import by_name
    from measure.subjects import carry_operands

    mma16, mma8 = CARDS[card]["mma16"], CARDS[card]["mma8"]
    routes, _v, facts, carve, _u = carry_operands(by_name(name))
    window = CARDS[card]["window"]
    merge = window // FACTS_WINDOW
    windows = -(-routes.tokens // window)
    BH = routes.batch * routes.write_heads
    read_level = carve.readout.lists[0] if carve.readout.lists else None
    write_level = carve.fold.lists[0] if carve.fold.lists else None
    widths = carve.widths
    if facts.read is not None and read_level is not None:
        mr = membership(facts.read, False, windows, merge)
    else:
        mr = dense_membership(BH, routes.tokens, widths[1] if len(widths) > 1 else widths[0], window)
    if facts.write is not None and write_level is not None:
        mw = membership(facts.write, True, windows, merge)
    else:
        mw = dense_membership(BH, routes.tokens, widths[0], window)
    rboxes, wboxes = side_boxes(mr, box.digits), side_boxes(mw, box.lists)
    hand = handoff(box)
    out = []
    for bh in range(BH):
        for w in range(windows):
            rws = [readout(rboxes[bh, w, b], box, read_level is None, pb, transit, batch, mma16)
                   for b in range(rboxes.shape[2])]
            fws = [fold(wboxes[bh, w, b], box, write_level is None, mma16, mma8) for b in range(wboxes.shape[2])]
            fls = {(i, j): fills(rboxes[bh, w, i], wboxes[bh, w, j], box, read_level is None, write_level is None)
                   for i in range(rboxes.shape[2]) for j in range(wboxes.shape[2])}
            for (i, j), fl in fls.items():
                win = Window()
                for part in (rws[i], fws[j], hand, fl):
                    for k, p in part.phases.items():
                        win.ph(k).add(p)
                out.append(((bh, i, j), win))
    return out, BH, windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--box", action="append", default=None, help="lists x digits x channels (default 32x16x64, 32x32x32)")
    ap.add_argument("--card", action="append", default=None, choices=sorted(CARDS))
    ap.add_argument("--panel", choices=("f16", "fp32"), default="f16")
    ap.add_argument("--transit", type=int, default=8, help="a dense side's transit group, digits")
    ap.add_argument("--batch", type=int, default=128, help="a dense side's token batch a warp, tokens")
    a = ap.parse_args()
    boxes = [Box.parse(s) for s in (a.box or ["32x16x64", "32x32x32"])]
    pb = 2 if a.panel == "f16" else 4
    for name in a.cells:
        for card in a.card or ["sm_80"]:
            for box in boxes:
                wins, BH, nw = cell_windows(name, box, card, pb, a.transit, a.batch)
                for scale in ((1.0, 0.5) if card == "sm_80" else (1.0,)):
                    priced = [price(win, card, scale) for _k, win in wins]
                    over = np.array([p["_overlapped"] for p in priced])
                    turn = np.array([p["_in_turn"] for p in priced])
                    boxes_n = len(wins) // (BH * nw)
                    per_box = over.reshape(BH, nw, boxes_n).sum(1)  #: a CTA's whole call
                    mean = {k: np.mean([p["_totals"][k] for p in priced]) for k in priced[0]["_totals"]}
                    phases = {k: np.mean([p[k] for p in priced], 0) for k in priced[0] if not k.startswith("_")}
                    today = TODAY_SM80.get(name) if card == "sm_80" else None
                    head = (f"{name} {card} box {box} ({box.token_groups} token x {box.channel_groups} channel groups) "
                            f"W {CARDS[card]['window']} L2 x{scale:.1f} panel {a.panel}")
                    print(head)
                    print(f"  floor overlapped {over.mean():8,.0f}  in turn {turn.mean():8,.0f} cycles a CTA-window"
                          + (f"   today {today:,} ({today / over.mean():.1f}x the overlapped floor)" if today else "")
                          + (f"   ~{over.mean() * MS_PER_CYCLE:.2f} ms carry (the record's alt-k4 scaling)"
                             if name == "nl64k-alt-k4" and card == "sm_80" else ""))
                    print(f"  totals: issue {mean['issue']:7,.0f}  shared {mean['shared']:7,.0f}  tensor "
                          f"{mean['tensor']:7,.0f}  global {mean['global']:7,.0f} ({mean['lines']:,.0f} lines, "
                          f"{mean['requests']:,.0f} requests);  CTA spread (slowest box / mean) "
                          f"{per_box.max() / per_box.mean():.2f}")
                    print("  phases (issue / shared / tensor): " + "  ".join(
                        f"{k} {v[0]:,.0f}/{v[1]:,.0f}/{v[2]:,.0f}" for k, v in phases.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
