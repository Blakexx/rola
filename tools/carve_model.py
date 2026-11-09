#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARVE MODEL: the carry's mechanics for any carve on any card, over a real cell's lists -- each phase's MMAs,
shared-memory bytes and instructions a warp, its registers and shared-memory ledger, its floors, and the window's
predicted cost. It ranks no carve until it reproduces the measured forms (KERNEL_STANDARDS §32.2; the validation rows
in docs/internals/tools/carve_model.md#validation).

    bazel run //tools:carve_model -- stats nl64k-alt-k4 [nl64k-dense ...]   # the cells' statistics
    bazel run //tools:carve_model -- events nl64k-alt-k4 [...]              # today's carve's events a CTA-window
    bazel run //tools:carve_model -- measured <SourceCounters csv> --cta-windows N

Docs: docs/internals/tools/carve_model.md
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: the box every carve in this model partitions: lists (the fold's list level, the readout's dense level and K), digits
#: (the fold's dense level, the readout's list level), channels; a window of tokens
BOX_LISTS = 32
WINDOW = 512


#: the lists' boundaries fall every this many union slots (the facts pass's prefix rows' columns)
SEGMENT = 64


@dataclass(frozen=True)
class Side:
    """One side of one box-window: its union's slots and each of its 32 lists' entries, by the union's 64-slot segments
    (a dense side: every slot in every list, `listed` False)."""
    listed: bool
    union: int
    segments: np.ndarray  #: [32, S] a list's entries in each 64-slot segment of the union

    @property
    def entries(self) -> np.ndarray:
        return self.segments.sum(1)

    def in_chunk(self, first: int, slots: int) -> np.ndarray:
        """[32] each list's entries among union slots `first .. first + slots - 1` (`first` a multiple of 64)."""
        return self.segments[:, first // SEGMENT: -(-(first + slots) // SEGMENT)].sum(1)

    @property
    def total(self) -> int:
        return int(self.entries.sum())

    @property
    def refetch(self) -> float:
        """Rows a list-at-a-time stream fetches per row the union stages once."""
        return self.total / self.union if self.union else 0.0


@dataclass(frozen=True)
class CellStats:
    """A cell's box-windows: `write[i]`, `read[i]` the two sides of box-window `i` and `words[i]` its plan word on this
    card (the plan pass's rule), `group` query heads a write head."""
    name: str
    group: int
    write: tuple[Side, ...]
    read: tuple[Side, ...]
    words: tuple[int, ...]
    area: int = 0        #: the plan's area on this card, bytes
    read_row: int = 0    #: a read row's bytes in the pool
    write_row: int = 0
    window: int = WINDOW
    read_ring: int = 0      #: a listed read side's rows streamed into each warp's ring of this many stages; 0 resident
    fold_ring: bool = False  #: a listed write side's rows streamed through each warp's fold ring


def _sides(lists, plan, routes, bh: int, g: int, windows: int) -> list[Side]:
    """A side's box-windows at box group `g` of batch-head `bh`: each list's entries from the prefix rows (`pre[l][-1]`
    the boundaries of lists < l over the whole union, modulo 2^16; a list's entries the difference)."""
    out = []
    for w in range(windows):
        tokens = min(WINDOW, routes.tokens - w * WINDOW)
        if not plan.lists:
            seg = np.zeros((BOX_LISTS, -(-tokens // SEGMENT)), dtype=np.int64)
            for j in range(seg.shape[1]):
                seg[:, j] = min(SEGMENT, tokens - j * SEGMENT)
            out.append(Side(False, tokens, seg))
            continue
        pre = lists.pre[bh, w, g].long().cpu().numpy() & 0xFFFF
        before = (pre[1:BOX_LISTS + 1] - pre[:BOX_LISTS]) & 0xFFFF  #: [32, columns] a list's entries before each boundary
        seg = np.diff(np.concatenate([np.zeros((BOX_LISTS, 1), dtype=np.int64), before], 1), axis=1) & 0xFFFF
        out.append(Side(True, int(lists.count[bh, w, g]), seg))
    return out


def cell_stats(name: str, forced: str | None = None) -> CellStats:
    """Every box-window's two sides on a registry cell, from the facts pass's lists as the carry reads them; the plan
    words the rule's, or a forced form's (`rola.ops.carry.forced_word`) where it fits."""
    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import facts as facts_ops

    routes, _values, facts, carve, _unit = carry_operands(by_name(name))
    windows = -(-routes.tokens // WINDOW)
    write = facts_ops.side_lists(facts, "write") if carve.fold.lists else None
    read = facts_ops.side_lists(facts, "read") if carve.readout.lists else None
    groups = lambda plan: max(1, carve.widths[plan.lists[0]] // BOX_LISTS) if plan.lists else 1  # noqa: E731
    from tests.oracle import carry_plan as cp

    ws, rs = [], []
    for bh in range(routes.batch * routes.write_heads):
        for gr in range(groups(carve.readout)):
            for gw in range(groups(carve.fold)):
                rs += _sides(read, carve.readout, routes, bh, gr, windows)
                ws += _sides(write, carve.fold, routes, bh, gw, windows)
    words = tuple(rec.word for box in cp.plans(routes, facts, carve, forced) for rec in box)
    assert len(words) == len(ws), "the plan records' order is the sides'"
    from rola.ops import carry as carry_ops

    key = carry_ops.arm_key(carve)
    f = carry_ops.form(key)
    return CellStats(name, routes.group, tuple(ws), tuple(rs), words, cp.area_of(f.smem, f.area, routes.group),
                     f.read_row, f.write_row)


def _sums(side: Side) -> tuple:
    """A side's segment sums as the plan reads them: the lists' entries before each segment's end."""
    return tuple(int(x) for x in np.cumsum(side.segments.sum(0)))


def merged(s: CellStats, k: int) -> CellStats:
    """THE WINDOW AT `k` x 512 TOKENS: each `k` consecutive windows of a box merged -- their unions added (their tokens
    are disjoint), each list's segments concatenated -- and each merged window planned by the same rule in the area
    its larger tables leave (the window inputs and the read tables are two windows' each: union tokens, prefix rows,
    units)."""
    from tests.oracle import carry_plan as cp

    if k == 1:
        return s
    w = WINDOW * k
    grow_fixed = 2 * (w - WINDOW) * 2 + 2 * 33 * (w - WINDOW) // SEGMENT * 2
    grow_tables = (2 * s.group * (w - WINDOW) * 2 + 2 * 33 * (w - WINDOW) // SEGMENT * s.group * 2
                   + 2 * s.group * (w - WINDOW))
    area = s.area - grow_fixed - grow_tables
    per_box = len(s.words)  # windows run in order inside each box's records; boxes follow one another
    reads, writes, words = [], [], []
    i = 0
    boxes_windows = _box_windows(s)
    for n in boxes_windows:
        for j in range(0, n, k):
            idx = list(range(i + j, min(i + j + k, i + n)))
            rd = _merge([s.read[t] for t in idx])
            wr = _merge([s.write[t] for t in idx])
            reads.append(rd)
            writes.append(wr)
            c = cp.Counts(rd.union, _sums(rd) if rd.listed else (0,) * rd.segments.shape[1],
                          wr.union, _sums(wr) if wr.listed else (0,) * wr.segments.shape[1], wr.listed)
            rows = tuple(0 if (ring and side.listed) else row
                         for row, side, ring in ((s.read_row, rd, s.read_ring), (s.write_row, wr, s.fold_ring)))
            words.append(cp.win_of(c, -1, *rows, area).word)
        i += n
    assert i == per_box
    return CellStats(f"{s.name}@W{w}", s.group, tuple(writes), tuple(reads), tuple(words), area, s.read_row,
                     s.write_row, w, s.read_ring, s.fold_ring)


#: A STREAMED SIDE'S COST beyond its MMA phase on sm_86, a fraction of the model's phase: the fold's from the built fold
#: stream (2026-10-19, nl64k-alt-k4 at equal area: fold +898, fills and landed -960 cycles a warp-window, against the
#: 3,320 of fills the model removes; alt-k16 and cohort-k16 within 3%); the readout's (with its drain) the `read_stream`
#: lab's 1.34x of its bare phase rescaled to the model's (0.648), unvalidated -- the fold's lab factor (0.375) was
#: `LAB_MISS` short of its build, and `window` prints the read stream at both
STREAM = {"fold": 1.05, "readout": 0.34 * 0.648}
LAB_MISS = 2.8
#: THE READ RING: its stages' bytes (8 rows of 64 bytes and their entry words) and each depth's cycles a CTA-window
#: beyond 5 stages' (4: the first tiles' block holds the ring, nothing ahead); the fold's ring (4 stages of 8 rows of
#: 128 bytes, a 16-token step's two and two ahead) takes the image's place, a window's read chunks all run before its
#: fold, and its prologue's tenant (each warp's header and first stages' tokens, staged a window ahead) takes
#: `FOLD_TENANT` of the area (the built fold stream, 2026-10-19: that area alone cost alt-k16 +11.7%)
READ_STAGE = 8 * 64 + 64
READ_RING = {4: 264.0, 5: 0.0}
FOLD_TENANT = 8 * 272


def streamed(s: CellStats, read_stages: int = 0, fold: bool = False) -> CellStats:
    """THE FORM WITH A SIDE'S ROWS STREAMED: no chunk of a streamed listed side holds rows or entries -- a read chunk is
    its panel -- and every entry fetches its row into its warp's ring (a token in several lists fetched once a list),
    each warp's entries arriving as its window's stage record. The read ring, `read_stages` deep (0: the read side
    resident), takes the area's share beside the image; the fold's (`fold`) takes the image's place and its tenant. A
    dense side keeps its rows resident (every token is in all 32 lists). Re-planned by the same rule."""
    from tests.oracle import carry_plan as cp

    read_ring = read_stages if s.read[0].listed else 0
    fold_ring = fold and s.write[0].listed
    area = s.area - 8 * read_ring * READ_STAGE - (FOLD_TENANT if fold_ring else 0)
    rows = (0 if read_ring else s.read_row, 0 if fold_ring else s.write_row)
    words = []
    for rd, wr in zip(s.read, s.write):
        c = cp.Counts(rd.union, _sums(rd) if rd.listed else (0,) * rd.segments.shape[1],
                      wr.union, _sums(wr) if wr.listed else (0,) * wr.segments.shape[1], wr.listed)
        words.append(cp.win_of(c, -1, *rows, area).word)
    name = s.name + (f"+read{read_ring}" if read_ring else "") + ("+fold" if fold_ring else "")
    return CellStats(name, s.group, s.write, s.read, tuple(words), area, s.read_row, s.write_row, s.window, read_ring,
                     fold_ring)


def forms(base: CellStats) -> list[tuple[str, CellStats]]:
    """The forms the model ranks: resident, the read side streamed (each ring depth), the fold streamed, both."""
    out = [("resident", base)]
    out += [(f"read stream, ring {d}", streamed(base, d)) for d in READ_RING]
    out += [("fold stream", streamed(base, fold=True))]
    out += [(f"both streams, ring {d}", streamed(base, d, True)) for d in READ_RING]
    return out


#: THE FOLD'S RING: 8 warps x 4 stages x 8 rows of 128 bytes, in the image's place once the read chunks have run
FOLD_RING = 8 * 4 * 8 * 128


def ring_beside_panel(s: CellStats, area: int) -> float:
    """The share of a streamed form's box-windows whose fold ring fits in `area` (the plan's whole area: the image and
    the read ring are dead once the read chunks have run) beside its last read chunk's panel (64 bytes a slot), so
    that panel can drain while the fold runs; elsewhere the drain ends before the fold begins."""
    from tests.oracle import carry_plan as cp

    fit = 0
    for rd, word in zip(s.read, s.words):
        _mode, rc, _fc, nr, _nf = cp.chunks(word)
        last = rd.union - rc * (nr - 1) if nr else 0
        fit += -(-last * cp.PANEL_ROW // 128) * 128 + FOLD_RING <= area
    return fit / len(s.words)


def _merge(sides: list[Side]) -> Side:
    """Consecutive windows' sides as one: the later window's union slots follow the earlier's (its tokens rank after
    them), each list's entries spread evenly over its segment's slots and re-binned at the merged union's 64-slot
    boundaries (a segment's entries' places within it are not in the facts' prefix rows)."""
    per_slot = []
    for x in sides:
        dens = np.repeat(x.segments / SEGMENT, SEGMENT, axis=1)[:, :x.union]
        if x.union and x.segments[:, -1].any() and x.union % SEGMENT:
            last = x.union // SEGMENT  # the last segment holds fewer than 64 slots
            dens[:, last * SEGMENT:] = (x.segments[:, last] / (x.union - last * SEGMENT))[:, None]
        per_slot.append(dens)
    allslots = np.concatenate(per_slot, axis=1)
    union = allslots.shape[1]
    pad = -(-union // SEGMENT) * SEGMENT - union
    binned = np.pad(allslots, ((0, 0), (0, pad))).reshape(BOX_LISTS, -1, SEGMENT).sum(2)
    return Side(sides[0].listed, union, binned)


def _box_windows(s: CellStats) -> list[int]:
    """Each box's windows (the records run box by box, every box the call's windows)."""
    from measure.cells import by_name

    tokens = by_name(s.name.split("@")[0].split("+")[0]).tokens
    n = -(-tokens // WINDOW)
    return [n] * (len(s.words) // n)


@dataclass
class Region:
    """A measured region of one launch, per CTA-window: its executed warp instructions, MMAs by SASS shape and shared
    wavefronts."""
    instructions: float = 0.0
    wavefronts: float = 0.0
    mmas: dict = None

    def add(self, n: float, op: str, wf: float) -> None:
        self.instructions += n
        self.wavefronts += wf
        if op.startswith(("HMMA", "IMMA")):
            self.mmas = self.mmas or {}
            self.mmas[op] = self.mmas.get(op, 0.0) + n


def measured(csv: Path, so: Path, cta_windows: int) -> dict[tuple, Region]:
    """THE MEASURED SIDE: one launch's `ncu --section SourceCounters --page source --print-source sass --csv` dump of the
    carry, each SASS line's executed instructions and shared wavefronts attributed to (the innermost MMA loop holding it,
    or None; its innermost carry function), a CTA-window's worth. The arm is the dump's kernel."""
    import re

    sys.path.insert(0, str(ROOT / "tools"))
    import region_ledger
    import sass
    import sass_gate

    text = csv.read_text()
    kernel = re.search(r'"Kernel Name","([^"]+)"', text).group(1)
    arm = int(re.search(r"Arm<\(int\)(\d+)", kernel).group(1))
    member = f"carry_arm_{arm}"
    arch = sass.device_arch()
    cubin = sass.cubin(so, member, arch)
    listing = sass.section(sass.disassemble(cubin), "carry_kernel", cubin.name)
    lines, labels = [], {}
    for line in listing.splitlines():
        m = re.match(r"^(\$?[\w.$]+):\s*$", line)
        if m:
            labels[m.group(1)] = len(lines)
            continue
        m = sass.INSTRUCTION.match(line)
        if m:
            w = sass_gate.WHOLE.match(line)
            pred = (m.group(2) or "").strip()
            lines.append((int(w.group(1), 16), w.group(3), w.group(4).strip(), pred) if w
                         else (int(m.group(1), 16), m.group(3), "", pred))
    loop_of = {}
    for i, (t, k) in enumerate(sass_gate.mma_loops(lines, labels)):
        for a, _, _, _ in lines[t:k + 1]:
            loop_of.setdefault(a, i)
    frames = region_ledger.frames_map(so, member, arch, "carry_kernel")
    from region_ledger import fn_of, function_map
    fmaps = {p.name: function_map(p.read_text().splitlines()) for p in (ROOT / "csrc/rola/src/carry").rglob("*.cuh")}
    rows = region_ledger.read_source_counters(csv)
    detail = region_ledger.read_source_detail(csv)
    base = rows[0][0]
    out: dict[tuple, Region] = {}
    for addr, n, _samples, _stalls in rows:
        off = addr - base
        chain = [(f, ln) for f, ln in frames.get(off, []) if f in fmaps]
        fn = fn_of(fmaps[chain[0][0]], chain[0][1]) if chain else "?"
        op, wf, _ideal = detail.get(addr, ("?", 0, 0))
        out.setdefault((loop_of.get(off), fn), Region()).add(n / cta_windows, op, wf / cta_windows)
    return out


def _steps(n: float) -> tuple[int, int]:
    """A list's fold steps over `n` entries (fold.cuh's `steps`): 16-token steps while more than 8 remain, then an 8-token
    tail where any remain (a merged window's `n` rounded to the nearest entry)."""
    n = int(round(n))
    s16 = max(0, -(-(n - 8) // 16)) if n > 8 else 0
    return s16, int(n - 16 * s16 > 0)


def _trips(n: np.ndarray) -> int:
    """A chunk's runs staged, each list's (its run's shuffles) and its warp trips of 32 entries (`entries_fill`)."""
    return sum(1 + -(-int(round(x)) // 32) for x in n)


@dataclass
class Events:
    """Today's carve's work a CTA-window, from the plan words and the lists (each box-window summed, then averaged)."""
    read_tiles: float = 0.0     #: readout tile-steps a warp issues (a pair's or a digit's), summed over warps
    read_mmas: float = 0.0
    read_digits: float = 0.0    #: digits stepped (state fragments loaded), summed over warps
    fold_steps16: float = 0.0
    fold_steps8: float = 0.0
    fold_mmas16: float = 0.0
    fold_mmas8: float = 0.0
    fold_lists: float = 0.0     #: (list, fold chunk) pairs
    images: float = 0.0         #: whole images stored (a half counts a half)
    read_chunks: float = 0.0
    read_rows: float = 0.0      #: union slots filled, read side
    fold_rows: float = 0.0
    staged_entries: float = 0.0  #: entries staged into chunks (a resident listed side's)
    entry_trips: float = 0.0    #: a resident listed side's runs staged (a list's in a chunk) and their 32-entry warp trips
    staged_chunks: float = 0.0   #: chunks whose rows or entries are staged (every side's but a streamed listed one's)
    drained: float = 0.0        #: panel slots drained
    read_entries: float = 0.0   #: a listed side's entries (a streamed side fetches a row each)
    fold_entries: float = 0.0
    read_stages: float = 0.0    #: a listed read side's 8-entry stages, a digit's entries rounded up
    fold_stages: float = 0.0    #: a listed fold side's steps (a 16- or 8-token step a stage)
    steps: float = 0.0          #: the window's steps (entries + 1)
    read_imbalance: float = 1.0  #: the slowest warp's readout tile-steps over the mean warp's, each window
    fold_imbalance: float = 1.0  #: the same for the fold's MMA pipe time


def events_today(s: CellStats) -> Events:
    """THE EVENTS of today's carve (8 warps, a warp 4 fold lists and the read digits `w + 8 i`, two at a time; the rule's
    chunks), a CTA-window's mean."""
    from tests.oracle import carry_plan as cp

    ev = Events()
    read_max = read_mean = fold_max = fold_mean = 0.0
    for rd, wr, word in zip(s.read, s.write, s.words):
        rwarp, fwarp = np.zeros(8), np.zeros(8)
        mode, rc, fc, nr, nf = cp.chunks(word)
        ev.read_chunks += nr
        ev.staged_chunks += (0 if (s.read_ring and rd.listed) else nr) + (0 if (s.fold_ring and wr.listed) else nf)
        ev.steps += nr + nf + 1
        ev.images += (1 if mode == cp.WHOLE else nr) if nr else 0
        for first in range(0, rd.union, rc or 1):
            if not nr:
                break
            slots = min(rc, rd.union - first)
            n = rd.in_chunk(first, slots)
            ev.read_rows += 0 if (s.read_ring and rd.listed) else slots
            ev.staged_entries += n.sum() if (rd.listed and not s.read_ring) else 0
            ev.entry_trips += _trips(n) if (rd.listed and not s.read_ring) else 0
            if rd.listed:
                ev.read_entries += n.sum()
                ev.read_stages += sum(-(-int(round(x)) // 8) for x in n)
            ev.drained += slots
            for w in range(8):
                for pair in ((w, w + 8), (w + 16, w + 24)):
                    ev.read_digits += 2
                    if rd.listed:
                        t = max(-(-int(round(n[d])) // 8) for d in pair)
                        ev.read_tiles += t
                        ev.read_mmas += 8 * t
                        rwarp[w] += t
                    else:
                        t = -(-slots // 8)
                        ev.read_tiles += 2 * t
                        ev.read_mmas += 8 * t
                        rwarp[w] += 2 * t
        for first in range(0, wr.union, fc or 1):
            if not nf:
                break
            slots = min(fc, wr.union - first)
            n = wr.in_chunk(first, slots)
            ev.fold_rows += 0 if (s.fold_ring and wr.listed) else slots
            ev.staged_entries += n.sum() if (wr.listed and not s.fold_ring) else 0
            ev.entry_trips += _trips(n) if (wr.listed and not s.fold_ring) else 0
            if wr.listed:
                ev.fold_entries += n.sum()
            for l in range(32):
                a, b = _steps(int(n[l]))
                ev.fold_lists += 1
                ev.fold_steps16 += a
                ev.fold_steps8 += b
                ev.fold_mmas16 += 8 * a
                ev.fold_mmas8 += 8 * b
                ev.fold_stages += (a + b) if wr.listed else 0
                fwarp[l % 8] += 8 * a * PIPE["HMMA.16816.F32"] + 8 * b * PIPE["HMMA.1688.F32"]
        read_max += rwarp.max()
        read_mean += rwarp.mean()
        fold_max += fwarp.max()
        fold_mean += fwarp.mean()
    k = len(s.words)
    for f in ev.__dataclass_fields__:
        setattr(ev, f, getattr(ev, f) / k)
    ev.read_imbalance = read_max / read_mean if read_mean else 1.0
    ev.fold_imbalance = fold_max / fold_mean if fold_mean else 1.0
    return ev


#: THE DRIVERS: each carry function's phase and the event its executed instructions and wavefronts scale with (a
#: function inside an MMA loop keyed `name@L`); the window's bookkeeping, per step, is `GLUE`'s, and a function in
#: neither is refused (`unmapped`)
DRIVERS = {
    "readout": {"read_tiles": "tile_mma slot_of coef_of token_scale add_h2 panel_word read_chunk@L ?@L chunk_addr@L "
                              "Entry::slots@L chunk_addr",
                "read_digits": "slice_row den_row exp_slot pow2_wide",
                "read_chunks": "read_chunk"},
    "fold": {"fold_steps16": "step16 step16@L", "fold_steps8": "step8",
             "fold_steps": "steps steps@L entry_of entry_of@L",
             "fold_lists": "fold_chunk", "one": "fold_units next_unit pow2_half2 list_units dense_exp"},
    "handoff": {"images": "store_batches store_row_batches store_list_batches store_frame list_scale",
                "one": "post_keys post_row_keys post_list_keys post_exps box_top ceil_exp pow2 den_tables_max "
                       "den_tables_out den_exp red_max_u32"},
    "fills": {"rows": "fill", "entry_trips": "entries_fill", "chunks": "read_fill fold_fill chunk_runs chunk_cols "
                                                                  "list_row Flow::slot_landed stage_bytes"},
    "drain": {"drained": "drain red_add_h2 red_add_f32 token_row word_at"},
}
_DRIVER_OF = {fn: (phase, event) for phase, evs in DRIVERS.items() for event, fns in evs.items() for fn in fns.split()}
#: THE WINDOW'S BOOKKEEPING: the plan's and the flow's reads, the staging, the places and the instruments ("?" a line
#: with no frame)
GLUE = frozenset(("carry_kernel", "stream_of", "entry_at", "step_of", "window_of", "parity_of", "lists_in", "facts_in",
                  "stage_window", "cols_of", "union_of", "pre_of", "origin_of", "addr_in", "place_of",
                  "box_classes", "level_stride", "fold_probe", "?"))
GLUE_KINDS = ("Flow", "StepWord", "PlanWord", "Entry")
#: the most of a launch's instructions the model may book to functions it does not know
UNMAPPED_MOST = 0.005


def driver_of(loop, fn: str) -> tuple[str, str]:
    """A measured region's (phase, event); a function no driver and no bookkeeping names is ("?", "steps")."""
    keyed = f"{fn}@L" if loop is not None else fn
    found = _DRIVER_OF.get(keyed) or _DRIVER_OF.get(fn)
    if found:
        return found
    return ("glue", "steps") if fn in GLUE or fn.split("::")[0] in GLUE_KINDS else ("?", "steps")


def unmapped(regions: dict) -> None:
    """Refuse a launch whose functions the model cannot place: a renamed or new carry function would otherwise price
    as bookkeeping, silently (2026-10-19: the hand-off's after its rewrite, a stream build's fold)."""
    total = sum(r.instructions for r in regions.values())
    lost: dict[str, float] = {}
    for (loop, fn), r in regions.items():
        if driver_of(loop, fn)[0] == "?":
            lost[fn] = lost.get(fn, 0.0) + r.instructions
    if total and sum(lost.values()) > UNMAPPED_MOST * total:
        named = ", ".join(f"{fn} {100 * n / total:.1f}%" for fn, n in sorted(lost.items(), key=lambda kv: -kv[1]))
        raise ValueError(f"the carve model places no driver for: {named} -- add each to DRIVERS or GLUE")


def event_count(ev: Events, event: str) -> float:
    """An event's count a CTA-window, the drivers' names."""
    return {"one": 1.0, "read_tiles": ev.read_tiles, "read_mmas": ev.read_mmas, "read_digits": ev.read_digits,
            "read_chunks": ev.read_chunks, "fold_steps16": ev.fold_steps16, "fold_steps8": ev.fold_steps8,
            "fold_steps": ev.fold_steps16 + ev.fold_steps8, "fold_mmas16": ev.fold_mmas16,
            "fold_mmas8": ev.fold_mmas8, "fold_lists": ev.fold_lists, "images": ev.images,
            "rows": ev.read_rows + ev.fold_rows, "entry_trips": ev.entry_trips, "chunks": ev.staged_chunks,
            "drained": ev.drained, "steps": ev.steps}[event]


def calibrate(regions: dict, ev: Events) -> dict[tuple, tuple[float, float]]:
    """Each (phase, event)'s instructions and wavefronts per event on a calibration cell."""
    unmapped(regions)
    tot: dict[tuple, list] = {}
    for (loop, fn), r in regions.items():
        key = driver_of(loop, fn)
        t = tot.setdefault(key, [0.0, 0.0])
        t[0] += r.instructions
        t[1] += r.wavefronts
    return {k: (v[0] / max(event_count(ev, k[1]), 1e-9), v[1] / max(event_count(ev, k[1]), 1e-9))
            for k, v in tot.items()}


def predict(prices: dict, ev: Events) -> dict[str, list]:
    """Each phase's [instructions, wavefronts] a CTA-window from the prices and a cell's events."""
    out: dict[str, list] = {}
    for (phase, event), (i, w) in prices.items():
        n = event_count(ev, event)
        o = out.setdefault(phase, [0.0, 0.0])
        o[0] += i * n
        o[1] += w * n
    return out


def by_phase(regions: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for (loop, fn), r in regions.items():
        o = out.setdefault(driver_of(loop, fn)[0], [0.0, 0.0])
        o[0] += r.instructions
        o[1] += r.wavefronts
    return out


#: a scheduler's cycles an MMA by SASS shape on the calibration card (`common/hardware.json`'s product rates; an
#: m16n8k8 half its k16, lab `mma_k8`), and the schedulers an SM
PIPE = {"HMMA.16816.F32": 32.132, "HMMA.16816.F16": 16.131, "HMMA.1688.F32": 16.066, "HMMA.1688.F16": 8.066}
SCHEDULERS = 4
#: each model phase's phase-ledger columns (`fills` waits on its copies in `landed`; the bookkeeping runs in `wait`)
LEDGER = {"readout": ("readout",), "fold": ("fold",), "handoff": ("maxima", "store"), "fills": ("fill", "landed"),
          "drain": ("drain",), "glue": ("wait",)}


def floors(instructions: float, wavefronts: float, pipe: float) -> dict[str, float]:
    """A phase's floors in cycles a CTA-window: its instructions over the schedulers at one a clock, its shared
    wavefronts at one a clock, its MMAs' pipe time over the schedulers."""
    return {"issue": instructions / SCHEDULERS, "shared": wavefronts, "tensor": pipe / SCHEDULERS}


def cycles_of(instructions: float, wavefronts: float, pipe: float, eff: float) -> float:
    """A phase's cycles a CTA-window: its issue floor stretched by its efficiency (each warp's phase a dependent chain, 2
    warps a scheduler), its shared and tensor floors beneath it."""
    f = floors(instructions, wavefronts, pipe)
    return max(f["issue"] * eff, f["shared"], f["tensor"])


def phase_counts(regions: dict) -> dict[str, list]:
    """Each phase's [instructions, wavefronts, MMA pipe cycles] from measured regions."""
    out: dict[str, list] = {}
    for (loop, fn), r in regions.items():
        o = out.setdefault(driver_of(loop, fn)[0], [0.0, 0.0, 0.0])
        o[0] += r.instructions
        o[1] += r.wavefronts
        o[2] += sum(n * PIPE[k] for k, n in (r.mmas or {}).items())
    return out


def imbalance(ev: Events) -> dict[str, float]:
    """Each phase's slowest warp over its mean warp (a phase ends at a rendezvous, so its slowest warp sets its time)."""
    return {"readout": ev.read_imbalance, "fold": ev.fold_imbalance}


#: the events whose work differs between a phase's warps (its per-digit and per-list work does not)
VARYING = {"readout": ("read_tiles",), "fold": ("fold_steps16", "fold_steps8", "fold_steps")}


def varying_share(prices: dict, ev: Events) -> dict[str, float]:
    """Each MMA phase's instructions that vary between its warps, a share of the phase's."""
    out = {}
    for phase, events in VARYING.items():
        part = sum(i * event_count(ev, e) for (p, e), (i, _w) in prices.items() if p == phase and e in events)
        whole = sum(i * event_count(ev, e) for (p, e), (i, _w) in prices.items() if p == phase)
        out[phase] = part / whole if whole else 0.0
    return out


def idle(own: dict[str, float], share: dict[str, float], ev: Events) -> float:
    """THE RENDEZVOUS' IDLE: a phase ends when its slowest warp does, so the others wait the slowest warp's extra work --
    the phase's varying share times its imbalance less one (the ledger books it in `wait`)."""
    imb = imbalance(ev)
    return sum(own.get(p, 0.0) * share.get(p, 0.0) * (imb[p] - 1.0) for p in imb)


def efficiencies(counts: dict, ledger: dict, ev: Events, prices: dict) -> dict[str, float]:
    """Each phase's measured cycles over its issue floor on a calibration cell; the bookkeeping's after the
    rendezvous' idle is taken out of the wait it is booked in."""
    own = {p: sum(ledger[x] for x in LEDGER[p]) for p in counts}
    own["glue"] = own.get("glue", 0.0) - idle(own, varying_share(prices, ev), ev)
    return {p: own[p] / floors(*c)["issue"] for p, c in counts.items()}


#: THE VALIDATION CELLS: the prices are a listed side's on the first, a dense side's on the second; the rest are held out
CALIBRATION = ("nl64k-alt-k4", "nl64k-dense")
HELD_OUT = ("nl64k-alt-k4-g4", "flagship-cohort-k4", "flagship-alt-k4", "flagship-alt-k16", "flagship-dense")


def profile(cell: str) -> Path:
    """One carry launch's SourceCounters dump on `cell` (cached under build/carve_model by the binary's mtime)."""
    from rola_devtools import process

    so = ROOT / "rola_cu13" / "_C.abi3.so"
    out = ROOT / "build" / "carve_model" / f"{cell}.{int(so.stat().st_mtime)}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    launch = ("import sys, torch; sys.path.insert(0, '.'); from measure.cells import by_name; "
              "from measure.subjects import carry_operands; from rola.ops import carry; "
              f"ops = carry_operands(by_name({cell!r})); [carry.carry(*ops) for _ in range(3)]; "
              "torch.cuda.synchronize()")
    import dev_config

    done = process.run([dev_config.ncu(), "--kernel-name", "regex:carry_kernel", "--launch-skip", "2",
                        "--launch-count", "1", "--section", "SourceCounters", "--page", "source", "--print-source",
                        "sass", "--csv", sys.executable, "-c", launch], cwd=ROOT, timeout=3600)
    out.write_text(done.stdout)
    return out


def mma_pipe(ev: Events) -> dict[str, float]:
    """Each MMA phase's pipe cycles a CTA-window from today's carve's events (the readout's fp16-accumulate on this
    card, the fold's fp32)."""
    return {"readout": ev.read_mmas * PIPE["HMMA.16816.F16"],
            "fold": ev.fold_mmas16 * PIPE["HMMA.16816.F32"] + ev.fold_mmas8 * PIPE["HMMA.1688.F32"]}


def ledger_of(ledgers: Path, cell: str) -> dict:
    """A cell's phase ledger (`phase_ledger.py --json`), refused unless it was taken on the build this tree's binary is:
    its cycles set the phases' efficiencies against counts profiled from this binary."""
    import json

    from rola.ops import carry as carry_ops

    led = json.loads((ledgers / f"ledger_{cell}.json").read_text())
    stamp = carry_ops.build_stamp()
    if led.get("build_stamp") != stamp:
        raise ValueError(f"{ledgers}/ledger_{cell}.json was taken on build {led.get('build_stamp')}, this binary is "
                         f"{stamp}: re-take it (tools/phase_ledger.py {cell} --json ...)")
    return led["per_phase"]


class Model:
    """THE CALIBRATED MODEL: prices and efficiencies from the calibration cells (their SourceCounters dumps and phase
    ledgers), predicting a cell's or a form's window."""

    def __init__(self, ledgers: Path):
        sys.path.insert(0, str(ROOT / "tools"))
        import phase_ledger

        so = ROOT / "rola_cu13" / "_C.abi3.so"
        self.prices, self.effs = {}, {}
        for side, cal in (("listed", CALIBRATION[0]), ("dense", CALIBRATION[1])):
            reg = measured(profile(cal), so, phase_ledger.cta_windows(cal))
            ev = events_today(cell_stats(cal))
            led = ledger_of(ledgers, cal)
            self.prices[side] = calibrate(reg, ev)
            self.effs[side] = efficiencies(phase_counts(reg), led, ev, self.prices[side])

    def window(self, stats: CellStats, read_miss: float = 1.0) -> dict[str, float]:
        """A cell's (or a form's) predicted cycles a CTA-window, by phase; a streamed side's cost beyond its MMA phase
        (`STREAM` of the phase, the readout's with its drain times `read_miss`, and the read ring's depth) booked in
        `fills`."""
        side = "listed" if stats.read[0].listed else "dense"
        ev = events_today(stats)
        pred = predict(self.prices[side], ev)
        pipe = mma_pipe(ev)
        cyc = {p: cycles_of(i, w, pipe.get(p, 0.0), self.effs[side][p]) for p, (i, w) in pred.items()}
        if stats.fold_ring:
            cyc["fills"] = cyc.get("fills", 0.0) + STREAM["fold"] * cyc.get("fold", 0.0)
        if stats.read_ring:
            cyc["fills"] = (cyc.get("fills", 0.0) + read_miss * STREAM["readout"]
                            * (cyc.get("readout", 0.0) + cyc.get("drain", 0.0)) + READ_RING[stats.read_ring])
        cyc["glue"] = cyc.get("glue", 0.0) + idle(cyc, varying_share(self.prices[side], ev), ev)
        return cyc


#: THE CARDS the floors are priced on: each MMA's pipe cycles a scheduler (the readout fp16-accumulate, the fold fp32),
#: and the SM's global path (`common/hardware.json`'s `global`: an L2 line, an atomic request, every SM busy; sm_80's
#: announced from sm_86 -- `L2_BRACKET` prices its line at half as well, the A100's L2 serving more an SM)
CARD_PIPE = {"sm_86": PIPE, "sm_80": {"HMMA.16816.F32": 8.0, "HMMA.16816.F16": 8.0, "HMMA.1688.F32": 4.0,
                                      "HMMA.1688.F16": 4.0}}
L2_BRACKET = 0.5
#: a staged entry's bytes (a resident side's lists), a streamed side's record bytes a stage (read: 8 entry words of 8
#: bytes; fold: a 16-row step's 96)
ENTRY_BYTES, READ_RECORD, FOLD_RECORD = 4, 64, 96
#: a streamed side's producer, warp instructions a stage (the staged labs: `read_stream`'s rest tile 20, `row_stream`'s
#: 16-token step 27), and a copy's write into shared memory, shared-pipe cycles a 128 bytes (`smem_rates`)
PRODUCER = {"readout": 20, "fold": 27}
COPY_WRITE = 0.5


def global_path(stats: CellStats, ev: Events) -> tuple[float, float]:
    """A CTA-window's global path: (128-byte lines from L2, atomic requests). A resident side reads its union's rows
    once and its staged entries; a streamed listed side a row an entry (a pad is not copied) and its stage records; the
    drain a request a drained slot (a slot's half line at d_v 64)."""
    lines = (ev.read_rows * stats.read_row + ev.fold_rows * stats.write_row + ev.staged_entries * ENTRY_BYTES) / 128
    if stats.read_ring:
        lines += (ev.read_entries * 64 + ev.read_stages * READ_RECORD) / 128
    if stats.fold_ring:
        lines += (ev.fold_entries * 128 + ev.fold_stages * FOLD_RECORD) / 128
    return lines, ev.drained


def floors_on(model: Model, stats: CellStats, card: str, l2_scale: float = 1.0) -> dict[str, float]:
    """A form's window at its floors on `card`, a CTA-window: `in_turn` the phases' binding floors summed (their
    instructions and wavefronts from the calibration prices, their MMAs at the card's rates) with the global path a
    floor beside them; `overlapped` the largest of the window's totals -- issue, shared wavefronts, the tensor pipe,
    the global path -- every phase overlapped. A streamed side's producer (`PRODUCER` a stage) and every copy's write
    (`COPY_WRITE`) are added to the prices; the streams' bytes are the global path."""
    sys.path.insert(0, str(ROOT))
    from measure import hardware

    side = "listed" if stats.read[0].listed else "dense"
    ev = events_today(stats)
    pred = {k: list(v) for k, v in predict(model.prices[side], ev).items()}
    if stats.read_ring:
        pred["readout"][0] += PRODUCER["readout"] * ev.read_stages
        pred["readout"][1] += COPY_WRITE * (ev.read_entries * 64 + ev.read_stages * READ_RECORD) / 128
    if stats.fold_ring:
        pred["fold"][0] += PRODUCER["fold"] * ev.fold_stages
        pred["fold"][1] += COPY_WRITE * (ev.fold_entries * 128 + ev.fold_stages * FOLD_RECORD) / 128
    pred.setdefault("fills", [0.0, 0.0])[1] += COPY_WRITE * (ev.read_rows * stats.read_row
                                                              + ev.fold_rows * stats.write_row) / 128
    pipe = CARD_PIPE[card]
    tensor = {"readout": ev.read_mmas * pipe["HMMA.16816.F16"],
              "fold": ev.fold_mmas16 * pipe["HMMA.16816.F32"] + ev.fold_mmas8 * pipe["HMMA.1688.F32"]}
    rates = hardware.resolve(arch=card, product="rtx3080ti" if card == "sm_86" else None)
    lines, requests = global_path(stats, ev)
    glob = lines * rates["global.l2_line"].cycles * l2_scale + requests * rates["global.red_request"].cycles
    in_turn = sum(cycles_of(i, w, tensor.get(p, 0.0), 1.0) for p, (i, w) in pred.items())
    totals = {"issue": sum(i for i, _ in pred.values()) / SCHEDULERS, "shared": sum(w for _, w in pred.values()),
              "tensor": sum(tensor.values()) / SCHEDULERS, "global": glob}
    return {"in_turn": max(in_turn, glob), "overlapped": max(totals.values()), "global": glob, "lines": lines,
            "requests": requests, **{f"total_{k}": v for k, v in totals.items()}}


def rank_cards(model: Model, base: CellStats) -> list[str]:
    """THE FORMS RANKED WITH THE GLOBAL PATH, on each card at the floors: resident and streamed (each read ring), at
    W 512 and 1024, a 512 tokens; each line its phases-in-turn floor, its overlapped floor (the binding total named)
    and its global path."""
    out = []
    for card, scales in (("sm_86", (1.0,)), ("sm_80", (1.0, L2_BRACKET))):
        for scale in scales:
            rows = []
            for form, stats in forms(base):
                for k in (1, 2):
                    f = floors_on(model, merged(stats, k), card, scale)
                    bind = max(("issue", "shared", "tensor", "global"), key=lambda x: f[f"total_{x}"])
                    rows.append((f["overlapped"] / k, f"{base.name:20s} {card} L2 x{scale:.1f} {form:22s} W {WINDOW * k:4d}: "
                                 f"in turn {f['in_turn'] / k:7,.0f}  overlapped {f['overlapped'] / k:7,.0f} ({bind})  "
                                 f"global {f['global'] / k:6,.0f} ({f['lines'] / k:5,.0f} lines, "
                                 f"{f['requests'] / k:4,.0f} atomics)  tensor {f['total_tensor'] / k:6,.0f}  "
                                 f"shared {f['total_shared'] / k:6,.0f}  issue {f['total_issue'] / k:6,.0f}"))
            out += [line for _, line in sorted(rows)]
    return out


def rank(model: Model, base: CellStats) -> list[str]:
    """THE FORMS RANKED on this card at a cell: each -- resident, or streamed at each read ring -- at W 512 and 1024,
    best first, its cycles a 512 tokens against today's at the calibration cells' efficiencies, and with every phase at
    its floor (the phases still in turn)."""
    side = "listed" if base.read[0].listed else "dense"
    ref = sum(model.window(base).values())
    rows = []
    for form, stats in forms(base):
        for k in (1, 2):
            form_k = merged(stats, k)
            eff = model.effs[side]
            at_eff = sum(model.window(form_k).values()) / k
            model.effs[side] = dict.fromkeys(eff, 1.0)
            at_floor = sum(model.window(form_k).values()) / k
            model.effs[side] = eff
            rows.append((at_eff, f"{base.name:20s} {form:24s} W {WINDOW * k:5d}: {at_eff / ref:5.2f}x today, "
                                 f"phases at their floors {at_floor / ref:5.2f}x"))
    return [line for _, line in sorted(rows)]


def validate(cells: tuple[str, ...] = CALIBRATION + HELD_OUT, ledgers: Path | None = None) -> list[str]:
    """The gate's table: each cell's instructions and wavefronts a CTA-window predicted from the calibration cells'
    prices against its measured counts, by phase; with `ledgers` (a directory of `phase_ledger.py --json` readings,
    `ledger_<cell>.json`, the clock locked), each window's cycles from the phases' floors at the calibration cells'
    efficiencies against the ledger's."""
    sys.path.insert(0, str(ROOT / "tools"))
    import phase_ledger

    so = ROOT / "rola_cu13" / "_C.abi3.so"
    reg = {c: measured(profile(c), so, phase_ledger.cta_windows(c)) for c in cells}
    ev = {c: events_today(cell_stats(c)) for c in cells}
    prices = {"listed": calibrate(reg[CALIBRATION[0]], ev[CALIBRATION[0]]),
              "dense": calibrate(reg[CALIBRATION[1]], ev[CALIBRATION[1]])}
    led = {c: ledger_of(ledgers, c) for c in cells} if ledgers else {}
    effs = ({side: efficiencies(phase_counts(reg[cal]), led[cal], ev[cal], prices[side])
             for side, cal in (("listed", CALIBRATION[0]), ("dense", CALIBRATION[1]))} if ledgers else {})
    rows = []
    for c in cells:
        side = "dense" if not cell_stats(c).read[0].listed else "listed"
        pred, meas = predict(prices[side], ev[c]), by_phase(reg[c])
        if ledgers:
            pipe = mma_pipe(ev[c])
            cyc = {p: cycles_of(i, w, pipe.get(p, 0.0), effs[side][p]) for p, (i, w) in pred.items()}
            cyc["glue"] = cyc.get("glue", 0.0) + idle(cyc, varying_share(prices[side], ev[c]), ev[c])
            got = {p: sum(led[c][x] for x in LEDGER[p]) for p in LEDGER}
            tot_p, tot_m = sum(cyc.values()), sum(led[c].values())
            rows.append(f"{c:20s} {'cycles':12s} {tot_m:9.0f} predicted {tot_p:9.0f} ({100 * (tot_p / tot_m - 1):+5.1f}%)  "
                        + " ".join(f"{p} {100 * (cyc.get(p, 0) / got[p] - 1):+.0f}%" for p in sorted(got) if got[p]))
        for k, what in ((0, "instructions"), (1, "wavefronts")):
            total_m = sum(v[k] for v in meas.values())
            total_p = sum(v[k] for v in pred.values())
            phases = " ".join(f"{ph} {100 * (pred.get(ph, [0, 0])[k] / meas[ph][k] - 1):+.0f}%"
                              for ph in sorted(meas) if meas[ph][k] > 0.02 * total_m)
            rows.append(f"{c:20s} {what:12s} {total_m:9.0f} predicted {total_p:9.0f} "
                        f"({100 * (total_p / total_m - 1):+5.1f}%)  {phases}")
    return rows


def describe(s: CellStats) -> str:
    def side(name, sides):
        u = np.array([x.union for x in sides])
        e = np.stack([x.entries for x in sides])
        per = e.flatten()
        if not sides[0].listed:
            return f"{name} dense: union {u.mean():.0f}, every slot in all {BOX_LISTS} lists"
        return (f"{name}: union {u.mean():.0f} (max {u.max()}), entries {e.sum(1).mean():.0f}, a list mean "
                f"{per.mean():.1f} p50 {np.percentile(per, 50):.0f} p90 {np.percentile(per, 90):.0f} max {per.max()}, "
                f"refetch {np.mean([x.refetch for x in sides]):.2f}x")
    return (f"{s.name} (g {s.group}, {len(s.write)} box-windows)\n  {side('write', s.write)}\n"
            f"  {side('read', s.read)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="the cells' box-window statistics")
    p.add_argument("cells", nargs="+")
    p = sub.add_parser("events", help="today's carve's events a CTA-window on cells")
    p.add_argument("cells", nargs="+")
    p = sub.add_parser("validate", help="the validation table: calibration cells' prices against the held-out cells")
    p.add_argument("--ledgers", type=Path, default=None,
                   help="a directory of ledger_<cell>.json phase-ledger readings (clock locked): adds cycles")

    p = sub.add_parser("forms", help="predicted window cycles of forced forms against the rule's, on a cell")
    p.add_argument("cell")
    p.add_argument("forms", nargs="+")
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("window", help="predicted cycles a 512 tokens at windows of k x 512 tokens, on cells")
    p.add_argument("cells", nargs="+")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2])
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("rank-cards", help="every form on cells at its floors on sm_86 and sm_80, the global path in")
    p.add_argument("cells", nargs="+")
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("rank", help="every form on cells, best first: its cycles a 512 tokens at the calibration cells' "
                                    "efficiencies and with every phase at its floor")
    p.add_argument("cells", nargs="+")
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("measured", help="a launch's per-region counts a CTA-window, from its SourceCounters dump")
    p.add_argument("csv", type=Path)
    p.add_argument("--so", type=Path, default=ROOT / "rola_cu13" / "_C.abi3.so")
    p.add_argument("--cta-windows", type=int, required=True)
    p.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    if a.cmd == "stats":
        for name in a.cells:
            print(describe(cell_stats(name)))
    elif a.cmd == "events":
        for name in a.cells:
            ev = events_today(cell_stats(name))
            print(name, " ".join(f"{k} {v:.1f}" for k, v in ev.__dict__.items()))
    elif a.cmd == "validate":
        print("\n".join(validate(ledgers=a.ledgers)))
    elif a.cmd == "forms":
        model = Model(a.ledgers)
        base = sum(model.window(cell_stats(a.cell)).values())
        print(f"{a.cell} rule: {base:.0f} cycles a CTA-window")
        for form in a.forms:
            cyc = model.window(cell_stats(a.cell, form))
            print(f"  {form:18s} {sum(cyc.values()):8.0f} ({sum(cyc.values()) / base:.3f}x)  "
                  + " ".join(f"{k} {v:.0f}" for k, v in sorted(cyc.items())))
    elif a.cmd == "window":
        model = Model(a.ledgers)
        for name in a.cells:
            base = cell_stats(name)
            ref = sum(model.window(base).values())
            for form, stats0 in forms(base):
                for k in a.k:
                    form_k = merged(stats0, k)
                    cyc = model.window(form_k)
                    tot = sum(cyc.values()) / k
                    beside = (f"  fold ring beside the last panel {100 * ring_beside_panel(form_k, base.area):.0f}%"
                              if form_k.fold_ring else "")
                    if form_k.read_ring:
                        worse = sum(model.window(form_k, LAB_MISS).values()) / k
                        beside += f"  read stream at {LAB_MISS}x its lab factor {worse / ref:.2f}x"
                    print(f"{name:20s} {form:24s} W {WINDOW * k:5d}: {tot:8.0f} a 512 tokens ({tot / ref:.2f}x)  "
                          + " ".join(f"{p} {v / k:.0f}" for p, v in sorted(cyc.items())) + beside)
    elif a.cmd == "rank-cards":
        model = Model(a.ledgers)
        for name in a.cells:
            print("\n".join(rank_cards(model, cell_stats(name))))
    elif a.cmd == "rank":
        model = Model(a.ledgers)
        for name in a.cells:
            for line in rank(model, cell_stats(name)):
                print(line)
    elif a.cmd == "measured":
        regions = measured(a.csv, a.so, a.cta_windows)
        total = sum(r.instructions for r in regions.values())
        print(f"{total:.0f} warp instructions a CTA-window, "
              f"{sum(r.wavefronts for r in regions.values()):.0f} shared wavefronts")
        for (loop, fn), r in sorted(regions.items(), key=lambda kv: -kv[1].instructions)[:a.top]:
            mm = " ".join(f"{k} {v:.0f}" for k, v in sorted((r.mmas or {}).items()))
            print(f"  loop {'-' if loop is None else loop:>3} {fn:22s} instr {r.instructions:8.0f} "
                  f"wf {r.wavefronts:7.0f} {mm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
