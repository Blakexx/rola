"""The whole-CTA replay: one CTA's traced instruction streams issued through the calibrated machine model with the
synchronization LEARNED from the trace, validated interval by interval against the real run's stamps and activity by
activity against the census's stall reasons.

    python tools/replay.py CELL [--census auto] [--learn-only]

The trace is the one `tools/warp_timeline.py --addrs --keep-trace` keeps. `learn` reads the mbarrier protocol off the
trace alone: the barrier words from the arrives' addresses, the counts and phases from the words' own values (the
tracer records what each poll's compare read), the outcomes from the predicate registers, the polarity from the spins.
`streams` collapses spins and gates the successful polls; `simulate` issues the eight warps; `compare` reads the
replay's stamps against the real ones. See docs/internals/tools/replay.md.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev_config  # noqa: E402
import sass_control as sc  # noqa: E402
from warp_trace import HEADER, MREF_BYTES, MREF_TAG, VALUE_BYTES, VALUE_TAG  # noqa: E402

SPIN_GAP = 100


def load_all(records: Path, listing: Path):
    """Every record in file order: headers as a structured array, and for the memory operands
    (record index -> list of 32 lane addresses per operand), joined to the LAST header of the same warp."""
    sass = {}
    for line in listing.read_text().splitlines():
        if line.startswith("#"):
            continue
        off, _opid, _nmref, text, loc = line.split("\t")
        sass[int(off)] = (text, loc)
    buf = np.fromfile(records, dtype=np.uint8)
    heads, mrefs = [], collections.defaultdict(list)
    value_of: dict[int, int] = {}  #: record index -> the value its compare read
    last_of_warp: dict[int, int] = {}
    i, n = 0, len(buf)
    while i + HEADER.itemsize <= n:
        tag = int.from_bytes(buf[i:i + 4].tobytes(), "little")
        if (tag & 0xFFFF0000) == VALUE_TAG:
            #: the value follows its compare's own record (the tool inserts the record's call first)
            w = tag & 0xFFFF
            if w in last_of_warp:
                value_of[last_of_warp[w]] = int.from_bytes(buf[i + 8:i + 12].tobytes(), "little")
            i += VALUE_BYTES
            continue
        if (tag & 0xFFFF0000) == MREF_TAG:
            if i + MREF_BYTES > n:
                break
            w = tag & 0xFFFF
            addrs = np.frombuffer(buf[i + 8:i + MREF_BYTES].tobytes(), dtype="<u8")
            if w in last_of_warp:
                mrefs[last_of_warp[w]].append(addrs.copy())
            i += MREF_BYTES
            continue
        rec = np.frombuffer(buf[i:i + HEADER.itemsize].tobytes(), dtype=HEADER)[0]
        w = int(rec["warp"])
        last_of_warp[w] = len(heads)
        heads.append(i)
        i += HEADER.itemsize
    idx = np.array(heads, dtype=np.int64)
    packed = np.stack([buf[idx + k] for k in range(HEADER.itemsize)], axis=1).tobytes()
    return np.frombuffer(packed, dtype=HEADER), dict(mrefs), sass, value_of


def opcode(text: str) -> str:
    return sc.opcode(text)


PRED = re.compile(r"^(?:@!?U?P\w+\s+)?[A-Z0-9_.]+\s+(U?P[0-6])\s*,")


def compare_form(text: str) -> str:
    """A compare's encoding with its registers and predicates anonymized: the polarity is learned per form."""
    return re.sub(r"\bU?P[0-6T]\b", "P", re.sub(r"\bU?R\d+\b|\bURZ\b|\bRZ\b", "R", text)).split(";")[0]


def learn(recs, mrefs: dict, sass: dict, value_of: dict, value_sites: dict) -> dict:
    """The CTA's synchronization, learned from the trace alone. BARRIER WORDS: every shared address an arrive names
    (`ATOMS.ARRIVE*`, `ARRIVES.LDGSTSBAR*`). POLLS: every shared load of a barrier word; the tracer records the word
    its compare tested (`value_sites`: compare pc -> load pc). THE WORD is hardware state: its high half is the phase
    bit (bit 31, the completed phases' parity) and the pending arrivals as a negative count, so the COUNT is the
    pending count right after a flip (the largest seen), and a poll's COMPLETIONS are the largest number with the
    phase bit's parity whose landed lanes the issued arrivals cover. OUTCOME: the predicate the compare wrote, read
    in the warp's next record; POLARITY per compare form from the spins (a run's last poll completed, the rest did
    not). A spin is a run of one warp's polls of one word, each within `SPIN_GAP` of its own records."""
    arrive_pcs = {pc for pc, (t, _l) in sass.items() if opcode(t) in ("ATOMS", "ARRIVES") and "ARRIVE" in t}
    arrivals: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)  #: word -> [(clock, lanes, pc)]
    for j, ms in mrefs.items():
        pc = int(recs["pc"][j])
        if pc in arrive_pcs:
            live = int(recs["active"][j]) & int(recs["pred"][j])
            if live:
                lane = (live & -live).bit_length() - 1
                arrivals[int(ms[0][lane])].append((int(recs["clock"][j]), bin(live).count("1"), pc))

    words = sorted(arrivals)
    word_of = {w + off: w for w in words for off in range(0, 8, 4)}

    load_to_cmp = {ld: cmp_ for cmp_, ld in value_sites.items()}
    by_warp = collections.defaultdict(list)
    for j in range(len(recs)):
        by_warp[int(recs["warp"][j])].append(j)

    events = []  #: (warp, word, position, clock, record, compare pc, value, predicate after)
    clock_pcs = {pc for pc, (t, _l) in sass.items() if "CLOCK" in t}
    stamps_before: dict[int, list[int]] = {}
    for w, js in by_warp.items():
        acc, n = [], 0
        for j in js:
            n += int(recs["pc"][j]) in clock_pcs
            acc.append(n)
        stamps_before[w] = acc
    for w, js in by_warp.items():
        for i, j in enumerate(js):
            pc = int(recs["pc"][j])
            if pc not in load_to_cmp or j not in mrefs:
                continue
            live = int(recs["active"][j]) & int(recs["pred"][j])
            if not live:
                continue
            word = word_of.get(int(mrefs[j][0][(live & -live).bit_length() - 1]))
            if word is None:
                continue
            cpc = load_to_cmp[pc]
            m = PRED.match(sass[cpc][0])
            bit = (8 if m.group(1).startswith("U") else 0) + int(m.group(1)[-1]) if m else None
            value = outcome = None
            for k in range(i + 1, min(i + 16, len(js))):
                if int(recs["pc"][js[k]]) == cpc:
                    value = value_of.get(js[k])
                    if bit is not None and k + 1 < len(js):
                        outcome = (int(recs["preds"][js[k + 1]]) >> bit) & 1
                    break
            events.append((w, word, i, int(recs["clock"][j]), j, cpc, value, outcome))

    #: counts off the hardware word
    counts = {}
    for word in words:
        vals = [e[6] for e in events if e[1] == word and e[6] is not None]
        counts[word] = max(0x80000000 - (v & 0x7FFFFFFF) for v in vals) if vals else 0

    #: spins and polarity
    runs = collections.defaultdict(list)
    run_id = {}
    for ev in sorted(events, key=lambda e: (e[0], e[1], e[2])):
        #: a spin: the same poll instruction again, within `SPIN_GAP` of the warp's records, with no stamp between
        key = (ev[0], ev[1])
        prev = runs[key][-1][-1] if runs[key] else None
        if (prev is not None and ev[2] - prev[2] < SPIN_GAP and int(recs["pc"][prev[4]]) == int(recs["pc"][ev[4]])
                and stamps_before[ev[0]][ev[2]] == stamps_before[ev[0]][prev[2]]):
            runs[key][-1].append(ev)
        else:
            runs[key].append([ev])
    votes = collections.defaultdict(collections.Counter)
    n = 0
    for rs in runs.values():
        for r in rs:
            for ev in r:
                run_id[ev[4]] = n
            n += 1
            if len(r) < 2:
                continue
            form = compare_form(sass[r[-1][5]][0])
            for ev in r[:-1]:
                if ev[7] is not None:
                    votes[form][("inner", ev[7])] += 1
            if r[-1][7] is not None:
                votes[form][("last", r[-1][7])] += 1

    polarity = {}
    for form, v in votes.items():
        done = max((0, 1), key=lambda x: v[("last", x)])
        polarity[form] = {"complete": done, "last_agree": v[("last", done)], "last_total": v[("last", 0)] + v[("last", 1)],
                          "inner_agree": v[("inner", 1 - done)], "inner_total": v[("inner", 0)] + v[("inner", 1)]}

    #: each successful poll's completions
    issued = {}
    for word in words:
        ev = sorted(arrivals[word])
        issued[word] = (np.array([c for c, _l, _p in ev]), np.cumsum([l for _c, l, _p in ev]))

    poll_events = {}
    agree = collections.Counter()
    for (w, word, i, clock, j, cpc, value, outcome) in events:
        form = compare_form(sass[cpc][0])
        success = form in polarity and outcome == polarity[form]["complete"]
        need = 0
        if value is not None and counts[word]:
            c = counts[word]
            clocks, lanes_so_far = issued[word]
            k = int(np.searchsorted(clocks, clock, side="left"))
            have = int(lanes_so_far[k - 1]) if k else 0
            current = c - (0x80000000 - (value & 0x7FFFFFFF))
            phase = value >> 31
            need = max(0, (have - current) // c)
            if need % 2 != phase:
                need -= 1
            #: the value's own verdict against the predicate's: a completed poll read the phase it needed
            agree[(success, value is not None)] += 1
        poll_events[j] = (word, success, run_id[j], False, max(0, need))

    out = {"polarity": polarity, "poll_events": poll_events, "counts": counts, "words": {}}
    for word in words:
        succ = collections.Counter(e[0] for e in events if e[1] == word and poll_events[e[4]][1])
        out["words"][word] = {"arrive_sites": sorted({pc for _c, _l, pc in arrivals[word]}), "arrivals": len(arrivals[word]),
                              "lanes": int(sum(l for _c, l, _p in arrivals[word])), "waiters": len(succ),
                              "polls": sum(1 for e in events if e[1] == word), "count": counts[word],
                              "successes": sum(succ.values())}
    return out


#: the replay's machine beyond `sass_control.MACHINES`: a copy lands `COPY_LATENCY` cycles after it issues (the global
#: side, L2 or DRAM) and holds the SM's memory pipe `COPY_BASE + COPY_LINE` a distinct global line its live lanes read
#: (calibration.md's sixteen-a-group copy rows, eight warps: sixteen lines 80.9 cycles a warp, five lines 55.3, so
#: 5.4 + 0.29 a line of the SM's pipe; a dead lane's zero fill is free); a CTA barrier releases `BAR_RELEASE` after
#: its last arrival
COPY_LATENCY = 600.0
COPY_BASE = 5.4
COPY_LINE = 0.29
BAR_RELEASE = 20.0
POLL_LATENCY = 30.0


def static_table(sass: dict, enc: dict, wavefronts: dict) -> dict:
    """pc -> (opcode, kind, control, latency, memory cost, extra): everything the replay reads per instruction."""
    import pipe_sim as ps

    out = {}
    #: the phase clock's constructor mark is a clock read that stamps nothing (`warp_trace.match` drops it too)
    marks = sorted(pc for pc, (t, _l) in sass.items() if "CLOCK" in t)[:1]
    for pc, (text, _loc) in sass.items():
        op = opcode(text)
        ctl = enc[pc][1] if pc in enc else {"stall": 1, "wr": 7, "rd": 7, "wait": 0}
        kind, extra = "", None
        if op == "BAR" and ".SYNC" in text:
            ids = re.findall(r"0x[0-9a-f]+", text.split("BAR.SYNC", 1)[1])
            #: DEFER_BLOCKING: the warp arrives and issues one more instruction before it blocks (the census samples
            #: the barrier's wait on the instruction after the next)
            kind, extra = "bar", (int(ids[0], 16) if ids else 0, int(ids[1], 16) // 32 if len(ids) > 1 else 0,
                                  1 if "DEFER_BLOCKING" in text else 0)
        elif op == "DEPBAR":
            m = re.search(r"SB0,\s*0x([0-9a-f]+)", text)
            kind, extra = "depbar", int(m.group(1), 16) if m else 0
        elif op == "LDGDEPBAR":
            kind = "commit"
        elif op == "LDGSTS":
            kind = "copy"
        elif op == "ARRIVES":
            kind = "copy_arrive"
        elif op == "ATOMS" and "ARRIVE" in text:
            kind = "arrive"
        elif op == "CS2R" and "CLOCK" in text and pc not in marks:
            kind = "clock"
        cost = sc.mem_cost(op, text, ps.MACHINE)
        if pc in wavefronts and op != "LDGSTS":
            cost = wavefronts[pc] * ps.MACHINE["mem_wavefront"]
        out[pc] = (op, kind, ctl, ps.LATENCY.get(op, ps.LATENCY["DEFAULT"]), cost, extra)
    return out


def streams(recs, mrefs: dict, value_of: dict, sass: dict, learned: dict) -> list[list[tuple]]:
    """Each warp's traced stream as the replay issues it: (pc, live, gate) with the failed polls of a spin removed and
    every successful barrier poll gated on (word, completions it observed), and each arrive's (word, lanes)."""
    word_of = {w + off: w for w in learned["counts"] for off in range(0, 8, 4)}
    polls = learned["poll_events"]  #: record index -> (word, success, run id, last in run)
    out = []
    by_warp = collections.defaultdict(list)
    for j in range(len(recs)):
        by_warp[int(recs["warp"][j])].append(j)
    for w in sorted(by_warp):
        js = by_warp[w]

        drop = set()
        runs = collections.defaultdict(list)
        for i, j in enumerate(js):
            if j in polls:
                runs[polls[j][2]].append(i)
        for idx in runs.values():
            if len(idx) >= 2:
                drop.update(range(idx[0], idx[-1]))

        seq = []
        for i, j in enumerate(js):
            if i in drop:
                continue
            pc = int(recs["pc"][j])
            live = int(recs["active"][j]) & int(recs["pred"][j])
            gate = None
            if j in polls and polls[j][1]:
                gate = (polls[j][0], polls[j][4])
            arrive = None
            if j in mrefs and live and opcode(sass[pc][0]) in ("ATOMS", "ARRIVES") and "ARRIVE" in sass[pc][0]:
                lane = (live & -live).bit_length() - 1
                word = word_of.get(int(mrefs[j][0][lane]))
                if word is not None:
                    arrive = (word, bin(live).count("1"))
            lines = None
            if opcode(sass[pc][0]) == "LDGSTS" and j in mrefs and live and len(mrefs[j]) >= 2:
                src = mrefs[j][1]
                lines = len({int(src[b]) >> 7 for b in range(32) if (live >> b) & 1})
            seq.append((pc, live != 0, gate, arrive, lines))
        out.append(seq)
    return out


def simulate(seqs: list[list[tuple]], table: dict, counts: dict, warps_per_cta: int) -> dict:
    """The whole CTA, event by event: the warp with the earliest ready instruction issues next. Returns each warp's
    clock-read times (its simulated stamps) and the waits it could not resolve."""
    import pipe_sim as ps

    mach = ps.MACHINE
    nsched = mach["schedulers"]
    inf = float("inf")
    pipe_done = [0.0] * nsched
    port_free = [0.0] * nsched
    port_owner = [(-1, "")] * nsched
    mem_free = 0.0
    landed: dict[int, list[tuple[float, int]]] = collections.defaultdict(list)  #: word -> [(time, lanes)]
    bars: dict[int, list[float]] = collections.defaultdict(list)  #: barrier id -> arrival times this generation
    bar_release: dict[int, list[float]] = collections.defaultdict(list)  #: id -> release time per generation
    st = [{"i": 0, "t": 0.0, "sb": [0.0] * 6, "sb_src": [""] * 6, "delay": [], "stall_gap": 0, "port_by": "", "bar_pending": None, "last": -1.0, "copy_max": 0.0, "groups": [], "open": 0.0, "bar_gen": collections.Counter(),
           "stamps": [], "waited": 0.0} for _ in seqs]

    def gate_time(word: int, need: int) -> float:
        if need <= 0:
            return 0.0
        target = need * counts[word]
        ev = sorted(landed[word])
        tot = 0
        for t, l in ev:
            tot += l
            if tot >= target:
                return t
        return inf

    def ready_of(w: int) -> tuple[float, str]:
        """The earliest the warp's next instruction can issue, and what sets it: `stall` (the previous instruction's
        stall count), `sb:<producer class>` (a scoreboard), `gate` (a barrier word's phase), `bar`, `depbar`."""
        s = st[w]
        if s["i"] >= len(seqs[w]):
            return inf, ""
        pc, live, gate, _arrive, lines = seqs[w][s["i"]]
        op, kind, ctl, _lat, _cost, extra = table[pc]
        if lines is not None:
            _cost = COPY_BASE + COPY_LINE * lines
        r, why = s["t"], "stall"
        for b in range(6):
            if (ctl["wait"] >> b) & 1 and s["sb"][b] > r:
                r, why = s["sb"][b], "sb:" + s["sb_src"][b]
        if gate is not None:
            g = gate_time(*gate) + POLL_LATENCY
            if g > r:
                r, why = g, "gate"
        pend = s["bar_pending"]
        if pend is not None and pend[2] == 0:
            bid, gen = pend[0], pend[1]
            if len(bar_release[bid]) <= gen:
                return inf, "bar"
            if bar_release[bid][gen] > r:
                r, why = bar_release[bid][gen], "bar"
        if op == "HMMA":
            if pipe_done[w % nsched] > r:
                r, why = pipe_done[w % nsched], "pipe"
        if _cost and live and mem_free - mach["mem_queue"] > r:
            r, why = mem_free - mach["mem_queue"], "memq"
        if kind == "depbar":
            n = extra
            if len(s["groups"]) > n:
                g = max(s["groups"][: len(s["groups"]) - n])
                if g > r:
                    r, why = g, "depbar"
        return r, why

    blocked_report = collections.Counter()
    while True:
        #: the earliest ready warp issues; a tie goes to the warp that issued least recently (the scheduler's
        #: arbitration is fair between its warps: the real stamps have both halves of the CTA in step)
        best, bt, bwhy, blast = None, inf, "", inf
        for w in range(len(seqs)):
            t, why = ready_of(w)
            if t < bt or (t == bt and t < inf and st[w]["last"] < blast):
                best, bt, bwhy, blast = w, t, why, st[w]["last"]
        if best is None:
            live = [w for w in range(len(seqs)) if st[w]["i"] < len(seqs[w])]
            for w in live:
                pc, _l, gate, _a, _n = seqs[w][st[w]["i"]]
                blocked_report[(w, pc, table[pc][1], str(gate))] += 1
            break
        w = best
        s = st[w]
        pc, live, gate, arrive, lines = seqs[w][s["i"]]
        op, kind, ctl, lat, cost, extra = table[pc]
        if lines is not None:
            cost = COPY_BASE + COPY_LINE * lines
        k = w % nsched
        t0 = s["t"]
        why = bwhy
        t = bt
        if port_free[k] > t:
            t, why = port_free[k], "port"
            s["port_by"] = f"{'partner' if port_owner[k][0] != w else 'self'}:{port_owner[k][1]}"
        pend = s["bar_pending"]
        if pend is not None:
            s["bar_pending"] = None if pend[2] == 0 else (pend[0], pend[1], pend[2] - 1)
        if kind == "bar":
            #: the warp arrives now; the generation releases at its last arrival, and the warp blocks `defer`
            #: instructions later
            bid, nwarps, defer = extra
            nwarps = nwarps or warps_per_cta
            gen = s["bar_gen"][bid]
            s["bar_gen"][bid] += 1
            bars[(bid, gen)].append(t)
            if len(bars[(bid, gen)]) == nwarps:
                bar_release[bid].append(max(bars[(bid, gen)]) + BAR_RELEASE)
            s["bar_pending"] = (bid, gen, defer)
        if op == "HMMA":
            pipe_done[k] = t + ps.HMMA_PIPE
        done = t + lat
        if cost and live:
            taken = max(float(t), mem_free)
            mem_free = taken + cost
            done = taken + lat
        port_free[k] = t + mach["issue"].get(op, 1)
        if kind == "copy" and live:
            land = t + COPY_LATENCY
            s["open"] = max(s["open"], land)
            s["copy_max"] = max(s["copy_max"], land)
        elif kind == "commit":
            s["groups"].append(s["open"] or t)
            s["open"] = 0.0
        elif kind == "copy_arrive" and arrive is not None:
            landed[arrive[0]].append((max(t + lat, s["copy_max"]), arrive[1]))
        elif kind == "arrive" and arrive is not None:
            landed[arrive[0]].append((t + lat, arrive[1]))
        elif kind == "clock":
            s["stamps"].append(t)
        if gate is not None:
            s["waited"] += max(0.0, t - s["t"])
        if ctl["wr"] < 6:
            s["sb"][ctl["wr"]] = done
            s["sb_src"][ctl["wr"]] = op
        if ctl["rd"] < 6:
            s["sb"][ctl["rd"]] = t + min(lat, 20)
            s["sb_src"][ctl["rd"]] = op + "(read)"
        #: the cycles since the warp's last issue: the stall count's share is `wait` (the profiler's fixed-latency
        #: stall), the rest is what `why` names
        gap_stall = s["stall_gap"]
        s["delay"].append((s["i"], t - t0, why, t, gap_stall, s["port_by"]))
        s["stall_gap"] = max(1, ctl["stall"]) - 1
        s["port_by"] = ""
        port_owner[k] = (w, op)
        s["last"] = t
        s["t"] = t + max(1, ctl["stall"])
        s["i"] += 1
    return {"stamps": [x["stamps"] for x in st], "end": [x["t"] for x in st], "delay": [x["delay"] for x in st],
            "unfinished": {f"warp {w} pc {pc} {kind} gate {g}": n for (w, pc, kind, g), n in blocked_report.items()}}


def compare(sim_stamps: list, child, real) -> dict:
    """Simulated against real, interval by interval: the simulated stamps carry the traced run's events (one per clock
    read), keyed by (window, event, ordinal) like the real run's."""
    from warp_trace import keyed

    rows = collections.defaultdict(lambda: [0, 0.0, 0.0])
    per_window = collections.defaultdict(lambda: [0.0, 0.0])
    for w in range(len(sim_stamps)):
        raw = child[w]
        raw = raw[raw != 0]
        ev = (raw & 0xFF).tolist()
        sim = sim_stamps[w]
        if len(sim) != len(ev):
            raise SystemExit(f"warp {w}: {len(sim)} simulated clock reads against {len(ev)} traced stamps")
        skey, swins = keyed([int(x) for x in sim], ev)
        rr = real[w]
        rr = rr[rr != 0]
        rkey, rwins = keyed((rr >> 8).tolist(), (rr & 0xFF).tolist())
        for key, (_j, s0, s1) in skey.items():
            if key in rkey:
                _jr, r0, r1 = rkey[key]
                row = rows[key[1]]
                row[0] += 1
                row[1] += s1 - s0
                row[2] += r1 - r0
        for wi in range(min(len(swins), len(rwins))):
            per_window[wi][0] += (swins[wi][1] - swins[wi][0]) / len(sim_stamps)
            per_window[wi][1] += (rwins[wi][1] - rwins[wi][0]) / len(sim_stamps)
    return {"by_activity": {k: {"n": v[0], "sim": v[1], "real": v[2]} for k, v in rows.items()},
            "windows": [per_window[i] for i in sorted(per_window)]}


#: the model's delay reasons as the profiler names its stall reasons
TO_CENSUS = {"pipe": "math", "stall": "wait", "memq": "mio", "port": "not_selected", "gate": "barrier/poll",
             "bar": "barrier/poll", "depbar": "long_sb", "issue": "selected"}
LONG = ("LDG", "RED", "ATOM", "ATOMG", "LDGSTS", "ARRIVES")


def model_reason(why: str) -> str:
    if why.startswith("sb:"):
        return "long_sb" if why[3:].split("(")[0] in LONG else "short_sb"
    return TO_CENSUS.get(why, why)


def intervals_of(seq: list, table: dict, child_row) -> list[str]:
    """Each instruction of a replay stream labelled with the traced run's activity it falls in (between clock reads)."""
    from phase_trace import intervals

    raw = child_row[child_row != 0]
    ev, cyc = (raw & 0xFF).tolist(), (raw >> 8).tolist()
    labs = [lab for (_s, _e, lab) in intervals(cyc, ev)]
    out, k = [], -1
    for pc, *_rest in seq:
        if table[pc][1] == "clock":
            k += 1
        out.append(labs[k] if 0 <= k < len(labs) else "start")
    return out


def census_mix(csv_path: Path, seqs: list, labels: list) -> dict:
    """The census's stall samples per activity by reason: an instruction's samples split over the activities by the
    replay streams' executions of it (the traced path with spins collapsed)."""
    from region_ledger import read_source_counters

    rows = read_source_counters(csv_path)
    base = min(r[0] for r in rows)
    at = {r[0] - base: r for r in rows}
    per = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for seq, labs in zip(seqs, labels):
        for (pc, *_r), lab in zip(seq, labs):
            per[lab][pc] += 1
            total[pc] += 1
    out = collections.defaultdict(collections.Counter)
    for lab, c in per.items():
        for pc, n in c.items():
            if pc in at and total[pc]:
                share = n / total[pc]
                for reason, v in at[pc][3].items():
                    out[lab][reason] += v * share
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--cta", type=int, default=0)
    ap.add_argument("--learn-only", action="store_true")
    ap.add_argument("--census", type=Path, default=None, help="`auto` or a stall_census export: measured wavefronts")
    a = ap.parse_args()
    import sass as sassmod
    import toolchains
    import torch

    stem = dev_config.scratch("warp_timeline") / f"{a.cell}-cta{a.cta}"
    recs, mrefs, sass, value_of = load_all(stem.with_suffix(".bin"), stem.with_suffix(".tsv"))
    value_sites = {int(l.split()[2]): int(l.split()[3]) for l in stem.with_suffix(".tsv").read_text().splitlines()
                   if l.startswith("# value")}
    print(f"{a.cell}: {len(recs)} records, {len(mrefs)} with addresses, {len(value_of)} compare values")
    learned = learn(recs, mrefs, sass, value_of, value_sites)
    print("polarity, per compare form (the value a spin exits on; inner polls must read the other):")
    for form, p in learned["polarity"].items():
        print(f"  {form[:60]:60s} complete={p['complete']}  last {p['last_agree']}/{p['last_total']}  inner {p['inner_agree']}/{p['inner_total']}")
    print(f"{'word':>6s} {'arrive sites':24s} {'arrivals':>8s} {'lanes':>6s} {'count':>5s} {'phases':>6s} {'waiters':>7s} {'polls':>6s} {'successes':>9s}")
    for addr, r in learned["words"].items():
        phases = r["lanes"] / r["count"] if r["count"] else 0
        print(f"{addr:>6x} {str(r['arrive_sites'])[:24]:24s} {r['arrivals']:8d} {r['lanes']:6d} {r['count']:5d} {phases:6.1f} "
              f"{r['waiters']:7d} {r['polls']:6d} {r['successes']:9d}")
    if a.learn_only:
        return 0
    arch = sassmod.device_arch()
    enc = sc.encoded(sassmod.cubin(toolchains.built_extension(), "carry_arm_0", arch))
    wavefronts = {}
    census = a.census
    if census is not None and str(census) == "auto":
        census = dev_config.scratch("stall_census") / f"{a.cell}.csv"
    if census is not None:
        from region_ledger import read_source_counters, read_source_detail

        rows = read_source_counters(census)
        base = min(r[0] for r in rows)
        execs = {r[0] - base: r[1] for r in rows}
        for addr, (_op, wf, _ideal) in read_source_detail(census).items():
            if execs.get(addr - base):
                wavefronts[addr - base] = wf / execs[addr - base]
    table = static_table(sass, enc, wavefronts)
    seqs = streams(recs, mrefs, value_of, sass, learned)
    print(f"replay streams: {[len(x) for x in seqs]} instructions a warp (traced {len(recs)} in all)")
    res = simulate(seqs, table, learned["counts"], len(seqs))
    if res["unfinished"]:
        print("UNFINISHED (deadlock or an unresolved wait):", list(res["unfinished"].items())[:8])
    child = torch.load(stem.with_suffix(".child.pt"))[a.cta]
    real = torch.load(stem.with_suffix(".real.pt"))[a.cta]
    cmp_ = compare(res["stamps"], child, real)
    tot_s = sum(v["sim"] for v in cmp_["by_activity"].values())
    tot_r = sum(v["real"] for v in cmp_["by_activity"].values())
    print(f"{'activity':20s} {'n':>5s} {'real a warp':>11s} {'replay':>9s} {'ratio':>6s}")
    for lab, v in sorted(cmp_["by_activity"].items(), key=lambda kv: -kv[1]["real"]):
        print(f"{lab:20s} {v['n']:5d} {v['real'] / 8:11.0f} {v['sim'] / 8:9.0f} {v['sim'] / max(1, v['real']):6.2f}")
    print(f"{'ALL MATCHED':20s} {'':5s} {tot_r / 8:11.0f} {tot_s / 8:9.0f} {tot_s / max(1, tot_r):6.2f}")
    ws = cmp_["windows"]
    print("windows (real, replay):", [(round(r), round(sv)) for sv, r in ws[:6]], "...")
    labels = [intervals_of(seq, table, child[w]) for w, seq in enumerate(seqs)]
    model = collections.defaultdict(collections.Counter)
    port_ops = collections.defaultdict(collections.Counter)
    for w, dl in enumerate(res["delay"]):
        for i, d, why, _t, stall_gap, port_by in dl:
            lab = labels[w][i]
            model[lab]["selected"] += 1
            model[lab]["wait"] += stall_gap
            if d > 0:
                model[lab][model_reason(why)] += d
            if why == "port":
                port_ops[lab][(table[seqs[w][i][0]][0], port_by)] += d
    mix = census_mix(census, seqs, labels) if census is not None else {}
    reasons = ("selected", "math", "wait", "short_sb", "long_sb", "mio", "not_selected", "barrier/poll", "lg", "branch_resolving", "no_inst")
    print("\nWHY, per activity: the model's cycles by reason beside the census's samples by reason (each as % of its row)")
    for lab in sorted(model, key=lambda k: -sum(model[k].values()))[:10]:
        m, c = model[lab], mix.get(lab, collections.Counter())
        mt, ct = sum(m.values()) or 1, sum(c.values()) or 1
        cc = collections.Counter()
        for k, v in c.items():
            cc["barrier/poll" if k == "barrier" else k] += v
        print(f"  {lab:18s} model " + " ".join(f"{r.split('/')[0][:8]}={100 * m[r] / mt:3.0f}" for r in reasons if m[r] / mt >= 0.02))
        print(f"  {'':18s} census" + " ".join(f" {r.split('/')[0][:8]}={100 * cc[r] / ct:3.0f}" for r in reasons if cc[r] / ct >= 0.02))
    print("\nthe model's issue-port waits by (the waiting instruction, who held the port), cycles a warp:")
    for lab in ("readout.tile", "fold.fragment", "readout.drain", "fold.fill"):
        print(f"  {lab:14s}", [(k, round(v / 8)) for k, v in port_ops[lab].most_common(5)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
