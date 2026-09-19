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
    arrivers: dict[int, set] = collections.defaultdict(set)
    for j, ms in mrefs.items():
        pc = int(recs["pc"][j])
        if pc in arrive_pcs:
            live = int(recs["active"][j]) & int(recs["pred"][j])
            if live:
                lane = (live & -live).bit_length() - 1
                arrivals[int(ms[0][lane])].append((int(recs["clock"][j]), bin(live).count("1"), pc))
                arrivers[int(ms[0][lane])].add(int(recs["warp"][j]))

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

    #: a PRIVATE word has one arriving warp (a warp's own ring): its waits are for the warp's own latest arrivals,
    #: however many tiles the warp took -- a relative gate, where a shared word's phase is absolute
    private = {word for word in words if len(arrivers[word]) == 1}
    out = {"polarity": polarity, "poll_events": poll_events, "counts": counts, "private": private, "words": {}}
    for word in words:
        succ = collections.Counter(e[0] for e in events if e[1] == word and poll_events[e[4]][1])
        out["words"][word] = {"arrive_sites": sorted({pc for _c, _l, pc in arrivals[word]}), "arrivals": len(arrivals[word]),
                              "lanes": int(sum(l for _c, l, _p in arrivals[word])), "waiters": len(succ),
                              "polls": sum(1 for e in events if e[1] == word), "count": counts[word],
                              "successes": sum(succ.values())}
    return out


#: the replay's machine beyond `sass_control.MACHINES`: a copy lands `COPY_LATENCY` cycles after it issues (calibration
#: row `async_copy_latency_1w`: one 16-byte copy a lane from L2, 325; DRAM is not yet a row) and holds the SM's memory pipe `COPY_BASE + COPY_LINE` a distinct global line its live lanes read
#: (calibration.md's sixteen-a-group copy rows, eight warps: sixteen lines 80.9 cycles a warp, five lines 55.3, so
#: 5.4 + 0.29 a line of the SM's pipe; a dead lane's zero fill is free); a CTA barrier releases `BAR_RELEASE` after
#: its last arrival
COPY_LATENCY = 325.0
#: where a deferred barrier blocks: a convergence or control instruction, or any memory access (the barrier orders
#: memory, so none may issue before it releases)
BLOCKS_AT = frozenset({"BSSY", "BSYNC", "BRA", "WARPSYNC", "EXIT", "BAR", "LDS", "STS", "LDSM", "LDG", "STG", "LDGSTS",
                       "ATOMS", "ATOM", "ATOMG", "RED", "ARRIVES", "LD", "ST"})
COPY_BASE = 5.4
COPY_LINE = 0.29
BAR_RELEASE = 20.0
#: a taken branch: its target issues `TAKEN_BRANCH` cycles after it (calibration row `branch_taken_1w`, 10.9: a loop of
#: taken uniform branches, branch to branch), and it holds the scheduler's branch path `BRANCH_PORT` cycles
#: (`branch_taken_2w`, 14.0 a warp with two warps a scheduler: 7.0 a branch)
TAKEN_BRANCH = 10.9
#: how a scheduler breaks a tie: `pipe` (an HMMA tie to the tensor pipe's owner, else least recently issued), `greedy`
#: (the scheduler's last issuer), `fair` (least recently issued); set by the arbitration rows
ARBITRATION = "pipe"
#: where a global access's issue cost (`sass_control.MACHINES` `issue`: a reduction 12, a global load or store 4) is
#: paid: `warp` (the issuing warp alone: the kernel's census shows a drain's partner blocked 2% of the time, and the
#: drain reads 1.01 / 1.08 this way against 1.15 / 1.16 on the port) or `port` (the scheduler's issue port)
GLOBAL_ISSUE = "warp"
BRANCH_PORT = 7.0
CONTROL = frozenset({"BRA", "JMP", "JMX", "BRX", "CALL", "RET", "BSYNC"})
#: a reconvergence: the instruction after a `BSYNC` issues `RECONVERGE` cycles after it (measured in place: the
#: census's branch-resolution wait at the instruction after ~50 `BSYNC` sites, a median 13.2 and 13.9 cycles an
#: execution on two cells; ptxas predicates the probe rows' blocks, so no row isolates it yet)
RECONVERGE = 13.0
#: memory instructions: a predicated-off one still waits for room in the memory queue (the census's `mio` at
#: `@!PT LDS RZ, [RZ]`)
MEMORY_OPS = frozenset({"LDS", "STS", "LDSM", "LDG", "STG", "LDGSTS", "ATOMS", "ATOM", "ATOMG", "RED", "ARRIVES",
                        "LD", "ST", "SHFL"})
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
            #: DEFER_BLOCKING: the warp arrives and keeps issuing until an instruction of `BLOCKS_AT` (the census
            #: samples every barrier's wait on the first `BSSY` after it, never on the arithmetic between)
            kind, extra = "bar", (int(ids[0], 16) if ids else 0, int(ids[1], 16) // 32 if len(ids) > 1 else 0,
                                  "DEFER_BLOCKING" in text)
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
                word = polls[j][0]
                gate = (word, -1) if word in learned["private"] else (word, polls[j][4])
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


def real_order(seqs: list, table: dict, child, real) -> tuple[list, list, dict]:
    """Each warp's replay stream re-ordered into the REAL run's order of activities: the stream is cut at its clock
    reads into segments, one an interval of the traced run, keyed by (window, event, ordinal); the in-window segments
    are issued in the order the real run's stamps name the same keys, and the rest keep their traced order. An
    activity's content does not depend on timing (the k-th walk is chunk k's), its place does: the traced run is ~1000x
    slower, so its fills fire at other polls and it waits where the real run does not. A successful poll LICENSES the
    activity after it, so its gate moves to the next traced segment's first instruction and travels with it; a real
    activity with no traced instance (a wait the traced run did not have) is issued as its own clock read, so its time
    is booked where the real stamps book it. Returns the streams, each stream's key per clock read (None outside the
    windows) and what could not be placed."""
    from warp_trace import keyed

    out, keys_out = [], []
    report = collections.Counter()
    for w, seq in enumerate(seqs):
        cuts = [i for i, item in enumerate(seq) if table[item[0]][1] == "clock"]
        raw = child[w]
        raw = raw[raw != 0]
        ckey, _ = keyed((raw >> 8).tolist(), (raw & 0xFF).tolist())
        by_index = {j: key for key, (j, _s, _e) in ckey.items()}
        segs = [list(seq[:cuts[0]])] + [list(seq[cuts[k]:cuts[k + 1] if k + 1 < len(cuts) else len(seq)])
                                        for k in range(len(cuts))]
        seg_key = [None] + [by_index.get(k) for k in range(len(cuts))]

        #: every successful poll's gate moves onto the next segment's first instruction
        carry: list = []
        for i, sg in enumerate(segs):
            if carry and sg:
                pc0, live0, gate0, arr0, lines0 = sg[0]
                gates = tuple(g for g in ((gate0,) if gate0 and not isinstance(gate0[0], tuple) else (gate0 or ())) if g)
                sg[0] = (pc0, live0, tuple(carry) + gates, arr0, lines0)
                carry = []
            for n, item in enumerate(sg):
                if item[2] is not None and not (n == 0 and i > 0 and isinstance(item[2][0], tuple)):
                    carry.append(item[2])
                    sg[n] = (item[0], item[1], None, item[3], item[4])
        clock_pc = {}
        for i, key in enumerate(seg_key):
            if key is not None and segs[i]:
                clock_pc.setdefault(key[1], segs[i][0][0])

        rr = real[w]
        rr = rr[rr != 0]
        rkey, _ = keyed((rr >> 8).tolist(), (rr & 0xFF).tolist())
        real_keys = [key for key, _v in sorted(rkey.items(), key=lambda kv: kv[1][0])]
        have = {k: i for i, k in enumerate(seg_key) if k is not None}
        first = min(have.values()) if have else len(segs)
        last = max(have.values()) if have else -1
        new, nkeys = [], []
        for i in range(first):
            new.extend(segs[i])
            nkeys.extend([None] * (1 if i else 0))
        pending: list = []
        used = set()
        for key in real_keys:
            i = have.get(key)
            if i is None:
                report["real activity with no traced instance (its clock read issued alone)"] += 1
                if key[1] in clock_pc:
                    new.append((clock_pc[key[1]], True, None, None, None))
                    nkeys.append(key)
                continue
            sg = list(segs[i])
            used.add(i)
            if pending and sg:
                pc0, live0, gate0, arr0, lines0 = sg[0]
                sg[0] = (pc0, live0, tuple(pending) + (gate0 or ()), arr0, lines0)
                pending = []
            new.extend(sg)
            nkeys.append(key)
        for i in range(first, last + 1):
            if i not in used and seg_key[i] is not None:
                report["traced activity the real run lacks (dropped)"] += 1
                if any(item[3] is not None or table[item[0]][1] in ("bar", "copy") for item in segs[i]):
                    report["...of them carrying arrives, barriers or copies (appended)"] += 1
                    new.extend(segs[i])
                    nkeys.append(None)
        for i in range(last + 1, len(segs)):
            new.extend(segs[i])
            nkeys.append(None)
        out.append(new)
        keys_out.append(nkeys)
    return out, keys_out, dict(report)


def compare_keyed(sim_stamps: list, keys: list, real) -> dict:
    """Simulated against real when the replay issued the real run's order: simulated interval i (clock read i to i+1)
    is the segment keyed `keys[i]`, compared with the real interval of that key."""
    from warp_trace import keyed

    rows = collections.defaultdict(lambda: [0, 0.0, 0.0])
    windows_sim = collections.defaultdict(float)
    windows_real = collections.defaultdict(float)
    for w in range(len(sim_stamps)):
        sim = sim_stamps[w]
        rr = real[w]
        rr = rr[rr != 0]
        rkey, _ = keyed((rr >> 8).tolist(), (rr & 0xFF).tolist())
        for i in range(len(sim) - 1):
            key = keys[w][i] if i < len(keys[w]) else None
            if key is None or key not in rkey:
                continue
            _j, r0, r1 = rkey[key]
            row = rows[key[1]]
            row[0] += 1
            row[1] += sim[i + 1] - sim[i]
            row[2] += r1 - r0
            windows_sim[key[0]] += (sim[i + 1] - sim[i]) / len(sim_stamps)
            windows_real[key[0]] += (r1 - r0) / len(sim_stamps)
    return {"by_activity": {k: {"n": v[0], "sim": v[1], "real": v[2]} for k, v in rows.items()},
            "windows": [(windows_sim[i], windows_real[i]) for i in sorted(windows_real)]}


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
    branch_free = [0.0] * nsched
    greedy = [-1] * nsched  #: the warp that issued each scheduler's last instruction
    pipe_owner = [-1] * nsched  #: the warp whose HMMA each scheduler's tensor pipe took last
    mem_free = 0.0
    landed: dict[int, list[tuple[float, int]]] = collections.defaultdict(list)  #: word -> [(time, lanes)]
    bars: dict[int, list[float]] = collections.defaultdict(list)  #: barrier id -> arrival times this generation
    bar_release: dict[int, list[float]] = collections.defaultdict(list)  #: id -> release time per generation
    st = [{"i": 0, "t": 0.0, "sb": [0.0] * 6, "sb_src": [""] * 6, "delay": [], "stall_gap": 0, "port_by": "", "bar_pending": None, "last": -1.0, "copy_max": 0.0, "groups": [], "open": 0.0, "bar_gen": collections.Counter(),
           "stamps": [], "waited": 0.0} for _ in seqs]

    own = collections.defaultdict(int)  #: (warp, word) -> lanes the warp has arrived with

    def gate_time(word: int, need: int, w: int) -> float:
        if need == -1:
            #: a private word: the phase the warp's own arrivals so far complete
            need = own[(w, word)] // counts[word]
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
            for one in (gate if isinstance(gate[0], tuple) else (gate,)):
                g = gate_time(one[0], one[1], w) + POLL_LATENCY
                if g > r:
                    r, why = g, "gate"
        pend = s["bar_pending"]
        if pend is not None and (not pend[2] or op in BLOCKS_AT):
            bid, gen = pend[0], pend[1]
            if len(bar_release[bid]) <= gen:
                return inf, "bar"
            if bar_release[bid][gen] > r:
                r, why = bar_release[bid][gen], "bar"
        if op == "HMMA":
            if pipe_done[w % nsched] > r:
                r, why = pipe_done[w % nsched], "pipe"
        if (_cost or op in MEMORY_OPS) and mem_free - mach["mem_queue"] > r:
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
        #: the earliest ready warp issues. A tie between HMMAs goes to the TENSOR PIPE'S OWNER, the warp whose HMMA the
        #: pipe took last (calibration rows `pair_phase_*`: a pair settles a burst apart, the pipe handed over where a
        #: burst leaves a tail and kept where it does not); any other tie goes to the warp that issued least recently
        best, bt, bwhy, bkey = None, inf, "", None
        for w in range(len(seqs)):
            t, why = ready_of(w)
            if t == inf:
                continue
            k_w = w % nsched
            is_hmma = table[seqs[w][st[w]["i"]][0]][0] == "HMMA"
            owner = ARBITRATION == "pipe" and is_hmma and pipe_owner[k_w] == w
            last_issuer = ARBITRATION == "greedy" and greedy[k_w] == w
            key = (t, 0 if (owner or last_issuer) else 1, st[w]["last"])
            if bkey is None or key < bkey:
                best, bt, bwhy, bkey = w, t, why, key
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
        nxt = seqs[w][s["i"] + 1][0] if s["i"] + 1 < len(seqs[w]) else None
        taken = op in CONTROL and nxt is not None and nxt != pc + 16
        if taken and branch_free[k] > t:
            t, why = branch_free[k], "branch"
        if port_free[k] > t:
            t, why = port_free[k], "port"
            s["port_by"] = f"{'partner' if port_owner[k][0] != w else 'self'}:{port_owner[k][1]}"
        pend = s["bar_pending"]
        if pend is not None and (not pend[2] or op in BLOCKS_AT):
            s["bar_pending"] = None
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
            pipe_owner[k] = w
        done = t + lat
        taken_at = None
        if cost and live:
            taken = max(float(t), mem_free)
            taken_at = taken
            mem_free = taken + cost
            done = taken + lat
        hold = mach["issue"].get(op, 1)
        port_free[k] = t + (hold if GLOBAL_ISSUE == "port" else 1)
        if kind == "copy" and live:
            land = t + COPY_LATENCY
            s["open"] = max(s["open"], land)
            s["copy_max"] = max(s["copy_max"], land)
        elif kind == "commit":
            s["groups"].append(s["open"] or t)
            s["open"] = 0.0
        elif kind == "copy_arrive" and arrive is not None:
            landed[arrive[0]].append((max(t + lat, s["copy_max"]), arrive[1]))
            own[(w, arrive[0])] += arrive[1]
        elif kind == "arrive" and arrive is not None:
            landed[arrive[0]].append((t + lat, arrive[1]))
            own[(w, arrive[0])] += arrive[1]
        elif kind == "clock":
            s["stamps"].append(t)
        if gate is not None:
            s["waited"] += max(0.0, t - s["t"])
        #: a scoreboard COUNTS its producers: it releases when every one has completed, not the last one set
        if ctl["wr"] < 6:
            if done > s["sb"][ctl["wr"]]:
                s["sb"][ctl["wr"]] = done
                s["sb_src"][ctl["wr"]] = op
        if ctl["rd"] < 6:
            #: a memory instruction reads its registers when the memory pipe takes it
            read = (taken_at if taken_at is not None else t) + min(lat, 20)
            if read > s["sb"][ctl["rd"]]:
                s["sb"][ctl["rd"]] = read
                s["sb_src"][ctl["rd"]] = op + "(read)"
        #: the cycles since the warp's last issue: the stall count's share is `wait` (the profiler's fixed-latency
        #: stall), the rest is what `why` names
        gap_stall = s["stall_gap"]
        s["delay"].append((s["i"], t - t0, why, t, gap_stall, s["port_by"]))
        s["stall_gap"] = max(1, ctl["stall"]) - 1
        s["port_by"] = ""
        port_owner[k] = (w, op)
        s["last"] = t
        greedy[k] = w
        s["t"] = t + max(1, ctl["stall"], hold if GLOBAL_ISSUE == "warp" else 1)
        if taken:
            branch_free[k] = t + BRANCH_PORT
            s["t"] = max(s["t"], t + TAKEN_BRANCH)
        elif op == "BSYNC":
            s["t"] = max(s["t"], t + RECONVERGE)
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
             "branch": "branch_resolving",
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


def per_instruction(csv_path: Path, seqs: list, labels: list, delays: list, table: dict, sass: dict, arch: str,
                    real_per_warp: float, top: int) -> None:
    """Instruction by instruction: the cycles a warp spends at each instruction in the model (the wait before it
    issues, its stall-count wait and its issue cycle) beside the census's (its share of the launch's samples, scaled
    to the real run's cycles a warp), with the reasons each side gives. A census sample is a warp-cycle at an
    instruction; the replay's cycles at an instruction are the same quantity."""
    from region_ledger import read_source_counters
    from warp_trace import source_lines

    rows = read_source_counters(csv_path)
    base = min(r[0] for r in rows)
    samples = {r[0] - base: (r[2], r[3]) for r in rows}
    total_samples = sum(v[0] for v in samples.values())
    lines = source_lines(arch)
    model = collections.Counter()
    model_why = collections.defaultdict(collections.Counter)
    where = collections.defaultdict(collections.Counter)
    for w, dl in enumerate(delays):
        for i, d, why, _t, stall_gap, _port in dl:
            pc = seqs[w][i][0]
            model[pc] += (d + stall_gap + 1) / len(seqs)
            if d > 0:
                model_why[pc][model_reason(why)] += d / len(seqs)
            if stall_gap:
                model_why[pc]["wait"] += stall_gap / len(seqs)
            where[pc][labels[w][i]] += 1
    census = {pc: v[0] / total_samples * real_per_warp for pc, v in samples.items()}
    pcs = set(model) | {pc for pc in census if census[pc] > 0}
    ranked = sorted(pcs, key=lambda pc: -abs(model.get(pc, 0.0) - census.get(pc, 0.0)))[:top]
    print(f"\\nPER INSTRUCTION: cycles a warp over the run, the model beside the census ({len(pcs)} instructions; "
          f"model {sum(model.values()):.0f}, census {sum(census.values()):.0f})")
    print(f"{'pc':>6s} {'activity':14s} {'line':22s} {'model':>7s} {'census':>7s}  {'SASS':38s} model reasons | census reasons")
    for pc in ranked:
        act = where[pc].most_common(1)[0][0] if where[pc] else "-"
        m, c = model.get(pc, 0.0), census.get(pc, 0.0)
        mr = ", ".join(f"{k} {100 * v / max(m, 1e-9):.0f}" for k, v in model_why[pc].most_common(2))
        cs = samples.get(pc, (0, {}))[1]
        ct = sum(cs.values()) or 1
        cr = ", ".join(f"{k} {100 * v / ct:.0f}" for k, v in sorted(cs.items(), key=lambda kv: -kv[1])[:2] if v)
        print(f"{pc:6d} {act:14s} {lines.get(pc, ('?', '?'))[0]:22s} {m:7.0f} {c:7.0f}  {sass[pc][0][:38]:38s} {mr} | {cr}")


def pair_overlap(sim_stamps: list, keys: list, real, nwarps: int) -> None:
    """For each fold fragment run: its duration and the share of it during which the scheduler's other warp was also in
    an HMMA activity (a fragment run or a readout tile), the real run beside the replay. A pair in step shares the pipe
    the whole run (two warps' HMMAs); a pair out of step gives each warp the pipe alone part of the time."""
    from warp_trace import keyed

    burst = ("fold.fragment", "readout.tile")

    def spans(get):
        return [[(s0, s1, lab) for (s0, s1, lab) in get(w)] for w in range(nwarps)]

    def real_ints(w):
        rr = real[w]
        rr = rr[rr != 0]
        rkey, rwins = keyed((rr >> 8).tolist(), (rr & 0xFF).tolist())
        base = [t0 for (t0, _t1) in rwins]
        return [(base[k[0]] + a, base[k[0]] + b, k[1]) for k, (_j, a, b) in rkey.items()]

    def sim_ints(w):
        sim = sim_stamps[w]
        return [(sim[i], sim[i + 1], keys[w][i][1]) for i in range(len(sim) - 1) if i < len(keys[w]) and keys[w][i]]

    for name, get in (("real", real_ints), ("replay", sim_ints)):
        sp = spans(get)
        durs, shares = [], []
        for w in range(nwarps):
            partner = sp[(w + 4) % nwarps] if nwarps == 8 else []
            pb = sorted((a, b) for a, b, lab in partner if lab in burst)
            for a, b, lab in sp[w]:
                if lab != "fold.fragment" or b <= a:
                    continue
                ov = sum(max(0.0, min(b, y) - max(a, x)) for x, y in pb if y > a and x < b)
                durs.append(b - a)
                shares.append(ov / (b - a))
        durs.sort()
        shares.sort()
        n = len(durs)
        print(f"  fold fragment runs, {name:6s}: median {durs[n // 2]:.0f} cycles, partner in an HMMA activity "
              f"{100 * sum(shares) / n:.0f}% of a run (median {100 * shares[n // 2]:.0f}%)")


def as_timeline(cell: str, sim_stamps: list, keys: list, seqs: list, table: dict) -> dict:
    """The replay in `tools/warp_timeline.py`'s record format: per warp, per window, each interval's event, cycles from
    the window's first replayed interval, and its instruction mix, so the timeline page draws it beside the real run."""
    from warp_timeline import summaries
    from warp_trace import CLASSES

    warps = []
    for w, sim in enumerate(sim_stamps):
        cut = [i for i, item in enumerate(seqs[w]) if table[item[0]][1] == "clock"]
        by_window: dict[int, list] = collections.defaultdict(list)
        for i in range(len(sim) - 1):
            key = keys[w][i] if i < len(keys[w]) else None
            if key is None:
                continue
            seg = seqs[w][cut[i] + 1:cut[i + 1]] if i + 1 < len(cut) else []
            mix = collections.Counter(CLASSES.get(table[item[0]][0], "alu") for item in seg)
            mix["n"] = len(seg)
            by_window[key[0]].append((sim[i], sim[i + 1], key, dict(mix)))
        wins = []
        for wi in range(max(by_window) + 1 if by_window else 0):
            ivs = sorted(by_window.get(wi, []))
            t0 = ivs[0][0] if ivs else 0.0
            wins.append({"len": (ivs[-1][1] - t0) if ivs else 0.0,
                         "ivs": [{"ev": k[1], "k": k[2], "t0": a - t0, "t1": b - t0, "mix": m} for a, b, k, m in ivs]})
        warps.append({"warp": w, "windows": wins})
    return {"cell": f"{cell} replay", "warps": warps, "windows": summaries(warps), "unmatched_intervals": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--cta", type=int, default=0)
    ap.add_argument("--learn-only", action="store_true")
    ap.add_argument("--census", type=Path, default=None, help="`auto` or a stall_census export: measured wavefronts")
    ap.add_argument("--json", type=Path, default=None, help="write the replay as a timeline record (real order only)")
    ap.add_argument("--order", choices=("real", "traced"), default="real",
                    help="issue each warp's activities in the real run's order (validation) or the traced run's")
    ap.add_argument("--per-instruction", type=int, default=0, metavar="N",
                    help="print the N instructions whose modelled cycles differ most from the census's")
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
    child = torch.load(stem.with_suffix(".child.pt"))[a.cta]
    real = torch.load(stem.with_suffix(".real.pt"))[a.cta]
    keys = None
    if a.order == "real":
        seqs, keys, report = real_order(seqs, table, child, real)
        print(f"issued in the real run's order: {report}")
    print(f"replay streams: {[len(x) for x in seqs]} instructions a warp (traced {len(recs)} in all)")
    res = simulate(seqs, table, learned["counts"], len(seqs))
    if res["unfinished"]:
        print("UNFINISHED (deadlock or an unresolved wait):", list(res["unfinished"].items())[:8])
    cmp_ = compare_keyed(res["stamps"], keys, real) if keys is not None else compare(res["stamps"], child, real)
    if keys is not None:
        pair_overlap(res["stamps"], keys, real, len(seqs))
        if a.json is not None:
            import json

            a.json.write_text(json.dumps(as_timeline(a.cell, res["stamps"], keys, seqs, table)))
    tot_s = sum(v["sim"] for v in cmp_["by_activity"].values())
    tot_r = sum(v["real"] for v in cmp_["by_activity"].values())
    print(f"{'activity':20s} {'n':>5s} {'real a warp':>11s} {'replay':>9s} {'ratio':>6s}")
    for lab, v in sorted(cmp_["by_activity"].items(), key=lambda kv: -kv[1]["real"]):
        print(f"{lab:20s} {v['n']:5d} {v['real'] / 8:11.0f} {v['sim'] / 8:9.0f} {v['sim'] / max(1, v['real']):6.2f}")
    print(f"{'ALL MATCHED':20s} {'':5s} {tot_r / 8:11.0f} {tot_s / 8:9.0f} {tot_s / max(1, tot_r):6.2f}")
    ws = cmp_["windows"]
    print("windows (real, replay):", [(round(r), round(sv)) for sv, r in ws[:6]], "...")
    if keys is None:
        labels = [intervals_of(seq, table, child[w]) for w, seq in enumerate(seqs)]
    else:
        labels = []
        for w, seq in enumerate(seqs):
            lab, k = [], -1
            for item in seq:
                if table[item[0]][1] == "clock":
                    k += 1
                key = keys[w][k] if 0 <= k < len(keys[w]) else None
                lab.append(key[1] if key else "outside")
            labels.append(lab)
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
    if census is not None and a.per_instruction:
        per_instruction(census, seqs, labels, res["delay"], table, sass, arch, tot_r / len(seqs), a.per_instruction)
    print("\nthe model's issue-port waits by (the waiting instruction, who held the port), cycles a warp:")
    for lab in ("readout.tile", "fold.fragment", "readout.drain", "fold.fill"):
        print(f"  {lab:14s}", [(k, round(v / 8)) for k, v in port_ops[lab].most_common(5)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
