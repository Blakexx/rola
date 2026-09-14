# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PART HARNESS (KERNEL_STANDARDS §19): each component of the carry kernel alone, on a
cell's real call, checked against a pure reference and timed against the model's budget in
`tools/budgets/carry.json`. The drivers are `carry_parts/carry_parts.cu`, built here from the
repo's headers (a bench instrument, never shipped). Lock the clock first (tools/clock_lock.py).

    python benchmarks/unit/bench_carry_parts.py --part head --cell flagship-alt-k4
    python benchmarks/unit/bench_carry_parts.py --part head --all --ncu   # instructions too

A part's row: its reference check, its time per CTA-window in cycles, and with `--ncu` its
executed warp-instructions per CTA-window beside the budget -- red when over.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

TILE = 16
WINDOW = 512
BOXES = 16


def build():
    """The driver, built AHEAD OF TIME beside the arm (`ROLA_BUILD_PARTS=1`, setup.py); a
    torch JIT of the same sources ran for minutes and was killed by the host's memory watchdog."""
    try:
        from rola import _C_parts
    except ImportError as ex:
        raise SystemExit("the part harness's driver is not built: "
                         "ROLA_BUILD_PARTS=1 ROLA_CUDA_ARCHS=86 ROLA_CARRY_ARMS=0 "
                         "python -m pip install -e . --no-build-isolation") from ex
    return _C_parts


def call_args(spec, schedule_name: str = "first"):
    """The kernel's positional call for a cell -- the one binding `carry_call` makes."""
    from benchmarks.cells import carry_call
    from rola.ops import carry as c

    drawn, kw = carry_call(spec)
    routes, desc, geom, launch = kw["routes"], kw["descriptor"], kw["geometry"], kw["launch"]
    sched = c.CarrySchedule(order=schedule_name)
    num = torch.zeros((1, spec.tokens, desc.DV), dtype=torch.float32, device="cuda")
    den = torch.zeros((1, spec.tokens), dtype=torch.float32, device="cuda")
    args = (routes.read, routes.write, routes.gain, kw["v"], num, den, list(desc.B), int(desc.DV),
            int(desc.page_bits), int(launch.warps_per_cta), geom.carve_order, sched.flat(),
            kw["liveness"].words, kw["activity"])
    return drawn, kw, args


# ------------------------------------------------------------------------ references

def side_shifts(ranks, span_bits):
    """level -> bit offset of its digit in the side's carve order (rank 0 innermost)."""
    D = len(ranks)
    return [sum(span_bits[m] for m in range(D) if ranks[m] < ranks[l]) for l in range(D)]


def box_liveness(words: np.ndarray, g: dict, side: str, owner: int, t0: int, wlen: int) -> np.ndarray:
    """[512, 16] bool: token x box liveness of CTA box `owner` on `side`, window at `t0`,
    from the class-1 words alone -- every level, straddles included."""
    D = g["D"]
    s = g["r"] if side == "read" else g["w"]
    span_bits = [int(x).bit_length() - 1 for x in g["span"][:D]]
    grid = [int(x) for x in g["grid"][:D]]
    owner_div = [int(x) for x in g["owner_div"][:D]]
    shifts = side_shifts(list(s["rank"])[:D], span_bits)
    sidx = 0 if side == "read" else 1
    rounds = (wlen + 31) // 32
    w0 = t0 // 32
    live = np.zeros((WINDOW, BOXES), dtype=bool)
    bits = np.zeros((WINDOW,), dtype=bool)
    for b in range(BOXES):
        any_pos = np.zeros((WINDOW,), dtype=bool)
        for pos in range(TILE):
            leaf = b * TILE + pos
            all_l = np.ones((WINDOW,), dtype=bool)
            for l in range(D):
                digit = (leaf >> shifts[l]) & ((1 << span_bits[l]) - 1)
                odig = (owner // owner_div[l]) % grid[l]
                row = int(g["col_base"][l]) + odig * (1 << span_bits[l]) + digit
                wrow = words[0, sidx, row, w0:w0 + rounds].astype(np.uint32)
                bits[:] = False
                for r in range(rounds):
                    bits[32 * r:32 * r + 32] = ((wrow[r] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool)
                all_l &= bits
            any_pos |= all_l
        live[:, b] = any_pos
    live[wlen:] = False
    return live


def head_reference(live: np.ndarray, wlen: int):
    """(order tokens, tile masks, live count) of the first-live-box sort: stable by token."""
    masks = (live.astype(np.int64) << np.arange(BOXES)).sum(1)
    key = np.where(masks != 0, np.argmax(live, axis=1), BOXES)
    order = np.lexsort((np.arange(WINDOW), key))
    tm = np.bitwise_or.reduce(masks[order].reshape(-1, TILE), axis=1)
    return order, tm, int((key < BOXES).sum()), masks


def words_of(live: np.ndarray) -> np.ndarray:
    """[16 rounds] uint32: bit t of round r set when token 32 r + t is live in ANY column."""
    bits = live.any(1).astype(np.uint64).reshape(16, 32)
    return (bits << np.arange(32, dtype=np.uint64)).sum(1).astype(np.uint32)


def check_head(spec, kw, out, owner_count: int, warps: int = 8):
    from rola.ops import carry as c

    order, tilemask, livec, warpwords, unionw, prefix = (t.cpu().numpy() for t in out)
    g = c.geometry(list(spec.widths), level_modes=spec.level_modes, bc=256, nsr=8, nsw=8)
    words = kw["liveness"].words.cpu().numpy()
    L = spec.tokens
    bad = 0
    for owner in range(owner_count):
        for win in range((L + WINDOW - 1) // WINDOW):
            t0 = win * WINDOW
            wlen = min(WINDOW, L - t0)
            rl = box_liveness(words, g, "read", owner, t0, wlen)
            ref_order, ref_tm, ref_live, _ = head_reference(rl, wlen)
            wl = box_liveness(words, g, "write", owner, t0, wlen)
            ref_union = words_of(wl)
            ref_ww = np.stack([words_of(wl[:, [w, w + warps]]) for w in range(warps)])
            counts = np.array([bin(int(x)).count("1") for x in ref_union])
            ref_prefix = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint16)
            got_order = order[0, owner, win].astype(np.uint16)
            ok = (np.array_equal(got_order, ref_order.astype(np.uint16))
                  and np.array_equal(tilemask[0, owner, win].astype(np.uint16), ref_tm.astype(np.uint16))
                  and int(livec[0, owner, win, 0]) == ref_live
                  and int(livec[0, owner, win, 1]) == int(ref_prefix[-1])
                  and np.array_equal(warpwords[0, owner, win].astype(np.uint32), ref_ww)
                  and np.array_equal(unionw[0, owner, win].astype(np.uint32), ref_union)
                  and np.array_equal(prefix[0, owner, win].astype(np.uint16), ref_prefix))
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH owner {owner} win {win}: order eq {np.array_equal(got_order, ref_order.astype(np.uint16))}"
                          f" tm eq {np.array_equal(tilemask[0, owner, win].astype(np.uint16), ref_tm.astype(np.uint16))}"
                          f" live {livec[0, owner, win].tolist()} vs {[ref_live, int(ref_prefix[-1])]}"
                          f" ww eq {np.array_equal(warpwords[0, owner, win].astype(np.uint32), ref_ww)}"
                          f" union eq {np.array_equal(unionw[0, owner, win].astype(np.uint32), ref_union)}"
                          f" prefix eq {np.array_equal(prefix[0, owner, win].astype(np.uint16), ref_prefix)}")
    return bad


def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, round to nearest even."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) >> 16
    return u.astype(np.uint16)


def bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def check_fill(spec, kw, slots, geom_list, owner_count: int):
    """Every copied chunk against the gather the union names: V rows chunk-swizzled, inner
    runs and gain-scaled outer runs in the row32 layout, the zero row zero; rows past the
    union's total unspecified."""
    from rola.ops import carry as c

    P, rows, slot_bytes, vrow_bytes, v_off, in_off, out_off, vchunks, gain_off = geom_list
    g = c.geometry(list(spec.widths), level_modes=spec.level_modes, bc=256, nsr=8, nsw=8)
    D = g["D"]
    wrank = list(g["w"]["rank"])[:D]
    inner_l, outer_l = wrank.index(0), wrank.index(1)
    words = kw["liveness"].words.cpu().numpy()
    write_plane = kw["routes"].write.view(torch.int16).cpu().numpy().astype(np.uint16)[0]
    gain = kw["routes"].gain.view(torch.int16).cpu().numpy().astype(np.uint16)[0]
    vv = kw["v"].view(torch.int16).cpu().numpy().astype(np.uint16)[0]
    got = slots.cpu().numpy()
    L = spec.tokens
    groups = min(vchunks, 8)
    bad = 0
    for owner in range(owner_count):
        abase = []
        for l in range(D):
            odig = (owner // int(g["owner_div"][l])) % int(g["grid"][l])
            abase.append(int(g["col_base"][l]) + odig * int(g["span"][l]))
        for win in range((L + WINDOW - 1) // WINDOW):
            t0 = win * WINDOW
            wlen = min(WINDOW, L - t0)
            live = box_liveness(words, g, "write", owner, t0, wlen)
            union = np.nonzero(live.any(1))[0]
            chunks = (len(union) + P - 1) // P
            for ch in range(chunks):
                want = np.zeros(slot_bytes // 2, dtype=np.uint16)
                valid = np.zeros(slot_bytes // 2, dtype=bool)
                for row in range(P):
                    r = ch * P + row
                    if r >= len(union):
                        continue
                    t = t0 + int(union[r])
                    for cc in range(vchunks):
                        sw = cc ^ (row & (groups - 1))
                        off = (v_off + row * vrow_bytes + sw * 16) // 2
                        want[off:off + 8] = vv[t, cc * 8:cc * 8 + 8]
                        valid[off:off + 8] = True
                    inner = write_plane[t, abase[inner_l]:abase[inner_l] + 16]
                    outer = write_plane[t, abase[outer_l]:abase[outer_l] + 16]
                    for cc in range(2):
                        sw = cc ^ ((row >> 2) & 1)
                        o_in = (in_off + row * 32 + sw * 16) // 2
                        o_out = (out_off + row * 32 + sw * 16) // 2
                        want[o_in:o_in + 8] = inner[cc * 8:cc * 8 + 8]
                        want[o_out:o_out + 8] = outer[cc * 8:cc * 8 + 8]
                        valid[o_in:o_in + 8] = True
                        valid[o_out:o_out + 8] = True
                    #: the gain pair: the aligned word of the gain row holding token t.
                    gp = (gain_off + row * 4) // 2
                    te = t & ~1
                    assert te < gain.shape[0], (t, t0, r, int(union[r]), wlen, gain.shape, len(union))
                    want[gp] = gain[te]
                    want[gp + 1] = gain[te + 1] if te + 1 < gain.shape[0] else 0
                    valid[gp:gp + 2] = True
                # the zero row
                zr = P
                for cc in range(vchunks):
                    off = (v_off + zr * vrow_bytes + cc * 16) // 2
                    valid[off:off + 8] = True
                for cc in range(2):
                    valid[(in_off + zr * 32 + cc * 16) // 2:(in_off + zr * 32 + cc * 16) // 2 + 8] = True
                    valid[(out_off + zr * 32 + cc * 16) // 2:(out_off + zr * 32 + cc * 16) // 2 + 8] = True
                g16 = got[owner, win, ch].view(np.uint16)
                if not np.array_equal(g16[valid], want[valid]):
                    bad += 1
                    if bad <= 3:
                        d = np.nonzero((g16 != want) & valid)[0]
                        byte = d * 2
                        fields = []
                        for b in byte:
                            if b < in_off:
                                fields.append(("V", (b - v_off) // vrow_bytes, ((b - v_off) % vrow_bytes) // 16))
                            elif b < out_off:
                                fields.append(("in", (b - in_off) // 32, ((b - in_off) % 32) // 16))
                            elif b < gain_off:
                                fields.append(("out", (b - out_off) // 32, ((b - out_off) % 32) // 16))
                            else:
                                fields.append(("gain", (b - gain_off) // 4, 0))
                        print(f"  MISMATCH owner {owner} win {win} chunk {ch}: {len(d)} halfwords; fields {sorted(set(fields))[:24]}")
    return bad


def run_fill(ext, spec, args, reps: int):
    return ext.fill_part(*args, reps)


def check_fold(spec, kw, out, owner_count: int, warps: int = 8, reps: int = 1):
    """The warps' state and masses against the Kronecker fold over every token of every
    window, the kernel's arithmetic emulated: outer x gain rounded to bf16, inner x that
    rounded to bf16, the products against V summed in fp32."""
    from rola.ops import carry as c

    state, mass = (t.cpu().numpy() for t in out)
    g = c.geometry(list(spec.widths), level_modes=spec.level_modes, bc=256, nsr=8, nsw=8)
    D = g["D"]
    wrank = list(g["w"]["rank"])[:D]
    inner_l, outer_l = wrank.index(0), wrank.index(1)
    write_plane = bf16_to_f32(kw["routes"].write.view(torch.int16).cpu().numpy().astype(np.uint16)[0])
    gain = bf16_to_f32(kw["routes"].gain.view(torch.int16).cpu().numpy().astype(np.uint16)[0])
    vv = bf16_to_f32(kw["v"].view(torch.int16).cpu().numpy().astype(np.uint16)[0])
    dv = vv.shape[1]
    L = spec.tokens
    bad = 0
    worst = 0.0
    for owner in range(owner_count):
        abase = []
        for l in range(D):
            odig = (owner // int(g["owner_div"][l])) % int(g["grid"][l])
            abase.append(int(g["col_base"][l]) + odig * int(g["span"][l]))
        inner = write_plane[:, abase[inner_l]:abase[inner_l] + 16]          # [L, 16 pos]
        outer = write_plane[:, abase[outer_l]:abase[outer_l] + 16]          # [L, 16 box]
        sg = bf16_to_f32(bf16_round(outer * gain[:, None]))                  # [L, 16 box]
        #: the kernel folds only union-live tokens, and their coefficients are exact zeros
        #: elsewhere, so the sum over all tokens is the same sum.
        C = np.zeros((L, 256), dtype=np.float32)
        for b in range(16):
            C[:, b * 16:(b + 1) * 16] = bf16_to_f32(bf16_round(inner * sg[:, b:b + 1]))
        S = (C.T.astype(np.float64) @ vv.astype(np.float64)) * reps            # [256 leaf][dv]
        M = C.astype(np.float64).sum(0) * reps                                # [256 leaf]
        for w in range(warps):
            for lane in range(32):
                r, q = lane >> 2, lane & 3
                for bi in range(16 // warps):
                    b = w + bi * warps
                    for j in range(dv // 8):
                        for e in range(4):
                            leaf = b * 16 + r + 8 * (e >> 1)
                            ch = 8 * j + 2 * q + (e & 1)
                            got = state[0, owner, w, lane, bi, j, e]
                            want = S[leaf, ch]
                            err = abs(got - want)
                            tol = 1e-3 * abs(want) + 2e-3
                            worst = max(worst, err / tol)
                            if err > tol:
                                bad += 1
                                if bad <= 5:
                                    print(f"  MISMATCH owner {owner} warp {w} lane {lane} box {b} leaf {leaf} ch {ch}: got {got} want {want}")
                    if q == 0:
                        for h in range(2):
                            leaf = b * 16 + r + 8 * h
                            got = mass[0, owner, w, lane, bi, 2 * h]
                            want = M[leaf]
                            err = abs(got - want)
                            tol = 1e-3 * abs(want) + 2e-3
                            worst = max(worst, err / tol)
                            if err > tol:
                                bad += 1
                                if bad <= 5:
                                    print(f"  MISMATCH mass owner {owner} warp {w} lane {lane} leaf {leaf}: got {got} want {want}")
    print(f"  fold check: worst error {worst:.3f} of tolerance")
    return bad


def run_fold(ext, spec, args, reps: int):
    return ext.fold_part(*args, reps)


# ----------------------------------------------------------------------------- timing

def time_ms(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def ncu_instructions(argv_base: list[str], reps: int) -> float:
    """smsp__inst_executed.sum of the part kernel under ncu, this script re-run with `reps`."""
    cmd = ["ncu", "--metrics", "smsp__inst_executed.sum", "-k", "regex:_part_kernel", "--csv",
           sys.executable, __file__, *argv_base, "--reps", str(reps), "--launch-only"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    vals = []
    for line in out.splitlines():
        if "smsp__inst_executed.sum" in line and line.startswith('"'):
            cells = [c.strip('"') for c in line.split('","')]
            vals.append(float(cells[-1].replace(",", "")))
    if not vals:
        raise SystemExit("ncu returned no smsp__inst_executed.sum row:\n" + out[-2000:])
    return vals[-1]


def run_head(ext, spec, args, reps: int):
    return ext.head_part(*args, reps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="head", choices=["head", "fill", "fold"])
    ap.add_argument("--cell", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="every budget cell")
    ap.add_argument("--reps", type=int, default=8, help="passes of the part per window in one launch")
    ap.add_argument("--ncu", action="store_true", help="also count instructions under ncu")
    ap.add_argument("--launch-only", action="store_true", help="(ncu child) one launch, no checks")
    ap.add_argument("--budget", default=str(ROOT / "tools/budgets/carry.json"))
    a = ap.parse_args(argv)

    from benchmarks.bench.carry_model import BUDGET_CELLS
    from benchmarks.cells import by_name
    from rola.ops import carry as c

    cells = list(BUDGET_CELLS) if a.all else (a.cell or ["flagship-alt-k4"])
    ext = build()
    budgets = json.loads(Path(a.budget).read_text())["cells"]
    ghz = c.sm_clock_ghz()
    for name in cells:
        spec = by_name(name)
        drawn, kw, args = call_args(spec)
        head_out = run_head(ext, spec, args, 1)
        torch.cuda.synchronize()
        owner_count = head_out[0].shape[1]
        windows = head_out[0].shape[2]
        cta_windows = head_out[0].shape[0] * owner_count * windows
        if a.part == "head":
            run, check = run_head, (lambda out: check_head(spec, kw, out, owner_count))
        elif a.part == "fill":
            run, check = run_fill, (lambda out: check_fill(spec, kw, out, list(ext.pool_geometry()), 2))
        else:
            run, check = run_fold, (lambda out: check_fold(spec, kw, out, 2))
        run(ext, spec, args, a.reps)
        torch.cuda.synchronize()
        if a.launch_only:
            continue
        bad = check(run(ext, spec, args, 1))
        t1 = time_ms(lambda: run(ext, spec, args, 1))
        tr = time_ms(lambda: run(ext, spec, args, a.reps))
        per_pass_ms = (tr - t1) / (a.reps - 1)
        cyc = per_pass_ms * 1e-3 * (ghz or float("nan")) * 1e9 / cta_windows
        b = budgets.get(name, {}).get(a.part)
        line = (f"{a.part:8s} {name:20s} check {'OK ' if bad == 0 else f'FAIL({bad})'}  "
                f"{per_pass_ms * 1e3:8.1f} us/pass  {cyc:8.0f} cyc/CTA-window"
                + (f"  clock {ghz:.3f} GHz" if ghz else "  clock ?"))
        if a.ncu:
            base = ["--part", a.part, "--cell", name]
            i0 = ncu_instructions(base, 0)
            i1 = ncu_instructions(base, a.reps)
            instr = (i1 - i0) / a.reps / cta_windows
            flag = "" if b is None else ("  OVER BUDGET" if instr > b else "  within budget")
            line += f"  {instr:8.0f} instr/CTA-window (budget {b if b is not None else '?'}){flag}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
