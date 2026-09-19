"""The warp trace: every dynamic instruction of one CTA's warps, recorded from the installed binary by the NVBit tool
`tools/nvbit/warp_trace`, and matched to the real run's stamp intervals.

    python tools/warp_trace.py --child CELL --cta 0 --out records.bin --sass listing.tsv --stamps stamps.pt [--addrs]

runs one launch inside an injected process (the parent sets `CUDA_INJECTION64_PATH`); `record()` is the parent side.
The instrumented launch is ~1000x slower, so its timing is discarded: `match()` takes each activity's INSTRUCTION
CONTENT from it and its CYCLES from the uninstrumented run's stamps, by (window, event, ordinal) -- the k-th walk is
chunk k's, the k-th fragment event run k's, the k-th tile tile k's. See docs/internals/tools/warp_timeline.md.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev_config  # noqa: E402
from phase_trace import intervals, windows  # noqa: E402

HERE = Path(__file__).resolve().parent
PIN = HERE / "nvbit_pin.json"
TOOL_SRC = HERE / "nvbit" / "warp_trace"
HEADER = np.dtype([("pc", "<u4"), ("active", "<u4"), ("pred", "<u4"), ("op", "<u2"), ("warp", "<u2"), ("clock", "<u8")])
MREF_TAG = 0xFFFF0000
MREF_BYTES = 264
#: instruction classes the intervals are summed by: opcode -> class; anything else is `alu`
CLASSES = {"HMMA": "hmma", "LDSM": "ldsm", "LDGSTS": "copy", "LDGDEPBAR": "copy", "DEPBAR": "copy", "LDG": "ldg",
           "STG": "stg", "LDS": "lds", "STS": "sts", "SHFL": "shfl", "BAR": "bar", "RED": "red", "ATOM": "red",
           "ARRIVES": "mbar", "SYNCS": "mbar", "BRA": "bra", "CS2R": "clk"}
KEYS = ("n", "hmma", "ldsm", "copy", "ldg", "stg", "lds", "sts", "shfl", "bar", "mbar", "red", "back")


def pin() -> dict:
    return json.loads(PIN.read_text())


def nvbit_root() -> Path:
    """The pinned NVBit release, fetched once into the scratch and checked against the pin's digest."""
    p = pin()
    root = dev_config.scratch("nvbit") / f"nvbit-{p['version']}"
    if (root / "core" / "libnvbit.a").exists():
        return root
    tarball = dev_config.scratch("nvbit") / p["release_asset"]
    if not tarball.exists():
        urllib.request.urlretrieve(p["release_url"], tarball)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != p["tarball_sha256"]:
        raise SystemExit(f"{tarball.name}: sha256 {digest} is not the pin's {p['tarball_sha256']}; refused")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            member.name = str(Path(*parts))
            tar.extract(member, root)
    return root


def tool_so(arch: str) -> Path:
    """The tracer built against the pinned NVBit for `arch`, rebuilt when its sources or the pin change."""
    root = nvbit_root()
    key = hashlib.sha256()
    for f in sorted(TOOL_SRC.rglob("*")):
        if f.is_file() and f.suffix in (".cu", ".h") or f.name == "Makefile":
            key.update(f.read_bytes())
    key.update(pin()["version"].encode())
    out = dev_config.scratch("nvbit") / f"warp_trace-{arch}-{key.hexdigest()[:12]}.so"
    if out.exists():
        return out
    subprocess.run(["make", "-C", str(TOOL_SRC), f"NVBIT_PATH={root / 'core'}", f"NVCC={dev_config.cuda_bin('nvcc')}",
                    f"ARCH={arch}", f"OUT={out}"], check=True, capture_output=True, text=True)
    subprocess.run(["make", "-C", str(TOOL_SRC), "clean"], check=False, capture_output=True)
    return out


def record(cell: str, cta: int, out: Path, sass: Path, stamps: Path, arch: str, addrs: bool = False) -> None:
    """One launch of `cell` in a child process under the tracer: records, the SASS listing, the child's own stamps."""
    env = dict(os.environ, CUDA_INJECTION64_PATH=str(tool_so(arch)), NOBANNER="1", TRACE_CTA=str(cta),
               TRACE_ADDRS="1" if addrs else "0", TRACE_OUT=str(out), TRACE_SASS=str(sass), TRACE_KERNEL="carry_kernel")
    done = subprocess.run([sys.executable, __file__, "--child", cell, "--cta", str(cta), "--stamps", str(stamps)],
                          env=env, capture_output=True, text=True, timeout=7200)
    if done.returncode != 0 or not out.exists():
        raise SystemExit(f"the traced launch of {cell} failed ({done.returncode}):\n{done.stdout[-2000:]}{done.stderr[-2000:]}")


def child(cell: str, cta: int, stamps: Path) -> None:
    import torch
    from phase_trace import take_stamps

    rows, _warps, _ctas = take_stamps(cell, cta + 1, 32768, warmup=0)
    torch.save(rows, stamps)


def load(records: Path, listing: Path):
    """The records as a structured array (memory-operand records dropped) and the listing: pc -> (opcode class, text)."""
    sass = {}
    for line in listing.read_text().splitlines():
        if line.startswith("#"):
            continue
        off, _opid, _nmref, text, loc = line.split("\t")
        op = text.split()[1 if text.startswith("@") else 0].split(".")[0]
        sass[int(off)] = (CLASSES.get(op, "alu"), text, loc, op)
    buf = np.fromfile(records, dtype=np.uint8)
    starts = []
    i, n = 0, len(buf)
    while i + HEADER.itemsize <= n:
        tag = int.from_bytes(buf[i:i + 4].tobytes(), "little")
        if (tag & MREF_TAG) == MREF_TAG:
            i += MREF_BYTES
            continue
        starts.append(i)
        i += HEADER.itemsize
    idx = np.array(starts, dtype=np.int64)
    heads = np.stack([buf[idx + k] for k in range(HEADER.itemsize)], axis=1).tobytes()
    return np.frombuffer(heads, dtype=HEADER), sass


def keyed(cyc: list[int], ev: list[int]) -> tuple[dict, list]:
    """(window, event, ordinal) -> (interval index, start, end), and the windows."""
    ints, wins = intervals(cyc, ev), windows(cyc, ev)
    out = {}
    for wi, (t0, t1) in enumerate(wins):
        seen: dict[str, int] = collections.defaultdict(int)
        for j, (s, e, lab) in enumerate(ints):
            if t0 <= s < t1:
                out[(wi, lab, seen[lab])] = (j, s - t0, e - t0)
                seen[lab] += 1
    return out, wins


def match(recs, sass: dict, child_stamps, real_stamps, warps: int) -> tuple[list[dict], dict, collections.Counter]:
    """Per warp, the real run's windows of intervals, each with the instrumented run's instruction mix for the same
    (window, event, ordinal); `mix` is None where the real run had an activity the instrumented one did not. Also
    the traced run's instruction histograms: event -> (pc -> executions inside that event), and pc -> executions."""
    clock_sites = sorted(pc for pc, (_c, text, _l, _op) in sass.items() if "CLOCK" in text)[1:]
    cls_of = {pc: c for pc, (c, _t, _l, _op) in sass.items()}
    out = []
    by_event: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total: collections.Counter = collections.Counter()
    for w in range(warps):
        raw = child_stamps[w]
        raw = raw[raw != 0]
        iev, icyc = (raw & 0xFF).tolist(), (raw >> 8).tolist()
        m = np.flatnonzero(recs["warp"] == w)
        pcs = recs["pc"][m]
        at = np.flatnonzero(np.isin(pcs, clock_sites))
        if len(at) != len(iev):
            raise SystemExit(f"warp {w}: {len(at)} clock reads traced against {len(iev)} stamps recorded; "
                             "the trace and the stamps are not the same run")
        mixes = []
        labels = [lab for (_s, _e, lab) in intervals(icyc, iev)]
        for i in range(len(at) - 1):
            seg = pcs[at[i] + 1:at[i + 1]].tolist()
            counts = collections.Counter(cls_of[pc] for pc in seg)
            back = int(np.sum(pcs[at[i] + 1:at[i + 1] + 1] < pcs[at[i]:at[i + 1]]))
            mixes.append({"n": len(seg), "back": back, **{k: int(v) for k, v in counts.items()}})
            hist = collections.Counter(seg)
            by_event[labels[i]].update(hist)
            total.update(hist)
        ikey, _ = keyed(icyc, iev)
        rr = real_stamps[w]
        rr = rr[rr != 0]
        rkey, rwins = keyed((rr >> 8).tolist(), (rr & 0xFF).tolist())
        wins = [{"len": t1 - t0, "ivs": []} for (t0, t1) in rwins]
        for (wi, lab, k), (_j, s, e) in sorted(rkey.items(), key=lambda kv: (kv[0][0], kv[1][1])):
            hit = ikey.get((wi, lab, k))
            wins[wi]["ivs"].append({"ev": lab, "k": k, "t0": s, "t1": e, "mix": mixes[hit[0]] if hit else None})
        out.append({"warp": w, "windows": wins})
    return out, by_event, total


def source_lines(arch: str) -> dict[int, tuple[str, str]]:
    """pc -> (innermost frame, outermost frame) as `file:line`, off the installed extension's cubin: the tracer's own
    line info is empty for inlined code, the disassembler's inline chain is not."""
    import sass as sassmod
    import toolchains

    cubin = sassmod.cubin(toolchains.built_extension(), "carry_arm_0", arch)
    chains = sassmod.frames(sassmod.disassemble(cubin, "--print-line-info-inline", "-gi"))
    return {pc: (f"{c[0][0]}:{c[0][1]}", f"{c[-1][0]}:{c[-1][1]}") for pc, c in chains.items() if c}


def census_join(csv_path: Path, sass: dict, by_event: dict, total: collections.Counter, arch: str) -> dict:
    """The profiler's per-instruction stall samples (`stall_census.py --csv`, the whole launch) joined to the traced
    instructions by offset: `instructions` = [pc, text, innermost line, outermost line, class, executions traced,
    samples, [reason samples]], `reasons` the reason names, `by_event` = event -> [[pc, executions inside the event]].
    An instruction's samples are the launch's; the page attributes them to an event by the traced executions' share."""
    from region_ledger import read_source_counters

    rows = read_source_counters(csv_path)
    base = min(a for a, _e, _s, _r in rows)
    reasons = sorted(rows[0][3]) if rows else []
    at = {a - base: (e, smp, r) for a, e, smp, r in rows}
    lines = source_lines(arch)
    instructions = []
    for pc, n in sorted(total.items()):
        cls, text, _loc, _op = sass[pc]
        inner, outer = lines.get(pc, ("?", "?"))
        e, smp, r = at.get(pc, (0, 0, {}))
        instructions.append([pc, text, inner, outer, cls, n, smp, [r.get(k, 0) for k in reasons]])
    return {"reasons": reasons, "instructions": instructions,
            "by_event": {lab: [[pc, n] for pc, n in sorted(c.items())] for lab, c in by_event.items()},
            "unjoined": sorted(pc for pc in total if pc not in at)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", metavar="CELL", required=True)
    ap.add_argument("--cta", type=int, default=0)
    ap.add_argument("--stamps", type=Path, required=True)
    a = ap.parse_args()
    child(a.child, a.cta, a.stamps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
