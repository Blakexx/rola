# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""tools/experiments/dram_by_activity.py -- DRAM/L2/L1 COUNTERS WITH THE ACTIVITY BYTE CONTROLLED.

The committed form of the ad hoc "plane-count instrument" whose first run reported
its read-only/written DRAM columns from (`0.657x` on read-only atoms at `k16_w384`;
`+9.6%` on written atoms at `k4_w512_record`).  That instrument had no committed
invocation, so its written column could not be reproduced -- the SKILLS
FEEDBACK made exactly that point, and this file is the answer to it: one script, one
definition of the experiment, run by BOTH binaries' own venv python out of ONE source
file, so a cross-tip delta is a fact about the extension and not about the harness.

WHAT IT CONTROLS.  The facts pass publishes one ACTIVITY byte per atom
(`rola.engine.facts.planes.atom_bits`: bit 0 = WRITTEN, bit 1 = READ), and the
kernel's state sweeps read it as "load iff resident and (read|written), store iff
written".  The experiment overrides that byte, leaving everything else -- the draw,
the arm, the page table, the launch -- identical:

  * ``asis``    -- the byte the facts pass emits for this cell.
  * ``read``    -- every TOUCHED atom forced READ-ONLY (`ATOM_READ`): loads only, and
                   on a split-plane binary a load of the hi plane only.
  * ``written`` -- every TOUCHED atom forced WRITTEN|READ: both planes both ways.
  * ``none``    -- `--state none`, the NULL-STATE control: no state I/O at all, so the
                   two binaries' numbers must agree, and their difference IS this
                   host's DRAM noise floor (measured at ~0.6%).

Every frozen cell has read-set == write-set == every atom, so on those cells
``asis`` and ``written`` request the same work: they are a second noise control, and
any spread between them is measurement, not mechanism.

HOW IT MEASURES.  One `ncu` process per (binary, variant, cell) profiles `--count`
steady-state launches after `--launch-skip` warmups, under `gpu_lock()` (never an
external `flock`), and the reported figure is the MEDIAN over launches with
the min/max kept beside it.  Binaries are interleaved per variant (A,B,A,B -- the
lesson), never "all of A then all of B".  Counters are asked in READ and WRITE halves
because the mechanism under suspicion (a partially-covered sector on the store side)
moves one half and not the other, and in REQUESTS as well as SECTORS because
sectors-per-request is the coalescing/alignment question the wall-clock run could not ask.

`--source` additionally runs one SourceCounters capture per (binary, cell) and
aggregates it by source line, which is how the three blocks of a split page (hi values,
lo values, the two 16-bit mass runs -- `docs/internals/common/state_page.md`) are told
apart: they are three distinct loops in `common/state_page.cuh`.

The rows are stored through `rola_results` at `dram_by_activity`, one sample a run, its stage in
the provenance.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
REPO = THIS_FILE.parents[2]
#: This harness REUSES `tools/probe_cells.py`'s frozen cell table, draw, arm resolution
#: and call construction rather than copying them: the cells must be the same cells the
#: campaign's timing numbers were taken at, and a second copy of the draw is a second
#: thing to drift (`rola-probe`: the cell specs are already a deliberate frozen mirror).
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO / "benchmarks"))

import dev_config  # noqa: E402
import probe_cells as pc  # noqa: E402
from gpu_lock import GPU_LOCK_DEFAULT, gpu_lock  # noqa: E402

NCU_BIN = pc.NCU_BIN
KERNEL_REGEX = "regex:carry.*kernel"

#: The counter set. READ and WRITE are asked separately at every level of the
#: hierarchy, and requests are asked beside sectors: `sectors / request` is the
#: alignment fact, and a DRAM byte total alone cannot separate "more bytes asked for"
#: from "the same bytes in worse sectors".
METRICS = [
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
]

VARIANTS = ("asis", "read", "written", "none")


# ---------------------------------------------------------------------------
# WORKER: runs in the TARGET binary's venv python, with that worktree as cwd.
def worker_main(args):
    import torch  # noqa: PLC0415

    pc.torch = torch
    import rola  # noqa: PLC0415
    from rola.engine.facts import planes  # noqa: PLC0415
    from rola.ops import carry as carry_ops  # noqa: PLC0415
    from rola.ops._ext import extension  # noqa: PLC0415

    ext = extension()
    spec = pc.CELL_SPECS[args.cell]
    carve, arm_index, arm_row = pc.resolve_arm(
        carry_ops, spec["widths"], spec["k"], spec["m"], spec["d_v"], spec["window"],
        pc.CHUNK, spec["level_modes"])
    read_bf, write_bf, g_bf, v_bf = pc.draw_cell(spec)
    pr, pw, gw, v = pc.pack_operands(carry_ops, read_bf, write_bf, g_bf, v_bf)

    s_out = page_tbl = atom_bits = None
    census = {}
    if args.variant != "none":
        s_out = carry_ops.state_plane(spec["widths"], pr.shape[0], spec["d_v"])
        n_atoms = s_out.shape[1]
        page_tbl = torch.arange(pr.shape[0] * n_atoms, device="cuda",
                                dtype=torch.int32).view(pr.shape[0], n_atoms)
        bits = planes.atom_bits(pw.reshape(1, *pw.shape), spec["widths"],
                                read_plane=pr.reshape(1, *pr.shape))
        touched = bits != 0
        census = {
            "atoms": int(bits.numel()),
            "touched": int(touched.sum()),
            "written_asis": int(planes.written_atoms(bits).sum()),
            "read_asis": int(planes.read_atoms(bits).sum()),
        }
        if args.variant == "read":
            #: FORCED READ-ONLY: the written bit is cleared on every touched atom, so
            #: the sweep loads and never stores. The untouched atoms stay untouched --
            #: the override changes an atom's ACTIVITY, never the RESIDENT set.
            bits = torch.where(touched, torch.full_like(bits, planes.ATOM_READ),
                               torch.zeros_like(bits))
        elif args.variant == "written":
            bits = torch.where(
                touched,
                torch.full_like(bits, planes.ATOM_READ | planes.ATOM_WRITTEN),
                torch.zeros_like(bits))
        atom_bits = bits.contiguous()

    if args.continuation and args.variant != "none":
        #: A CONTINUATION: the entry plane IS the exit plane (the in-place law -- the op
        #: seam refuses two distinct planes under an activity bitmap,
        #: `docs/internals/state.md#activity`). This is the ONLY shape in which a
        #: READ-ONLY atom moves any bytes at all: with a null `s_in` -- what
        #: `probe_cells.make_call` passes, and therefore what every `--state paged`
        #: number in this campaign measured -- there is no entry sweep, so "read-only"
        #: means "no state I/O", and the split planes' read-side saving is unobservable
        #: by construction.
        call_args = (pr, pw, gw, v, list(spec["widths"]), int(spec["k"]), int(spec["m"]),
                     int(spec["window"]), int(pc.CHUNK), int(carve),
                     spec["level_modes"], s_out, s_out, page_tbl, atom_bits)
        try:
            ext.carry_forward_inter(*call_args)
        except TypeError as ex:
            raise RuntimeError(
                "REFUSED: this binary's carry_forward_inter does not take "
                "(s_in, s_out, page_tbl, atom_bits) -- the continuation shape cannot "
                f"be requested here: {ex}") from ex
        call = lambda a=call_args: ext.carry_forward_inter(*a)  # noqa: E731
        took_modes = took_planes = took_bits = True
    else:
        call, took_modes, took_planes, took_bits = pc.make_call(
            ext, pr, pw, gw, v, spec["widths"], spec["k"], spec["m"], spec["window"],
            pc.CHUNK, carve, spec["level_modes"], s_out, page_tbl, atom_bits)
        if args.variant != "none" and not took_bits:
            raise RuntimeError(
                "REFUSED: this binary's carry_forward_inter takes no activity bitmap, so "
                "the activity byte cannot be controlled and the variant is meaningless.")

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.launches):
        call()
    torch.cuda.synchronize()

    props = torch.cuda.get_device_properties(0)
    try:
        from rola_cu13._build_config import BUILD_CONFIG  # noqa: PLC0415
        manifest_sha256, ptxas = BUILD_CONFIG["manifest_sha256"], BUILD_CONFIG["ptxas"]
    except Exception:  # noqa: BLE001
        manifest_sha256 = ptxas = None
    print(json.dumps({
        "continuation": bool(args.continuation),
        "rola_file": rola.__file__, "device_name": torch.cuda.get_device_name(0),
        "sm": f"sm_{props.major}{props.minor}", "driver_cuda": torch.version.cuda,
        "torch": torch.__version__, "manifest_sha256": manifest_sha256, "ptxas": ptxas,
        "resolved_carve": carve, "arm_index": arm_index, "arm_row": list(arm_row),
        "level_modes_applied": took_modes, "caller_owned_planes": took_planes,
        "activity_gated": took_bits, "census": census,
    }))


# ---------------------------------------------------------------------------
# ORCHESTRATOR.
def _worker_cmd(binary, cell, variant, warmup, launches, continuation=False, extra=()):
    return [pc.venv_python(binary["venv"]), str(THIS_FILE), "--worker", "--cell", cell,
            "--variant", variant, "--warmup", str(warmup), "--launches", str(launches),
            *(["--continuation"] if continuation else []), *extra]


def _parse_ncu_csv(text):
    """`{metric: [value per profiled launch]}` plus the kernel name, from ncu --csv."""
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if head is None:
        return None, {}, "\n".join(lines[-15:])
    reader = csv.DictReader(io.StringIO("\n".join(lines[head:])))
    per_metric, kernel = {}, None
    for r in reader:
        kernel = r.get("Kernel Name", kernel)
        name, val = r.get("Metric Name"), r.get("Metric Value")
        if not name:
            continue
        try:
            per_metric.setdefault(name, []).append(float(str(val).replace(",", "")))
        except (TypeError, ValueError):
            continue
    return kernel, per_metric, None


def run_variant(binary, cell, variant, warmup, launches, lock_path, out_dir,
                continuation=False, timeout=1800):
    cmd = [NCU_BIN, "--target-processes", "all", "-k", KERNEL_REGEX,
           "--launch-skip", str(warmup), "-c", str(launches),
           "--print-kernel-base", "mangled", "--metrics", ",".join(METRICS), "--csv",
           *_worker_cmd(binary, cell, variant, warmup, launches, continuation)]
    with gpu_lock(lock_path):
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                              timeout=timeout)
    tag = f"{cell}_{variant}{'_cont' if continuation else ''}"
    Path(out_dir, f"ncu_{binary['label']}_{tag}.csv").write_text(proc.stdout)
    kernel, per_metric, tail = _parse_ncu_csv(proc.stdout)
    #: The worker's one-line JSON stamp, found by DECODING candidate lines rather than
    #: by matching a leading key: a key added at the front of the stamp silently
    #: unstamped every recorded row once already (the `unknown-smunknown` hardware
    #: profile in this stage's own fourth cycle).
    stamp = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            cand = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "rola_file" in cand:
            stamp = cand
    stamp["rola_file"], outside = pc.rola_file_in(binary["worktree"], stamp.get("rola_file"))
    row = {"binary": binary["label"], "cell": cell, "variant": variant, "rc": proc.returncode,
           "kernel_name": kernel, "stamp": stamp, "n_launches": 0}
    if outside or not per_metric:
        row["error"] = pc.portable_error(outside or f"no ncu CSV (rc={proc.returncode}): {tail or proc.stderr[-1500:]}",
                                         binary["worktree"])
        return row
    for name, vals in per_metric.items():
        #: EVERY profiled launch's value is kept, not just the median: the diagnosis
        #: found this cell's DRAM READ term BIMODAL launch to launch, and a summary
        #: that reports only a median hides the mode a delta was actually read from.
        row[name] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals),
                     "n": len(vals), "values": vals}
        row["n_launches"] = max(row["n_launches"], len(vals))
    return row


def run_source(binary, cell, variant, warmup, lock_path, out_dir,
               continuation=False, timeout=2400):
    """One SourceCounters capture, aggregated by source line -- the block attribution."""
    cmd = [NCU_BIN, "--target-processes", "all", "-k", KERNEL_REGEX,
           "--launch-skip", str(warmup), "-c", "1", "--section", "SourceCounters",
           "--import-source", "no", "--page", "source", "--csv",
           *_worker_cmd(binary, cell, variant, warmup, 1, continuation)]
    with gpu_lock(lock_path):
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                              timeout=timeout)
    path = Path(out_dir, f"src_{binary['label']}_{cell}_{variant}.csv")
    path.write_text(proc.stdout)
    return {"binary": binary["label"], "cell": cell, "variant": variant,
            "rc": proc.returncode, "csv": path.name}


def fmt(v, scale=1.0, digits=3):
    return "--" if v is None else f"{v['median'] / scale:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", action="append", default=[], type=pc.parse_binary_spec,
                    help="worktree:PATH,venv:PATH[,label:NAME]; repeatable, interleaved")
    ap.add_argument("--cells", default="k4_w512_record")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--launches", type=int, default=6, help="profiled launches per run")
    ap.add_argument("--rounds", type=int, default=1, help="interleaved A/B repetitions")
    ap.add_argument("--lock", default=GPU_LOCK_DEFAULT)
    ap.add_argument("--out-dir", default=str(dev_config.scratch("dram_by_activity")))
    ap.add_argument("--source", action="store_true", help="also capture SourceCounters")
    ap.add_argument("--continuation", action="store_true",
                    help="entry plane IS the exit plane, so the ENTRY sweep runs "
                         "(the only shape where a read-only atom moves bytes)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--agent", default=None)
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_main(args)
        return

    cells = args.cells.split(",")
    variants = args.variants.split(",")
    for c in cells:
        if c not in pc.CELL_SPECS:
            ap.error(f"unknown cell {c!r}; known: {sorted(pc.CELL_SPECS)}")
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}; known: {VARIANTS}")
    if not args.binary:
        ap.error("need at least one --binary worktree:PATH,venv:PATH")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in args.binary:
        b["git_sha"] = pc.git_sha(b["worktree"])
        print(f"BINARY {b['label']}: {b['worktree']} sha={b['git_sha'][:8]}")

    rows = []
    for cell in cells:
        for variant in variants:
            for _ in range(args.rounds):
                for b in args.binary:  # INTERLEAVED per variant
                    r = run_variant(b, cell, variant, args.warmup, args.launches,
                                    args.lock, out_dir, args.continuation)
                    rows.append(r)
                    print(f"  {b['label']:<14}{cell:<18}{variant:<9}"
                          f"dram={fmt(r.get('dram__bytes.sum'), 1e6):>10} MB  "
                          f"launches={r['n_launches']} {r.get('error', '')[:80]}")
    if args.source:
        for cell in cells:
            for variant in variants:
                if variant == "none":
                    continue
                for b in args.binary:
                    s = run_source(b, cell, variant, args.warmup, args.lock, out_dir,
                                   args.continuation)
                    print(f"  source {b['label']}/{cell}/{variant} -> {Path(out_dir, s['csv'])} rc={s['rc']}")

    print()
    hdr = ("cell", "variant", "binary", "dram MB", "dram rd MB", "dram wr MB",
           "L2 rd sect", "L2 wr sect", "ld req", "st req", "ld sect", "st sect",
           "inst M", "us")
    print("".join(f"{h:<13}" for h in hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['cell']:<13}{r['variant']:<13}{r['binary']:<13}ERROR {r['error'][:90]}")
            continue
        cols = [r["cell"], r["variant"], r["binary"],
                fmt(r.get("dram__bytes.sum"), 1e6),
                fmt(r.get("dram__bytes_read.sum"), 1e6),
                fmt(r.get("dram__bytes_write.sum"), 1e6),
                fmt(r.get("lts__t_sectors_op_read.sum"), 1e3, 1),
                fmt(r.get("lts__t_sectors_op_write.sum"), 1e3, 1),
                fmt(r.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"), 1e3, 1),
                fmt(r.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"), 1e3, 1),
                fmt(r.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"), 1e3, 1),
                fmt(r.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"), 1e3, 1),
                fmt(r.get("smsp__inst_executed.sum"), 1e6, 2),
                fmt(r.get("gpu__time_duration.sum"), 1e3, 1)]
        print("".join(f"{c:<13}" for c in cols))

    dump = out_dir / "rows.jsonl"
    with dump.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows to {dump}")

    if not args.no_record:
        from rola_results import Store, checkout  # noqa: PLC0415

        arms = []
        for b in args.binary:
            stamp0 = next((r["stamp"] for r in rows if r["binary"] == b["label"] and "error" not in r), {})
            arms.append({**{k: v for k, v in checkout(b["worktree"]).items() if k in ("git_sha", "diff_sha256")},
                         "manifest_sha256": stamp0.get("manifest_sha256")})
        semantics = {"arms": arms, "cells": sorted({r["cell"] for r in rows}), "variants": args.variants.split(","),
                     "warmup": args.warmup, "launches": args.launches, "rounds": args.rounds, "source": args.source,
                     "continuation": args.continuation, "metrics": METRICS}
        provenance = {"stage": args.stage, "agent": args.agent, "purpose": args.purpose,
                      "arms": [{"label": b["label"], **checkout(b["worktree"])} for b in args.binary]}
        store = Store("dram_by_activity")
        if any("error" not in r for r in rows):
            sample = store.put(semantics, output=rows, provenance=provenance)
        else:
            sample = store.put(semantics, error="no row measured", provenance=provenance, rows=rows)
        print(f"record: {store.location} sample {sample['n']} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
