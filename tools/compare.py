#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMPARISON: arms of rola checkouts and of other libraries on the cells of one point, interleaved call by call.

    python tools/compare.py --point L4096-N4096-dv64 --registry points.json \\
        --arm label:master,arm:prefill_op,worktree:/path/rola-a,venv:/path/venv-a \\
        --arm label:tip,arm:prefill_op,worktree:/path/rola-b,venv:/path/venv-b \\
        [--foreign label:attention,runner:attention,provider:rola_bench.measure.attention:arms,arm:flash,python:PY,cwd:DIR] \\
        [--cells a,b] [--reference master] [--rounds 8 --reps 11 --warmup 10] [--out FILE] [--record]
    python tools/compare.py --cells flat-small-alt-k16 --arm label:first,arm:carry_forward \\
        --arm label:identity,arm:carry_forward@schedule=identity

The method is rola-devtools' interleaving driver on a POINT (`rola_devtools.cells`): a named group of cells by runner
and what it holds equal, from the central cell registry (`rola_devtools.cells`) and the `--registry` files. `--cells`
narrows the point to the named cells; without `--point` it is a point of its own, those rola cells for the rola runner.
One worker process per arm environment; every row (an arm on a cell) warmed past the floor and then called once per rep
in a fresh random order, each sample the arm's own CUDA-event stopwatch. A rola ARM (`--arm`, runner `rola`) is a
subject with its dials, built by its checkout's own `bench.provider` under its own venv on each cell the point sends the
rola runner, so the only thing that differs between two rola arms is what the arm names; `worktree` and `venv` default
to this checkout and its pointer venv. A FOREIGN arm (`--foreign`) is another library's runner (`module:function`) run by
the python it names, on the cells the point sends its `runner` (default: its label).

This file holds the device for the run (the GPU lock, exclusive) and the host's clock lock, proven by the device's own
clock read before the first call and after the last; a run whose second read is off the lock is refused. It prints each
row's median and paired ratio, writes the driver's whole result with `--out`, and with `--record` stores it through
`rola_results` at `compare`. Docs: docs/internals/tools/compare.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))

import dev_config  # noqa: E402
from rola_devtools.locks import clock as clock_lock  # noqa: E402
from rola_devtools.locks.gpu import gpu_lock  # noqa: E402


def _fields(spec: str, required: tuple[str, ...], flag: str) -> dict:
    out = {}
    for part in spec.split(","):
        key, _, value = part.partition(":")
        out[key.strip()] = value.strip()
    missing = [key for key in required if not out.get(key)]
    if missing:
        raise argparse.ArgumentTypeError(f"{flag} needs {', '.join(k + ':...' for k in required)}; {spec!r} lacks "
                                         f"{missing}")
    return out


def rola_arm(spec: str) -> dict:
    """``label:NAME,arm:ARM[,worktree:PATH,venv:PATH]``: an arm of a rola checkout (default this one)."""
    out = _fields(spec, ("label", "arm"), "--arm")
    out.setdefault("worktree", str(ROOT))
    out.setdefault("venv", str(Path(dev_config.get("workspace.worktrees")) / f"venv-{Path(out['worktree']).name}"))
    return out


def foreign_arm(spec: str) -> dict:
    """``label:NAME,provider:MODULE:FUNCTION,arm:ARM,python:PATH,cwd:DIR[,runner:NAME]``: another library's arm."""
    return _fields(spec, ("label", "provider", "arm", "python", "cwd"), "--foreign")


def arm_specs(rola_arms: list[dict], foreign: list[dict]):
    from rola_devtools.interleave import ArmSpec

    specs = []
    for arm in rola_arms:
        worktree = str(Path(arm["worktree"]).resolve())
        python = Path(arm["venv"]) / "bin" / "python"
        if not python.is_file():
            raise SystemExit(f"--arm {arm['label']}: no python at {python}")
        specs.append(ArmSpec(arm["label"], "bench.provider:arms", arm["arm"], python=str(python), cwd=worktree,
                             env={"PYTHONPATH": f"{worktree}:{worktree}/benchmarks"}, runner="rola"))
    specs += [ArmSpec(arm["label"], arm["provider"], arm["arm"], python=arm["python"], cwd=arm["cwd"],
                      runner=arm.get("runner", "")) for arm in foreign]
    return specs


def point_of(args) -> dict:
    """The point the run is on: `--point` from the registries, narrowed by `--cells`; or `--cells` alone, for rola."""
    from rola_devtools.cells import FILES, Registry

    reg = Registry.load([*FILES, *args.registry])
    wanted = args.cells.split(",") if args.cells else None
    if not args.point:
        if not wanted:
            raise SystemExit("name a --point, or --cells for a point of rola cells")
        return reg.adhoc({"rola": wanted})
    point = reg.point(args.point)
    if wanted:
        unknown = sorted(set(wanted) - {c["name"] for cells in point["runners"].values() for c in cells})
        if unknown:
            raise SystemExit(f"--cells {unknown} are not cells of point {point['name']}")
        point = {**point, "runners": {runner: [c for c in cells if c["name"] in wanted]
                                      for runner, cells in point["runners"].items()}}
        point["runners"] = {runner: cells for runner, cells in point["runners"].items() if cells}
    return point


def compare(args) -> dict:
    from rola_devtools.interleave import interleave

    point = point_of(args)

    def read_ghz():
        from rola.ops import carry as carry_ops

        return carry_ops.sm_clock_ghz()

    clock = clock_lock.engage(read_ghz)
    result = interleave(point, arm_specs(args.arm, args.foreign), rounds=args.rounds, reps=args.reps,
                        warmup=args.warmup, reference=args.reference, seed=args.seed, hold=gpu_lock)
    after = read_ghz()
    held = clock is not None and clock_lock.within(after, clock)
    if clock is not None and not held:
        raise SystemExit(f"CLOCK: the device reads {after} GHz after the run, off the lock at {clock['ghz']} GHz; "
                         "refusing the comparison")
    return {**result, "clock": {"locked_ghz": clock["ghz"] if clock else None, "ghz_after": after, "held": held}}


def record(args, result: dict) -> None:
    from rola_results import Store, checkout, portable

    arms = []
    for arm in args.arm:
        facts = checkout(arm["worktree"])
        arms.append({"label": arm["label"], "arm": arm["arm"], "git_sha": facts["git_sha"],
                     "diff_sha256": facts["diff_sha256"]})
    arms += [{"label": arm["label"], "provider": arm["provider"], "runner": arm.get("runner", arm["label"]),
              "arm": arm["arm"]} for arm in args.foreign]
    semantics = {"point": result["point"], "arms": arms, "rounds": args.rounds,
                 "reps": args.reps, "warmup": args.warmup, "seed": args.seed, "reference": result["reference"]}
    output = json.loads(portable(json.dumps(result), ROOT))
    store = Store("compare")
    sample = store.put(semantics, output=output,
                       provenance={"checkouts": [checkout(arm["worktree"]) for arm in args.arm]})
    print(f"record: {store.location} sample {sample['n']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", default=None, help="a registered point (benchmarks/cells and the --registry files)")
    ap.add_argument("--registry", action="append", default=[], help="another registry file of cells and points; repeatable")
    ap.add_argument("--cells", default=None, help="narrow the point to these cells, or without --point, a point of rola cells")
    ap.add_argument("--arm", action="append", default=[], type=rola_arm, help="a rola arm; repeatable")
    ap.add_argument("--foreign", action="append", default=[], type=foreign_arm, help="another library's arm; repeatable")
    ap.add_argument("--reference", default=None, help="the label (paired within each cell) or one row ratios divide by")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the whole result as JSON")
    ap.add_argument("--record", action="store_true", help="store the result through rola_results at `compare`")
    args = ap.parse_args()
    if not args.arm and not args.foreign:
        ap.error("need at least one --arm or --foreign")
    result = compare(args)
    for row in result["arms"]:
        ratios = "".join(f"  x{p['ratio_median']:.4f} vs {p['reference']} (iqr {p['ratio_iqr']:.4f})" for p in row["paired"])
        print(f"{row['row']:<44}{row['arm']:<42}{row['median_ms']:.4f} ms (iqr {row['iqr_ms']:.4f}){ratios}")
    print(f"clock: {result['clock']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    if args.record:
        record(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
