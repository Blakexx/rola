#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMPARISON: arms of rola checkouts and of other libraries at one registered cell, interleaved call by call.

    python tools/compare.py --cell flat-small-alt-k16 \\
        --arm label:master,arm:carry_forward,worktree:/path/rola-a,venv:/path/venv-a \\
        --arm label:tip,arm:carry_forward@schedule=identity,worktree:/path/rola-b,venv:/path/venv-b \\
        [--foreign label:attention,provider:rola_bench.measure.attention:arms,arm:flash,python:PY,cwd:DIR] \\
        [--matching "capacity at N = L"] [--reference master] [--rounds 8 --reps 11 --warmup 10] [--out FILE] [--record]

The method is rola-devtools' interleaving driver: one worker process per arm environment, every arm warmed past the
floor and then called once per rep in a fresh random order, each sample the arm's own CUDA-event stopwatch. A rola ARM is
a subject with its dials, built by its checkout's own `bench.provider` under its own venv, so the only thing that
differs between two rola arms is what the arm names; `worktree` and `venv` default to this checkout and its pointer
venv. A FOREIGN arm is another library's provider (`module:function`) run by the python it names. The point is the
cell, with the tokens and value width a foreign arm reads from it; with a foreign arm the matching rule is required,
because what makes the comparison fair is a claim to state.

This file holds the device for the run (the GPU lock, exclusive) and the host's clock lock, proven by the device's own
clock read before the first call and after the last; a run whose second read is off the lock is refused. It prints each
arm's median and paired ratio, writes the driver's whole result with `--out`, and with `--record` stores it through
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

import clock_lock  # noqa: E402
import dev_config  # noqa: E402
from gpu_lock import gpu_lock  # noqa: E402


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
    """``label:NAME,provider:MODULE:FUNCTION,arm:ARM,python:PATH,cwd:DIR``: another library's arm."""
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
                             env={"PYTHONPATH": f"{worktree}:{worktree}/benchmarks"}))
    specs += [ArmSpec(arm["label"], arm["provider"], arm["arm"], python=arm["python"], cwd=arm["cwd"])
              for arm in foreign]
    return specs


def compare(args) -> dict:
    from rola_devtools.interleave import interleave

    from bench.driver import CELLS

    if args.cell not in CELLS:
        raise SystemExit(f"unknown cell {args.cell!r}; registered: {sorted(CELLS)}")
    if args.foreign and not args.matching:
        raise SystemExit("a comparison with another library's arm states its matching rule: --matching")
    _kind, spec = CELLS[args.cell]
    point = {"cell": args.cell, "tokens": spec.tokens, "d_v": spec.dv}

    def read_ghz():
        from rola.ops import carry as carry_ops

        return carry_ops.sm_clock_ghz()

    clock = clock_lock.engage(read_ghz)
    result = interleave(point, arm_specs(args.arm, args.foreign), matching=args.matching or "the same registered cell",
                        rounds=args.rounds, reps=args.reps, warmup=args.warmup, reference=args.reference,
                        seed=args.seed, hold=gpu_lock)
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
    arms += [{"label": arm["label"], "provider": arm["provider"], "arm": arm["arm"]} for arm in args.foreign]
    semantics = {"point": result["point"], "matching": result["matching"], "arms": arms, "rounds": args.rounds,
                 "reps": args.reps, "warmup": args.warmup, "seed": args.seed, "reference": result["reference"]}
    output = json.loads(portable(json.dumps(result), ROOT))
    store = Store("compare")
    sample = store.put(semantics, output=output,
                       provenance={"checkouts": [checkout(arm["worktree"]) for arm in args.arm]})
    print(f"record: {store.location} sample {sample['n']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="a registered cell (benchmarks/cells)")
    ap.add_argument("--arm", action="append", default=[], type=rola_arm, help="a rola arm; repeatable")
    ap.add_argument("--foreign", action="append", default=[], type=foreign_arm, help="another library's arm; repeatable")
    ap.add_argument("--matching", default=None, help="the rule that makes the point fair; required with --foreign")
    ap.add_argument("--reference", default=None, help="the label paired ratios divide by (default: the first arm)")
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
        paired = row.get("paired")
        ratio = f"  x{paired['ratio_median']:.4f} vs {paired['reference']} (iqr {paired['ratio_iqr']:.4f})" if paired else ""
        print(f"{row['label']:<18}{row['arm']:<42}{row['median_ms']:.4f} ms (iqr {row['iqr_ms']:.4f}){ratio}")
    print(f"clock: {result['clock']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    if args.record:
        record(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
