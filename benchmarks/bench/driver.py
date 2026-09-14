"""The process that runs the harness: argument parsing, the discipline, the ledger write.

Thin by design. Every driver under `benchmarks/unit/` and `benchmarks/integration/` is
this module plus the name of the subject it prices, so the discipline cannot be acquired
in nine slightly different ways.

An ARM SPEC is `subject@cell`, or `subject@cell@calls=N` for a subject priced as N carried
calls (`bench.subjects.Subject.calls`) -- every part is a registry name or a declared count,
so a command line is citable and a row's `arm` field reads the same way. `--ab A:B` runs the two specs
interleaved rep by rep through `bench.pairing.interleaved_ab`; without it each spec is
measured alone, which is the BASELINE mode.

Usage (invoked BARE -- `bench.discipline.disciplined` takes `gpu_lock()` itself):

    python benchmarks/unit/bench_liveness.py --tier landing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.discipline import disciplined
from bench.pairing import INSTRUMENTS, REPS_PER_ROUND, Arm, interleaved_ab, measure_arm
from bench.regression import CANONICAL_INSTRUMENT
from bench.stats import MIN_ROUNDS, paired_verdict
from bench.subjects import SUBJECTS, applicable
from benchmarks.cells import carry_cells
from benchmarks.cells.layer import build as build_layer_fixture
from benchmarks.cells.layer import layer_cells

#: THE REGISTRY, read once: `{name -> (kind, record)}` over BOTH cell kinds. A name is
#: unique across the registry, so a command line names a cell and the kind follows.
REPO = Path(__file__).resolve().parents[2]

CELLS = {**{c.name: ("carry", c) for c in carry_cells()},
         **{c.name: ("layer", c) for c in layer_cells()}}

#: Named GROUPS, so a run is cited by a name rather than by a slice of the registry.
GROUPS = {
    "all": list(CELLS),
    "carry": [n for n, (kind, _) in CELLS.items() if kind == "carry"],
    "layer": [n for n, (kind, _) in CELLS.items() if kind == "layer"],
    "decode": [n for n, (kind, c) in CELLS.items()
               if kind == "layer" and c.decode_steps > 0],
}

#: The probe version of THIS harness. It is inside every fingerprint, so a redefinition
#: of what a subject measures starts a new series rather than extending the old one.
PROBE_VERSION = 2

#: Rounds per tier. `landing` is `MIN_ROUNDS`, which is DERIVED from the paired test's
#: alpha: below it no outcome can be significant and the test is undefined, not weak.
TIER_ROUNDS = {"light": 3, "landing": MIN_ROUNDS}


def parse_spec(spec: str, subjects) -> tuple[str, str, int]:
    subject, _, rest = spec.partition("@")
    cell, _, count = rest.partition("@")
    if not cell or (count and not count.startswith("calls=")):
        raise SystemExit(f"arm spec {spec!r} is not `subject@cell[@calls=N]`")
    calls = int(count.removeprefix("calls=")) if count else 1
    if subject not in subjects:
        raise SystemExit(f"{subject!r} is not one of this driver's subjects: {sorted(subjects)}")
    if cell not in CELLS:
        raise SystemExit(f"{cell!r} is not a registered cell: {sorted(CELLS)}")
    kind, spec = CELLS[cell]
    if subject not in applicable(spec, kind, calls):
        raise SystemExit(f"{subject!r} does not apply to {cell!r} at {calls} call(s) "
                         f"(applicable: {applicable(spec, kind, calls)})")
    return subject, cell, calls


def fixture(cell: str, cache: dict) -> dict:
    """The fixture is built ONCE per cell per process: two subjects on one cell must see
    the same fixture, or the pairing is across two draws as well as two arms.

    A CARRY cell's fixture is the record itself -- the draw is realized by whichever
    subject reads it, from the cell's own seed, so there is nothing to build ahead. A
    LAYER cell's is the constructed layer, which is expensive and shared.
    """
    if cell not in cache:
        kind, spec = CELLS[cell]
        cache[cell] = ({"cell": spec} if kind == "carry"
                       else {**build_layer_fixture(spec), "cell": spec})
    return cache[cell]


def arm_name(subject: str, cell: str, calls: int) -> str:
    return f"{subject}|{cell}" + (f"|calls={calls}" if calls != 1 else "")


def build_arm(subject: str, cell: str, calls: int, cache: dict) -> Arm:
    #: the count goes on a COPY: the cached fixture is shared by every subject on the cell.
    arm = SUBJECTS[subject].build({**fixture(cell, cache), "calls": calls})
    return Arm(name=arm_name(subject, cell, calls), call=arm.call)


def record(run, out: list, args, rounds: int, specs: list[str]) -> None:
    """The run as one sample in `rola_results` at `bench_driver`: the semantics are the arm specs, the counts, the
    instrument, the tree (commit and tracked diff), the binary's manifest and the device."""
    from rola_results import Store, checkout

    here = checkout(REPO)
    semantics = {"specs": specs, "tier": args.tier, "rounds": rounds, "reps_per_round": args.reps,
                 "instrument": args.instrument, "probe_version": PROBE_VERSION, "git_sha": here["git_sha"],
                 "diff_sha256": here["diff_sha256"], "manifest_sha256": run.env.get("manifest_sha256"),
                 "device": run.env.get("gpu")}
    results = json.loads(json.dumps(out, default=str))
    sample = Store("bench_driver").put(semantics, output=results, provenance={
        **here, "run_id": run.run_id, "stage": args.stage, "clocks": json.loads(json.dumps(run.clocks, default=str))})
    print(f"record: bench_driver sample {sample['n']} ({len(results)} result(s))")


def main(subjects, argv=None, *, description: str = "") -> int:
    """One driver's whole body. `subjects` is the roster this driver is allowed to run."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cells", default="all",
                   help=f"comma-separated cell names, or a group: {sorted(GROUPS)}")
    p.add_argument("--subject", default=None,
                   help=f"one of {sorted(subjects)}; default is all of this driver's")
    p.add_argument("--ab", default=None, metavar="A:B",
                   help="two `subject@cell` specs, run interleaved rep by rep")
    p.add_argument("--tier", default="light", choices=sorted(TIER_ROUNDS))
    p.add_argument("--rounds", type=int, default=None, help="override the tier's rounds")
    p.add_argument("--reps", type=int, default=REPS_PER_ROUND, help="reps per round")
    p.add_argument("--instrument", default=CANONICAL_INSTRUMENT, choices=sorted(INSTRUMENTS))
    p.add_argument("--json", default=None, help="also write the results here")
    #: THE RECORD: the run as one sample in `rola_results` at `bench_driver`, one shared call
    #: site (below) for every unit driver.
    p.add_argument("--no-record", action="store_true",
                   help="do not store the results (a scratch run, not a record)")
    p.add_argument("--stage", default=None, help="the queue stage this run belongs to")
    args = p.parse_args(argv)

    rounds = args.rounds or TIER_ROUNDS[args.tier]
    cmd = f"bench {args.ab or args.subject or 'all'} --tier {args.tier}"
    names = GROUPS[args.cells] if args.cells in GROUPS else args.cells.split(",")
    out = []

    with disciplined(tier=args.tier, cmd=cmd) as run:
        print(f"run {run.run_id}  sha {run.sha[:12]}{' DIRTY' if run.dirty else ''}  "
              f"clock {run.clocks['sm_mhz']:.0f} MHz  {run.env['gpu']}")
        cache: dict = {}

        if args.ab:
            a_spec, _, b_spec = args.ab.partition(":")
            a_unit, b_unit = parse_spec(a_spec, subjects), parse_spec(b_spec, subjects)
            specs = [a_spec, b_spec]
            a, b = build_arm(*a_unit, cache), build_arm(*b_unit, cache)
            result = interleaved_ab(a, b, rounds=rounds, reps_per_round=args.reps,
                                    instrument=args.instrument)
            summary = result.summary()
            summary["significance"] = paired_verdict(result.diffs())
            #: THE RAW PAIRED SAMPLES, in run order: the alternation is the measurement design
            summary["a_times"], summary["b_times"] = list(result.a_times), list(result.b_times)
            out.append(summary)
            print(json.dumps(summary, indent=1, sort_keys=True, default=str))
        else:
            wanted = [args.subject] if args.subject else sorted(subjects)
            units = [(subject, cell, calls) for cell in names for subject in wanted
                     for calls in SUBJECTS[subject].calls
                     if subject in applicable(CELLS[cell][1], CELLS[cell][0], calls)]
            specs = [f"{s}@{c}" + (f"@calls={n}" if n != 1 else "") for s, c, n in units]
            for subject, cell, calls in units:
                arm = build_arm(subject, cell, calls, cache)
                res = measure_arm(arm, rounds=rounds, reps_per_round=args.reps,
                                  instrument=args.instrument)
                res["cell"], res["subject"], res["calls"] = cell, subject, calls
                out.append(res)
                print(f"  {subject:<20} {cell:<22} {calls:>2} call(s) "
                      f"{res['median_ms']:9.4f} ms  IQR {res['iqr_ms']:.4f}")
        if not args.no_record:
            record(run, out, args, rounds, specs)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
    return 0
