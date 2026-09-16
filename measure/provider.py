# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S RUNNER: this checkout's bench subjects as timed arms, built on central cells for the timing system.

`arms(cell)` takes a central carry cell (a shape, a draw and the state it binds) or a central layer cell (an input)
and returns a builder for every arm this checkout runs on it, or refuses the cell by raising:
- any cell, when the binary lacks an arm its tree ships (`tools/manifests/shipped_set.json`): an iteration build
  (`ROLA_CARRY_ARMS`) measures a subset of the tree;
- a carry cell whose carry arm (D, DV, warps_per_cta at `measure.cells.WARPS_PER_CTA`) the binary does not carry;
- data that is neither kind.
`measure/executors.py`'s timing entry calls it on the cell a registration hands it; a profiler launches one arm
through `oneshot_argv`.
An arm is a subject that applies to the cell (`measure.subjects.applicable`, the cell's own facts) and whose kernel this
binary carries at the cell's shape (the intra arm at the cell's depth and window for `intra_forward` and `carry_intra`,
the decode arm for `decode_step`), with its dials: its name is the subject, then `@calls=N` for a call count other than
one and `@schedule=S` for a carry order other than `first`, each only where the subject reads that dial
(`Subject.calls`, `Subject.dials`), and on a layer cell `@layer=C` for each RoLA construction declared for that input
(`measure.cells.layer.CONSTRUCTIONS`). Only the arm asked for is built.
Building one proves two things before anything is timed, and refuses the arm by name when either fails: from a fact the
device produces, that this binary carries the subject's family (its stamp entry; a path or a hash cannot catch a stale
binary), and that `import rola` resolved inside this checkout.

A built arm is a `rola_devtools.timing.Timed`: its call times one launch between two CUDA events (the canonical
instrument, `cuda_events`) and returns milliseconds; its reset is the launch's (a carried state restored, a decode state
re-seeded), run untimed before every call. What an arm reports is the cell's facts beside the arm's dials and the
binary's: the manifest digest, the family stamp, the device, torch, the assembler, and the SM clock read when the arm
was built (docs/measurement.md).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

#: this checkout: `import rola` must resolve inside it
CHECKOUT = Path(__file__).resolve().parents[1]


def arm_name(subject: str, *, calls: int = 1, schedule: str = "first", construction: str | None = None) -> str:
    """An arm's name: the subject, then each dial that is not its default, then a layer arm's construction."""
    return subject + (f"@calls={calls}" if calls != 1 else "") + (f"@schedule={schedule}" if schedule != "first" else "") \
        + (f"@layer={construction}" if construction else "")


def arms(data) -> dict:
    """Every arm this checkout runs on a cell's data, each as a builder; raises to refuse the cell."""
    from itertools import product

    from rola_devtools.cells.carry import CarryCell
    from rola_devtools.cells.layer import LayerCell

    from measure.cells import arm_key
    from measure.cells.layer import CONSTRUCTIONS
    from measure.subjects import SUBJECTS, applicable
    from rola.ops import carry
    from rola.ops.carry import ORDER_POLICIES

    _refuse_a_partial_binary()
    if isinstance(data, CarryCell):
        kind, constructions = "carry", [None]
        if arm_key(data) not in {tuple(arm) for arm in carry.arms()}:
            raise LookupError(f"{data.name}: this binary carries no carry arm {arm_key(data)} (D, DV, warps_per_cta); it "
                              f"carries {sorted(tuple(arm) for arm in carry.arms())}")
    elif isinstance(data, LayerCell):
        kind, constructions = "layer", [c for c in CONSTRUCTIONS.values() if data.name in c.cells]
        if not constructions:
            raise LookupError(f"{data.name}: no RoLA construction is declared for this layer cell")
    else:
        raise TypeError(f"rola's runner takes a central carry cell or layer cell, got {type(data).__name__}")
    out = {}
    for subject in SUBJECTS.values():
        schedules = tuple(ORDER_POLICIES) if "schedule" in subject.dials else ("first",)
        for construction, calls, schedule in product(constructions, subject.calls, schedules):
            if subject.name in applicable(data, kind, calls) and _kernel_carried(subject.name, data, construction):
                name = arm_name(subject.name, calls=calls, schedule=schedule,
                                construction=construction and construction.name)
                out[name] = partial(_build, subject.name, kind, data, calls, schedule, construction)
    return out


def _refuse_a_partial_binary() -> None:
    from rola.ops import carry

    if str(CHECKOUT / "tools") not in sys.path:
        sys.path.insert(0, str(CHECKOUT / "tools"))
    import gen_shards

    declared = {tuple(row) for row in gen_shards.CARRY_ARMS}
    built = {tuple(arm) for arm in carry.arms()}
    if declared - built:
        raise RuntimeError(f"this binary carries {sorted(built)} and lacks the shipped arms {sorted(declared - built)}: an "
                           "iteration build (ROLA_CARRY_ARMS) measures a subset of the tree; build the shipped set")


def _kernel_carried(subject: str, spec, construction) -> bool:
    """Whether this binary carries the kernel `subject` launches at the cell's shape, or a layer cell's construction
    (the cell's own facts are `measure.subjects.applicable`'s)."""
    from rola.ops import carry, decode, intra
    from rola.ops.prefill import _intra_level_width

    if subject == "carry_intra":
        return len(set(spec.widths)) == 1 and _intra_level_width(len(spec.widths)) == spec.widths[0]
    if subject == "intra_forward":
        widths = {lw for levels, lw, window, _smem in intra.arms() if levels == len(spec.widths) and window == carry.WINDOW}
        return bool(widths) and max(spec.widths) <= min(widths)
    if subject == "decode_step":
        rows = decode.arms()
        padded = [dv for dv, _d, _decay in rows if dv >= spec.dv]
        return bool(padded) and (min(padded), len(construction.widths), False) in rows
    return True


def _build(name: str, kind: str, spec, calls: int, schedule: str, construction):
    import torch
    from rola_devtools.timing import Timed

    import rola
    from measure.subjects import SUBJECTS
    from rola.ops import carry as carry_ops
    from rola.ops._ext import extension

    subject = SUBJECTS[name]
    stamp = getattr(extension(), subject.stamp, None)
    if stamp is None:
        raise RuntimeError(f"this binary carries no {subject.stamp} entry, so it cannot state that it has {name}'s family")
    rola_file = Path(rola.__file__).resolve()
    if not rola_file.is_relative_to(CHECKOUT):
        raise RuntimeError(f"`import rola` resolved outside this checkout ({CHECKOUT.name}); the arm would time another "
                           "tree's binary under this tree's name")
    fx = {"cell": spec} if kind == "carry" else _layer_fixture(spec, construction)
    fx.update(schedule=schedule, calls=calls)
    launch = subject.build(fx)
    try:
        from rola_cu13._build_config import BUILD_CONFIG
    except ImportError:
        BUILD_CONFIG = {}
    props = torch.cuda.get_device_properties(0)

    def call() -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        launch.call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    widths = spec.widths if kind == "carry" else construction.widths
    cell = {"cell": spec.name, "kind": kind, "subject": name, "level": subject.level, "calls": calls,
            "schedule": fx["schedule"],
            "state": getattr(spec, "state", None), "backing": getattr(spec, "backing", None),
            "construction": construction and construction.name, "tokens": spec.tokens, "d_v": spec.dv,
            "widths": list(widths),
            "manifest_sha256": BUILD_CONFIG.get("manifest_sha256"), "ptxas": BUILD_CONFIG.get("ptxas"),
            "family_stamp": int(stamp()), "device": torch.cuda.get_device_name(0), "sm": f"sm_{props.major}{props.minor}",
            "torch": torch.__version__, "rola_file": rola_file.relative_to(CHECKOUT).as_posix(),
            "sm_ghz_at_build": carry_ops.sm_clock_ghz()}
    return Timed(call=call, built=cell, instrument="cuda_events", reset=launch.reset,
                 outside_allocator=launch.outside_allocator)


def _layer_fixture(spec, construction):
    from measure.cells.layer import build

    return build(spec, construction)


def oneshot_argv(cell: str, arm: str = "carry_forward") -> list[str]:
    """One untimed launch of `arm` on `cell` by this checkout's runner, under this interpreter: what a profiler wraps.
    The arm is built exactly as a comparison builds it, so what a profile counts is the launch a comparison times."""
    #: AS A MODULE, from the checkout: a package file run by path has its own directory for a root and cannot import
    #: its package (the `measure/` rename turned every profiled launch into `No module named 'measure'`)
    return [sys.executable, "-m", "measure.provider", "--oneshot", cell, arm]


def _oneshot(cell: str, arm: str) -> None:
    import torch
    from rola_devtools.cells import central

    from measure.cells import by_name

    if cell not in central().cells:
        raise SystemExit(f"{cell!r} is not a central cell (rola_devtools.cells)")
    offered = arms(by_name(cell))
    if arm not in offered:
        raise SystemExit(f"{cell}: this checkout runs no arm {arm}; it runs {sorted(offered)}")
    offered[arm]().call()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--oneshot":
        raise SystemExit("usage: provider.py --oneshot CELL ARM")
    sys.path[:0] = [str(CHECKOUT)]
    _oneshot(sys.argv[2], sys.argv[3])
