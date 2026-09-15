# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S RUNNER: this checkout's bench subjects as arms of the interleaving driver (`rola_devtools.interleave`).

    ArmSpec(label, "bench.provider:arms", "carry_forward@schedule=identity", runner="rola",
            python=<the checkout's venv python>, cwd=<the checkout>,
            env={"PYTHONPATH": "<the checkout>:<the checkout>/benchmarks"})

The driver calls `arms` with the DATA of each cell a point sends the rola runner (`rola_devtools.cells.build`): a carry
cell (`benchmarks.cells:carry_cell`, a shape and a draw) or a layer cell (`benchmarks.cells.layer:layer_cell`, a
constructor). It returns a builder for every arm this checkout runs on that cell, or refuses the cell by raising:
- any cell, when the binary lacks an arm its tree ships (`tools/manifests/shipped_set.json`): an iteration build
  (`ROLA_CARRY_ARMS`) measures a subset of the tree;
- a carry cell whose carry arm (D, DV, warps_per_cta) the binary does not carry;
- data that is neither kind.
An arm is a subject that applies to the cell (`bench.subjects.applicable`, the cell's own facts) and whose kernel this
binary carries at the cell's shape (the intra arm at the cell's depth and window for `intra_forward` and `prefill_op`,
the decode arm for `decode_step`), with its dials: its name is the subject, then `@calls=N` for a call count other than
one, `@schedule=S` for a carry order other than `first`, and `@state=S` for a state arm other than `fresh`, each only
where the subject reads that dial (`Subject.calls`, `Subject.dials`). Only the arms a comparison asks for are built.
Building one proves two things before anything is timed, and refuses the arm by name when either fails: from a fact the
device produces, that this binary carries the subject's family (its stamp entry; a path or a hash cannot catch a stale
binary), and that `import rola` resolved inside this checkout.

The call times one launch between two CUDA events (the canonical instrument, `cuda_events`) and returns milliseconds.
What an arm reports is the cell's facts beside the arm's dials and the binary's: the manifest digest, the family stamp,
the device, torch, the assembler, and the SM clock read when the arm was built (docs/measurement.md).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

#: this checkout: `import rola` must resolve inside it
CHECKOUT = Path(__file__).resolve().parents[2]


def arm_name(subject: str, calls: int = 1, schedule: str = "first", state: str = "fresh") -> str:
    """An arm's name: the subject, then each dial that is not its default."""
    return subject + (f"@calls={calls}" if calls != 1 else "") + (f"@schedule={schedule}" if schedule != "first" else "") \
        + (f"@state={state}" if state != "fresh" else "")


def arms(data) -> dict:
    """Every arm this checkout runs on a cell's data, each as a builder; raises to refuse the cell."""
    from itertools import product

    from bench.subjects import STATE_ARMS, SUBJECTS, applicable
    from benchmarks.cells import CellSpec
    from benchmarks.cells.layer import LayerCellSpec
    from rola.ops import carry
    from rola.ops.carry import ORDER_POLICIES

    _refuse_a_partial_binary()
    if isinstance(data, CellSpec):
        kind = "carry"
        if data.arm not in {tuple(arm) for arm in carry.arms()}:
            raise LookupError(f"{data.name}: this binary carries no carry arm {data.arm} (D, DV, warps_per_cta); it "
                              f"carries {sorted(tuple(arm) for arm in carry.arms())}")
    elif isinstance(data, LayerCellSpec):
        kind = "layer"
    else:
        raise TypeError(f"rola's runner takes a carry cell (benchmarks.cells:carry_cell) or a layer cell "
                        f"(benchmarks.cells.layer:layer_cell), got {type(data).__name__}")
    out = {}
    for subject in SUBJECTS.values():
        if not _kernel_carried(subject.name, data):
            continue
        schedules = tuple(ORDER_POLICIES) if "schedule" in subject.dials else ("first",)
        states = STATE_ARMS if "state" in subject.dials else ("fresh",)
        for calls, schedule, state in product(subject.calls, schedules, states):
            if subject.name in applicable(data, kind, calls):
                out[arm_name(subject.name, calls, schedule, state)] = partial(_build, subject.name, kind, data, calls,
                                                                              schedule, state)
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


def _kernel_carried(subject: str, spec) -> bool:
    """Whether this binary carries the kernel `subject` launches at the cell's shape (the cell's own facts are
    `bench.subjects.applicable`'s)."""
    from rola.ops import carry, decode, intra
    from rola.ops.prefill import _intra_level_width

    if subject == "prefill_op":
        return len(set(spec.widths)) == 1 and _intra_level_width(len(spec.widths)) == spec.widths[0]
    if subject == "intra_forward":
        widths = {lw for levels, lw, window, _smem in intra.arms() if levels == len(spec.widths) and window == carry.WINDOW}
        return bool(widths) and max(spec.widths) <= min(widths)
    if subject == "decode_step":
        rows = decode.arms()
        padded = [dv for dv, _d, _decay in rows if dv >= spec.dv]
        return bool(padded) and (min(padded), len(spec.widths), False) in rows
    return True


def _build(name: str, kind: str, spec, calls: int, schedule: str, state: str):
    import torch
    from rola_devtools.interleave import Arm

    import rola
    from bench.subjects import SUBJECTS
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
    fx = {"cell": spec} if kind == "carry" else {**_layer_fixture(spec), "cell": spec}
    fx.update(state_arm=state, schedule=schedule, calls=calls)
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

    cell = {"cell": spec.name, "kind": kind, "subject": name, "calls": calls, "state": fx["state_arm"],
            "schedule": fx["schedule"], "tokens": spec.tokens, "d_v": spec.dv, "widths": list(spec.widths),
            "manifest_sha256": BUILD_CONFIG.get("manifest_sha256"), "ptxas": BUILD_CONFIG.get("ptxas"),
            "family_stamp": int(stamp()), "device": torch.cuda.get_device_name(0), "sm": f"sm_{props.major}{props.minor}",
            "torch": torch.__version__, "rola_file": rola_file.relative_to(CHECKOUT).as_posix(),
            "sm_ghz_at_build": carry_ops.sm_clock_ghz()}
    return Arm(cell=cell, call=call, instrument="cuda_events")


def _layer_fixture(spec):
    from benchmarks.cells.layer import build

    return build(spec)


def oneshot_argv(cell: str, arm: str = "carry_forward") -> list[str]:
    """One untimed launch of `arm` on `cell` by this checkout's runner, under this interpreter: what a profiler wraps.
    The arm is built exactly as a comparison builds it, so what a profile counts is the launch a comparison times."""
    return [sys.executable, str(Path(__file__).resolve()), "--oneshot", cell, arm]


def _oneshot(cell: str, arm: str) -> None:
    import torch

    from benchmarks.cells.registry import CELLS

    if cell not in CELLS:
        raise SystemExit(f"{cell!r} is not a registered cell")
    offered = arms(CELLS[cell][1])
    if arm not in offered:
        raise SystemExit(f"{cell}: this checkout runs no arm {arm}; it runs {sorted(offered)}")
    offered[arm]().call()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--oneshot":
        raise SystemExit("usage: provider.py --oneshot CELL ARM")
    sys.path[:0] = [str(CHECKOUT), str(CHECKOUT / "benchmarks")]
    _oneshot(sys.argv[2], sys.argv[3])
