# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PROVIDER: this checkout's bench subjects as arms of the interleaving driver (`rola_devtools.interleave`).

    ArmSpec(label, "bench.provider:arms", "carry_forward@schedule=identity", python=<the checkout's venv python>,
            cwd=<the checkout>, env={"PYTHONPATH": "<the checkout>:<the checkout>/benchmarks"})

A point names a registered cell (`benchmarks/cells`), and may state the facts another library's arms read from it
(`tokens`, `d_v`), which must be that cell's: the point is what a comparison holds equal. An ARM is a subject that applies to
that cell (`bench.subjects.SUBJECTS`) with its dials, which belong to the arm and never to the run: its name is the
subject, then `@calls=N` for a call count other than one, `@schedule=S` for a carry order other than `first`, and
`@state=S` for a state arm other than `fresh`, each only where the subject reads that dial (`Subject.calls`,
`Subject.dials`). Only the arms a comparison asks for are built. Building one proves two things before anything is
timed, and refuses the arm by name when either fails: from a fact the device produces, that this binary carries the
subject's family (its stamp entry; a path or a hash cannot catch a stale binary), and that `import rola` resolved inside
this checkout.

The call times one launch between two CUDA events (`bench.pairing`'s canonical instrument) and returns milliseconds.
The cell an arm reports is the registry cell's facts beside the arm's dials and the binary's: the manifest digest, the
family stamp, the device, torch, the assembler, and the SM clock read when the arm was built (docs/measurement.md).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

#: this checkout: `import rola` must resolve inside it
CHECKOUT = Path(__file__).resolve().parents[2]


def arm_name(subject: str, calls: int = 1, schedule: str = "first", state: str = "fresh") -> str:
    """An arm's name: the subject, then each dial that is not its default."""
    return subject + (f"@calls={calls}" if calls != 1 else "") + (f"@schedule={schedule}" if schedule != "first" else "") \
        + (f"@state={state}" if state != "fresh" else "")


def arms(point: dict) -> dict:
    """Every arm `point`'s cell carries, each as a builder."""
    from itertools import product

    from bench.driver import CELLS
    from bench.subjects import STATE_ARMS, SUBJECTS, applicable
    from rola.ops.carry import ORDER_POLICIES

    if not set(point) <= {"cell", "tokens", "d_v"} or point.get("cell") not in CELLS:
        raise KeyError(f"a point names one registered cell (benchmarks/cells) and at most its tokens and d_v, got {point!r}")
    kind, spec = CELLS[point["cell"]]
    stated = {"tokens": spec.tokens, "d_v": spec.dv}
    if any(point[key] != value for key, value in stated.items() if key in point):
        raise ValueError(f"the point states {point}, but {spec.name} is {stated}")
    out = {}
    for subject in SUBJECTS.values():
        schedules = tuple(ORDER_POLICIES) if "schedule" in subject.dials else ("first",)
        states = STATE_ARMS if "state" in subject.dials else ("fresh",)
        for calls, schedule, state in product(subject.calls, schedules, states):
            if subject.name in applicable(spec, kind, calls):
                out[arm_name(subject.name, calls, schedule, state)] = partial(_build, subject.name, kind, spec, calls,
                                                                              schedule, state)
    return out


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
        from rola._build_config import BUILD_CONFIG
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
