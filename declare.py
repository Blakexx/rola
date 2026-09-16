# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S DECLARATIONS: this checkout's targets for rola-devtools' declared build system (`rola_devtools.build`).

    python -m rola_devtools.build plan declare.py:all
    python -m rola_devtools.build run  declare.py:all --arg cells=flagship-dense,flagship-alt-k4

`declare(g, env, cells=..., timing=...)` declares one checkout's targets under the graph's scope and returns them (the
timing registrations by arm name), so a root elsewhere (rola-bench's) loads this file from each checkout it measures,
calls it per checkout with one shared timing server, and composes its own sessions. `root(g, **args)` is rola measuring itself: this checkout, its own server,
one session and one memory pass over the cells, every result stored.

Per checkout: `binary` (the gated build, held on every host slot, cached while its binary stands), `environment` (the
machine facts, every run), the instruments of `INSTRUMENTS` (SASS and the register walk once, cached; the phase clock,
pipe counters, stall census, timeline and the intra roofline on every carry cell, every run, holding the GPU), and with
a timing server the timing registrations: `carry_forward` and `carry_intra` on the carry cells, `entmax_solve@layer=C`
and `decode_step@layer=C` for each RoLA construction on the layer cells declared for it, and the clock reader. Every
target depends on the binary and the environment, whose outputs reach its key. Every cell is a NODE (`rola_devtools.cells.declare`) whose output is its record and the digest of the code that drew it;
a target that runs on cells takes those nodes as its data inputs, so nothing here or in the build system resolves a cell
by name at run time. This file imports nothing of rola's: it reads the constructions from `benchmarks/cells/layer.py` by
path.
"""
from __future__ import annotations

import sys
from pathlib import Path

from rola_devtools.build.declare import Env, load
from rola_devtools.cells import central
from rola_devtools.cells.declare import cells as cell_nodes
from rola_devtools.store import store
from rola_devtools.timing.declare import (
    measure_memory,
    measure_timing,
    register_clock_reader,
    register_timing,
    start_timing_server,
    stop_timing_server,
)

HERE = Path(__file__).resolve().parent
EXECUTORS = "executors:"
#: where this checkout's repository-local imports resolve, for code digests
IMPORT_ROOTS = ["tools", "benchmarks", "."]
#: what the extension is built from, besides the files `setup.py` imports
BUILD_DATA = ["csrc", "pyproject.toml", "tools/manifests", "tools/toolchains", "tools/sccache_pin.json",
              "tools/mold_pin.json", "tools/devtools.txt"]
#: name -> (tool, arguments, runs per carry cell, holds, data files its result depends on)
INSTRUMENTS = {
    "sass": ("tools/sass_gate.py", ["{binary}"], False, {"host_cpu": 1}, []),
    "registers": ("tools/life_ranges.py", ["--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], False,
                  {"host_cpu": 1}, ["csrc/rola/src"]),
    "phases": ("tools/phase_ledger.py", ["{cell}", "--launches", "1"], True, {"gpu": "all"}, []),
    "counters": ("tools/pipe_counters.py", ["{cell}"], True, {"gpu": "all"}, []),
    "census": ("tools/stall_census.py", ["{cell}"], True, {"gpu": "all"},
               ["tools/budgets/carry.json", "csrc/rola/src/carry/carry_kernel.cuh"]),
    "timeline": ("tools/pipe_timeline.py", ["--cell", "{cell}"], True, {"gpu": "all"}, []),
    "roofline": ("tools/roofline.py", ["--cells", "{cell}"], True, {"gpu": "all"}, []),
}
CARRY_ARMS = ("carry_forward", "carry_intra")
LAYER_ARMS = ("entmax_solve", "decode_step")
#: seconds an instrument may run on one cell
INSTRUMENT_TIMEOUT_S = 3600


def checkout(path, *, python: str, label: str) -> Env:
    path = Path(path).resolve()
    return Env(label, python, str(path), {"PYTHONPATH": f"{path}:{path}/benchmarks:{path}/tools"})


def kind(cell: str) -> str:
    """A central cell's kind, by its data provider's module (`rola_devtools.cells.carry`, `.layer`, `.qkv`)."""
    return central().cell(cell)["data"].split(":")[0].rsplit(".", 1)[-1]


def _code(entry: str, data=()) -> dict:
    return {"entry": entry, "roots": IMPORT_ROOTS, "data": list(data)}


def declare(g, env: Env, *, cells, timing=None, instruments=tuple(INSTRUMENTS)) -> dict:
    carry = [c for c in cells if kind(c) == "carry"]
    layer = [c for c in cells if kind(c) == "layer"]
    binary = g.node("binary", executor=EXECUTORS + "compile_kernel", env=env, holds={"host_cpu": "all"},
                    verify=EXECUTORS + "binary_present", code=_code("setup.py", BUILD_DATA))
    environment = g.node("environment", executor=EXECUTORS + "probe_environment", env=env, cache=False)
    facts = {"binary": binary, "environment": environment}
    out = {"binary": binary, "environment": environment, "instruments": {}, "entries": {}, "clock": None}
    for name in instruments:
        tool, args, per_cell, holds, data = INSTRUMENTS[name]
        if per_cell and not carry:
            continue
        out["instruments"][name] = g.node(
            name, executor=EXECUTORS + "run_tool", env=env, deps=facts,
            inputs=cell_nodes(g, carry) if per_cell else (), holds=holds,
            params={"tool": tool, "args": args, "per_cell": per_cell, "timeout": INSTRUMENT_TIMEOUT_S},
            cache=not per_cell, code=_code(tool, data))
    if timing is None:
        return out
    timed = _code("benchmarks/executors.py", ["benchmarks/bench"])
    for arm in CARRY_ARMS if carry else ():
        out["entries"][arm] = register_timing(g, arm, server=timing, env=env, executor=EXECUTORS + "timed",
                                              cells=cell_nodes(g, carry), params={"arm": arm}, deps=facts, code=timed)
    constructions = load(Path(env.cwd) / "benchmarks" / "cells" / "layer.py")["CONSTRUCTIONS"]
    for construction in constructions.values():
        takes = [c for c in layer if c in construction.cells]
        for arm in LAYER_ARMS:
            wanted = [c for c in takes if arm != "decode_step" or central().cell(c)["params"]["decode_steps"] > 0]
            if wanted:
                name = f"{arm}@layer={construction.name}"
                out["entries"][name] = register_timing(g, name, server=timing, env=env, executor=EXECUTORS + "timed",
                                                       cells=cell_nodes(g, wanted), params={"arm": name}, deps=facts,
                                                       code=timed)
    out["clock"] = register_clock_reader(g, "clock", server=timing, env=env, executor=EXECUTORS + "read_clock", deps=facts,
                                         code=_code("benchmarks/executors.py"))
    return out


def root(g, python: str = sys.executable, label: str = "rola", cells: str = "flagship-dense,flagship-alt-k4",
         instruments: str = ",".join(INSTRUMENTS), rounds: str = "8", reps: str = "11", store_root: str = "") -> dict:
    names = cells.split(",")
    server = start_timing_server(g)
    mine = declare(g.scoped(label), checkout(HERE, python=python, label=label), cells=names, timing=server,
                   instruments=[i for i in instruments.split(",") if i])
    root_dir = store_root or None
    stores = [store(g, f"store/{name}", source=target, location=f"rola/{name}", cache=target.cache, root=root_dir)
              for name, target in mine["instruments"].items()]
    session = measure_timing(g, "session", server=server, entries=list(mine["entries"].values()), clock=mine["clock"],
                             cells=names, rounds=int(rounds), reps=int(reps))
    memory = measure_memory(g, "memory", server=server, entries=list(mine["entries"].values()), cells=names)
    stores += [store(g, "store/session", source=session, location="timing/session", root=root_dir),
               store(g, "store/memory", source=memory, location="timing/memory", root=root_dir)]
    stop = stop_timing_server(g, server=server, after=[session, memory, *stores])
    return {"all": g.group("all", [*stores, stop])}
