# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S DECLARATIONS: this checkout's targets for rola-devtools' declared build system (`rola_devtools.build`).

    python -m rola_devtools.build plan declare.py:all
    python -m rola_devtools.build run  declare.py:all --only 'rola/phases' --skip '*/timeline'

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
by name at run time. This file imports nothing of rola's: it reads the constructions from `measure/cells/layer.py` by
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
EXECUTORS = "measure.executors:"
#: where this checkout's repository-local imports resolve, for code digests
IMPORT_ROOTS = ["tools", "."]
#: what the extension is built from, besides the files `setup.py` imports
BUILD_DATA = ["csrc", "pyproject.toml", "tools/manifests", "tools/toolchains", "tools/sccache_pin.json",
              "tools/mold_pin.json", "tools/devtools.txt"]
#: name -> (tool, arguments, runs per carry cell, holds, data files its result depends on)
INSTRUMENTS = {
    "sass": ("tools/sass_gate.py", ["{binary}"], False, {"host_cpu": 1}, [], 600),
    "registers": ("tools/life_ranges.py", ["--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], False,
                  {"host_cpu": 1}, ["csrc/rola/src"], 600),
    "phases": ("tools/phase_ledger.py", ["{cell}", "--launches", "1"], True, {"gpu": "all"}, [], 120),
    "counters": ("tools/pipe_counters.py", ["{cell}"], True, {"gpu": "all"}, [], 900),
    "census": ("tools/stall_census.py", ["{cell}"], True, {"gpu": "all"},
               ["tools/budgets/carry.json", "csrc/rola/src/carry/carry_kernel.cuh"], 900),
    "timeline": ("tools/pipe_timeline.py", ["--cell", "{cell}"], True, {"gpu": "all"}, [], 900),
    "roofline": ("tools/roofline.py", ["--cells", "{cell}"], True, {"gpu": "all"}, [], 120),
}
CARRY_ARMS = ("carry_forward", "carry_intra")
LAYER_ARMS = ("entmax_solve", "decode_step")
#: seconds an instrument may run on one cell
#: EACH INSTRUMENT'S TIMEOUT IS PER CELL AND SIZED TO THE INSTRUMENT: a phase-clock launch is seconds on the largest
#: cell (37 cells in 1.1 minutes, measured 2026-09-16), an ncu replay minutes. A hang then costs one timeout and lands
#: as that cell's recorded failure; under one hour for every instrument, `nl64k-dense`'s device hang cost a baseline
#: four hours of waiting and never a record.
#: a whole test tier is minutes, not one launch
TIER_TIMEOUT_S = 3600


def checkout(path, *, python: str, label: str) -> Env:
    path = Path(path).resolve()
    return Env(label, python, str(path), {"PYTHONPATH": f"{path}:{path}/tools"})


def kind(cell: str) -> str:
    """A central cell's kind, by its data provider's module (`rola_devtools.cells.carry`, `.layer`, `.qkv`)."""
    return central().cell(cell)["data"].split(":")[0].rsplit(".", 1)[-1]


def _code(entry: str, data=()) -> dict:
    return {"entry": entry, "roots": IMPORT_ROOTS, "data": list(data)}


def declare(g, env: Env, *, timing=None, instruments=tuple(INSTRUMENTS)) -> dict:
    """EVERY target this checkout has. A node declares the cells it runs on; nothing here takes a cell list, and a build
    that wants fewer prunes by label."""
    cells = sorted(central().cells)
    #: A MEASUREMENT RUNS ON CELLS SIZED FOR MEASUREMENT: the registry states who each carry cell is sized for
    #: (`tier`: `oracle` for the fp64 reference, `probe` for a measurement, `both`), and an instrument or a timing
    #: arm over a 16-token reference cell measures nothing -- the oracle-tier cells are the diff surfaces' and the
    #: tier node's (`SURFACES`, `oracle-tier`), not the instruments'. Every carry cell declared here took a run from 12
    #: cells to 49 and an ncu replay apiece (2026-09-16).
    carry = [c for c in cells if kind(c) == "carry" and central().cell(c)["params"].get("tier") in ("probe", "both")]
    layer = [c for c in cells if kind(c) == "layer"]
    binary = g.node("binary", executor=EXECUTORS + "compile_kernel", env=env, holds={"host_cpu": "all"},
                    verify=EXECUTORS + "binary_present", code=_code("setup.py", BUILD_DATA))
    environment = g.node("environment", executor=EXECUTORS + "probe_environment", env=env, cache=False)
    facts = {"binary": binary, "environment": environment}
    out = {"binary": binary, "environment": environment, "instruments": {}, "entries": {}, "clock": None,
           "sides": _sides(g, env, facts), "diffs": {}}
    out["diffs"]["carry-vs-oracle"] = _kernel_vs_oracle(g, env, out["sides"])
    #: THE ORACLE TIER AS ONE TARGET: pytest over `tests/oracle`, keyed on the tier's code and the binary, cached while
    #: both stand; its output is the outcome and the margins, stored beside the instruments (`rola/oracle-tier`)
    out["instruments"]["oracle-tier"] = g.node(
        "oracle-tier", executor=EXECUTORS + "pytest_tier", env=env, deps=facts, holds={"gpu": 1},
        params={"paths": ["tests/oracle"], "timeout": TIER_TIMEOUT_S},
        code={"files": ["tests/oracle", "tests/conftest.py", "rola", "measure"]})
    for name in instruments:
        tool, args, per_cell, holds, data, timeout = INSTRUMENTS[name]
        if per_cell and not carry:
            continue
        out["instruments"][name] = g.node(
            name, executor=EXECUTORS + "run_tool", env=env, deps=facts,
            inputs=cell_nodes(g, carry) if per_cell else (), holds=holds,
            params={"tool": tool, "args": args, "per_cell": per_cell, "timeout": timeout},
            cache=not per_cell, code=_code(tool, data))
    if timing is None:
        return out
    timed = _code("measure/executors.py", ["measure/bench"])
    for arm in CARRY_ARMS if carry else ():
        out["entries"][arm] = register_timing(g, arm, server=timing, env=env, executor=EXECUTORS + "timed",
                                              cells=cell_nodes(g, carry), params={"arm": arm}, deps=facts, code=timed)
    constructions = load(Path(env.cwd) / "measure" / "cells" / "layer.py")["CONSTRUCTIONS"]
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
                                         code=_code("measure/executors.py"))
    return out


#: THE DIFF SURFACES (`tests/oracle/sides.py`): what each IS -- the side executor, the cells it runs on (a cell kind
#: and, where it has one, a tier), and what the side holds while it runs. Exposed for rola-bench, which pairs a surface
#: across checkouts and OWNS the rule it compares them under (its `RULES`), as it owns the timing methodology; the
#: only diff this file declares itself is the kernel against its own oracle, below. What rola states about a
#: surface's numerics is a FACT, in `NUMERICS`: the carry kernel's readout fan-in is an fp32 atomic reduction whose
#: association order varies run to run, so `num` and `den` reassociate -- MEASURED 2026-09-16, one binary against
#: itself, three runs over every oracle-tier cell: den within 7e-7 relative, num within 1.9e-9 absolute where a
#: cancelled slot made the relative error meaningless; `state`, written once per leaf, is exact.
SURFACES = {
    "oracle": ("tests.oracle.sides:oracle", "carry", "oracle", {"host_cpu": "all"}),
    "producer": ("tests.oracle.sides:producer", "producer", None, {"host_cpu": "all"}),
    "carry-kernel": ("tests.oracle.sides:carry_kernel", "carry", "oracle", {"gpu": "all"}),
}
NUMERICS = {
    "carry-kernel": {"reassociation": {"num": {"rtol": 7e-7, "atol": 1.9e-9}, "den": {"rtol": 7e-7, "atol": 2.4e-7}},
                     "exact": ["state"]},
}
#: the kernel-vs-oracle diff INSIDE one checkout: the carry kernel's slots against the fp64 reference's, PER SLOT under
#: each output kind's clauses and envelope (`tests/oracle/tolerances.py`) -- the numeric half of the oracle tier
KERNEL_VS_ORACLE = ("carry-kernel", "tests.oracle.sides:carry_reference",
                    {"num": "CARRY_NUM", "den": "CARRY_DEN", "state": "CARRY_STATE"})


#: THE BACKWARD'S BUDGET. The oracle side runs `naive_rola` forward AND backward in fp64, and autograd retains the
#: recurrence's state at every token: `B * H * T * N * (DV + 1) * 8` bytes of retained state, and MEASURED (2026-09-16)
#: a peak above nine times that -- `flat-small-dense`, 1.0 GiB of retained state, could not run under a 9 GiB cap, and
#: the run over every oracle-tier cell was killed by the host. A surface that runs the backward therefore admits only
#: the cells whose retained state is under this budget, stated here so the excluded cells are a consequence a reader can
#: compute and not a run that died. The forward alone is graded on every oracle-tier cell by the carry reference.
RETAINED_STATE_BUDGET = 512 * 2**20
BACKWARD_SURFACES = ("oracle",)


def retained_state_bytes(record: dict, batch: int = 1, heads: int = 2) -> int:
    params = record["params"]
    leaves = 1
    for width in params["widths"]:
        leaves *= width
    return batch * heads * params["tokens"] * leaves * (params["dv"] + 1) * 8


def surface_cells(surface: str) -> list:
    """The cells one surface runs on: its kind, at its tier where it has one, and under the backward's budget where the
    surface runs a backward."""
    _executor, kind_, tier, _holds = SURFACES[surface]
    return sorted(name for name, record in central().cells.items()
                  if kind(name) == kind_ and (tier is None or record["params"].get("tier") == tier)
                  and (surface not in BACKWARD_SURFACES or retained_state_bytes(record) < RETAINED_STATE_BUDGET))


SIDE_CODE = ("tests/oracle/sides.py", ["tests/oracle", "rola", "measure"])


def _sides(g, env: Env, facts: dict) -> dict:
    """One side TARGET per surface, in this checkout, over the cells the surface runs on. A side that launches the
    kernel reads the binary; a pure-torch reference does not, so a rebuilt binary re-keys the one and not the other."""
    from rola_devtools.diff import side

    return {name: side(g, f"side/{name}", env=env, executor=executor, cells=cell_nodes(g, surface_cells(name)),
                       code=_code(*SIDE_CODE), binds="rola", holds=holds, deps=facts if "gpu" in holds else {})
            for name, (executor, _kind, _tier, holds) in SURFACES.items()}


def _kernel_vs_oracle(g, env: Env, sides: dict):
    """The carry kernel against the fp64 reference, in this checkout: the clause rule per output kind."""
    from rola_devtools.diff import diff, side

    kernel_surface, reference_executor, kinds = KERNEL_VS_ORACLE
    outputs = load(Path(env.cwd) / "tests" / "oracle" / "tolerances.py")
    nodes = cell_nodes(g, surface_cells(kernel_surface))
    reference = side(g, "side/carry-reference", env=env, executor=reference_executor, cells=nodes,
                     code=_code(*SIDE_CODE), binds="rola", holds={"host_cpu": "all"})
    per_quantity = {q: {"clauses": [list(c) for c in outputs[k].clauses], "rtol": outputs[k].rtol,
                        "envelope_from": f"{q}_envelope"} for q, k in kinds.items()}
    #: RECORDED, not a build failure: the verdict is stored per cell, red cells included (an arm the binary does not
    #: ship is red by decision until kernel work ships it), and the rest of the build measures on
    return diff(g, "diff/carry-vs-oracle", left=sides[kernel_surface], right=reference, strategy="per-slot",
                params={"per_quantity": per_quantity}, minimum=len(nodes), on_difference="record")


def root(g, python: str = sys.executable, label: str = "rola", rounds: str = "8", reps: str = "11",
         store_root: str = "") -> dict:
    """rola measuring and gating itself: EVERY node this checkout has, and the CLI prunes by label (`--only`, `--skip`)."""
    server = start_timing_server(g)
    env = checkout(HERE, python=python, label=label)
    mine = declare(g.scoped(label), env, timing=server)
    root_dir = store_root or None
    stores = [store(g, f"store/{name}", source=target, location=f"rola/{name}", cache=target.cache, root=root_dir)
              for name, target in mine["instruments"].items()]
    session = measure_timing(g, "session", server=server, entries=list(mine["entries"].values()), clock=mine["clock"],
                             rounds=int(rounds), reps=int(reps))
    memory = measure_memory(g, "memory", server=server, entries=list(mine["entries"].values()))
    stores += [store(g, "store/session", source=session, location="timing/session", root=root_dir),
               store(g, "store/memory", source=memory, location="timing/memory", root=root_dir),
               store(g, "store/carry-vs-oracle", source=mine["diffs"]["carry-vs-oracle"],
                     location="rola/carry-vs-oracle", root=root_dir)]
    stop = stop_timing_server(g, server=server, after=[session, memory, *stores])
    return {"all": g.group("all", [*stores, stop])}
