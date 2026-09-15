# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S MEASUREMENT REGISTRY: this checkout's build, clock reader, instruments and timed arms, registered with
rola-devtools' measurement service (`rola_devtools.measure`) on the central cells.

    python -m rola_devtools.measure plan benchmarks.registry:registry --cells flagship-dense --units carry.phases
    python -m rola_devtools.measure run  benchmarks.registry:registry --session carry_forward@flagship-dense,flagship-alt-k4

rola-bench's composer runs this registry once per checkout it measures, so a node run here and the same node run there
reach one record. Every unit reads only this checkout: its binary, its tools, its reading of the central cells.

    build                   the extension, built by the checkout's own gated build under the host budget
    clock                   the SM clock as this binary reads it, which proves the host's clock lock
    carry.sass              the SASS signatures of the built carry arms                (`tools/sass_gate.py`)
    carry.registers         peak live registers by region, arm 0 with line info        (`tools/life_ranges.py`)
    carry.phases            per carry cell: the phase clock, cycles a warp a window     (`tools/phase_ledger.py`)
    carry.counters          per carry cell: the profiler's pipe and resource counters   (`tools/pipe_counters.py`)
    carry.census            per carry cell: every stall sample by component and line    (`tools/stall_census.py`)
    carry.timeline          per carry cell: the pipes over one launch, PM sampling      (`tools/pipe_timeline.py`)
    <arm>                   a timed arm of `bench.provider` on every cell that offers it: carry_forward, prefill_op,
                            and per RoLA construction entmax_solve@layer=C and decode_step@layer=C

Every unit depends on `build`. A unit's IDENTITY is the binary's sha256, its code (the entry and every checkout file it
imports, with the data it reads), its parameters and the ENVIRONMENT: the GPU and driver, the toolkit's assembler,
torch, the Nsight Compute CLI and the SM clock the host locks -- read from the machine and the dev config, never from
paths. A unit accepts a cell when this binary runs it there; the acceptance is read after the build, from the binary's
own arm tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import cache
from importlib import metadata
from pathlib import Path

from rola_devtools.build import identity as ident
from rola_devtools.measure import Arm, Build, ClockReader, Instrument, Refusal, Registration, Timed
from rola_devtools.process import run as run_command

CHECKOUT = Path(__file__).resolve().parents[1]
#: where this checkout's repository-local imports resolve
IMPORT_ROOTS = ("tools", "benchmarks", ".")
#: what the extension is built from, besides the files `setup.py` imports
BUILD_DATA = ("csrc", "pyproject.toml", "tools/manifests", "tools/toolchains", "tools/sccache_pin.json",
              "tools/mold_pin.json", "tools/devtools.txt")
#: seconds an instrument runs before its node fails; the SASS gate and the register walk state their own
INSTRUMENT_TIMEOUT_S = 3600
#: seconds a build may take
BUILD_TIMEOUT_S = 6 * 3600
#: the subjects registered as timed arms: every cell that offers the arm gets it
CARRY_SUBJECTS = ("carry_forward", "prefill_op")
LAYER_SUBJECTS = ("entmax_solve", "decode_step")


def _tools() -> None:
    for path in (CHECKOUT, CHECKOUT / "benchmarks", CHECKOUT / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def binary() -> Path:
    sos = sorted(CHECKOUT.glob("rola_*/_C.*so"))
    if not sos:
        raise Refusal(f"no built extension in {CHECKOUT.name} (rola_<toolchain>/_C*.so)")
    return sos[0]


@cache
def binary_key() -> str:
    return ident.digest(binary())


def _version(cmd: list[str]) -> str:
    """A tool's version output, or "" on a machine without it."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


@cache
def environment() -> dict:
    """The machine facts a number depends on."""
    _tools()
    import dev_config

    smi = _version(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]).strip()
    ptxas = _version([f"{dev_config.get('toolchain.cuda_home')}/bin/ptxas", "--version"]).split()
    ncu = _version([dev_config.get("toolchain.ncu"), "--version"])
    return {"gpu": smi, "ptxas": next((w for w in ptxas if w.startswith("V")), " ".join(ptxas[-2:])),
            "torch": metadata.version("torch"),
            "ncu": next((line.split("Version ")[1].split()[0] for line in ncu.splitlines() if "Version " in line), ""),
            "clock_ghz": dev_config.get("clock.ghz")}


@cache
def environment_key() -> str:
    return hashlib.sha256(json.dumps(environment(), sort_keys=True).encode()).hexdigest()


@cache
def code_key(entry: str, data: tuple[str, ...]) -> str:
    return ident.code(CHECKOUT, entry, IMPORT_ROOTS, data)


def _carried(cell) -> str | None:
    """None when this binary carries the carry arm the cell launches, else why not."""
    _tools()
    from rola_devtools.cells.carry import CarryCell

    from benchmarks.cells import arm_key
    from rola.ops import carry

    if not isinstance(cell, CarryCell):
        return f"{cell.name}: not a carry cell"
    carried = sorted(tuple(a) for a in carry.arms())
    if arm_key(cell) not in carried:
        return f"{cell.name}: this binary carries no carry arm {arm_key(cell)} (D, DV, warps_per_cta); it carries {carried}"
    return None


class RolaBuild(Build):
    """The extension, built in place by `pip install -e .` (setup.py takes the host budget itself). Its identity is what
    the build reads: `setup.py` and every checkout file it imports, the sources and declarations (`BUILD_DATA`), the build
    parameters set for this process, torch and the assembler; its output names the binary and its sha256, which is present
    while the checkout still holds that binary."""

    location = "rola/build"

    def identity(self, cell) -> dict:
        _tools()
        import dev_config

        return {"source": code_key("setup.py", BUILD_DATA),
                "parameters": {k: os.environ[k] for k in sorted(dev_config.BUILD_PARAMETERS) if k in os.environ},
                "torch": metadata.version("torch"), "ptxas": environment()["ptxas"]}

    def execute(self, ws: Path) -> None:
        done = run_command([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps"],
                           cwd=CHECKOUT, timeout=BUILD_TIMEOUT_S)
        (ws / "build.log").write_text(done.stdout[-20000:] + done.stderr[-20000:])
        if done.returncode:
            raise RuntimeError(f"the build exited {done.returncode}:\n{(done.stdout + done.stderr)[-3000:]}")

    def post(self, ws: Path, handle) -> None:
        so = binary()
        handle.output({"binary": so.relative_to(CHECKOUT).as_posix(), "sha256": ident.digest(so)})

    def present(self, output: dict) -> bool:
        path = CHECKOUT / output["binary"]
        return path.is_file() and ident.digest(path) == output["sha256"]


class Clock(ClockReader):
    def read_ghz(self) -> float | None:
        _tools()
        from rola.ops import carry

        return carry.sm_clock_ghz()


class Tool(Instrument):
    """One run of a checkout instrument through its JSON command line: `python <tool> <args> --json <ws>/out.json`; its
    record is the instrument's own output. In an argument, `{binary}` is the built extension's path in the checkout and
    `{cell}` the cell's name. A per-cell tool takes the carry cells whose carry arm this binary carries."""

    def __init__(self, location: str, tool: str, args: list, data: list, repeatable: bool, timeout: int,
                 per_cell: bool) -> None:
        self.location, self.tool, self.args, self.data = location, tool, list(args), list(data)
        self.repeatable, self.timeout, self.per_cell = repeatable, timeout, per_cell

    def accepts(self, cell) -> str | None:
        return _carried(cell) if self.per_cell else None

    def identity(self, cell) -> dict:
        return {"binary": binary_key(), "code": code_key(self.tool, tuple(self.data)),
                "environment": environment_key(), "args": self.args}

    def setup(self, cell, ws: Path):
        return {"binary": binary().relative_to(CHECKOUT).as_posix(), "cell": cell.name if cell is not None else ""}

    def argv(self, prepared) -> list[str]:
        return [a.replace("{binary}", prepared["binary"]).replace("{cell}", prepared["cell"]) for a in self.args]

    def execute(self, prepared, ws: Path) -> None:
        out = ws / "out.json"
        done = run_command([sys.executable, self.tool, *self.argv(prepared), "--json", str(out)], cwd=CHECKOUT,
                           timeout=self.timeout)
        if not out.exists():
            raise RuntimeError(f"{self.tool} exited {done.returncode} without output:\n{(done.stdout + done.stderr)[-1500:]}")

    def post(self, ws: Path, handle) -> None:
        handle.output(json.loads((ws / "out.json").read_text()))


class Timeline(Tool):
    """`tools/pipe_timeline.py` writes `<out>.json` beside its capture instead of taking `--json`."""

    def __init__(self) -> None:
        super().__init__("rola/carry.timeline", "tools/pipe_timeline.py", ["--cell", "{cell}"], [], True,
                         INSTRUMENT_TIMEOUT_S, True)

    def execute(self, prepared, ws: Path) -> None:
        done = run_command([sys.executable, self.tool, *self.argv(prepared), "--out", str(ws / "tl"), "--no-record"],
                           cwd=CHECKOUT, timeout=self.timeout)
        if not (ws / "tl.json").exists():
            raise RuntimeError(f"pipe_timeline exited {done.returncode} without output:\n{(done.stdout + done.stderr)[-1500:]}")
        (ws / "out.json").write_text((ws / "tl.json").read_text())


class Subject(Arm):
    """A timed arm of this checkout's runner (`bench.provider`), named as the runner names it: the subject, its dials and a
    layer arm's construction. It accepts every cell the runner offers it on."""

    location = "rola/memory"

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def _offered(self, cell):
        _tools()
        from bench.provider import arms

        try:
            return arms(cell)
        except (LookupError, TypeError, RuntimeError) as why:
            return str(why)

    def accepts(self, cell) -> str | None:
        offered = self._offered(cell)
        if isinstance(offered, str):
            return offered
        return None if self.arm in offered else f"{cell.name}: this checkout runs no arm {self.arm} there"

    def identity(self, cell) -> dict:
        return {"binary": binary_key(), "code": code_key("benchmarks/bench/provider.py", ()),
                "environment": environment_key(), "arm": self.arm}

    def setup(self, cell, ws: Path) -> Timed:
        offered = self._offered(cell)
        if isinstance(offered, str) or self.arm not in offered:
            raise Refusal(offered if isinstance(offered, str) else f"{cell.name}: no arm {self.arm}")
        built = offered[self.arm]()
        return Timed(call=built.call, built=built.cell, instrument=built.instrument,
                     outside_allocator=built.outside_allocator)


def registry() -> list[Registration]:
    here = f"{__name__}:"
    census_data = ["tools/budgets/carry.json", "csrc/rola/src/carry/carry_kernel.cuh"]
    units = [Registration("build", here + "RolaBuild"),
             Registration("clock", here + "Clock"),
             Registration("carry.sass", here + "Tool",
                          {"location": "rola/carry.sass", "tool": "tools/sass_gate.py", "args": ["{binary}"], "data": [],
                           "repeatable": False, "timeout": 900, "per_cell": False}, deps=("build",)),
             Registration("carry.registers", here + "Tool",
                          {"location": "rola/carry.registers", "tool": "tools/life_ranges.py",
                           "args": ["--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"],
                           "data": ["csrc/rola/src", "build/generated/carry_parts.inc"], "repeatable": False,
                           "timeout": 1800, "per_cell": False}, deps=("build",)),
             Registration("carry.timeline", here + "Timeline", deps=("build",))]
    for name, tool, extra, data in (("carry.phases", "tools/phase_ledger.py", ["--launches", "1"], []),
                                    ("carry.counters", "tools/pipe_counters.py", [], []),
                                    ("carry.census", "tools/stall_census.py", [], census_data)):
        units.append(Registration(name, here + "Tool",
                                  {"location": f"rola/{name}", "tool": tool, "args": ["{cell}", *extra], "data": data,
                                   "repeatable": True, "timeout": INSTRUMENT_TIMEOUT_S, "per_cell": True},
                                  deps=("build",)))
    _tools()
    from bench.provider import arm_name
    from benchmarks.cells.layer import CONSTRUCTIONS

    arms = [*CARRY_SUBJECTS, *(arm_name(subject, construction=c) for subject in LAYER_SUBJECTS for c in CONSTRUCTIONS)]
    units += [Registration(arm, here + "Subject", {"arm": arm}, deps=("build",)) for arm in arms]
    return units
