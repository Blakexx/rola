# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S MEASUREMENT GRAPH: this checkout's instruments, timed arms and memory as units of `rola_devtools.graph`.

    PYTHONPATH=.:benchmarks:tools python -m rola_devtools.graph plan benchmarks.graph:graph --select carry.phases@flagship-dense
    PYTHONPATH=.:benchmarks:tools python -m rola_devtools.graph run benchmarks.graph:graph --hold benchmarks.graph:hold ...

rola-bench includes this graph by reference, one instance per checkout it measures, so a node run here and the same node
run there reach one record. Every unit reads only this checkout: its binary (`rola_cu13/_C*.so`), its tools, its cells.

    carry.sass                the SASS signatures of the built carry arms              (`tools/sass_gate.py`)
    carry.registers@arm0      peak live registers by region, arm 0 with line info      (`tools/life_ranges.py`)
    carry.phases@<cell>       the phase clock: cycles a warp a window, per warp        (`tools/phase_ledger.py`)
    carry.counters@<cell>     the profiler's pipe and resource counters, one launch    (`tools/pipe_counters.py`)
    carry.census@<cell>       every stall sample by component, reason, source line     (`tools/stall_census.py`)
    carry.timeline@<cell>     the pipes over one launch, PM sampling                   (`tools/pipe_timeline.py`)
    time.<subject>@<cell>     a timed arm of `bench.provider` for a session to interleave
    memory.<subject>@<cell>   the arm's peak allocated and reserved device memory over its calls, alone

A unit's IDENTITY is the binary's sha256, its code (the instrument and every checkout file it imports, with the data
files it reads), its parameters and the ENVIRONMENT: the GPU and driver, the toolkit's assembler, torch, the Nsight
Compute CLI and the SM clock the host locks, read from the machine and the dev config, never from paths.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
from functools import cache
from importlib import metadata
from pathlib import Path

from rola_devtools.graph import Node, Refusal, Timed, Unit
from rola_devtools.graph import identity as ident
from rola_devtools.process import run as run_command

CHECKOUT = Path(__file__).resolve().parents[1]
#: where this checkout's repository-local imports resolve
IMPORT_ROOTS = ("tools", "benchmarks", ".")
CELLS = ("benchmarks/cells/carry_cells.json", "benchmarks/cells/layer_cells.json")
#: the subjects a cell's timed and memory nodes are declared for, at their default dials; an arm the binary or the cell
#: does not take is refused in setup
SUBJECTS = {"carry": ("carry_forward", "prefill_op"), "layer": ("entmax_solve", "decode_step")}
#: seconds an instrument runs before its node fails; the SASS gate and the register walk state their own
INSTRUMENT_TIMEOUT_S = 3600
#: the calls a memory node's peak is taken over, after one warm call
MEMORY_CALLS = 5


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
    try:
        return ident.digest(binary())
    except Refusal:
        return "none"


def _version(cmd: list[str]) -> str:
    """A tool's version output, or "" on a machine without it (a CPU-only job describes the graph too)."""
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
def code_key(entry: str, data: tuple[str, ...]) -> str:
    return ident.code(CHECKOUT, entry, IMPORT_ROOTS, data)


@cache
def environment_key() -> str:
    return hashlib.sha256(json.dumps(environment(), sort_keys=True).encode()).hexdigest()


class Instrument(Unit):
    """One run of a checkout instrument through its JSON command line: `python <tool> <args> --json <ws>/out.json`; its
    record is the instrument's own output. `{binary}` in an argument is the built extension's path in the checkout.
    `cell` is the carry cell the instrument launches, or None: a cell whose carry arm the binary does not carry is a
    refusal, not a failure of the instrument."""

    def __init__(self, location: str, tool: str, args: list, data: list, repeatable: bool, timeout: int,
                 cell: str | None) -> None:
        self.location, self.tool, self.args, self.data = location, tool, list(args), list(data)
        self.repeatable, self.timeout, self.cell = repeatable, timeout, cell

    def identity(self) -> dict:
        return {"binary": binary_key(), "code": code_key(self.tool, tuple(self.data)),
                "environment": environment_key(), "args": self.args}

    def setup(self, ws: Path):
        path = binary().relative_to(CHECKOUT).as_posix()
        if self.cell is not None:
            _tools()
            from benchmarks.cells import by_name
            from rola.ops import carry

            arm = by_name(self.cell).arm
            carried = sorted(tuple(a) for a in carry.arms())
            if arm not in carried:
                raise Refusal(f"{self.cell}: this binary carries no carry arm {arm} (D, DV, warps_per_cta); it carries "
                              f"{carried}")
        return path

    def execute(self, prepared, ws: Path) -> None:
        args = [a.replace("{binary}", prepared) for a in self.args]
        out = ws / "out.json"
        done = run_command([sys.executable, self.tool, *args, "--json", str(out)], cwd=CHECKOUT, timeout=self.timeout)
        if not out.exists():
            raise RuntimeError(f"{self.tool} exited {done.returncode} without output:\n{(done.stdout + done.stderr)[-1500:]}")

    def post(self, ws: Path) -> dict:
        return json.loads((ws / "out.json").read_text())


class Timeline(Instrument):
    """`tools/pipe_timeline.py` writes `<out>.json` beside its capture instead of taking `--json`."""

    def __init__(self, cell: str) -> None:
        super().__init__("rola/carry.timeline", "tools/pipe_timeline.py", ["--cell", cell], [CELLS[0]], True, INSTRUMENT_TIMEOUT_S,
                         cell)

    def execute(self, prepared, ws: Path) -> None:
        done = run_command([sys.executable, self.tool, "--cell", self.cell, "--out", str(ws / "tl"), "--no-record"],
                           cwd=CHECKOUT, timeout=self.timeout)
        if not (ws / "tl.json").exists():
            raise RuntimeError(f"pipe_timeline exited {done.returncode} without output:\n{(done.stdout + done.stderr)[-1500:]}")
        (ws / "out.json").write_text((ws / "tl.json").read_text())


def _arm(cell: str, arm: str):
    """The provider's arm `arm` on `cell`, built; a cell or arm this checkout does not run is a refusal."""
    _tools()
    from bench.provider import arms
    from benchmarks.cells.registry import CELLS as REGISTERED

    try:
        offered = arms(REGISTERED[cell][1])
    except (LookupError, RuntimeError) as why:
        raise Refusal(f"{cell}: {why}") from why
    if arm not in offered:
        raise Refusal(f"{cell}: this checkout runs no arm {arm} there; it runs {sorted(offered)}")
    return offered[arm]()


class Arm(Unit):
    """A timed arm of this checkout's runner (`bench.provider`) on a registered cell, interleaved by a session."""

    location = "rola/time"
    timed = True

    def __init__(self, cell: str, arm: str) -> None:
        self.cell, self.arm = cell, arm

    def identity(self) -> dict:
        return {"binary": binary_key(), "code": code_key("benchmarks/bench/provider.py", CELLS),
                "environment": environment_key(), "cell": self.cell, "arm": self.arm}

    def setup(self, ws: Path):
        built = _arm(self.cell, self.arm)
        return Timed(call=built.call, built=built.cell, instrument=built.instrument)

    def post(self, ws: Path) -> dict:
        samples = json.loads((ws / "samples.json").read_text())
        return {"calls": len(samples["ms"])}


class Memory(Unit):
    """The arm alone: the device's peak allocated and reserved bytes over `MEMORY_CALLS` calls after one warm call, and
    what stays allocated after them. A timed session cannot say this: its members share one allocator per worker."""

    location = "rola/memory"
    repeatable = True

    def __init__(self, cell: str, arm: str) -> None:
        self.cell, self.arm = cell, arm

    def identity(self) -> dict:
        return {"binary": binary_key(), "code": code_key("benchmarks/bench/provider.py", CELLS),
                "environment": environment_key(), "cell": self.cell, "arm": self.arm, "calls": MEMORY_CALLS}

    def setup(self, ws: Path):
        import torch

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        built = _arm(self.cell, self.arm)
        built.call()
        return built, before

    def execute(self, prepared, ws: Path) -> None:
        import torch

        built, before = prepared
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(MEMORY_CALLS):
            built.call()
        torch.cuda.synchronize()
        (ws / "memory.json").write_text(json.dumps({
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_after_bytes": torch.cuda.memory_allocated(), "allocated_before_build_bytes": before,
            "built": built.cell}))

    def post(self, ws: Path) -> dict:
        return json.loads((ws / "memory.json").read_text())


def cells() -> dict[str, str]:
    """Every registered cell, by name, with its kind: what a run measures is its selection's."""
    carry = json.loads((CHECKOUT / CELLS[0]).read_text())["cells"]
    layer = json.loads((CHECKOUT / CELLS[1]).read_text())["cells"]
    return {**{c["name"]: "carry" for c in carry}, **{c["name"]: "layer" for c in layer}}


def graph() -> list[Node]:
    here = f"{__name__}:"
    census_data = [CELLS[0], "tools/budgets/carry.json", "csrc/rola/src/carry/carry_kernel.cuh"]
    nodes = [Node("carry.sass", here + "Instrument", {"location": "rola/carry.sass", "tool": "tools/sass_gate.py",
                                                       "args": ["{binary}"], "data": [], "repeatable": False,
                                                       "timeout": 900, "cell": None}),
             Node("carry.registers@arm0", here + "Instrument",
                  {"location": "rola/carry.registers", "tool": "tools/life_ranges.py",
                   "args": ["--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"],
                   "data": ["csrc/rola/src", "build/generated/carry_parts.inc"], "repeatable": False, "timeout": 1800,
                   "cell": None})]
    for cell, kind in cells().items():
        if kind == "carry":
            for module, tool, extra, data in (("carry.phases", "tools/phase_ledger.py", ["--launches", "1"], [CELLS[0]]),
                                              ("carry.counters", "tools/pipe_counters.py", [], [CELLS[0]]),
                                              ("carry.census", "tools/stall_census.py", [], census_data)):
                nodes.append(Node(f"{module}@{cell}", here + "Instrument",
                                  {"location": f"rola/{module}", "tool": tool, "args": [cell, *extra], "data": data,
                                   "repeatable": True, "timeout": INSTRUMENT_TIMEOUT_S, "cell": cell}))
            nodes.append(Node(f"carry.timeline@{cell}", here + "Timeline", {"cell": cell}))
        for subject in SUBJECTS[kind]:
            nodes.append(Node(f"time.{subject}@{cell}", here + "Arm", {"cell": cell, "arm": subject}))
            nodes.append(Node(f"memory.{subject}@{cell}", here + "Memory", {"cell": cell, "arm": subject}))
    return nodes


_CLOCK: list = []


@contextlib.contextmanager
def hold():
    """The device for a setup and an execute (the GPU lock, exclusive, inherited by every process started inside), and
    the host's clock lock, engaged once a process and proven by the device's clock read after each hold."""
    _tools()
    import clock_lock
    from gpu_lock import gpu_lock

    def read_ghz():
        from rola.ops import carry as carry_ops

        return carry_ops.sm_clock_ghz()

    with gpu_lock():
        if not _CLOCK:
            _CLOCK.append(clock_lock.engage(read_ghz))
        yield
        after = read_ghz()
        if _CLOCK[0] is not None and not clock_lock.within(after, _CLOCK[0]):
            raise RuntimeError(f"CLOCK: the device reads {after} GHz after the hold, off the lock at {_CLOCK[0]['ghz']} GHz")
