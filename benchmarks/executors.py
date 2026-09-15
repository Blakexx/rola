# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S EXECUTORS: what rola's declared targets (`declare.py`) run, in this checkout's environment.

The build system's worker runs them from this checkout's directory with `.`, `benchmarks` and `tools` on the path.
Each takes the target's context (`rola_devtools.build.context.Context`) and returns its JSON output, keeping raw files
in its workspace:

    compile_kernel      the extension, by this checkout's own gated build (`pip install -e .`, which takes the host
                        budget itself); its output is the binary's path in the checkout and its sha256, and
                        `binary_present` confirms a cached build still stands on this machine
    probe_environment   the machine facts a number depends on (GPU and driver, assembler, torch, Nsight Compute, the
                        clock the host locks), read from the machine and the dev config, never from a path
    run_tool            a checkout instrument through its `--json` command line, once or on each cell it takes: a cell
                        the tool cannot run is recorded as that cell's failure (the instrument's domain), a cell-less
                        analysis that writes nothing fails the build
    timed               a timing entry: this checkout's runner (`bench.provider`) arm on a cell, as a Timed with its reset
    read_clock          the SM clock as this binary reads it, for the timing system's proof
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
#: seconds a build may take
BUILD_TIMEOUT_S = 6 * 3600


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    from rola_devtools.process import run

    return run(cmd, cwd=CHECKOUT, timeout=timeout)


def _portable(text: str) -> str:
    return text.replace(str(CHECKOUT) + "/", "").replace(str(Path.home()) + "/", "~/")


def binary() -> Path:
    sos = sorted(CHECKOUT.glob("rola_*/_C*.so"))
    if not sos:
        raise RuntimeError(f"no built extension in {CHECKOUT.name} (rola_<toolchain>/_C*.so)")
    return sos[0]


def compile_kernel(ctx) -> dict:
    from rola_devtools.build.identity import digest

    done = _run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps"], BUILD_TIMEOUT_S)
    (ctx.workspace / "build.log").write_text(_portable(done.stdout[-50000:] + done.stderr[-50000:]))
    if done.returncode:
        raise RuntimeError(f"the build exited {done.returncode}:\n{_portable((done.stdout + done.stderr)[-3000:])}")
    so = binary()
    return {"binary": so.relative_to(CHECKOUT).as_posix(), "sha256": digest(so)}


def binary_present(output: dict) -> bool:
    from rola_devtools.build.identity import digest

    path = CHECKOUT / output["binary"]
    return path.is_file() and digest(path) == output["sha256"]


def _version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def probe_environment(ctx) -> dict:
    import dev_config

    smi = _version(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]).strip()
    ptxas = _version([f"{dev_config.get('toolchain.cuda_home')}/bin/ptxas", "--version"]).split()
    ncu = _version([dev_config.get("toolchain.ncu"), "--version"])
    return {"gpu": smi, "ptxas": next((w for w in ptxas if w.startswith("V")), " ".join(ptxas[-2:])),
            "torch": metadata.version("torch"),
            "ncu": next((line.split("Version ")[1].split()[0] for line in ncu.splitlines() if "Version " in line), ""),
            "clock_ghz": dev_config.get("clock.ghz")}


def run_tool(ctx) -> dict:
    p = ctx.params
    binary_path = ctx.deps["binary"].output["binary"]
    cells = [record["name"] for record in ctx.inputs] if p["per_cell"] else [""]
    results, statuses = {}, {}
    for cell in cells:
        out = ctx.workspace / f"{cell or 'out'}.json"
        args = [a.replace("{binary}", binary_path).replace("{cell}", cell) for a in p["args"]]
        done = _run([sys.executable, p["tool"], *args, "--json", str(out)], p["timeout"])
        if out.exists():
            results[cell] = json.loads(out.read_text())
            statuses[cell] = {"status": "ok", "exit": done.returncode}
        elif p["per_cell"]:
            statuses[cell] = {"status": "failed", "exit": done.returncode,
                              "error": _portable((done.stdout + done.stderr)[-2000:])}
        else:
            raise RuntimeError(f"{p['tool']} exited {done.returncode} without output:\n"
                               f"{_portable((done.stdout + done.stderr)[-2000:])}")
    (ctx.workspace / "results.json").write_text(json.dumps(results, sort_keys=True))
    return {"file": "results.json", "cells": statuses}


def timed(cell, params):
    from bench.provider import arms

    offered = arms(cell)
    if params["arm"] not in offered:
        raise LookupError(f"{cell.name}: this checkout runs no arm {params['arm']} there; it runs {sorted(offered)}")
    return offered[params["arm"]]()


def read_clock() -> float | None:
    from rola.ops import carry

    return carry.sm_clock_ghz()
