#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CLOCK LOCK SEAM: the host says how its clock is locked; the harness only asks.

A GPU's boost governor moves the SM clock with power, and a kernel number is comparable
across runs only at one clock (`docs/internals/common/sm_clock.md`). How a clock is locked
is a fact about the HOST -- `nvidia-smi -lgc` as root on Linux, an elevated scheduled task
on a Windows host under WSL, nothing at all on a box that forbids it -- so it lives in the dev
config's `clock.json` (`tools/dev_config.py`; `python tools/dev.py clock --mhz N` writes it):

    {"ghz": 1.665,
     "lock":   ["/mnt/c/Windows/System32/schtasks.exe", "/run", "/tn", "gpu-lock"],
     "unlock": ["/mnt/c/Windows/System32/schtasks.exe", "/run", "/tn", "gpu-unlock"]}

The harness locks at start, proves the lock with the in-kernel clock read, unlocks on every
exit path, and refuses a row whose measured clock is off the lock. Without a clock a run is
UNLOCKED: rows carry their measured clock and are marked so in the ledger, as dirty rows are.
"""
from __future__ import annotations

import atexit
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import

TOLERANCE = 0.01


def load() -> dict | None:
    cfg = {key: dev_config.get(f"clock.{key}") for key in ("ghz", "lock", "unlock")}
    if cfg["ghz"] is None:
        return None
    missing = [key for key in ("lock", "unlock") if not cfg[key]]
    if missing:
        raise SystemExit(f"{dev_config.directory() / 'clock.json'}: ghz is set but {missing} is not")
    return cfg


def _run(cmd: list[str]) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if done.returncode:
        raise SystemExit(f"CLOCK: {cmd[0]} failed ({done.returncode}): {(done.stderr or done.stdout).strip()[-400:]}; the host's "
                         "lock is broken -- python tools/dev.py clock --mhz N re-registers and proves it")


def within(ghz: float | None, cfg: dict | None) -> bool:
    """whether a measured clock sits on the declared lock; an unlocked run is not judged."""
    return cfg is None or (ghz is not None and abs(ghz - cfg["ghz"]) <= TOLERANCE * cfg["ghz"])


def engage(read_ghz) -> dict | None:
    """lock, prove it with the device read, and arrange the unlock for every exit."""
    cfg = load()
    if cfg is None:
        print("CLOCK: unlocked (no clock in the dev config); rows carry their measured clock",
              file=sys.stderr)
        return None
    _run(cfg["lock"])

    def release(*_):
        try:
            _run(cfg["unlock"])
        finally:
            pass

    atexit.register(release)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (release(), sys.exit(128 + s)))
    ghz = read_ghz()
    if ghz is None or not within(ghz, cfg):
        raise SystemExit(f"CLOCK: lock to {cfg['ghz']} GHz not proven; the device reads "
                         f"{ghz} GHz. Refusing to measure.")
    print(f"CLOCK: locked at {ghz:.3f} GHz (declared {cfg['ghz']})", file=sys.stderr)
    return cfg
