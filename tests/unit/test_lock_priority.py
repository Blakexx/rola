# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""GATE for the NICE stage (Blake, 2026-08-29): every compile slot from
`build_lock.acquire()` and every `gpu_lock()` hold lowers the ACQUIRING
process's own CPU niceness before anything real runs inside it, so a child it
launches (a compiler, a test, a probe) inherits nice 10 -- os.nice() cannot be
lowered back down by an unprivileged process once raised, so each of the two
conditions below (`host.nice` true vs. false in a test dev config) runs in its OWN fresh driver
subprocess rather than reusing this test process across both.

docs/internals/tools/{build_lock,gpu_lock}.md carries the mechanism; this file
is the proof each names: read /proc/<pid>/stat field 19 (nice) of a child
spawned from inside the lock's own `with` block.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"

# field 19 (1-indexed, proc(5)) = nice; `comm` (field 2) can itself contain
# spaces/parens, so this reads past its closing `)` rather than splitting the
# raw line naively.
_READ_NICE = (
    "import pathlib, sys\n"
    "raw = pathlib.Path('/proc/self/stat').read_text()\n"
    "fields = raw[raw.rindex(')') + 2:].split()\n"
    "sys.stdout.write(fields[19 - 3])\n"
)

_BUILD_LOCK_DRIVER = (
    "import subprocess, sys\n"
    "import build_lock\n"
    "with build_lock.acquire(gate=False, desired=1):\n"
    "    r = subprocess.run([sys.executable, '-c', " + repr(_READ_NICE) + "],\n"
    "                       capture_output=True, text=True, check=True)\n"
    "    sys.stdout.write(r.stdout)\n"
)

_GPU_LOCK_DRIVER = (
    "import subprocess, sys\n"
    "import gpu_lock\n"
    "with gpu_lock.gpu_lock(sys.argv[1]):\n"
    "    r = subprocess.run([sys.executable, '-c', " + repr(_READ_NICE) + "],\n"
    "                       capture_output=True, text=True, check=True)\n"
    "    sys.stdout.write(r.stdout)\n"
)


def _run_driver(driver_src, env, *extra_args) -> int:
    """Run `driver_src` as a fresh interpreter (cwd=tools/, so `import
    build_lock`/`import gpu_lock` resolve without touching sys.path) and
    return the nice value its own grandchild reported."""
    result = subprocess.run(
        [sys.executable, "-c", driver_src, *extra_args],
        capture_output=True, text=True, check=True,
        cwd=str(_TOOLS_DIR), env=env,
    )
    # `build_lock.acquire()` prints its own slot-accounting line to stdout
    # (by design -- it is a build's progress log); the nice value is always
    # the LAST thing the driver writes.
    return int(result.stdout.strip().splitlines()[-1])


def test_build_lock_acquire_lowers_child_priority(tmp_path, dev_config_env):
    locks = str(tmp_path)
    assert _run_driver(_BUILD_LOCK_DRIVER, dev_config_env(host={"lock_dir": locks, "nice": True})) == 10
    assert _run_driver(_BUILD_LOCK_DRIVER, dev_config_env(host={"lock_dir": locks, "nice": False})) == 0


def test_gpu_lock_lowers_child_priority(tmp_path, dev_config_env):
    lock_path = str(tmp_path / "gpu.lock")
    assert _run_driver(_GPU_LOCK_DRIVER, dev_config_env(host={"nice": True}), lock_path) == 10
    assert _run_driver(_GPU_LOCK_DRIVER, dev_config_env(host={"nice": False}), lock_path) == 0
