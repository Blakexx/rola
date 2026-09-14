#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HOST-COMPUTE BUDGET, ONE POOL FOR EVERY CPU-HEAVY TOOL (LOCKS brief,
Blake 2026-08-29: "add locks around our tooling where it makes sense ...
maybe semaphores, allow some controlled level of parallelism").

**WHY ONE MORE POOL, WHEN `build_lock.py` ALREADY HAD ONE.** `build_lock.py`'s
`GLOBAL_CICC_SLOTS` counted `cicc` processes only -- but a `compute-sanitizer`
run, a `clang-tidy --cuda-host-only` pass over one translation unit, and a
pytest tier that forks workers (`pytest-xdist`) are ALL host-CPU-heavy in the
same way a compile is, and none of them drew from the SAME budget a compile
did. Three of those running alongside a full-budget compile is the exact "12+
cicc on one host" shape (`build_lock.py`'s own docstring) with a different
process name on each contender. Slot COUNTS now live in exactly one place --
here -- and `build_lock.py` draws its compile slots from this pool rather than
keeping a second, uncoordinated one (its own module docstring records the
delegation).

**THE MECHANISM**, the same one `build_lock.py`/`gpu_lock.py` already used,
generalized to N callers rather than one: `HOST_BUDGET_SLOTS` lock files
(`host.lock_dir` in the dev config, default `/tmp` -- a test battery can
point every acquirer at a throwaway directory instead of the real, possibly
contended, `/tmp` locks another agent on this host is holding). `acquire(n)`
waits for at least one free slot (never proceeds with zero), then
opportunistically takes up to `n` MORE that are free right now without
waiting further for them -- so N concurrent callers each make some progress
rather than one winning the whole budget or all of them deadlocking.
`acquire(n, exclusive=True)` instead waits for and holds EVERY slot (a "gate"
draw), mirroring `build_lock.py`'s own gate/iteration split.

**REENTRANT BY CONSTRUCTION** (the `gpu_lock()` shape, extended here): an
acquire sets `ROLA_HOST_BUDGET_HELD` in the process environment; a NESTED
acquire that finds the marker already set is a no-op, so an outer tool that
already holds slots and calls an inner tool that also acquires cannot
self-deadlock (the shape `gpu_lock.py`'s docstring names, generalized to
this pool). `host.locks_trace` prints every acquire/release with the holder
pid and the slot count, for every lock in this file and `gpu_lock.py` alike.

**THE THIRD CATEGORY: FILE LOCKS.** A handful of tools write ONE shared
artifact from potentially-concurrent invocations (the records database
epilogue, a ratify manifest, the generated selection header) -- not a
counting resource, just mutual exclusion over one output. `file_lock(name)`
below is that primitive: one named exclusive lock, short-held, reentrant the
same way.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import


def _default_slots() -> int:
    """cores - 2 for headroom, capped by host memory: a cicc peaks near 2.3 GB on the templated TUs (3 GB with headroom) and the host also runs the
    coordinating pythons (torch imports ~1-2 GB each), so 8 GB is reserved before the budget
    never admits more compilers than the machine can hold without swapping (the 2026-08-29
    freeze was 13 cicc on a 23 GB host)."""
    cores = max(1, (os.cpu_count() or 4) - 2)
    try:
        with open("/proc/meminfo") as f:
            total_kb = int(next(l for l in f if l.startswith("MemTotal:")).split()[1])
        gb = total_kb / 1024 / 1024
        by_memory = max(1, int((gb - 8.0) / 3.0))  # 8 GB reserved for the agents' pythons + the shell
    except (OSError, StopIteration, ValueError):
        by_memory = 4
    return max(1, min(cores, by_memory))


#: `os.cpu_count() - 2` leaves headroom for the interactive shell and whatever
#: agent process is driving the tool itself -- the same reasoning `nproc`-based
#: defaults elsewhere in this tree (`build_flags.default_max_jobs`) already use,
#: just applied to the WHOLE host-compute pool rather than one build's jobs.
HOST_BUDGET_SLOTS = dev_config.get("host.budget_slots") or _default_slots()


_HELD_MARKER = "ROLA_HOST_BUDGET_HELD"


def _lock_dir() -> Path:
    """`host.lock_dir`: a test battery points it at a throwaway directory; host and containers share the real one."""
    return Path(dev_config.get("host.lock_dir"))


def _slot_path(i: int) -> Path:
    return _lock_dir() / f"rola_host_budget_slot_{i}.lock"


def trace(msg: str) -> None:
    if dev_config.get("host.locks_trace"):
        print(f"[locks pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def lower_priority(who: str) -> None:
    """A locked process lowers its own CPU and IO priority (`host.nice`); children it starts inherit both."""
    if not dev_config.get("host.nice"):
        return
    with contextlib.suppress(OSError):
        os.nice(10)
    ionice = shutil.which("ionice")
    if ionice is None:
        print(f"[{who}] ionice not on PATH; CPU niceness only", flush=True)
        return
    subprocess.run([ionice, "-c", "3", "-p", str(os.getpid())], check=False, capture_output=True)


def _try_acquire(i: int):
    fd = open(_slot_path(i), "w")  # noqa: SIM115 -- lock handle outlives this call
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


def _blocking_acquire(i: int, label: str):
    fd = open(_slot_path(i), "w")  # noqa: SIM115 -- see _try_acquire
    trace(f"waiting for host-budget slot {i} ({label}) ...")
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


@contextmanager
def acquire(n: int = 1, *, exclusive: bool = False, label: str = "host-budget"):
    """Hold `held` slots (yielded) of the host-compute budget for the block.

    `exclusive=True`: waits for and holds ALL `HOST_BUDGET_SLOTS` slots -- a
    gate draw, for a caller that wants the whole host to itself.
    `exclusive=False` (default): waits for at least one free slot, then takes
    up to `n` more that are free right now without waiting further.
    """
    if _HELD_MARKER in os.environ:
        trace(f"reentrant acquire ({label}); an ancestor already holds the budget")
        yield HOST_BUDGET_SLOTS if exclusive else max(1, n)
        return

    total = HOST_BUDGET_SLOTS
    n = total if exclusive else max(1, min(n, total))
    held: list = []
    try:
        if exclusive:
            # An exclusive draw collects every slot; two exclusive drawers collecting
            # at once would each hold a subset and wait on the other's forever, so
            # the collection itself runs under one admission mutex.
            trace(f"exclusive draw ({label}): waiting for all {total} slot(s) ...")
            with open(_lock_dir() / "rola_host_budget_exclusive.lock", "a+") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                try:
                    for i in range(total):
                        held.append(_blocking_acquire(i, label))
                finally:
                    fcntl.flock(admission, fcntl.LOCK_UN)
        else:
            for i in range(total):
                if len(held) >= n:
                    break
                fd = _try_acquire(i)
                if fd is not None:
                    held.append(fd)
            if not held:
                #: NEVER a hardcoded slot index here: with more than one
                #: caller contending, WHICH slot each already holds is a
                #: race, so waiting on slot 0 specifically can wait forever
                #: on a slot nobody is about to free while a DIFFERENT one
                #: frees immediately (found by this module's own gate: a
                #: release of "the other" holder never unblocked a waiter
                #: pinned to slot 0). Poll the whole sweep instead -- ANY
                #: slot freeing is progress.
                while not held:
                    for i in range(total):
                        fd = _try_acquire(i)
                        if fd is not None:
                            held.append(fd)
                            break
                    if not held:
                        time.sleep(0.01)
                for i in range(total):
                    if len(held) >= n:
                        break
                    fd = _try_acquire(i)
                    if fd is not None:
                        held.append(fd)
        trace(f"holding {len(held)}/{total} host-budget slot(s) ({label})")
        os.environ[_HELD_MARKER] = str(os.getpid())
        try:
            yield len(held)
        finally:
            os.environ.pop(_HELD_MARKER, None)
    finally:
        for fd in held:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
        if held:
            trace(f"released {len(held)} host-budget slot(s) ({label})")


#: One marker per named file lock, so holding the manifest lock does not make
#: a nested acquire of the (unrelated) records-db lock a no-op -- only a
#: SECOND acquire of the SAME name is reentrant.
def _file_held_marker(name: str) -> str:
    return f"ROLA_FILE_LOCK_HELD_{name}"


@contextmanager
def file_lock(name: str, *, label: str | None = None):
    """One short-held EXCLUSIVE lock over a single named shared artifact.

    Not a counting resource like `acquire()` -- this is mutual exclusion over
    ONE output (the records database, a ratify manifest, the generated
    selection header), so two concurrent writers of the same artifact
    serialize instead of interleaving their writes.
    """
    label = label or name
    marker = _file_held_marker(name)
    if marker in os.environ:
        trace(f"reentrant file_lock({name})")
        yield
        return
    path = _lock_dir() / f"rola_file_lock_{name}.lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        trace(f"waiting for file_lock({name}) ...")
        fcntl.flock(fd, fcntl.LOCK_EX)
        trace(f"holding file_lock({name}) ({label})")
        os.environ[marker] = str(os.getpid())
        try:
            yield
        finally:
            os.environ.pop(marker, None)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        trace(f"released file_lock({name})")


def _main(argv: list[str]) -> int:
    """`python tools/host_budget.py [--slots N] [--exclusive] -- <command...>`

    The one CLI for running a command under the host budget: a census compile, a clang-tidy
    translation unit, anything CPU-heavy that is not `setup.py` (which acquires itself).
    `--exclusive` is a gate draw. The command runs niced (`host.nice`).
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--label", default="cli")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        print("usage: host_budget.py [--slots N] [--exclusive] -- <command...>",
              file=sys.stderr)
        return 2
    with acquire(a.slots, exclusive=a.exclusive, label=a.label):
        lower_priority("host_budget")
        return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
