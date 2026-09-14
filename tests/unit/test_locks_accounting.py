# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""GATE for the LOCKS brief (Blake, 2026-08-29): "add locks around our
tooling ... maybe semaphores, allow some controlled level of parallelism".
Every unit here uses FAKE children -- small driver scripts that hold a lock
and wait to be told to release, never a real compile or a real GPU kernel --
so the whole file runs in well under a second and never touches the real
`/tmp` lock files another agent on this host may be holding
(`host.lock_dir` in a test dev config/an explicit `gpu_lock(path=...)` point every case at
`tmp_path` instead).

Proves, per the brief's Gates line:
  * N+1 acquirers of an N-slot pool: the (N+1)th blocks.
  * Releasing one of the N frees a slot for the blocked (N+1)th.
  * Reentrancy: a nested acquire inside an ancestor's `with` is a no-op.
  * `gpu_lock`'s two GPU modes: SHARED holders share a fixed pool; an
    EXCLUSIVE acquire waits for every shared holder, and a shared acquire
    waits for an exclusive holder.
"""
from __future__ import annotations

import os
import select
import subprocess
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"

#: Holds ONE host-budget slot, announces it, then blocks on stdin until told
#: to let go -- the "fake child" the brief's Gates line asks for.
_HOLD_HOST_BUDGET = (
    "import sys\n"
    "import host_budget\n"
    "with host_budget.acquire(1, label='test-hold'):\n"
    "    print('ACQUIRED', flush=True)\n"
    "    sys.stdin.readline()\n"
    "print('RELEASED', flush=True)\n"
)

#: Same shape, for `gpu_lock` -- `mode` is passed as argv[2] ("exclusive" or
#: "shared"), the lock path as argv[1] so every case points at `tmp_path`.
_HOLD_GPU_LOCK = (
    "import sys\n"
    "import gpu_lock\n"
    "with gpu_lock.gpu_lock(sys.argv[1], mode=sys.argv[2]):\n"
    "    print('ACQUIRED', flush=True)\n"
    "    sys.stdin.readline()\n"
    "print('RELEASED', flush=True)\n"
)

_TIMEOUT_S = 5.0


def _spawn(driver_src: str, env: dict, *extra_args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", driver_src, *extra_args],
        cwd=str(_TOOLS_DIR), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )


def _readline_or_none(proc: subprocess.Popen, timeout: float) -> str | None:
    """One line of `proc.stdout`, or `None` if none arrives within `timeout` --
    the non-blocking half of "the (N+1)th acquirer blocks": a real readline()
    would hang forever on a process that never prints, which is exactly the
    outcome under test."""
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        return None
    return proc.stdout.readline()


def _release(proc: subprocess.Popen) -> None:
    proc.stdin.write("\n")
    proc.stdin.close()
    assert proc.wait(timeout=_TIMEOUT_S) == 0, proc.stderr.read()


def _base_env(make, tmp_path: Path, *, slots: int) -> dict:
    env = make(host={"lock_dir": str(tmp_path), "budget_slots": slots, "nice": False})
    env.pop("ROLA_HOST_BUDGET_HELD", None)
    return env


def test_n_plus_one_acquirers_block_then_release_frees_a_slot(tmp_path, dev_config_env):
    locks = tmp_path / "locks"
    locks.mkdir()
    env = _base_env(dev_config_env, locks, slots=2)
    holders = [_spawn(_HOLD_HOST_BUDGET, env) for _ in range(2)]
    try:
        for h in holders:
            assert _readline_or_none(h, _TIMEOUT_S) == "ACQUIRED\n"

        blocked = _spawn(_HOLD_HOST_BUDGET, env)
        try:
            # The pool is full: the (N+1)th acquirer must NOT get in promptly.
            assert _readline_or_none(blocked, 1.0) is None

            # Releasing one of the N frees exactly the slot the blocked one
            # was waiting on.
            _release(holders[0])
            holders[0] = None
            assert _readline_or_none(blocked, _TIMEOUT_S) == "ACQUIRED\n"
        finally:
            _release(blocked)
    finally:
        for h in holders:
            if h is not None:
                _release(h)


def test_reentrant_acquire_is_a_noop_under_an_ancestors_marker(tmp_path, dev_config_env):
    """A nested acquire that finds `ROLA_HOST_BUDGET_HELD` already set must
    not open or wait on any lock file -- proven here by asking for MORE slots
    than exist (`slots=1`, request `n=5`) and observing it does not block."""
    import host_budget

    locks = tmp_path / "locks"
    locks.mkdir()
    env = _base_env(dev_config_env, locks, slots=1)
    os.environ["ROLA_DEV_CONFIG"] = env["ROLA_DEV_CONFIG"]
    os.environ["ROLA_HOST_BUDGET_HELD"] = "999999"  # an "ancestor" holding it
    try:
        with host_budget.acquire(5, label="nested") as held:
            # Reentrant path yields `n` back, never blocking on a pool that
            # could not possibly satisfy a real request for 5 of 1 slot.
            assert held == 5
        # Reentrant path never opened a lock file at all.
        assert list(locks.iterdir()) == []
    finally:
        os.environ.pop("ROLA_HOST_BUDGET_HELD", None)
        os.environ.pop("ROLA_DEV_CONFIG", None)


def test_gpu_lock_shared_pool_admits_two_and_blocks_a_third(tmp_path, dev_config_env):
    lock_path = str(tmp_path / "gpu.lock")
    env = dev_config_env(host={"gpu_shared_slots": 2, "nice": False})
    env.pop("ROLA_GPU_LOCK_HELD", None)

    a = _spawn(_HOLD_GPU_LOCK, env, lock_path, "shared")
    b = _spawn(_HOLD_GPU_LOCK, env, lock_path, "shared")
    try:
        assert _readline_or_none(a, _TIMEOUT_S) == "ACQUIRED\n"
        assert _readline_or_none(b, _TIMEOUT_S) == "ACQUIRED\n"

        c = _spawn(_HOLD_GPU_LOCK, env, lock_path, "shared")
        try:
            assert _readline_or_none(c, 1.0) is None
            _release(a)
            a = None
            assert _readline_or_none(c, _TIMEOUT_S) == "ACQUIRED\n"
        finally:
            _release(c)
    finally:
        for h in (a, b):
            if h is not None:
                _release(h)


def test_gpu_lock_exclusive_waits_for_every_shared_holder_and_vice_versa(tmp_path, dev_config_env):
    lock_path = str(tmp_path / "gpu.lock")
    env = dev_config_env(host={"gpu_shared_slots": 2, "nice": False})
    env.pop("ROLA_GPU_LOCK_HELD", None)

    shared = _spawn(_HOLD_GPU_LOCK, env, lock_path, "shared")
    try:
        assert _readline_or_none(shared, _TIMEOUT_S) == "ACQUIRED\n"

        exclusive = _spawn(_HOLD_GPU_LOCK, env, lock_path, "exclusive")
        try:
            # One shared holder is enough to block an exclusive acquire --
            # it needs BOTH slots, not just the one free one.
            assert _readline_or_none(exclusive, 1.0) is None

            # A second shared acquire is ALSO blocked while the exclusive
            # acquire is queued on the second slot (it already holds slot 0).
            other_shared = _spawn(_HOLD_GPU_LOCK, env, lock_path, "shared")
            try:
                assert _readline_or_none(other_shared, 1.0) is None
            finally:
                other_shared.terminate()
                other_shared.wait(timeout=_TIMEOUT_S)

            _release(shared)
            shared = None
            assert _readline_or_none(exclusive, _TIMEOUT_S) == "ACQUIRED\n"
        finally:
            _release(exclusive)
    finally:
        if shared is not None:
            _release(shared)
