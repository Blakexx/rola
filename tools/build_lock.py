# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MACHINE-WIDE CICC BUDGET, TOOL-ENFORCED (KERNEL_STANDARDS §14 addendum 2,
Blake ruling 2026-08-28-night: "three agents' MAX_JOBS=4 each -> 12+ cicc" sat
this host at 100% CPU for 10+ minutes and forced a WSL shutdown mid-session).

**WHY THIS CANNOT BE A BRIEF INSTRUCTION.** An external shell wrapper (the
same day, since deleted) had already unified the iteration-slot and gate-lock
mechanisms into one -- but it was a wrapper a caller had to remember to invoke, and three concurrent agents each forgetting it (or each reasonably
believing their OWN `MAX_JOBS=4` was the whole host's budget, not a share of
it) is exactly what happened. `RoLABuildExtension.run()` (`setup.py`) now
acquires this budget ITSELF, unconditionally, before the actual compile --
`pip install -e .` cannot start `cicc` without going through it, from any
worktree, any agent, any invocation path, brief or no brief.

**THE MECHANISM, NOW DRAWN FROM `tools/host_budget.py` (LOCKS brief,
2026-08-29).** `cicc` used to count against its own private pool
(`GLOBAL_CICC_SLOTS`/`ROLA_GLOBAL_CICC_SLOTS`, `/tmp/rola_cicc_slot_<i>.lock`),
separate from every other host-CPU-heavy tool -- a sanitizer run, a
clang-tidy device pass, a pytest worker -- so three of THOSE running
alongside a full-budget compile was the same "12+ heavy processes on one
host" shape this file exists to prevent, wearing a different process name,
and this file's slot count could not see them. Slot counts now live in
exactly one place, `tools/host_budget.py`'s `HOST_BUDGET_SLOTS`
(`host.budget_slots` in the dev config, default `os.cpu_count() - 2`) --
`GLOBAL_CICC_SLOTS` below is that same number, kept as a name this module's
own callers and docs already cite; `ROLA_GLOBAL_CICC_SLOTS` is superseded.
A GATE build (the shipped arm set) takes EVERY slot of the shared pool,
blocking until the whole budget is free (`python tools/host_budget.py --exclusive -- <cmd>` is the same draw
for a command that is not `setup.py`). An ITERATION build takes however many slots are free RIGHT NOW,
from 1 up to its own desired `MAX_JOBS`, BLOCKING (never proceeding with
zero) until at least one is free -- so three iteration builds started at once
each still get SOME progress rather than one winning everything or all three
deadlocking. `MAX_JOBS` is then set to the COUNT ACTUALLY HELD, never to what
the caller originally asked for -- the whole enforcement is worthless if a
build keeps its self-chosen `MAX_JOBS` after being handed fewer slots than
that.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_budget  # noqa: E402 -- path insert must precede this import

#: Kept as a name: callers and docs elsewhere in this tree say
#: `build_lock.GLOBAL_CICC_SLOTS` -- it is now an alias for the one shared
#: pool every host-CPU-heavy tool draws from.
GLOBAL_CICC_SLOTS = host_budget.HOST_BUDGET_SLOTS


@contextmanager
def acquire(*, gate: bool, desired: int):
    """Hold `held` slots (yielded) for the wrapped compile; releases on exit.

    `gate=True`: waits for and holds ALL of `host_budget`'s shared slots -- the
    machine-wide budget, exclusively, the same discipline a milestone ratify
    build already required against a single lock file (KERNEL_STANDARDS §15
    addendum), now against the shared pool every host-CPU-heavy tool (not
    just a compile) draws from.
    `gate=False`: waits for AT LEAST ONE free slot (never proceeds with zero),
    then opportunistically takes up to `desired` MORE slots that are free
    RIGHT NOW without waiting further for them -- an iteration build gets
    *some* parallelism promptly rather than either the whole budget or an
    unbounded wait for it.
    """
    desired = max(1, desired)
    label = "gate" if gate else "iteration"
    with host_budget.acquire(desired, exclusive=gate, label=f"build_lock:{label}") as held:
        host_budget.lower_priority("build_lock")
        yield held
