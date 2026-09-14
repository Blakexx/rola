# `tools/host_budget.py` — one host-compute budget for every CPU-heavy tool

Blake, 2026-08-29 (the LOCKS brief): "add locks around our tooling where it
makes sense … maybe semaphores, allow some controlled level of parallelism."
Before this file, `cicc` (`tools/build_lock.py`'s `GLOBAL_CICC_SLOTS`) had a
pool of its own, but a `compute-sanitizer` run, a `clang-tidy
--cuda-host-only` pass over one translation unit, and a `pytest-xdist`
worker were all just as CPU-heavy and drew from nothing — three of THOSE
running alongside a full-budget compile is the same "12+ heavy processes on
one host" shape `build_lock.py`'s own module docstring records (the
2026-08-28-night incident), wearing a different process name. Slot counts
now live in exactly one place: `HOST_BUDGET_SLOTS`
(`host.budget_slots` in the dev config, default `os.cpu_count() - 2` bounded by memory).

## The mechanism

`acquire(n=1, exclusive=False, label=...)` is a context manager, the same
gate/iteration shape `tools/build_lock.py` already used before this file
existed (that file now delegates to this one — see below):

* `exclusive=True` (a "gate" draw): waits for and holds **every** slot.
* `exclusive=False` (default, an "iteration" draw): waits for **at least
  one** free slot (never proceeds with zero), then opportunistically takes
  up to `n` more that are free *right now* without waiting further — three
  iteration callers started at once each make some progress rather than one
  winning the whole budget or all three deadlocking.

When no slot is free at all, the fallback poll sweeps **every** slot index
each iteration rather than blocking on a hardcoded one — an earlier version
of both this file and `tools/gpu_lock.py` waited on slot 0 specifically,
which can wait forever on a slot nobody is about to free while a different
one frees immediately (found by this file's own gate, `tests/unit/
test_locks_accounting.py`: releasing "the other" holder never unblocked a
waiter pinned to slot 0, since which physical slot a given holder ends up
with is itself a race).

Reentrant by construction, the same shape `tools/gpu_lock.py`'s
`ROLA_GPU_LOCK_HELD` uses: acquiring sets `ROLA_HOST_BUDGET_HELD` in the
process environment, and a nested acquire that finds it already set is a
no-op — neither opening nor waiting on any lock file a second time. Because
`os.environ` is inherited across `fork`+`exec`, this covers both same-process
re-entry and the subprocess case (a tool that holds the budget launching a
child that also acquires it).

`host.locks_trace: true` prints every acquire/release (holder pid, slot count)
for this file's locks and `tools/gpu_lock.py`'s alike. The slot locks live in `host.lock_dir`, which a test
points at a throwaway directory through a test dev config.

`python tools/host_budget.py [--slots N] [--exclusive] -- <cmd>` is the one CLI for a CPU-heavy command that is
not `setup.py` (a census compile, a clang-tidy translation unit): it holds the slots, lowers its priority per
`host.nice` (`lower_priority`, shared by the build and GPU locks), and runs the command.

## `tools/build_lock.py` now draws from here

`build_lock.acquire(gate, desired)`'s public API is unchanged; internally it
now calls `host_budget.acquire(desired, exclusive=gate, ...)`.
`GLOBAL_CICC_SLOTS` is kept as a name (some docs/callers still say
`build_lock.GLOBAL_CICC_SLOTS`) but is now simply an alias for
`host_budget.HOST_BUDGET_SLOTS`, set by `host.budget_slots`.

## Other draws

* `tools/sanitize_oracle.py`: `host_budget.acquire(2, ...)` — a sanitizer
  run is CPU-heavy (instrumenting every memory access) independent of the
  wrapped pytest process's own cost.
* `tools/lint/run_clang_tidy_device.sh`: one `python tools/host_budget.py
  --slots 1 -- clang-tidy ...` per translation unit — a `clang-tidy
  --cuda-host-only` pass is a full front-end parse, as CPU-heavy as a
  compile.
* `tests/conftest.py`: a session-scoped, autouse fixture draws 1 slot for
  the lifetime of a pytest run, but only when `PYTEST_XDIST_WORKER` is set
  (i.e. only inside a forked `pytest-xdist` worker; a bare `pytest` run
  draws nothing here).

## `file_lock(name)` — the third category

Not a counting resource: one short-held **exclusive** lock over a single
named shared artifact, for tools that write ONE output potentially
concurrently — a ratify manifest (`tools/ratify.py`'s `--write`), and the
generated carry-selection header (`tools/ratify.py`'s unconditional
per-build write, and `tools/gen_shards.py --write`, both targeting the same
`CARRY_SELECTION_INC` path). Reentrant per name, the same marker shape as
`acquire()`.

## Gate

`tests/unit/test_locks_accounting.py` proves, with fake children (small
driver scripts that acquire and then block on stdin, never a real compile or
GPU kernel), all pointed at a throwaway `tmp_path` lock directory so the
gate never contends with another agent's real locks on this host:

* N+1 acquirers of an N-slot pool: the (N+1)th blocks; releasing one of the
  N frees it.
* A nested acquire under an ancestor's marker is a no-op (asks for more
  slots than exist and does not block).
* `tools/gpu_lock.py`'s two GPU modes (see `gpu_lock.md`).
