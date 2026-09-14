# `tools/gpu_lock.py` — the GPU lock also lowers its own priority

`gpu_lock()` is the one primitive every GPU-touching entry point (pytest,
`tools/probe_cells.py`, `tools/sanitize_oracle.py`,
`tools/compare.py`) takes itself, reentrant by construction
(this file's own module docstring covers the mechanism).

## Two modes (LOCKS brief, 2026-08-29)

`mode="exclusive"` (default, every pre-existing bare `gpu_lock()` call keeps
this behavior) excludes every other holder, either mode — for MEASURED work:
`tools/probe_cells.py`, `ncu`, `tools/compare.py`.
`mode="shared"` admits up to `GPU_SHARED_SLOTS` (default 2) concurrent
holders — for CORRECTNESS work that does not corrupt another correctness
run's answer by sharing the device: the pytest oracle/integration/cuda-marked
unit tiers (`tests/conftest.py`), and `tools/sanitize_oracle.py`.

Built on one reader/writer lock file (`{base}.rw`, native `flock(LOCK_SH |
LOCK_EX)`) plus the `GPU_SHARED_SLOTS`-sized counting pool, plus one
admission mutex (`{base}.wq`) that closes a starvation gap plain `flock()`
leaves open: Linux does not give a *pending* exclusive request priority over
a *new* shared one, so a steady stream of shared acquirers could keep an
exclusive acquirer waiting forever. Every acquirer takes `{base}.wq` first
and holds it for its own admission step (an exclusive acquirer across its
whole, possibly long, wait for `LOCK_EX`; a shared acquirer only across its
own — always uncontended — `LOCK_SH` grab), so no new shared acquirer can
join once an exclusive request is outstanding. Found and fixed by this
file's own gate (`tests/unit/test_locks_accounting.py`): a three-process
repro where a second shared acquirer walked straight past a pending
exclusive one.

The fallback that blocks when no counting slot is free polls the whole slot
sweep rather than a hardcoded index, for the same reason `host_budget.md`
gives: which physical slot a given holder ends up with is a race, so waiting
on slot 0 specifically can wait on a slot nobody is about to free while a
different one frees immediately.

The NICE stage
(Blake, 2026-08-29) adds the same treatment `tools/build_lock.py` gives a
compile slot: the process that actually acquires the lock file (never a
nested no-op acquirer — those inherit the ancestor's already-lowered state)
lowers its own CPU niceness (`os.nice(10)`) and, if `ionice` is on `PATH`,
its I/O class to idle (`ionice -c 3 -p <pid>`) before the `with` block's body
— a test, a probe launch, a sanitizer run, a bench — runs. A missing
`ionice` binary is a logged skip, not a failure.

`host.nice: false` in the dev config skips this (the coordinator's interactive gate runs;
`docs/internals/tools/dev_config.md`). Default is off. This changes nothing about
what a GPU-locked run measures: the probe and bench harnesses' numbers are
GPU-timed, and niceness only affects host CPU/IO scheduling around the GPU
work, not the work itself.

`tests/unit/test_gpu_lock_lowers_child_priority` (in
`tests/unit/test_lock_priority.py`, shared with the build-lock proof) is the
gate: a child process launched from inside `gpu_lock()`'s `with` block
reports nice 10 in `/proc/<pid>/stat` (field 19), and 0 under
`host.nice: false` — each condition its own fresh driver subprocess, for the
same reason `build_lock.md` states (niceness cannot be lowered back down by
an unprivileged process). The lock path used is a throwaway `tmp_path` file,
never the real `/tmp/rola_gpu.lock`, so the test never contends with actual
GPU work.
