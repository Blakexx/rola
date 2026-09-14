# Measurement — the method, the instruments and the record

Every performance number this repository publishes is taken by one method, with one stopwatch per comparison, and
stored with its provenance. This document is what that sentence means.

| layer | where | answers |
|---|---|---|
| cells | `benchmarks/cells/` (`registry.py`) | WHAT shape and draw is measured: one name, one cell, everywhere; `runnable()` is which cells a binary runs, refusing one without a shipped arm |
| subjects | `benchmarks/bench/subjects.py`, `bench/provider.py` | WHICH launch is timed, and each arm's dials |
| method | rola-devtools' `rola_devtools.interleave`, run by `tools/compare.py` | HOW: interleaved call by call, paired within a rep |
| preconditions | `tools/compare.py`, `bench/provider.py` | what the box and the binary must be first |
| statistics | rola-devtools' `rola_devtools.verdict` | whether a difference is real |
| record | `rola_results` (the measurements store) | where a number lives afterwards, and which stored samples are a baseline |

The benchmark repository's suite (rola-bench, `rola_bench/measure`) times through `tools/compare.py`, so a suite number
and a number taken here are the same measurement. `tools/probe_cells.py` runs the same subjects across two binaries with
`ncu` beside the timing.

## The method

`tools/compare.py` (`docs/internals/tools/compare.md`) runs the interleaving driver over arms of rola checkouts and of
other libraries at one registered cell. Each arm environment gets one worker process; every arm is warmed past the
driver's floor of 10 calls; then every rep of every round calls every arm once, in a fresh random order, and each sample
is one call on the arm's own stopwatch. Reps are odd, so a round's median is one of its samples, and a paired ratio is
taken within a rep, so drift slower than a rep is common to both arms. The driver refuses a warmup under the floor, an
even rep count, and two stopwatches in one comparison.

Call by call, because drift on this host is the size of the effects measured: its sustained clock has two states ~17 %
apart with a minutes-long time constant, and the same binary in both arms once read 0.864 vs 0.739 ms under a fixed
order. Blocks let a clock change land between arms, and a fixed order charges drift to one arm. The null gate (one arm in
two workers, rola-devtools' `tests/test_interleave.py`) is the check that the method adds no ratio of its own.

## Preconditions, which the tool acquires

Not a checklist: `tools/compare.py` and the provider do each of these, and no flag skips one.

| step | what happens |
|---|---|
| GPU lock | `/tmp/rola_gpu.lock`, exclusive, through `tools/gpu_lock.py`'s `gpu_lock()`, held for the whole comparison. The tool is invoked BARE — never wrapped in an external `flock` on the same path, which self-deadlocks (per-open-file-description semantics; the K38 incident). `gpu_lock()` is reentrant (`ROLA_GPU_LOCK_HELD`), so a self-locking tool may launch another. `host.gpu_lock` in the dev config names the path (host and containers share it). |
| clock lock | the host's own lock (`tools/clock_lock.py`, the dev config's `clock.json`), proven by the device's clock read before the first call and after the last; a comparison whose second read is off the lock is refused. A host without one runs unlocked, and the result says so. |
| binary identity | building an arm asks the device for the subject's family stamp and refuses the arm without it: a path or a hash passes against a stale extension, a device-side fact does not. |
| extension identity | `import rola` must resolve inside the arm's own checkout, or the arm is refused: an editable install in a shared venv otherwise answers with another tree's kernel while the record carries this tree's commit. |
| tree identity | `--record` stores each rola arm's commit and the sha256 of its tracked diff; a dirty tree is stamped, not refused. |

## The reference

Every reported carry number sits beside FlashAttention at the same cell, measured in the
same interleaved, clock-locked run (`tools/compare.py`): rola-bench's attention arm
(`rola_bench/measure/attention.py`, a provider of the interleaving driver), causal, one head of
width `dv`, `L` tokens, bf16, through torch's flash backend: Dao's FlashAttention-2 compiled
into torch, or FA3 (Hopper) and FA4 (Blackwell) once `torch.nn.attention.activate_flash_attention_impl`
registers them. The backend is forced, so torch refuses a call flash cannot take rather than
timing its math or memory-efficient backend, and the row records torch's version and the
implementation. The capacity-fair point is `N = L`; a cell states its `L`. The layer
comparison is the inter term plus the intra term against that one number.

## The stopwatch

There is one, it is named, and its name travels with every number.

| instrument | what it measures |
|---|---|
| **`cuda_events`** (committed) | device-side elapsed time between two events straddling the launch |
| `perf_counter` | host wall time around a full device sync on both sides — the launch and sync cost included |
| `ncu` | a counter, or a duration taken under profiling conditions: replayed, caches flushed, base clocks |

**A comparison across two of them is refused, not warned about.** The gap is not
noise; it is the launch and sync cost one instrument includes. Measured on this box
(RTX 3080 Ti, sm_86, 30 interleaved reps per instrument):

| cell | `cuda_events` | `perf_counter` | ratio |
|---|---:|---:|---:|
| `topo-D2-w64x64-BC64` global | 0.0725 ms | 0.2942 ms | 4.06× |
| `atscale-L2048-BC16` global | 12.8251 ms | 13.4350 ms | 1.05× |

The host cost is roughly constant (0.22 ms here), so it is nearly the whole
measurement on a sub-millisecond cell and a rounding error at scale. The
`ncu` duration for the same small cell reads 0.0906 ms — 25 % above the event time —
which is why an ncu duration is a third instrument and not a cross-check on the
first two.

Every arm names its stopwatch (`Arm.instrument`; rola's arms time with `cuda_events`), and the driver refuses a
comparison whose arms name two.

## The flagging rule: three gates, all must fire

`rola_devtools.verdict.classify` judges samples; which stored samples it is given is the store's query (below).

1. **Effect size** — `threshold_ms`: `median + 3·IQR/1.349` over the baseline's session medians. `ALARM_SIGMA = 3.0` is
   the only free choice, and it is stated as a false-alarm rate (~0.1 % one-sided per cell per run), never as a
   percentage of anything.
2. **Significance** — `paired_verdict`: an exact Wilcoxon signed-rank test over the judged session's per-round
   differences, candidate minus reference, at `ALPHA = 0.01`. **The round floor is derived, not chosen**: the smallest
   two-sided exact p reachable with `n` non-zero differences is `2/2ⁿ`, so below `ceil(log2(2/ALPHA)) = 8` rounds no
   outcome can be significant and the test is undefined rather than underpowered. `tools/compare.py` and the suite run 8;
   fewer report `insufficient_data`.
3. **Persistence** — `classify_flags`: a regression needs a trailing run of at least two violations. One thermally
   unlucky run on a box with logged power capping is not a kernel change. A run of three that has since healed is
   `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first says the design cannot decide, the
second says it decided.

**THE SPREAD IS THE SPREAD OF THE QUANTITY THE COMPARISON CROSSES — a measured correction, not a preference.** A
baseline recorded from one session's reps, on the chunk arm (tag `baseline/pre-k31`), put 17 of 44 arms over their
limit on the same binary a minute later: a within-session spread is reps taken back to back on one warm fixture, often
exactly zero, while a comparison against a baseline crosses processes (the fixture rebuilt, the allocator cold, the
launch order different). The threshold is therefore derived from session medians, and a baseline needs three sessions
(`MIN_BASELINE`).

**AND NEVER A SPREAD SMALLER THAN THE STOPWATCH RESOLVES.** That correction left two arms over their limit on an
unchanged binary, both with a recorded spread of exactly `0.0000 ms`: the device-event timer quantizes, sessions of a
small kernel land on the same value, and the limit sits at the median the next tick exceeds. The spread used is
`max(IQR, resolution)`, where the resolution is the smallest gap between the distinct values the judged samples hold —
measured on every judgement, because it is a property of the stopwatch and the box — and the verdict says when it bound
(`floored_at_resolution`).

## Which samples are a baseline

    python -m rola_results verdict [--cell C] [--subject S] [--baseline LABEL] [--json]

reads the suite's timing sessions (the `session_arms` view) per unit: a cell, a subject, a call count, both arms' names,
and the reference's device and torch. For the newest session of each candidate commit, the baseline is the reference
arm's samples over that unit's newest ten sessions; the runs are the candidate's sessions at its commit, oldest first;
the paired differences are the judged session's per-round medians. Nothing is stored: a verdict is a reading of the
records, taken again whenever it is asked for. The suite prints them at the end of every run that times a session.

## SASS

`python tools/ratify.py` and `python tools/sass_bodies.py` are the instruments. Both binaries in a
comparison no longer need to have been built at the same path (P19,
`docs/build.md`'s path-trap section) — they must agree on `ratify.SASS_HASH_VERSION`,
which `--compare` checks and refuses on mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `python tools/ratify.py --census` there, read the spill
column — and never a flag on the shipped tool. `--force-ctas` was that flag; it
was deleted once nothing in `csrc/` read its macro, because a tool that
can force a bound is a tool that can ratify one.

## The sanitizer gap

No sanitizer measurement has been taken of the carry family. `compute-sanitizer` runs a kernel under conditions no timing
instrument reproduces, so it is a third instrument like `ncu` and needs its own refusal rules and record shape before a
number from it enters the record. Named here because an undelivered item named is not a delivered one.

## What is deliberately not here

* **A second method.** The in-process A/B driver and its statistics were deleted once `tools/compare.py` timed the same
  subjects (`docs/internals/DELETIONS.md`): two methods are how a harness comes to disagree with itself.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in a landing. Any query artifact is
  derived and ignored — see the record below, which is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in the surveyed field; replacing it
  would reintroduce the invented number the derivation exists to avoid.

## The measurements record: `rola_results`

Every measuring tool stores what it measured through one library, `rola_results`, in the rola-results repository:
`store.root` in the dev config (default `rola-results` beside the checkouts; the container mounts it at
`/workspace/store`), importable from every interpreter `tools/dev.py init` provisions. A result measured in any
worktree, host or container lands in that one repository, visible to every other and to the dashboard at once.

A tool opens a store at its OWN LOCATION and hands it the SEMANTICS of what it measured -- the inputs its numbers
depend on, reduced to identities (a commit and its tracked diff, a binary's manifest or file digest, the cells, the
counts, the device's software, the clock lock). The KEY is sha256 over the semantics, so the same measurement taken
again is another SAMPLE of the same record and a changed input is a new record. A sample is the raw output or the
failure, when, how long, and its PROVENANCE (the checkouts by directory name, branch and commit, the stage, the host).
Every sample is kept, failures included. Nothing relational is stored: a ratio against a baseline is the reader's,
and a timing only compares within the session that interleaved it.

| location | writer | the semantics | a sample's output |
|---|---|---|---|
| `probe_cells` | `tools/probe_cells.py` | each binary's commit, tree digest, manifest, family stamp and lane (bench, calls, schedule); the cells, state arm, counts, device software, clock lock | every (binary, cell) row |
| `compare` | `tools/compare.py --record` | the point, the matching rule, each arm's label and name with its commit and diff (a foreign arm's provider), the counts, the seed, the reference | the driver's whole result |
| `suite/<module>` | rola-bench's measurement suite | the module, unit, identities and dependencies | the instrument's raw JSON |
| `calibration` | `benchmarks/unit/bench_carry_calib.py` | the parts binary, device, owners, sizes, clock | the calibration rows |
| `pipe_timeline`, `pipe_timeline.scale` | `tools/pipe_timeline.py` | the cell, binary and scale; a calibration's composition | the series and summary; the plateau |
| `build_ledger`, `compose_ledger` | `tools/build_ledger.py`, `tools/compose_ledger.py` | the commit and diff, the tag or cells | the report |
| `dram_by_activity` | `tools/experiments/dram_by_activity.py` | each binary's commit, diff and manifest; the variants, counts | the rows |
| `environment` | `tools/dev.py container check` | the environment key | the proofs |

A tool run by another tool stores nothing (`--no-record`): the caller keeps the output in its own record. `python -m
rola_results sql "..."` queries every location through a derived SQLite index beside the records (`history` and
`latest` for one location; the views `timing_rows`, `timing_pairs`, `session_arms`, `driver_rows`, `cells` flatten the
tools' outputs, and the rola-results README gives the last, history, compare and baseline questions as SQL); `python -m
rola_results verdict` is the regression judgement above; `bench_driver` holds the deleted in-process driver's records; `python
tools/dev.py store commit -m <message>` commits the records
(`python -m rola_results commit`), after the repository's own check that every key recomputes from its semantics and
every output exists. Pushing is the owner's act. The record kept before the library -- the per-run JSONL, the perf
ledger, the ledger and composer reports -- left the store for the owner's archive (`docs/internals/DELETIONS.md`).
