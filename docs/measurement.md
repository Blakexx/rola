# Measurement — the method, the instruments and the record

Every performance number this repository publishes is taken by one method, with one stopwatch per comparison, and
stored with its provenance. This document is what that sentence means.

| layer | where | answers |
|---|---|---|
| cells | rola-devtools' central registry `rola_devtools.cells` (rola's reading: `benchmarks/cells/`) | WHAT is measured: an input, named once for every package, with its draw, seed and proven regime |
| subjects | `benchmarks/bench/subjects.py`, `bench/provider.py` (rola's runner) | WHICH launch is timed on a cell, at which LEVEL, each arm's dials, its untimed reset, and which cells this binary cannot run |
| declarations | `declare.py` (this checkout's targets), rola-bench's `declare.py` (checkouts composed) | WHAT RUNS: the build, the instruments, the timing registrations and sessions, the stores, as targets of `rola_devtools.build` |
| method | rola-devtools' `rola_devtools.timing` (`measure_timing`) | HOW: interleaved call by call, one call at a time, in a fresh random order each rep |
| preconditions | the targets' requirements (`rola_devtools.build.resources`), `bench/provider.py` | what the box and the binary must be first |
| statistics | rola-devtools' `rola_devtools.verdict` | whether a difference is real |
| record | `rola_results` (the measurements store) | where a number lives afterwards, and which stored samples are a baseline |

rola measures itself with `python -m rola_devtools.build run declare.py:all`; rola-bench's root loads this file from each
checkout it compares, declares every checkout's targets beside its attention reference, and times them in shared
sessions. Both are one method: a number taken either way is the same measurement.

## The method

A session is a `measure_timing` target (`rola_devtools.timing`): the timing entries it depends on -- a checkout's arm on
a central cell, rola's `carry_forward` or rola-bench's `flash` -- are set up in their checkouts' worker processes behind a
barrier (nothing is timed until every entry has set up), each warmed past the floor of 10 calls, and then every rep of
every round calls every entry once, in a fresh random order, one call at a time: the next call is sent only when the
last has replied. Each sample is one launch on the entry's own stopwatch, taken inside its worker, so the pipe between
processes is never in a sample; an entry whose launch changes what its next call reads (a carried state, a decode step)
is reset to exactly what its first call saw before every call, untimed, so every call does the same work on the same
data. Reps are odd, so a round's median is one of its samples. The session keeps every sample in the order taken, with
its round, rep and position; which entry is the reference is chosen when the samples are read, never when they are
taken. An entry that cannot set up (a kernel this binary lacks) is recorded in the session and the rest are timed; two
stopwatches in one session fail it.

A NULL GATE (`measure_null_gate`) times one registration's entries against copies of themselves in second workers of the
same checkout: where the per-rep ratios of the two copies put one outside their interquartile range, a worker's bias is
found, and a ratio between two checkouts' workers on that cell is read beside it. rola-bench's root declares one over
the target's `carry_forward`.

Call by call, because drift on this host is the size of the effects measured: its sustained clock has two states ~17 %
apart with a minutes-long time constant, and the same binary in both arms once read 0.864 vs 0.739 ms under a fixed
order. Blocks let a clock change land between arms, and a fixed order charges drift to one arm. A timing only compares
within the session that interleaved it.

## Preconditions, which the targets acquire

Not a checklist: the requirements a target holds and the runner's setup do each of these, and no flag skips one.

| step | what happens |
|---|---|
| GPU lock | a session holds `gpu: all` (`rola_devtools.build.resources`): `/tmp/rola_gpu.lock` exclusive through `gpu_lock()` (`rola_devtools.locks.gpu`), taken by the build system for the whole session and handed to the entry workers as the lock-held marker, so a self-locking tool a target starts does not wait on it. Nothing is wrapped in an external `flock` on the same path, which self-deadlocks (per-open-file-description semantics: the wrapper's lock blocks the tool's own `flock` call forever). `host.gpu_lock` in the dev config names the path (host and containers share it). |
| clock lock | a session holds `clock: 1`: the host's own lock (`rola_devtools.locks.clock`, the dev config's `clock.json`), proven by this binary's clock read (`register_clock_reader`) before the first call and after the last; a session whose read is off the lock fails. A host without one runs unlocked, and the session records its reads. |
| binary identity | building an arm asks the device for the subject's family stamp and refuses the arm without it: a path or a hash passes against a stale extension, a device-side fact does not. |
| extension identity | `import rola` must resolve inside the entry's own checkout, or the arm is refused: an editable install in a shared venv otherwise answers with another tree's kernel while the record carries this tree's commit. |
| tree identity | every stored sample carries each checkout's commit and the sha256 of its tracked diff; a dirty tree is stamped, not refused. |

## The three levels, and what a comparison may cross

Blake, 2026-09-15: the LAYER holds the projections and produces the routing (RoLA's analogue of QKV); the OP takes those
bare operands and a state and produces the next state and the readout -- the facts, the paging, the packing and the
kernels; a KERNEL runs its own launches and nothing around them. Every subject states the level it prices
(`Subject.level`) and every sample records it, so a reading pairs entries of ONE level and `python -m rola_results
verdict` refuses a pair of two. A comparison between this library and another is therefore a LAYER comparison: RoLA's
own layer against attention's, because RoLA's number includes the state its capacity buys.

What this line can measure today: the KERNEL level (`carry_forward`, `intra_forward`, `carry_intra` -- the carry and the
intra as one operator over facts the caller built -- `liveness_pass`, `entmax_solve`) and one OP-level path
(`decode_step`, through the engine's decode DAG, which runs the facts pass and commits the pages it writes). There is no
op-level prefill and no layer subject: `rola.interface.rola_op` refuses on prefill since the box-native rebuild deleted
the shipped chunk consumer, so the chunk DAG has no live path to time. The flash reference is a layer entry, and until a
RoLA layer runs beside it, a session that holds both is two levels measured in one place, not a comparison.

## The reference

Every reported carry number sits beside FlashAttention at the same point, measured in the
same interleaved, clock-locked session: rola-bench's attention entry
(`rola_bench/measure/attention.py`) on the group's QKV cell, causal, one head of
width `dv`, `L` tokens, bf16, through torch's flash backend: Dao's FlashAttention-2 compiled
into torch, or FA3 (Hopper) and FA4 (Blackwell) once `torch.nn.attention.activate_flash_attention_impl`
registers them. The backend is forced, so torch refuses a call flash cannot take rather than
timing its math or memory-efficient backend, and the entry records torch's version and the
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

Every timed entry names its stopwatch (`Timed.instrument`; rola's arms time with `cuda_events`), and a session refuses
entries that name two.

## The flagging rule: three gates, all must fire

`rola_devtools.verdict` judges a candidate against a reference timed in the same sessions; which stored sessions and
which reference it is given is the store's query (below). Timing compares only within the session that interleaved it,
so every gate reads within-session quantities: `session(candidate, reference)` takes the two members' samples by round
and gives, per round, each member's median, their RATIO and their DIFFERENCE.

1. **Effect size** — `ratio_limit`: the session's median ratio over `1 + 3·IQR/1.349` of its own per-round ratios.
   `ALARM_SIGMA = 3.0` is the only free choice, and it is stated as a false-alarm rate (~0.1 % one-sided per cell per
   session), never as a percentage of anything.
2. **Significance** — `paired_verdict`: an exact Wilcoxon signed-rank test over the session's per-round differences,
   candidate minus reference, at `ALPHA = 0.01`. **The round floor is derived, not chosen**: the smallest two-sided
   exact p reachable with `n` non-zero differences is `2/2ⁿ`, so below `ceil(log2(2/ALPHA)) = 8` rounds no outcome can
   be significant and the test is undefined rather than underpowered. Sessions run 8; fewer report `insufficient_data`.
3. **Persistence** — `persistence`: a regression needs a trailing run of at least two sessions over their own limits, at
   the same code on both sides. One thermally unlucky session on a box with logged power capping is not a kernel change.
   A run of two that has since healed is `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first says the design cannot decide, the
second says it decided.

**THE SPREAD IS THE SPREAD OF THE QUANTITY JUDGED — measured corrections, not preferences.** A baseline once recorded
from one session's reps, on the chunk arm (tag `baseline/pre-k31`), put 17 of 44 arms over their limit on the same
binary a minute later: reps taken back to back on one warm fixture spread less than a comparison that crosses processes.
The verdict then derived its limit from a reference's medians across sessions, which carried the host's drift instead
(two clock states ~17 % apart). What is judged now is the paired ratio inside one session, so the limit is that ratio's
own round-to-round spread, drift both members share cancels in it, and the one thing the ratio crosses that its spread
cannot see -- two worker processes -- is the null gate's.

**AND NEVER A SPREAD SMALLER THAN THE STOPWATCH RESOLVES.** Two arms once sat over their limit on an unchanged binary,
both with a recorded spread of exactly `0.0000 ms`: the device-event timer quantizes, a small kernel's calls land on the
same value, and the limit sits at the median the next tick exceeds. The spread used is `max(IQR, resolution)`, where the
resolution is the smallest gap between the distinct values one member's samples hold, relative to the reference's
median -- measured on every judgement, because it is a property of the stopwatch and the box -- and the verdict says
when it bound (`floored_at_resolution`).

## Which samples are judged

    python -m rola_results verdict --reference LABEL [--candidate LABEL] [--cell C] [--arm A] [--json]

reads the stored timing sessions (the `timing_members` and `timing_samples` views). In every session that timed the
reference label's arm on a cell, each other label's member of that arm on that cell is paired with it and judged inside
the session. A unit is the arm, the cell, the two labels and the code each ran (commit and tracked diff); its sessions,
oldest first, are the runs, and the newest is judged. Nothing is stored: a verdict is a reading of the records, taken
again whenever it is asked for. `python -m rola_results dashboard --out FILE --reference LABEL` is one run as a page.

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

* **A second method.** The in-process A/B driver, and later `tools/compare.py` and the interleaving driver, were deleted
  once the declared build system timed the same subjects (`docs/internals/DELETIONS.md`): two methods are how a harness
  comes to disagree with itself.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in a landing. Any query artifact is
  derived and ignored — see the record below, which is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in the surveyed field; replacing it
  would reintroduce the invented number the derivation exists to avoid.

## The measurements record: `rola_results`

Every measuring tool stores what it measured through one library, `rola_results`, in the rola-results repository:
`store.root` in the dev config (default `rola-results` beside the checkouts; the container mounts it at
`/workspace/store`), importable from every interpreter `tools/dev.py init` provisions. A result measured in any
worktree, host or container lands in that one repository, visible to every other at once.

A tool opens a store at its OWN LOCATION and hands it the SEMANTICS of what it measured -- the inputs its numbers
depend on, reduced to identities (a commit and its tracked diff, a binary's manifest or file digest, the cells, the
counts, the device's software, the clock lock). The KEY is sha256 over the semantics, so the same measurement taken
again is another SAMPLE of the same record and a changed input is a new record. A sample is the raw output or the
failure, when, how long, and its PROVENANCE (the checkouts by directory name, branch and commit, the stage, the host).
Every sample is kept, failures included. Nothing relational is stored: a ratio against a baseline is the reader's,
and a timing only compares within the session that interleaved it.

| location | writer | the semantics | a sample's output |
|---|---|---|---|
| `rola/<instrument>` | a store target of `declare.py` or rola-bench's root | the instrument target's semantics: its tool and arguments, code digest, the cells, the binary's and environment's outputs | the instrument's JSON per cell, with each cell's failure |
| `timing/session`, `timing/memory` | a store target over `measure_timing` / `measure_memory` | the session's semantics: every registration's executor, cells, code and binary, the timing parameters | every sample in order with its round, rep and position, each entry's status, the clock reads; each entry's peak memory |
| `timing/null` | a store target over `measure_null_gate` (rola-bench's root) | the gate's semantics: the registration, its cells, the timing parameters | the session of the two copies and, per cell, whether they agree |
| `calibration` | `benchmarks/unit/bench_carry_calib.py` | the parts binary, device, owners, sizes, clock | the calibration rows |
| `pipe_timeline.scale` | `tools/pipe_timeline.py --calibrate` | the cell, the binary and the calibration's composition | the plateau the tool reads back as later runs' scale |
| `compose_ledger` | `tools/compose_ledger.py` | the commit and diff, the cells | the report |
| `environment` | `tools/dev.py container check` | the environment key | the proofs |

A tool run by another tool stores nothing (`--no-record`): the caller keeps the output in its own record. `python -m
rola_results sql "..."` queries every location through a derived SQLite index beside the records (`history` and
`latest` for one location; views flatten the stored outputs, and the rola-results README lists them with the last,
history, compare and baseline questions as SQL); `python -m rola_results verdict` is the regression judgement above; `python
tools/dev.py store commit -m <message>` commits the records
(`python -m rola_results commit`), after the repository's own check that every key recomputes from its semantics and
every output exists. Pushing is the owner's act. The record kept before the library -- the per-run JSONL, the perf
ledger, the ledger and composer reports -- left the store for the owner's archive (`docs/internals/DELETIONS.md`).
