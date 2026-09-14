# Measurement — the instruments, the discipline, and the record

> **K31 (2026-08-20).** The chunk arm and its two-scan backward are DELETED from the
> master line (baseline = tag `baseline/pre-k31`; `docs/internals/DELETIONS.md`).
> `bench_chunk.py`, the consumer subjects, the latency baseline and their harness
> gates went with them; the surviving roster is `chunk_facts`, `chunk_block_bits`,
> `decode_step` and `entmax_solve` through `bench/driver.py`. Sections below that
> speak of the consumer harness describe the baseline tag's tree.

Every performance number this repository publishes is produced by one harness, taken
with one stopwatch, and written to one file with its provenance. This document is
what those three sentences mean.

**THE SUCCESSOR HAS LANDED.** The chunk arm's harness is `benchmarks/bench_chunk.py`
over `benchmarks/{unit,integration}/`, and the owed list below is DELIVERED — item by item,
with the file that discharges it, in [the owed list](#what-the-chunk-arm-harness-owed-and-where-it-landed).
The table and the sections that follow describe the RETIRED tiled apparatus and remain the
predecessor's record; where a row says RETIRED, its successor is named beside it.

**HALF OF THIS HARNESS IS RETIRED, AND THIS DOCUMENT IS BOTH ITS SUCCESSOR'S
SPECIFICATION AND ITS PREDECESSOR'S RECORD.** Every fixture, every methodology
object and the driver itself named the tiled consumer — `probe/registry.py`'s
entries each named `consumer_forward_kernel` and a `bench.cells` cell — so when
that kernel was deleted the harness had no subject and could not even
import. It was deleted as ONE capability rather than left broken
([`internals/DELETIONS.md`](internals/DELETIONS.md); revival `git show
c7eaeb9:<path>`).

| Layer | Where | Answers | State |
|---|---|---|---|
| fixtures | `benchmarks/bench/cells.py` | WHAT is measured | **RETIRED** — every cell was a tiled arm |
| methodology | `benchmarks/bench/pairing.py` | HOW: interleaved paired arms, median of per-pair ratios | **RETIRED** — the engine is what the successor most owes |
| preconditions | `benchmarks/probe/discipline.py` | what the box must be doing first | **RETIRED**; `settle_clocks` survives, moved into `bench/regression.py` |
| statistics | `benchmarks/bench/stats.py`, `bench/regression.py` | whether a difference is real | **SURVIVES**, gated by `tests/unit/test_perf_ledger.py` |
| record | `rola_results` (the measurements store) | where the number lives afterwards | **REPLACED** 2026-09-12: the perf ledger and `benchmarks/bench/ledger.py` were deleted for the library every tool stores through |
| driver | `benchmarks/rola_probe.py`, `benchmarks/probe/` | the process that runs all of it | **RETIRED** |

## What the chunk-arm harness owed, and where it landed

Stated as a list so it cannot be half-remembered, and marked DELIVERED only where a file
in the tree discharges it.

| owed | state | where |
|---|---|---|
| the interleaved paired-ratio engine (`_paired_one_kernel` and its ~7x bias-reduction rule) | **DELIVERED** | `benchmarks/bench/pairing.py` — `interleaved_ab`, and `PairedResult` REFUSES a statistic over samples that were not alternated |
| the discipline gates (the lock, the dirty-tree refusal, the clock-settle refusal, the extension's identity, the warmup floor) | **DELIVERED** | `benchmarks/bench/discipline.py`, `bench/pairing.py`, `bench/chunk_baseline.py` |
| their planted-violation tests | **DELIVERED** | `tests/unit/test_chunk_bench_harness.py` — one planted violation per gate, with positive controls |
| per-cell latency baselines for the chunk arm | **DELIVERED** | `benchmarks/cells/chunk_latency_baseline.json`, recorded whole across seven sessions by `bench_chunk.py baseline --write`; the gate is `tests/unit/test_chunk_latency_regression.py`, which reads and never writes |
| the frozen cells the baselines are keyed to | **DELIVERED** | `benchmarks/cells/` (the one registry: `carry_cells.json`, `layer_cells.json` + `layer_manifest.json`) |
| its own LOCKING STATEMENT | **DELIVERED** | below, and enforced by `bench.discipline.disciplined` |
| a sanitizer measurement for the chunk arm | **STILL OWED** | never existed; see [the sanitizer gap](#the-sanitizer-gap) |

**THE LOCKING STATEMENT (K46 -- self-locking, superseding the WRAPPED design this
paragraph used to describe).** The chunk-arm harness takes `/tmp/rola_gpu.lock` ITSELF
now, through `tools/gpu_lock.py`'s `gpu_lock()` -- the primitive every GPU entry point
in this tree uses (`tests/conftest.py`, `tools/probe_cells.py`,
`tools/sanitize_oracle.py`, `bench.discipline.disciplined`). It is invoked BARE —

```bash
python benchmarks/unit/bench_liveness.py --tier landing
```

Before K46 this harness was invoked WRAPPED and `bench.discipline.require_gpu_lock`
CHECKED that it was, by finding the ancestor process holding the lock path open,
reasoning that the `rola-probe` self-lock exception retired with `rola-probe` and the
rule then had no exception left. K46 reverses that call: a caller having to remember
"under flock" by hand is exactly the rule a tool should enforce itself
(KERNEL_STANDARDS §18), and `gpu_lock()`'s reentrancy (`ROLA_GPU_LOCK_HELD`) means a
self-locking tool can safely call another self-locking tool without the K38 deadlock
that made the WRAPPED design seem necessary in the first place.

<a id="the-sanitizer-gap"></a>

**The sanitizer gap.** No sanitizer measurement has ever been taken of the chunk arm, and
this harness does not add one: `compute-sanitizer` runs a kernel under conditions no timing
instrument reproduces, so it is a THIRD instrument like `ncu` and needs its own refusal
rules and its own row shape before a number from it can enter the record. Stating it here
is the honest half of the delivery — an undelivered item named is not a delivered one.

The surviving pre-P72 benches (`bench_layer.py`, `bench_scaling.py`, `bench_paging.py`) run
today and use the same statistics, but they are curve and attribution studies rather than
gated cells.

Everything below describes the discipline as it was BUILT and measured on the tiled
apparatus. It is the specification the successor is held to; where the successor's
implementation differs it is noted at the row.

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

`bench.pairing.same_instrument` implements the refusal; `bench.regression.check_instrument`
applies it to the latency gate, and a baseline that does not say which stopwatch took
it is refused as unknown rather than assumed compatible.

## The discipline, which the harness acquires

Not a checklist. `probe.discipline.disciplined` did each of these with no flag
that skipped any of them, and the successor owes the same — enforced, not
documented.

| step | what happens |
|---|---|
| GPU lock | `/tmp/rola_gpu.lock`, taken exclusively via `tools/gpu_lock.py`'s `gpu_lock()`; if held, the harness waits. The harness is invoked BARE — never wrapped in an external `flock` on the same path, which self-deadlocks (per-open-file-description semantics; the K38 incident). `gpu_lock()` is REENTRANT (`ROLA_GPU_LOCK_HELD`), so a self-locking caller invoking another self-locking tool as a subprocess cannot deadlock either. `host.gpu_lock` in the dev config names the lock path (host and containers share it). |
| memory fraction | `set_per_process_memory_fraction(0.65)`, inside the lock and before the fixture is built — the value the two device-heavy test modules already pin after two whole-machine VRAM freezes |
| clock settling | `settle_clocks()` — moved to `bench/regression.py` when `bench/instruments.py` went, because `bench_scaling.py` still needs it; **`settled: False` aborts the run.** Clocks cannot be locked from inside WSL2 at all, so targeting the plateau is the only available protocol, not a convenience |
| tree identity | `git rev-parse` + `git status --porcelain` + `ratify.csrc_digest()`; a dirty tree does not block the run, it stamps `dirty: true`, and dirty rows are excluded from baselines |
| extension identity | `import rola`, then resolve its `__file__` against this checkout's root; **an extension resolved from OUTSIDE this checkout ABORTS the run.** `rola` installs editable into a shared venv, so a probe launched from a worktree can silently import the MAIN tree's compiled extension unless the worktree leads it on `sys.path` — the row would then carry this tree's sha and csrc digest against another tree's kernel, agreeing with itself the way a true negative also would. The resolved path is stamped into every row's `env` regardless |
| tier | `light` or `landing`, stamped into every row — a light-tier number can never be quoted as a landing one by a reader who did not see the command |
| record write | automatic: stored through `rola_results` at `bench_driver` with where it came from; `--no-record` marks a scratch run |

An identity that does not describe the measured binary is worse than no identity —
which is why the extension check runs before the clock settles and before any row
exists, alongside the tree's sha and csrc digest rather than after them.

## The record

A driver run is stored through `rola_results` at `bench_driver` ("The measurements record: `rola_results`" below): the
RAW per-rep samples in run order, and for an interleaved A/B both arms' samples beside the paired verdict, because the
run order carries the drift signal and the alternation is the measurement design. Medians and IQRs are computed on read.
The record's semantics carry the box (GPU, the binary's manifest) and the tree, so a number is never compared across
them by accident. The append-only perf ledger that held these rows before the library, with its markdown-era legacy
import, left the store with the rest of the pre-library record (`docs/internals/DELETIONS.md`).

### The committed latency baseline

`benchmarks/cells/latency_baseline.json` was the per-commit gate's reference,
recorded by `benchmarks/bench_regression.py --write` and never by a test. **Its
rows are the TILED consumer's and no chunk number may be compared against them**;
`benchmarks/cells/README.md` says so beside the files, which are parked rather
than deleted because a measurement is a record even when its subject is gone. Under the committed
instrument its arms read 0.0625–0.2376 ms with block IQRs of 0.001–0.005 ms — a
spread 30–60× tighter than the host timer's, because the launch and sync cost that
dominated it is no longer in the measurement.

**A tighter band exposes anything that was hiding inside the old one.** It exposed
one immediately: the gate had been reading its number from inside a benchmark
runner's round loop, and that runner's schedule — not the kernel — set the value. The
gate now takes the arm's median once, through the recorder's own entry point, so the
two sides are the same quantity by construction. `benchmarks/test_latency_regression.py`
recorded the measurement that forced it, and retired with the harness — the lesson
(the gate and the recorder must take the same quantity through the same entry
point) is CARRIED: `tests/unit/test_chunk_latency_regression.py` and
`bench_chunk.py baseline --write` both go through `bench.pairing.measure_arm` on a
fixture from the same registry, and neither owns a round loop of its own.

### The chunk arm's committed baseline

`benchmarks/cells/chunk_latency_baseline.json`, one row per `subject|cell` over the
registry in `benchmarks/cells/`. Recorded WHOLE, on a CLEAN tree —
`bench.chunk_baseline.write` refuses a dirty one, because a number taken over uncommitted
edits describes a tree nobody can check out. It carries the GPU, the arch, the driver,
torch, the assembler, the ratification digest, the recorded sha and the extension's own
device-side build stamp; it carries no hostname, username or path. The gate side reads it
and never writes it.

**EACH ROW IS THE MEDIAN AND IQR OVER SEVEN SESSION MEDIANS, NOT OVER ONE SESSION'S
BLOCKS — and that is a measured correction, not a preference.** The first chunk baseline
was recorded from one session, and the very next `gate` run, on the same binary a minute
later, put **17 of 44 arms over their limit**. The two spreads are different quantities: a
within-session block IQR is eleven reps taken back to back on one warm fixture, often
exactly `0.0000 ms`, while the gate's comparison crosses two PROCESSES — the fixture is
rebuilt, the caching allocator is cold, the launch order differs. A threshold derived from
the tighter one is derived from a spread the comparison never crosses. So the recorder
runs seven child processes (the retired recorder's own `35 reps x 5 blocks x 7 processes`
shape, reached here from the same evidence) and the row keeps `within_session_iqr_ms`
beside the cross-session one as the record of the contrast. `MIN_SESSIONS = 2` is enforced
on both sides: a single-session baseline can neither be written nor gated against.

**AND A SPREAD OF ZERO IS A STATEMENT ABOUT THE STOPWATCH, NOT THE KERNEL.** The
cross-session fix left two arms over their limit on an unchanged binary, both of them rows
whose recorded IQR was exactly `0.0000 ms`: the device-event timer quantizes, seven
sessions of a small kernel land on the same value every time, and `median + 3*IQR/1.349`
is then a limit exactly at the median that the next tick exceeds. The spread the threshold
uses is therefore `max(recorded IQR, the instrument's measured resolution)`, where the
resolution is the smallest nonzero gap between the DISTINCT values the run itself produced
— measured every run rather than stored, because it is a property of the stopwatch and the
box and a stored copy could disagree with the samples beside it. Both halves are measured;
neither is chosen. The verdict says which one bound it (`floored_at_resolution`).

The lesson generalizes past this file: **a threshold must be derived from the spread of
the quantity the comparison actually crosses, and never from a spread smaller than the
instrument can resolve.** The paired A/B engine crosses two reps in
one process, so its spread is the per-pair ratio's; the latency gate crosses two
processes, so its spread is the session median's.

## The flagging rule: three gates, all must fire

1. **Effect size** — `bench.regression.threshold_ms`: `median + 3·IQR/1.349`, derived
   from the window's own measured spread. `ALARM_SIGMA = 3.0` is the only free
   choice and it is stated as a false-alarm rate (~0.1 % one-sided per cell per run),
   never as a percentage of anything.
2. **Significance** — `bench.stats.paired_verdict`: an exact Wilcoxon signed-rank
   test over the per-round differences, at `ALPHA = 0.01`. **The round floor is
   derived, not chosen**: the smallest two-sided exact p reachable with `n` non-zero
   differences is `2/2ⁿ`, so below `ceil(log2(2/ALPHA)) = 8` rounds no outcome can be
   significant and the test is undefined rather than underpowered. A landing-tier run
   collects that many; anything shorter reports `insufficient_data`.
3. **Debounce** — `bench.stats.classify_flags`: a regression needs a trailing run of
   at least two violations. One thermally unlucky run on a box with 90 s of logged
   power capping is not a kernel change. A run of three that has since healed is
   reported as `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first
says the design cannot decide, the second says it decided.

## Running it

The five subcommands the retired driver carried, kept as the successor's surface
brief:

```bash
rola-probe duration <probe> --ab <arm>:<arm> --tier landing
rola-probe counters <probe> --arm <arm> --section <section>
rola-probe sanitize <probe> --tool memcheck
rola-probe sass --json bodies.json [--diff-against before.json]
rola-probe report [--probe <probe>] [--format md]
```

**No `flock` on those lines, and adding one HANGED the command** — the harness took
the lock itself, and a second acquisition of the same path from the parent process
blocks its child forever. `sass` and `report` took no lock at all because neither
runs a kernel.

It was not a console script, deliberately: the installed wheel ships `rola/`, not
`benchmarks/`, so a `rola-probe` on PATH would be a command that exists and cannot
import the harness it drives. The same holds of anything that replaces it.

The two commands that DO survive today are the ones that never needed the harness:
`python tools/ratify.py` and `python tools/sass_bodies.py` (below).

### Counters

The metric list was a committed section file, `benchmarks/probe/sections/*.section`,
not a flag string — two throwaway scripts holding the same 24-metric comma-joined
string is how the two drifted apart. Two ncu behaviours are handled by the harness
rather than remembered:

* `--section-folder` **replaces** the default search path instead of adding to it, so
  the stock folder is always passed alongside ours;
* counter capture runs at `--launch-count 1` with the timing loop disabled, because a
  warm-up loop pollutes counters the way it does not pollute a median.

The **replay mode is recorded in every row**. `kernel` replay restores only the memory
the kernel wrote, which is silently wrong for a kernel that carries state across
launches — the paged and stateful continuation arms — where `application` replay is
mandatory.

### SASS

`rola-probe sass` was a front door to `tools/ratify.py` and `tools/sass_bodies.py`
and re-implemented neither the disassembly nor the compile flags; **both tools are
in the tree and are the front door now.** Both binaries in a
comparison no longer need to have been built at the same path (P19,
`docs/build.md`'s path-trap section) — they must agree on `ratify.SASS_HASH_VERSION`,
which `--compare` checks and refuses on mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `python tools/ratify.py --census` there, read the spill
column — and never a flag on the shipped tool. `--force-ctas` was that flag; it
was deleted once nothing in `csrc/` read its macro, because a tool that
can force a bound is a tool that can ratify one.

## What is deliberately not here

* **Clock locking.** `nvidia-smi -lgc` has no effect from inside WSL2; the plateau
  protocol is the correct design on this box, not a workaround for it.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in
  a landing. Any query artifact is derived and ignored — see the record below, which
  is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in
  the surveyed field; replacing it would reintroduce the invented number the
  derivation exists to avoid.

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
| `bench_driver` | `benchmarks/bench/driver.py` (every unit driver) | the arm specs, tier, counts, instrument, commit and diff, manifest, device | the results with their samples |
| `suite/<module>` | rola-bench's measurement suite | the module, unit, identities and dependencies | the instrument's raw JSON |
| `calibration` | `benchmarks/unit/bench_carry_calib.py` | the parts binary, device, owners, sizes, clock | the calibration rows |
| `pipe_timeline`, `pipe_timeline.scale` | `tools/pipe_timeline.py` | the cell, binary and scale; a calibration's composition | the series and summary; the plateau |
| `build_ledger`, `compose_ledger` | `tools/build_ledger.py`, `tools/compose_ledger.py` | the commit and diff, the tag or cells | the report |
| `dram_by_activity` | `tools/experiments/dram_by_activity.py` | each binary's commit, diff and manifest; the variants, counts | the rows |
| `environment` | `tools/dev.py container check` | the environment key | the proofs |

A tool run by another tool stores nothing (`--no-record`): the caller keeps the output in its own record. `python -m
rola_results sql "..."` queries every location through a derived SQLite index beside the records (`history` and
`latest` for one location; the views `timing_rows`, `timing_pairs`, `driver_rows`, `cells` flatten the tools' outputs,
and the rola-results README gives the last, history, compare, baseline and regression questions as SQL); `python
tools/dev.py store commit -m <message>` commits the records
(`python -m rola_results commit`), after the repository's own check that every key recomputes from its semantics and
every output exists. Pushing is the owner's act. The record kept before the library -- the per-run JSONL, the perf
ledger, the ledger and composer reports -- left the store for the owner's archive (`docs/internals/DELETIONS.md`).
