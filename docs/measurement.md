# Measurement — the method, the instruments and the record

Every performance number this repository publishes is taken by one method, with one stopwatch per comparison, and
stored with its provenance. This document is what that sentence means.

| layer | where | answers |
|---|---|---|
| cells | rola-devtools' central registry `rola_devtools.cells` (rola's reading: `measure/cells/`) | WHAT is measured: an input, named once for every package, with its draw, seed and proven regime |
| subjects | `measure/subjects.py`, `measure/provider.py` (rola's runner) | WHICH launch is timed on a cell, at which LEVEL, each arm's dials, its untimed reset, and which cells this binary cannot run |
| declarations | `declare.py` (this checkout's targets), rola-bench's `declare.py` (checkouts composed) | WHAT RUNS: the build, the instruments, the timing registrations and sessions, the stores, as targets of `rola_devtools.build` |
| method | rola-devtools' `rola_devtools.timing` (`measure_timing`) | HOW: interleaved call by call, one call at a time, in a fresh random order each rep |
| preconditions | the targets' requirements (`rola_devtools.build.resources`), `measure/provider.py` | what the box and the binary must be first |
| statistics | rola-devtools' `rola_devtools.verdict` | whether a difference is real |
| record | `rola_results` (the measurements store) | where a number lives afterwards, and which stored samples are a baseline |

rola measures itself with `bazel run //:measure -- run declare.py:all`; rola-bench's root loads this file from each
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
| GPU lock | a session holds `gpu: all`: the GPU lock exclusive through `gpu_lock()`, taken by the build system for the whole session and handed to the entry workers as the lock-held marker, so a self-locking tool a target starts does not wait on it. The lock's rules are [`testing.md#running-it`](testing.md#running-it)'s. |
| clock lock | a session holds `clock: 1`: the host's own lock (`rola_devtools.locks.clock`, the dev config's `clock.json`), proven by this binary's clock read (`register_clock_reader`) before the first call and after the last; a session whose read is off the lock fails. A host without one runs unlocked, and the session records its reads. |
| binary identity | building an arm asks the device for the subject's family stamp and refuses the arm without it: a path or a hash passes against a stale extension, a device-side fact does not. |
| extension identity | `import rola` must resolve inside the entry's own checkout, or the arm is refused: a venv that names another checkout (an editable install in a shared venv, or another checkout's own venv) otherwise answers with that tree's kernel while the record carries this tree's commit. |
| tree identity | every stored sample carries each checkout's commit and the sha256 of its tracked diff; a dirty tree is stamped, not refused. |

## The three levels, and what a comparison may cross

Blake, 2026-09-15: the LAYER holds the projections and produces the routing (RoLA's analogue of QKV); the OP takes those
bare operands and a state and produces the next state and the readout -- the facts, the paging, the packing and the
kernels; a KERNEL runs its own launches and nothing around them. Every subject states the level it prices
(`Subject.level`) and every sample records it, so a reading pairs entries of ONE level and
`bazel run //:results -- verdict` refuses a pair of two.
A comparison between this library and another is therefore a LAYER comparison: RoLA's
own layer against attention's, because RoLA's number includes the state its capacity buys.

What this line can measure today: the KERNEL level (`carry_forward`, `intra_forward`, `facts_pass`, `entmax_solve`) and one OP-level path
(`decode_step`, through the engine's decode DAG, which runs the facts pass and commits the pages it writes). There is no
op-level prefill subject and no layer subject: `rola.interface.rola_op` walks the prefill DAG, whose carry leg is not
yet
wired to the rebuilt carry ([`open-work.md`](open-work.md)). The flash reference is a layer entry, and until a
RoLA layer runs beside it, a session that holds both is two levels measured in one place, not a comparison.

## The reference

Every reported carry number sits beside FlashAttention at the same point, measured in the
same interleaved, clock-locked session: rola-bench's attention entry
(`rola_bench/measure/attention.py`) on the group's QKV cell, causal, width `dv`, bf16, through torch's flash
backend: Dao's FlashAttention-2 compiled into torch, or FA3 (Hopper) and FA4 (Blackwell) once
`torch.nn.attention.activate_flash_attention_impl` registers them. The backend is forced, so torch refuses a call
flash cannot take rather than timing its math or memory-efficient backend, and the entry records torch's version and
the implementation.

<a id="incremental"></a>
THE INCREMENTAL CELL is the headline (Blake, 2026-10-18): the time to process `c` NEW tokens given `T` tokens already
absorbed, for a model of state count `N`. FlashAttention's `c` queries attend over a KV cache of `T + c` keys,
bottom-right causal (the entry's `context` parameter); RoLA's layer runs the `c` tokens' producer, facts pass, carry
from its `N`-leaf state and intra. FlashAttention's cost grows with `T`; RoLA's does not once its state holds the
history, so the comparison is a grid -- `T` in {0, 4K, 16K, 64K, 256K, 1M} by `c` in {1, 512, 4K, 16K}, the model's
hidden width and heads stated -- never one number. Each side's per-call overhead is in its time: FlashAttention's KV
read and launch; RoLA's state load and store (its HBM round trip, ~12.6 MB a head at `N` 64K), launch, partial last
window and wave quantization.

AT LAYER LEVEL, WITH BOTH SIDES' PROJECTIONS: FlashAttention's QKV projection of the `c` tokens (the entry's `hidden`:
one cuBLAS GEMM for every head, the library's own form) against RoLA's producer (its routing, gain and values: 1,088
columns a head against QKV's 192), so a number is what a user pays. The output projection is the same on both sides and
is in neither.

`N = L` -- `T = 0`, `c = L`, the whole sequence from zero tokens -- is the grid's whole-sequence case and the
capacity-fair point a single-length cell states (`N = L`); it averages FlashAttention over contexts from 0 to `L` and
holds one RoLA call's overheads, so it is never quoted alone. Until RoLA's carry starts from a stored state (the
segment boundary), RoLA's side of the grid is projected from the labs, the producer's floor and the state's bytes, each
number labelled projected.

<a id="quoting"></a>
## Quoting a comparison

- THE BASELINE is attention over its KV cache (and wide monolith layers), whose state grows with the context -- not
  single-state linear attention: the method's case is a bounded state against a growing cache, and at `N`
  proportional to `L` the state stays under the cache while its key width is small, so that regime's problem is
  compute, which the activation's sparsity decouples (the user's framing, 2026-07-01).
- CAPACITY-FAIR POINTS (Blake, 2026-08-01): a gap is never quoted from a cell where RoLA holds `N` states while
  FlashAttention attends over `L < N` tokens -- attention is quadratic in `L` and RoLA flat in it at fixed `N`, so the
  cell charges RoLA capacity rent at short-context prices. Headline figures put measured points at capacity-fair
  `(N, L)`, FlashAttention run at `L = N`, and share one x-axis, addressing capacity; an `L < N` cell extrapolates
  per-token latency only.
- EVERY RATIO CARRIES ITS LABELS (Blake, 2026-10-24): the card, the level (op or layer), the depth and the sparsity.
  The headline is end-to-end layer time with the op-level number beside it, at a head at scale, depths 1-4 shown; a
  GeForce op-level ratio is never the headline.
- THE OPERATING POINT `N ~ sqrt(L)` is quoted beside `N = L`, never instead (Blake, 2026-10-31: "sqrt(L) is context,
  not the bar"); `N = L` stays a target the kernel must do well at.
- AT SCALE (Blake, 2026-07-30: "we should be testing the large regime too"): every performance-bearing change is timed
  on at least one at-scale cell (`N >= 65536`, the largest `L` that fits) at both ends of density
  (`KERNEL_STANDARDS.md#s21`). The small cell is the SMOKE cell, never "flagship"; a conclusion drawn on toy cells
  alone is labelled cell-bound. A small state is a depth-1 cell (Blake, 2026-09-14: "If you want small, just use depth
  1"), never a sub-floor depth-2 rung.
- RECORDS FOLLOW THE DECLARATION: when a refactor changes what a record means, the change is re-baselined and the old
  records deleted rather than kept readable through compatibility code (Blake, 2026-09-15: "its ok to just delete old
  records if its too much work to maintain compatability"); what was deleted is said, and anything large is archived
  outside the store first.

## The stopwatch

There is one, it is named, and its name travels with every number.

| instrument | what it measures |
|---|---|
| **`cuda_events`** (committed) | device-side elapsed time between two events straddling the launch |
| `perf_counter` | host wall time around a full device sync on both sides — the launch and sync cost included |
| `ncu` | a counter, or a duration taken under profiling conditions: replayed, caches flushed, base clocks |

**A comparison across two of them is refused, not warned about.** The gap is not
noise; it is the launch and sync cost one instrument includes: roughly constant on this
box (~0.22 ms, RTX 3080 Ti), so nearly the whole measurement on a sub-millisecond cell
and a rounding error at scale. An `ncu` duration reads above the event time (25 % on a
small cell), which is why it is a third instrument and not a cross-check on the first
two.

Every timed entry names its stopwatch (`Timed.instrument`; rola's arms time with `cuda_events`), and a session refuses
entries that name two.

<a id="comparing-two-builds"></a>
## Comparing two builds

A change is judged against its baseline in ONE session: the baseline checkout (a worktree at the old commit, built,
with its venv) is a reference of rola-bench's suite, and the run is pruned to the timing sessions and their stores:

```bash
python -m rola_devtools.build run declare.py:suite \
    --arg target=worktree:<new checkout>,venv:<its venv> --arg references='worktree:<baseline>,venv:<its venv>' \
    --only 'store/session/*'
```

Each session interleaves both checkouts' entries call by call at the locked clock (above), so the ratio between them
is read from one session's samples (`bazel run //:results -- verdict`). Times from two separate runs are never
compared, and an unlocked clock's are never read at all (KERNEL_STANDARDS §32.7): this host's sustained clock has two
states ~17% apart (the method, above), larger than the effects a change is judged on. One launch's cycles each under `ncu`
(`sm__cycles_elapsed.max`) compare two builds without a session where the question is one kernel's cycles.

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

**THE SPREAD IS THE SPREAD OF THE QUANTITY JUDGED.** Reps taken back to back on one warm fixture spread less than a
comparison that crosses processes (a limit taken from one session's reps flagged most arms of an unchanged binary a
minute later), and a reference's medians across sessions carry the host's drift (two clock states ~17 % apart). What
is judged is the paired ratio inside one session, so the limit is that ratio's own round-to-round spread, drift both
members share cancels in it, and the one thing the ratio crosses that its spread
cannot see -- two worker processes -- is the null gate's.

**AND NEVER A SPREAD SMALLER THAN THE STOPWATCH RESOLVES.** The device-event timer quantizes, so a small kernel's calls
can land on one value, a recorded spread of `0.0000 ms`, and a limit at the median the next tick exceeds. The spread
used is `max(IQR, resolution)`, where the
resolution is the smallest gap between the distinct values one member's samples hold, relative to the reference's
median -- measured on every judgement, because it is a property of the stopwatch and the box -- and the verdict says
when it bound (`floored_at_resolution`).

## Which samples are judged

    bazel run //:results -- verdict --reference LABEL [--candidate LABEL] [--cell C] [--arm A] [--json]

reads the stored timing sessions (the `timing_members` and `timing_samples` views). In every session that timed the
reference label's arm on a cell, each other label's member of that arm on that cell is paired with it and judged inside
the session. A unit is the arm, the cell, the two labels and the code each ran (commit and tracked diff); its sessions,
oldest first, are the runs, and the newest is judged. Nothing is stored: a verdict is a reading of the records, taken
again whenever it is asked for. `bazel run //:results -- dashboard --out FILE --reference LABEL` is one run as a page.

## SASS

`bazel run //:ratify` and `bazel run //tools:sass_bodies` are the instruments. Two binaries in a
comparison may be built at different paths ([`build.md#kernel-identity`](build.md#kernel-identity)); they must agree
on `ratify.SASS_HASH_VERSION`, which `--compare` checks and refuses on mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `bazel run //:ratify -- --census` there, read the spill
column — and never a flag on the shipped tool, because a tool that can force a
bound is a tool that can ratify one.

## What is deliberately not here

* **A second method.** The declared build system is the one way a subject is timed: two methods are how a harness
  comes to disagree with itself.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in a landing. Any query artifact is
  derived and ignored — see the record below, which is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in the surveyed field; replacing it
  would reintroduce the invented number the derivation exists to avoid.

## The build: everything declared, a run pruned by label

A checkout's `declare.py` declares EVERY target it has -- its build (the build graph's `bazel run //:library`, run
every build: Bazel's cache is its key, and the library's digest, its output, keys every target that reads the binary),
its machine
facts, each instrument over the cells sized for measurement (`tier` probe or both), its timing arms, its diff sides
and its own kernel-vs-oracle diff, the oracle tier as one node -- and takes no selector. Which of it a build runs is pruned at the CLI by LABEL, which knows
nothing of cells, groups or instruments: `--only 'rola/phases'` runs that instrument and what it needs, `--skip
'*/timeline'` drops that instrument and everything that reads it. A node whose code, binary and inputs have not changed
is a cache hit and does not run. A full run of rola-bench's suite over one checkout is seven minutes (measured
2026-09-16, 212 targets); the dev loop is a pruned run, seconds.

THE ORACLE TIER IS A TARGET (`rola/oracle-tier`): the oracle battery run whole, keyed on its code and the binary, its
output the counts, WHICH cases failed and the per-comparison margins (`--oracle-margins`, the worst per output kind),
stored beside the instruments. A red cell is a stored fact.

THE DIFFS. A SIDE is an executor run over cell nodes in the environment it names, its tensors kept in the build cache;
a DIFF is two sides under one named rule -- `bit-identical`, the per-slot clause rule mirroring the oracle's, or
`support-equal` -- with a claim (`expect` same or different) and a consequence (a build failure or a recorded verdict).
A side proves which checkout's library it imported; a comparison that compared fewer quantities than it declared, or a
cell only one side could produce, refuses rather than passes; a cell both sides refuse is listed and not counted. Inside
Across checkouts rola-bench pairs each surface a checkout exposes (`SURFACES` in `declare.py`: the fp64 oracle and the
producer reference) under ITS rule for the surface (`RULES`) -- the oracle bit-identical, the producer support-equal --
stored at `diff/<surface>`. Verdicts are stored, tensors never. The carry kernel has no diff surface and no
kernel-vs-oracle diff: its oracle obligation is owed (`tools/coverage.json`).

## <a id="instruments"></a>The instruments

Each is a target of `declare.py` (`INSTRUMENTS`), run per cell where it reads a launch, its output stored beside the
timing session:

| target | tool | what it reads |
|---|---|---|
| `sass` | `tools/sass_gate.py` | the build's SASS signatures (KERNEL_STANDARDS §20) |
| `registers` | [`tools/life_ranges.py`](internals/tools/life_ranges.md) | peak live registers per source region |
| `phases` | [`tools/phase_ledger.py`](internals/tools/phase_ledger.md) | the kernel's phase clock on a cell |
| `counters` | [`tools/pipe_counters.py`](internals/tools/pipe_counters.md) | the profiler's pipe and resource counters |
| `census` | [`tools/stall_census.py`](internals/tools/stall_census.md) | every warp-stall sample, by source line |
| `timeline` | [`tools/pipe_timeline.py`](internals/tools/pipe_timeline.md) | the pipes' activity over a launch |
| `counters-intra`, `census-intra` | the same tools' `--kernel` ([`tools/kernel_targets.py`](internals/tools/kernel_targets.md)) | the intra's launches, on the cells of 16K tokens and more |
| `roofline` | `tools/roofline.py` | the intra's time beside the device's u8 IMMA ceiling |
| `oracle-tier` | the oracle battery | the counts, the failed cases, the margins |

A question about the machine itself, rather than a shipped kernel, is a lab experiment ([`lab/`](lab/README.md)).

## The measurements record: `rola_results`

Every measuring tool stores what it measured through one library, `rola_results`, in the rola-results repository:
`store.root` in the dev config (default `rola-results` beside the checkouts), importable from every interpreter
`bazel run //:init` provisions. A result measured in any worktree of a machine lands in that one repository, visible
to every other at once; the devbox container has its own clone (`docs/setup.md#the-devbox-image`), whose records
reach another machine's by git (`bazel run //:results -- commit`, then a push and a pull).

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
| `calibration` | `measure/harness/bench_carry_calib.py` | the parts binary, device, owners, sizes, clock | the calibration rows |
| `pipe_timeline.scale` | `tools/pipe_timeline.py --calibrate` | the cell, the binary and the calibration's composition | the plateau the tool reads back as later runs' scale |
| `compose_ledger` | `tools/compose_ledger.py` | the commit and diff, the cells | the report |
| `environment` | `bazel run //:container_check` | the devbox image's digest | the steps, each timed, and the cold start |
| `rola/oracle-tier` | a store target over the tier node | the tier's code, the binary, the environment | the counts, the failed cases, the margins |
| `diff/<surface>` | a store target over a diff node | both sides' semantics and the rule | the verdict per cell and quantity, never a tensor |

Reading across runs is a join, never a parser: `instrument_metrics` is every number every instrument recorded in
long form (run, cell, dotted metric, value), `diff_cells` every stored diff's verdict per cell and quantity, and
`bazel run //:results -- instruments --baseline RUN` is one run's numbers beside another's as deltas; the dashboard
(`--baseline RUN`) renders the same. Which run is the baseline, like which label is the reference, is the reader's.

A tool run by another tool stores nothing (`--no-record`): the caller keeps the output in its own record.
`bazel run //:results -- sql "..."` queries every location through a derived SQLite index beside the records
(`history` and `latest` for one location; views flatten the stored outputs, and the rola-results README lists them
with the last, history, compare and baseline questions as SQL); `bazel run //:results -- verdict` is the regression
judgement above; `bazel run //:results -- commit -m <message>` commits the records, after the repository's own check
that every key recomputes from its semantics and every output exists. Pushing is the owner's act.
The record kept before the library -- the per-run JSONL, the perf
ledger, the ledger and composer reports -- left the store for the owner's archive (`docs/internals/DELETIONS.md`).
