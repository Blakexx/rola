# Testing — the three tiers

The suite answers **different questions**, and conflating them is how a real bug hides. It is three directories, and
**each tier ASSUMES the tier below it** — that is what keeps the suites from re-proving everything everywhere.

| Directory | Question | Compares |
|---|---|---|
| **`tests/oracle`** | Does the kernel compute the right function, over the whole DISTRIBUTION SPACE the model can put through it? | the kernels against the **fp64 oracle**, per slot, at DECLARED regime cells |
| **`tests/integration`** | Does the COMPOSITION work, GIVEN that the kernels are trustworthy? | the layer against the op, a dispatch fork against what it dispatched to, and — the two-branches-agree class — a kernel against **itself** across geometries and both state backings |
| **`tests/unit`** | Does each file honour its own public contract? | one file's surface, under assumptions about how the others behave |

**There is no fourth tier**, and a campaign name never appears in a test path: `tools/lint/lint_standards.py` refuses
any directory under `tests/` other than `unit/`, `integration/` or `oracle/`. **A check that does not execute the code
under test is a lint, not a test** — it reads files, manifests or versions and asserts consistency, so it lives in
`tools/lint` (ruled 2026-08-29).

The oracle tier alone is insufficient because a per-geometry tolerance can absorb a geometry bug. The two-branches-agree
class alone is insufficient because two branches can agree with each other and both be wrong — this project has a bug
that passed exactly that way. So that class **never stands alone**: every branch it relates is independently anchored to
the oracle. And the regime cells exist because routing is a MOVING distribution (the gain knob alone sweeps candidate
fraction ~0.96 → ~0.09 in training), so fixed fixtures answer their question at a handful of unmarked points; declared
cells make the covered regimes named and the uncovered ones visible.

An integration test never re-grades arithmetic. It checks wiring, dispatch and state flow, taking the op's correctness
as established by the oracle tier. Wherever the layer selects a kernel by shape, length or cache, the LAYER is gated
forward and backward across every threshold (lengths spanning each switch), asserting finite, non-`None` gradients
on the input and every parameter for every form -- a forward-only branch under a grad-requiring call is a silent graph
break, not an error. A gate over a defaulted mode tests only the default, so a crown gate is parametrized over every
declaration the tree ships a body for, each cell asserts the kernel body it exercised (never default dispatch), and new
coverage is proved by a negative control (a known-broken variant must fail).

## The oracle tier — semantics

**Ground truth is `rola.naive_rola`, fp64, and nothing else.** A freshly written second reference proves
self-consistency only. This prohibition is not a preference; it is the direct lesson of a shared-router bug that agreed
with a sibling implementation and disagreed with the definition.

### What the oracle is, and what it promises

The oracle FACTORS, and each factor is trusted for a different reason. The recurrence is textbook gated linear
attention -- one state per leaf, a diagonal gate, a rank-one deposit, a linear readout -- written as the plain-torch
loop
a reader can check against any linear-attention paper, with normalization carried as a ones column of the value rather
than as a second recurrence. The feature map's hard part, entmax, is the DeepSPIN authors' own package, pinned. What is
OURS is the glue -- the leaf address map, the gate values, the ratio readout -- and it is deliberately small, because it
is the one shared blind spot the factoring does not remove. The oracle takes the bundle itself as its input, so a kernel
and its reference read the same quantized operands.

Three promises a caller can rely on:

* **Both outputs are `float64`, whatever the inputs were.** Casting back is the caller's business; a reference that
  rounded itself to an operand's precision would make an accuracy gate compare against a rounded reference.
* **An input the output does not depend on gets `None`, not a manufactured zero** -- autograd's ordinary answer, and the
  oracle states nothing else. A test that wants a zero asserts `grad is None` and supplies the zero itself.
* **The address map is the oracle's own**, derived from the widths rather than imported from `Topology`, so a wrong
  stride convention cannot agree with itself in every gate the repository has.

**Grouped queries are the ungrouped layer with each write head's side repeated** for its query heads (`docs/api.md`
[grouped queries](api.md#grouped-queries)): THE IDENTITY GATE, `tests/oracle/test_oracle_grouped.py`, holds the oracle
to it per slot and fails on the other head assignment; a grouped kernel's gate is the oracle at its own heads.

**An op's gate proves the math; the layer's measures quality** (Blake, 2026-09-26: "the layer oracle should measure
the end-to-end numerical quality, but each op oracle should just measure that we're actually implementing the math
correctly"). An op's gate compares the kernel with the fp64 op on the SAME quantized inputs, per slot, at its
arithmetic's own tolerance, and vetoes; its error against unquantized inputs is reported per slot and per token, never
gated, since for declared low-precision arithmetic a relative check against the unquantized op fails small channels
by construction. Quality thresholds and the precision knobs are the layer oracle's, per layer.

**Entmax gradients are graded only against the analytic VJP.** The feature map's gradient is non-smooth: a finite
difference is a chord across a kink, and autograd through the bisection flips branches near convergence, so neither
is an oracle. The reference is the hand-written implicit-function VJP composed through the canonical oracle, evaluated
at kink-free interior points (the support stable under the perturbation); a gap claimed to be fp32 noise is shown by
running the production gradient in fp64 against the fp64 oracle, where it collapses to ~1e-6.

**A derivation feeding kernel arithmetic is checked symbolically before its gates are trusted**: exact rationals on
small adversarial instances, chunk boundaries included (2026-08-15: a dropped term that manifested only across a chunk
boundary passed every seed-level gate and was caught by the exact pass).

**The oracle has no outside anchor** (Blake, 2026-09-15): no default-off cross-check against another library's loop is
kept, because a default-off gate reports a skip where it should report a failure. The recurrence's claim to BE the gated
linear-attention loop is held by review of the lines it is written in and by the mutants its own tier plants.

### Changing the oracle: the dual-run protocol

`rola/ops/naive.py` and the routing references are the oracle pair, so a change to them is not reviewed by reading the
diff -- the question is never "is this code reasonable" but "does it compute the same function". Both versions are RUN:

```bash
# in rola-bench: this checkout against another, every surface, as targets
python -m rola_devtools.build run declare.py:diffs --arg target=worktree:PATH --arg references=worktree:OLD
```

Both sides are DIFF SIDES (`tests/oracle/sides.py`) run by the build system in each checkout's own environment over the
same central cells: this checkout exposes its surfaces (`declare.py`'s `SURFACES`), and a diff target
(`rola_devtools.diff`) compares each under the rule rola-bench owns for it: the oracle under BIT-IDENTITY, forward
and backward, with "no gradient" an empty
tensor and never a zero one; the producer reference under support equality. A side proves which tree it imported, a
comparison that compared fewer quantities than declared refuses, and the raw tensors never leave the build cache.

**The backward's retained state is a declared BUDGET.** The oracle side runs `naive_rola` forward and backward in
fp64, and autograd retains the recurrence's state at every token (measured: a peak above nine times the retained-state
byte count). `RETAINED_STATE_BUDGET` (512 MiB) is checked per cell (`retained_state_bytes`), and only
`BACKWARD_SURFACES` runs the backward at all; a cell over budget is excluded from that surface, a consequence a reader
can compute from the declaration rather than a run that died.

**A venv imports the `rola` of the checkout it names, so a tool run from one WORKTREE with another checkout's venv imports that checkout unless it is stopped.**
The protocol binds and then CHECKS the library each side imported, and prints both. Test runs are unaffected -- pytest
puts the rootdir on the path -- but any new tool that imports `rola` and compares two trees inherits the trap.

### The cells

The cells are the central registry's (`rola_devtools.cells`, read through `measure/cells`): each states the regime it
proves — density, temporal coherence, read-write correlation, mass concentration, support cardinality, depth and
widths, length — and its draw proves that regime is present before any kernel comparison, so a fixture that cannot
demonstrate its claimed regime fails rather than passes vacuously. The oracle tier runs the carry cells at
`tier = oracle` (`tests/unit/test_central_cells.py`). **The length axis is not decoration**: an error that ACCUMULATES
along the sequence grows as `sqrt(T)`, so a short fixture under-charges it, and the multi-window cells carry it.

<a id="whole-call-gates"></a>
THE WHOLE-CALL GATES run on those cells and stay red, cell by cell, until the system under them is ported (ruling
2026-09-29: a red test is the plan's record of what is unported; base infrastructure is re-pointed at a new seam, never
deleted with the implementation it exercised): the carry's inter numerator against the fp64 inter term in the
recurrence's units (`tests/oracle/test_carry_vs_oracle.py`, its drawn-cell supplement
`test_carry_hypothesis_cells.py`), the op's readout and handed-back state against `naive_rola`
(`tests/oracle/test_prefill_vs_oracle.py`), and the state's backings bit-identical through a fresh call and a
continuation (`tests/integration/test_chunk_paging_equivalence.py`, red by plan). Its readouts' bit-identity is an
assertion a carry cannot meet -- its num differs run to run (its fp32 sums across CTAs reassociate) -- so the test's
port re-points it to that sum's bound; the state's bit-identity stands. Every one of them is red on this line by the
ops' named error (`rola.ops._ext.KernelNotBuilt`): no carry, intra or producer kernel is built. `declare.py`'s `carry-vs-oracle` diff stores the
carry's verdict per cell beside the instruments.

Decode is its own family (`tests/oracle/test_decode_family.py`): the prefill-to-decode hand-off, a 32-step horizon with
checkpoints ALONG it (a per-step check cannot see cross-call compounding), and frozen density declarations proven
performance-only under drifting realized routing.

### Tolerance discipline

1. **Every kernel-vs-oracle comparison is PER SLOT** (`tests.oracle.fixtures.assert_slots_close`) and names the OUTPUT
   KIND it grades (`tests/oracle/tolerances.py`). A kind holds each slot to its CLAUSES -- `(rtol, atol)` pairs, one of
   which fails a slot whose relative and absolute errors both exceed it, measured with their frontier and never tighter
   than the kind's own budget -- and, where the output is multilinear, to `rtol` times that slot's envelope, the sum of
   the sizes of the terms it adds up, which the same fp64 reference computes on `|v|` (`fixtures.oracle_run`). The
   tightest binds and the failure says which. Never a global max: the readout is a ratio whose denominator is
   accumulated write mass, so `|y|` at the cold start exceeds the rest of the sequence by orders of magnitude, and a
   single normalizer set by that token was measured BIT-IDENTICAL between two arms it was supposed to separate.
2. **A rule that cannot fail is the same defect as a tolerance that cannot fail**, so each gate plants errors on the
   kernel's own output that must fail (`fixtures.assert_planted_errors_fail`), and `tests/unit/test_oracle_rule.py`
   holds the rule itself. `tests/oracle/tolerances.py` records the measured margins.
3. **A kernel whose arithmetic is declared is graded against a reference that mirrors it** (the state's
   `state_rounded`), so the tolerance
   charges only accumulation order, never a rounding the reference skipped. Where a third party's arithmetic is opaque
   (a cuBLAS reduction order), the gate sits at the seam: the production intermediate is fed to the reference. A fused
   kernel is gated through a shipped probe composed of its own device components, so the agreement is a fact of the
   call graph and the gate runs the shipped binary.
4. **Every tolerance is either DECLARED with its derivation in a module comment or DERIVED in code.** A number with
   neither is a defect.
5. **Loosening a tolerance is a semantic change**: changelog entry, derivation, reviewer sign-off. Tightening needs none
   of that.

## The two-branches-agree class

The same question — do two geometries of one computation agree — asked one file per pair:

| file | the pair it relates |
|---|---|
| `test_decode_bc_blindness.py` | one decode step across prefill boxes that differ in `BC` |
| `test_decode_paging.py`, `test_page_arena_vmm.py`, `test_atom_bitmap.py` | the dense and paged backings, and the backing's own structures |
| `test_stream_discipline_fixture.py` | the crossing paths (below) |

The census of instantiation axes lives on the binary's side: each family's `arms()` answers out of the running
extension, and `tools/lint/lint_standards.py`'s `check_shard_partition` holds the shard partition.

### Class (a) — ENGINEERED IDENTITY

The two geometries are *designed* to produce byte-identical output. Tested with `torch.equal` — **no tolerance, no
`allclose`, ever.** The claim is always structural and the structure is stated. **Every such pair declares a VALIDITY
PRECONDITION and the test asserts it**: a bit-identity assertion whose precondition is unstated is a future flake.

### Class (b) — REASSOCIATION-EQUIVALENT

The two geometries legitimately reorder floating-point accumulation. Tested by **two assertions, both required**:

1. **Oracle anchor** — *each* geometry is independently within its tolerance of the *same* oracle point. This is what
   stops two wrong geometries from agreeing.
2. **Derived mutual bound** — the two are within a bound *derived* from the reassociation, never chosen.

Two prohibitions, both of which have cost this project time:

- **Never bitwise-vs-each-other.** An atomic fan-in's order is a random variable; one draw agreeing is not a property.
- **Never a loosened tolerance in place of a derived bound.** If a pair misses its bound, the answer is a bug hunt or a
  corrected derivation — never a bigger number.

### The classes are not a spectrum

A pair is (a) or (b), and where one comparison carries both, those are separate assertions with separate
preconditions. The decode step is the live example: its STATE is class (a) — the write positions partition the leaves,
so a written leaf's bytes are a function of its old bytes and the token alone — while its `y` is class (b), each head's
partials added in arrival order ([`internals/decode/decode.md#determinism`](internals/decode/decode.md#determinism)).

## The build-shape gates

Each of these guards a property of the BUILD whose failure leaves every other gate green.

**`bazel run //tools:gen_shards -- --check`** regenerates the per-arm carry units into memory and diffs them against the
checked-in ones, and holds the selection header — generated, never checked in — to its generator's schema for the
scopes a build or ratify run uses and for the empty selection
([`internals/tools/gen_shards.md`](internals/tools/gen_shards.md)).
The library's record action runs it on every build, and the commit gate's standards lint runs it (`check_shard_partition`, below).

**`check_shard_partition`** in `tools/lint/lint_standards.py`: the ratification on disk is the one `_build_config.py`
recorded at build time, its digest recomputed over the arch subset the build declares. CPU only; a tree that has not
been built has nothing to compare and is not failed.

**`check_arch_coverage`** is the same claim across ARCHITECTURES, the failure no single-card battery can see:
tabulated architectures with no ratified manifest are REPORTED, not failed — the *not ratified — will refuse to load*
row `tools/supported.py` prints — and the check asserts the generated table says so. Manifests only, no device.

**`tests/unit/test_closed_world_arch_gate.py`** holds the RUNTIME refusal: closed-world rule R4 forbids a launch on a
device whose arch was never measured, as a pair of clauses — `static_assert(tabulated(kArch))` in
`csrc/rola/src/common/arch_caps.cuh` and `check_arch_table()` (`csrc/rola/src/common/arch_runtime.cu`) as the first
statement of every launching entry. The test states the clause over every declared entry surface, so the next entry
inherits it.

**`bazel run //tools:header_selfcheck`** compiles every header under `csrc/rola/` ALONE, twice in one translation unit
under the extension's own flags, so a header that leans on another's include, or whose include guard is broken, fails
here rather than at a later caller. It needs `nvcc` but no device.

## The documentation gates

A document nothing checks is a document that drifts. These run in the commit gate, read files and never touch a device
([`internals/tools/lint.md`](internals/tools/lint.md)):

- **`tools/docrefs.py`**: every citation a doc or a skill makes of the tree is pinned to its target's content; a new
  citation is read and accepted, and a changed target waits in `pending` for the docs pass; a skill is held strictly
  ([`internals/tools/docrefs.md`](internals/tools/docrefs.md)).
- **`tools/lint/lint_standards.py`'s mirror rules**: every source file under `csrc/rola/` has a mirror doc at the same
  relative path (minus `src/`); every internals anchor a source cites resolves; a hazard stub's slug is its
  anchor; no doc defines an anchor twice; a page names only identifiers the tree carries (or declares them `ABSENT`);
  and a directory of GENERATED sources — `csrc/rola/src/instantiations/` — is exempt from the one-page rule and
  documented by one README, the exemption itself checked (the README exists, and the directory holds nothing its
  generator does not produce). A floor on the number of sources found keeps a broken glob from checking nothing.
- **`check_supported_table`** (the standards lint): the README's support table is `tools/supported.py`'s output, read
  from files alone. The README's other generated block, the activation registry's capability table, is produced by
  running the registry, so its check executes the code under test and is a test, not a lint: it runs in the CPU tier
  (`tests/unit/test_capability_table_docs.py`), not in the commit gate.
- **`tools/lint/portable_claude.py`**: the agent files name no machine path.

## The host-structure gates

The engine's DAG framework is host-only, so its battery is too.

**`tests/unit/test_facts_memo.py`** runs with no CUDA and no extension. It holds the caches the engine may declare —
`pure_of`'s plain dict and `per_state`'s `WeakKeyDictionary`, including the weakness itself (a dropped sequence drops
its buffers) and the TRANSIENTS-ONLY discipline (a `per_state` node's body runs against a state that raises on any
attribute read) — and the structural refusals that fire at DAG declaration: a need that arrives later than its reader,
a second Join, a second device-to-host read, a memo key that is not one of the node's inputs. It holds the decode
family's structural declarations too: every node of `DECODE_STEP` declares `capturable` and none declares `host_sync`,
and no shipped DAG contains a `reference_only` node, which keeps the growth gate's independent second derivation from
being wired in. The CAPTURE ITSELF is `tests/integration/test_decode_graph_step.py`'s.

**`tests/unit/test_facts_rules.py`** holds the RULE layer: every band-2 fold fed plain values and asserted on its
decision, plus the marker sweep that makes `rola/engine/rules/` an inventory (an unmarked function there fails). The
run table is derived over all eight liveness bytes against every row of
[`internals/engine/rules.md#run-table`](internals/engine/rules.md#run-table), whose row lines are matched against the
rule. It must pass under `CUDA_VISIBLE_DEVICES=""`, and it asserts that the battery never resolved the compiled
extension.

## Running it

Each tier is one target of the build graph (`tests/BUILD.bazel`, [`build-graph.md`](build-graph.md#checks)), and each
test is in exactly one:

```bash
bazel test //tests:cpu              # tests/unit -m "not cuda", with no device visible
bazel test //tests:unit_cuda        # tests/unit's CUDA-marked rows
bazel test //tests:oracle           # kernels vs the fp64 baselines
bazel test //tests:integration      # composition, dispatch, two-branches-agree
bazel test //:lane                  # all four
bazel test //tests:oracle --test_filter=carry_fold --test_output=streamed   # some of a tier, by pytest's -k
```

Every tier takes the built library -- collecting a tier reaches the plugin (a module that calls a kernel's entry at
import) -- so a source change rebuilds it first. The CPU tier is `tests/unit` alone: an oracle or integration row needs the device whatever its marker (with none
visible, such rows fail rather than skip), so the oracle tier runs `tests/oracle` whole.

**The commit lane is the whole battery above, inside five minutes.** The build-shape gates and `tools/ratify.py` run
additionally at a batch tip (and always when csrc, manifests or codegen flags moved); the latency gate and the sanitizer
lane run on their own cadence, never per-commit. **Never `pytest tests/` unscoped.**

**Every GPU entry point takes the GPU lock itself** — a test tier through a session fixture in `tests/conftest.py`,
`tools/sanitize_oracle.py`, rola-devtools' build system for a target that holds the GPU — through `gpu_lock()`
(`rola_devtools.locks.gpu`): shared for correctness runs, exclusive for measurement. Invoke them BARE: an external
`flock` on the lock's path around a self-locking tool deadlocks, and a lint rule in `tools/lint/lint_standards.py`
refuses one on a command line. A tier runs outside the sandbox, since the locks it takes live in `/tmp` by default
(the GPU lock `host.gpu_lock`, the host budget's slots in `host.lock_dir`) and a sandboxed test's `/tmp` is its own;
a device tier runs alone (`exclusive`), and each shard of a sharded tier draws a host slot, as an xdist worker does.

**THE `cuda` MARKER IS NOT THE LOCK'S QUESTION.** The marker's job is *collection-time deselection* — it lets the first
tier avoid initializing a CUDA context at all. It is **not** a device partition: `tests/integration` carries rows that
launch kernels without being `cuda`-marked (e.g. `test_facts_pass.py::test_the_words_are_the_model_bit_for_bit`,
skipped without a device), because in that tier a device is the premise rather than a property of a row. So the
`tests/conftest.py` fixture keys on TIER (`tests/oracle/`, `tests/integration/`) OR the `cuda` marker. Every committed
test is collected by one tier above. The `bench` marker is opt-in for the same reason in reverse: a latency measurement is not a correctness gate and
must never be able to fail a merge.

## Measurement, and the tests that guard it

Performance work has its own document — [`measurement.md`](measurement.md) — because the instruments are as much a
subject of review as the kernel. The timing system's own non-vacuity tests live with it, in rola-devtools and
rola-results; this repository's is `tests/unit/test_bench_provider.py`, which plants an uncarried carry arm, an
iteration build, another library's data and a binary without the kernel a subject launches, and requires each refusal.
`bazel run //:ratify -- --self-test` is the same discipline on the codegen side ([`ratification.md`](ratification.md)).

### Triaging a suspected stream-ordering bug

There is no runtime detector for a forgotten cross-stream dependency on already-allocated memory, and
`compute-sanitizer --tool racecheck` does not fill the gap — it detects on-chip **shared memory** hazards only, and both
bugs that escaped here were host-side Python. Two environment variables classify one before a debugger is opened,
because each removes a different precondition:

```bash
CUDA_LAUNCH_BLOCKING=1            # serializes launches: collapses the window a missing join needs
PYTORCH_NO_CUDA_MEMORY_CACHING=1  # every free becomes a device-synchronizing cudaFree
```

A bug that disappears under either is an allocator-lifetime or stream-ordering bug — a missing `record_stream` or a
missing join — not a device memory hazard.

### The stream-discipline fixture, and what every crossing path owes

`tests/integration/stream_discipline.py` is the fixture, and **a path that crosses a stream instantiates all four
checks**:

| check | makes deterministic |
|---|---|
| `OrderingSpy` | the publication is joined before the consumer reads it — and a join performed *inside* the publication is recorded distinctly, so it cannot satisfy the outer claim |
| `record_stream_spy` | a tensor crossing to another stream declares its lifetime there |
| `allocator_pressure` | the window a lifetime bug needs is actually opened |
| `assert_capture_rejoins` | a fork that never rejoins **refuses to close a CUDA graph capture** |

The last is the only *mechanical* detector of the missing-join class that exists anywhere, and it is not ours:
`cudaStreamEndCapture` returns `cudaErrorStreamCaptureUnjoined` when a stream forked inside the captured region never
joins back. Each attempt runs in its own process, because a failed capture leaves the context in an error state that
later calls inherit. **Its limits are measured too**: capture validates only the captured region, cannot see an event
recorded on the *wrong* stream, and requires the region to be capture-legal. `PageArena` records its side stream's
readiness event outside any capture, which a capture rejects as a cross-capture dependency, so the paged admission
cannot be wrapped in one; the detector is the contract for paths that *can* be captured, and
`test_stream_discipline_fixture.py` proves it reads on this driver in both directions.

Every check in that file is run against a planted violation as well as a correct path — an unjoined fork, a
publication read before its join, an inner join standing in for an outer one, a tensor crossing without
`record_stream`, a burst that allocates nothing.

### The sanitizer lane

`bazel test //tests:sanitizer` runs `tools/sanitize_oracle.py`, passing it the toolkit's own `compute-sanitizer`
(`@cuda_tools`, the release the library is built by; the script refuses any whose `--version` is not the toolchain
record's `sanitizer`, and takes none from PATH) over ONE named oracle cell per family (the facts pass, decode and the entmax solves), each pair in its own process
with a timeout, under the GPU lock; `--test_arg=--tool --test_arg=racecheck` or `--family` narrows or widens it. It is
`manual`: no suite holds it.
It runs **when a component lands, before a ratify, and on a schedule, never per commit**
(`docs/KERNEL_STANDARDS.md#sanitizer`) — the tools cost one to two orders of magnitude, and a gate that takes an hour
gets disabled — with `--error-exitcode=1`, an explicit skip-list whose entries carry their
reason, and an `ASSERTED` table for an adjudicated idiom that still runs and passes only if every record is that idiom.
Both are empty, which is the standard.

**RACECHECK IS DEFERRED for every kernel** (Blake, 2026-10-01). The lane runs synccheck and initcheck by default;
racecheck runs only by an explicit `--tool racecheck`, and nothing waits on it: it does not finish on the mbarrier
pipelines (polls tracked inside one barrier interval), it reports false hazards on a fence standing in for a barrier,
and its barrier table overflows past ~20 mbarriers. What covers races meanwhile: the repeat-launch bitwise tests,
synccheck, memcheck and the per-slot oracle.

**Its two honest limits.** `--leak-check full` is **not** used: torch's caching allocator retains its pools by design,
so a clean run reports "leaks", and a lane whose clean state is errors is a lane whose findings nobody reads. And an
overrun that stays **inside** a caching-allocator segment is invisible, since that memory is legitimately the
allocator's; the lane detects accesses that escape the segment.
