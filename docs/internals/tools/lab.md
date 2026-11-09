# `tools/lab.py`, `measure/lab/` — the lab: small CUDA experiments that become kernel rules

The kernels are written to rules about the hardware and the compiler — what a pipe costs beside the tensor pipe, what
ptxas does to a pattern, when two warps interfere. Those rules come from measurements, and a measurement taken in a
one-off probe is easy to get wrong and impossible to repeat. The lab makes an experiment cheap to write and safe to
trust, and keys every record by the architecture, so a rule says where it was measured and a new card or toolkit
re-runs the experiments its rules cite (`docs/KERNEL_STANDARDS.md` §26).

## The loop

1. A QUESTION about the machine, asked as one experiment, `measure/lab/experiments/<name>.cu`: its header states the
   QUESTION and the METHOD, its kernels are template instantiations (one per variant), and its `main` declares the SASS
   each kernel must contain and times the variants.
2. `bazel run //tools:lab -- run <name>` compiles it with the pinned toolkit for the running card, reads the SASS and
   REFUSES to measure unless every declared expectation holds, no kernel touches local memory and no kernel holds a
   convergence call nobody declared (the SASS gate's classification, `tools/sass_gate.py`: ptxas's loop-exit
   `CALL.REL.NOINC` and guarded fallbacks are not ones); checks the GPU is quiet; runs the repetitions; prints median, min and max and flags a result
   whose samples spread past 5%.
3. `--record` writes `measure/lab/results/<name>/<sm_XX>.json` — the device, driver, toolkit, commit, the results with
   their samples, and each kernel's registers and leading opcodes. A rule cites this file.
4. The finding becomes a law in `docs/KERNEL_STANDARDS.md` (§26), naming the experiment, the record and the
   architectures it holds on.

`bazel run //tools:lab -- list` names the experiments by their QUESTION;
`bazel run //tools:lab -- sass <name> [REGEX] [--full]` shows each kernel's registers and leading opcodes (or the whole
disassembly). `bazel run //tools:lab -- ncu <name> --metric M [--metric ...] [--record]` reads
hardware counters per kernel (the median over its launches) into `<sm_XX>.counters.json`, for a
question about traffic rather than time.
`bazel run //tools:lab -- ledger <name> [--kernel REGEX]` attributes a kernel's executed instructions to its phases --
the `//:` headings of the experiment's source -- per declared unit (KERNEL_STANDARDS §29.13; the method:
[budgets.md](budgets.md)). `run` also refuses a budgeted kernel whose loops hold convergence instructions (`BSSY`,
`BSYNC`, `BRA.DIV`) beyond what it declares as `LOOP_CONVERGENCE`.

An experiment that reads data -- a cell's routing, say -- has a `<name>.inputs.py` beside it whose `write(directory)`
writes the files and returns their names. `run` and `ncu` call it first (in the scratch space), pass the binary
`--inputs <directory>`, and the record carries each file's digest under `inputs`.

<a id="counters"></a>

## The standard counters

`run` reads the standard counter set of [`pipes.py`](pipes.md) for every kernel of the experiment (its first launch),
prints per card each pipe's cycles a unit (its MMAs, else its declared `Bench::units`, else a launch an SM) with the
binding one named, and on this card each pipe's busy share, L2 sectors and the three largest stall reasons; the record
keeps the counters under `counters` (KERNEL_STANDARDS §29.14). The tensor pipe is priced from a declared `Bench::mma`,
else from the work the tensor-pipe counter counts and the one MMA kind the SASS issues.

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
* `lab::WarpSpan` — a warp's start and end `clock64`; `Bench::span(warps)` is the mean block's span (last end less
  first start), the throughput when warps finish apart.
* The operand pieces the kernels use: `ldsm` and `ldsm_t` (`ldmatrix.x4`, `.trans`), `hmul2`, `pack_half2`, `smem`;
  the mbarrier as `ops.cuh` spells it: `mbar_init`, `mbar_arrive`, `mbar_wait` (polled), `mbar_test` (once).
* `lab::sink(out, v)` — a fold of everything a thread computed, stored unconditionally.
* The atoms as one PTX instruction each: `mma_f16_f32` (and `_fresh`), `mma_f16_f16`, `mma_bf16_f32`, `mma_u8u8`,
  `mma_u8s8`.
* `lab::Bench` — the host runner: `expect(kernel, op, cmp, count)` declares SASS (printed under `--expect`, run by the
  driver before anything is timed), `variant(launch, measure)` warms up and repeats a launch, `record(name, unit,
  samples)` prints a result, `cycles(warps)` and `mean(...)` read the warps' clocks. `--blocks N` holds the grid an experiment launches over `Bench::sms` to N CTAs, so a sanitizer
  can run an experiment at one CTA (racecheck's host memory grows with the CTAs it tracks: the carry's readout part
  harness, `read_alone`, at one CTA).
* `Bench::budget(kernel, class, max)` — KERNEL_STANDARDS §29's instruction budget, per MMA, gated on ncu's executed
  counts by `run`; `Bench::units(kernel, unit, n)` makes a kernel with no MMAs budgeted per unit, its first launch
  doing `n` of them (`color_union`: a box-window colored).
* `lab::kernel("tax", {3, 17})` — a kernel's demangled name as an expectation's pattern.

## What goes wrong, and what guards it

Every item below cost a wrong number before the lab existed; each is now a guard or a habit.

* <a id="build"></a>**A launch sized at the host pass's card.** The host pass compiles at the pinned card, so a
  harness that sized its launch from a plan at the compiling architecture launched the A100 with sm_86's shared bytes
  (2026-11-07, `read_alone`, `read_dense`). An experiment compiles for one architecture (`tools/lab.py`'s `build`:
  `-arch=sm_XX`, C++20 as the extension) and `-DROLA_LAB_CC` names it to both passes, so `lab::kLabBuild` -- that
  architecture's own build ([`common/hardware.md#builds`](../common/hardware.md#builds)) -- is the one plan a harness
  reads on the device and on the host.
* **Dead work.** ptxas deletes an instruction whose result is dead — including a result only tested against a value it
  can fold (a probe's MMAs vanished behind `acc == x && m == y` with `m` constant). Guard: `lab::sink` stores
  unconditionally, and the expectations refuse a kernel missing what it means to time.
* **Rewritten arithmetic.** `asm volatile` binds the front end, not ptxas. ptxas folds a repeated `x += k`, rewrites a
  whole integer-add recurrence (adds are exactly associative), merges two-input adds into the three-input `IADD3`, and
  issues adds as `IMAD.IADD` on the FMA pipe at will. Habit: an instruction's operands are loop-variant values from
  other chains; integer adds are timed paired with a non-associative op; an expectation may name alternatives
  (`IADD3|IMAD.IADD`).
* **Collectives under role-dependent control flow.** A warp collective under a branch on a thread-derived value — even
  a warp-uniform role like `threadIdx.x >> 5`, and even when the role only sets a loop's trip count — compiles into a
  CALL into a convergence subroutine. Guard: the driver refuses an undeclared CALL; `collective_branch` records the
  behaviour.
* **A sink that cancels.** XOR-folding a value twice folds to zero, and ptxas deleted a whole kernel's MMAs behind
  it. Guard: the expectations.
* **The mean warp undercounts.** When warps finish apart, the mean warp's cycles read below the tensor floor; the
  block's span is the throughput. Habit: `WarpSpan` wherever warps can drift.
* **Light register pressure.** A step that runs at its floor beside 150 registers may spill beside the kernel's 255.
  Habit: an experiment about a kernel's step holds the kernel's live accumulators (`fold_step`).
* **Runtime choices inside a timed loop.** A `kind` argument switched at run time inside the loop measured branches, not
  MMAs. Habit: every variant is a template parameter.
* **The operand pattern.** An instruction's cost beside the tensor pipe grows with the registers it reads: packed f16
  math with a constant operand measured ~4x cheaper than with two registers. Habit: write the operand pattern the
  kernels use.
* **Phase lock.** Two warps a scheduler can fall into a slow relative phase on some launches (up to 22% for a pure HMMA
  stream). Guard: repetitions; the driver flags spread.
* **A busy GPU.** Another consumer time-slices every kernel. Guard: the driver refuses above 5% utilization before a
  run (`--force` to override).

## <a id="parts"></a>Part harnesses

An experiment may include the kernels' own headers (`tools/lab.py` puts `csrc/rola/src` on the include path and builds
with the production flags), so a component is measured as the kernel runs it, alone and on real inputs, before its
cost is read in the composed ledger (KERNEL_STANDARDS section 19).

## <a id="experiments"></a>The experiments

Each experiment's question, method, record and verdict is its own page under [`docs/lab/`](../../lab/README.md), one
a source in `measure/lab/experiments/`. The anchors below are where the sources' citations land:

- <a id="tensor-tax"></a>[`tensor_tax`](../../lab/tensor_tax.md)
- <a id="mma-queue"></a>[`mma_queue`](../../lab/mma_queue.md)
- <a id="mma-pairing"></a>[`mma_pairing`](../../lab/mma_pairing.md)
- <a id="collective-branch"></a>[`collective_branch`](../../lab/collective_branch.md)
- <a id="fold-step"></a>[`fold_step`](../../lab/fold_step.md)
- <a id="mma-overlap"></a>[`mma_overlap`](../../lab/mma_overlap.md)
- <a id="pass-pipeline"></a>[`pass_pipeline`](../../lab/pass_pipeline.md)
- <a id="pool-layout"></a>[`pool_layout`](../../lab/pool_layout.md)
- <a id="sparse-pass"></a>[`sparse_pass`](../../lab/sparse_pass.md)
- <a id="state-publish"></a>[`state_publish`](../../lab/state_publish.md)
- <a id="pass-fixed"></a>[`pass_fixed`](../../lab/pass_fixed.md)
- <a id="list-fold"></a>[`list_fold`](../../lab/list_fold.md)
- <a id="read-step"></a>[`read_step`](../../lab/read_step.md)
- <a id="publish-rounds"></a>[`publish_rounds`](../../lab/publish_rounds.md)
- <a id="smem-rates"></a>[`smem_rates`](../../lab/smem_rates.md)
- <a id="warp-hiding"></a>[`warp_hiding`](../../lab/warp_hiding.md)
- <a id="row-span"></a>[`row_span`](../../lab/row_span.md)
- <a id="vote-stash"></a>[`vote_stash`](../../lab/vote_stash.md)
- <a id="facts-block"></a>[`facts_block`](../../lab/facts_block.md)
- <a id="facts-two"></a>[`facts_two`](../../lab/facts_two.md)
- <a id="read-image"></a>[`read_image`](../../lab/read_image.md)
- <a id="read-w1024"></a>[`read_w1024`](../../lab/read_w1024.md)
- <a id="image-ring"></a>[`image_ring`](../../lab/image_ring.md)
- <a id="fold-publish"></a>[`fold_publish`](../../lab/fold_publish.md)
- <a id="digit-box"></a>[`digit_box`, `digit_box32`, `list_box`](../../lab/digit_box.md)
- <a id="color-union"></a>[`color_union`](../../lab/color_union.md)
- <a id="mma-k8"></a>[`mma_k8`](../../lab/mma_k8.md)
- <a id="intra-stream"></a>[`intra_stream`](../../lab/intra_stream.md)
- <a id="issue-latency"></a>[`issue_latency`](../../lab/issue_latency.md)
- <a id="row-stream"></a>[`row_stream`](../../lab/row_stream.md)
- <a id="read-stream"></a>[`read_stream`](../../lab/read_stream.md)
- <a id="read-red"></a>[`read_red`](../../lab/read_red.md)
- <a id="read-group"></a>[`read_group`](../../lab/read_group.md)
- <a id="drain"></a>[`drain`](../../lab/drain.md)
