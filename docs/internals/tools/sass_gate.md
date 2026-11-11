# `tools/sass_gate.py` -- the ptxas signatures, on every build

The gate reads every function of every cubin a binary carries and is red on any of KERNEL_STANDARDS
[§20](../../KERNEL_STANDARDS.md#s20)'s ptxas signatures (local memory, a stack frame, convergence calls a converged
warp executes, predicated HMMAs, branches and gaps in an MMA step); guarded convergence
fallbacks are counted, cold code, not red. A step loop leaves out a fallback's return: the stub a `BRA.DIV` enters is
read to its unconditional branch back, however long (ptxas copies a step's remaining work into it). Every build runs it after the link (the library's gate action, [below](#graph)); by hand,
`bazel run //tools:sass_gate -- <.so | .cubin> [--kernel RE] [--member S] [--loops] [--json PATH]`. Each function's line
carries its registers (`regs`, off `cuobjdump -res-usage`), so an iteration build that skips the ratify still shows a
register change. The step cover (KERNEL_STANDARDS §32.4) is a family's machine model's reading, and the gate runs
none until a family's model exists.

## <a id="step-branches"></a>A branch inside a step

`mma_loop_branches` counts, in each innermost MMA loop (`mma_loops`: a backward branch and the lines from its target),
the convergence barriers (BSSY) and the forward conditional branches: each ends ptxas's scheduling block inside the step
([§20](../../KERNEL_STANDARDS.md#s20), §27). One shape is not counted, the loop's exit laid out as its LATCH: a forward
conditional branch out of the loop followed by nothing but unpredicated register copies (`MOV`, `IMAD.MOV`) and an
UNCONDITIONAL back-edge. ptxas lays a loop out so when its exit is not the fall-through -- the carry's readout passes,
one copy a reduce dtype and accumulate form, three of the four jumping to the copies' join (2026-11-06) -- and the block
it ends is the whole step: no step instruction follows it in the loop. The copies are the loop's carried values that
ptxas did not coalesce, moved on the back path only; a register's pressure is the spill check's (`ldl`) and the
register budget's (`tools/life_ranges.py`), not this lint's. The shape costs one issued branch a trip, and a copy each,
against the folded form (`@P BRA` back, the exit falling through). Other work between the exit and the back-edge, a
predicated copy, a branch back into the loop, or a conditional back-edge still counts (`tests/unit/test_sass_gate.py`).

## <a id="cost"></a>What a read costs

Every lint of a function reads the same two disassemblies of its cubin and the same MMA loops, so each is formed once:
a cubin's `nvdisasm` text (`sass.disassemble`) and its parsed `cuobjdump -sass` once a
process for each version of the file (its path, size and mtime), `mma_loops` once a function (`_LOOPS`). A whole read
of the sm_86 extension is ~20 s (measured 2026-11-01), which the
verdict cache leaves to the cubins that changed.

## <a id="cache"></a>The verdict cache

A cubin's verdict, report and statistics are recorded under the key of everything that decides them -- the cubin's
bytes, the gate's own source closure (every module of `tools/` it has loaded), the disassemblers' identity (`nvdisasm` and `cuobjdump` by
their version lines) and the arguments (`--kernel`, `--max-per-hmma`, `--loops`, and `--instrument`'s declarations, by
content) -- in the dev config's scratch
(`sass_gate_verdicts/`). A cubin whose key matches is replayed: its verdict is the reading of identical bytes under an
identical gate, the argument `tools/ratify.py`'s per-unit cache makes. Any change to the cubin or the gate misses and
re-reads. A no-change build's gate is ~0.1 s; a one-kernel edit re-reads that kernel's cubin and the stamp's.
`--no-cache` reads every cubin and records nothing.

## <a id="instrument"></a>An instrument's library

`--instrument DECLARATIONS` gates an instrument's library (`bazel/defs.bzl`'s `rola_instrument`: a family's part
harness, which lands with the family's first schedule): its kernels are PROBES, so it owes none of the source's
kernels, and a probe keeps a step signature -- branches, the conversion unit or global memory in an MMA step -- only
where its declarations name it. A calibration that puts a global reduction between its MMAs does so to measure that
very cost, so it declares it, by a pattern over the function's mangled name, with the cost it measures (a JSON file,
`kernels`: `{pattern: {lint: cost}}`, the lints `step_branches`, `step_xu`, `step_global`); `sm_XX <pattern>` declares
a signature one architecture's SASS keeps alone (ptxas schedules each apart). An undeclared
signature is red, as on the extension, and so is a declaration an architecture the binary carries does not show: a
plain one on each such architecture, an `sm_XX` one on its own (a one-card iteration build reads none of another
card's). The declarations say only what is so. Every other
signature (local memory, a stack frame, convergence calls, predicated HMMA) is red on a probe as on any kernel. This
is section 29's lab declaration (STEP_XU, STEP_GLOBAL) for an instrument, beside its source.

## <a id="graph"></a>The graph's gate

The graph's library rule runs the gate as an action with `--cuobjdump` and `--nvdisasm` naming its hermetic toolkit's
(`@cuda_tools`): `sass.use_graph` points every disassembly at those binaries and every scratch directory at one of the
process's own, and the verdict cache is off -- the action is cached by Bazel on its inputs, the library and the gate's
sources ([`../../build-graph.md#gates`](../../build-graph.md#gates)). Run from a checkout, the gate reads the dev
config's toolkit and scratch, and its own cache.
