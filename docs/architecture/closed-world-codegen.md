# Closed-world codegen

> **Only measured binaries run.**

The engine ships a fixed, enumerated set of kernel instantiations, compiled ahead
of time by a pinned assembler, each with a committed record of what that
assembler did to it. A device whose architecture has no such record is refused at
launch. There is no path by which unmeasured code executes.

The mechanics — the manifest format, the four gate outcomes, re-ratification,
one-file-per-architecture, the self-test — are documented once, in
[`docs/ratification.md`](../ratification.md). This document is the *why*, which
that file deliberately does not repeat.

## 1. The problem: codegen cannot be commanded, only pinned

Register allocation and instruction selection are not controllable from CUDA
source. PTX registers are virtual; the assembler decides. There are no supported
SASS assemblers. So the usual engineering answer — *specify the output* — is not
available.

What *is* available: **compilation is deterministic per (source, toolchain,
flags, architecture)**. The same four inputs give the same SASS, every time. So
the way to get a deterministic binary is not to constrain the allocator but to
**close the input space**.

*Reading the numbers back* — compile, measure registers and spills, assert they
are within bounds — is **not** the sufficiency argument. The defect, in the
project owner's ruling: "read them back is not structurally good enough when we
are only testing in a limited architectural regime … the only way to be sure is
deterministic code and rules" (user ruling, 2026-08-01). A read-back gate proves
a property of the binary you happened to build, on the architecture you happened
to have. It says nothing about the binary a different assembler would produce for
an architecture nobody measured. Read-back has a narrower job in this design:
**drift detection between ratifications**.

## 2. The rules

The rules and their enforcement points are one table, in
[`docs/ratification.md`](../ratification.md#the-four-closed-world-rules); this section
keeps only their argument.

R2 is the no-fallbacks rule applied to the compiler: an unknown architecture must
**fail to load**, not silently JIT unmeasured code. R3 and R4 are what make
"supported regime" a fact by construction rather than a claim in a README — the
supported set *is* the ratified set. Bringing up new hardware or a new compiler
therefore requires a ratification run (probe → manifest → gates) before anything
executes.

There is no environment variable that disables any of them.

## 3. The JIT fork is rejected

FlashAttention's active line compiles by **runtime JIT rather than AOT**.
That is a considered position by strong engineers, and the pressures behind it
are real: an AOT matrix over `(tile shape × dtype × head dim × architecture …)`
grows multiplicatively, compile times grow with it, and a JIT path compiles
exactly the one kernel a user's call needs.

**RoLA does not follow it.** The AOT-only build is a decision taken against that
alternative — not an oversight and not an un-revisited default — and this page is
its record.

**The reason is reproducibility for future readers — not process overhead.**
A JIT path means the binary that produced a published number does not exist
anywhere afterwards. It was assembled on the reader's machine, by the reader's
driver, from source the reader has plus a compiler the reader did not pin. Every
claim this project makes about the kernel — a register count, a spill count, a
latency, a bit-identity — is a claim about a *specific* stream of instructions.
Under JIT those claims degrade into claims about a *family* of instruction
streams that were never all measured, and the degradation is silent: the numbers
still print.

The failure mode is concrete rather than imagined, and this engine contains the
instance. One instantiation carries a **4-byte spill accepted as the measured
optimum**, because every alternative spills worse. A judgement of that kind
recorded only in a document is enforced by nothing: a codegen change can grow the
spill with no gate failing ([`ratification.md`](../ratification.md#why-it-exists)).
Under AOT it is instead an asserted invariant over 345 entries per architecture.
Under JIT it could not be asserted at all, because the entries would not exist
until run time, on a machine the project does not control.

The cost of the decision is honest and it is paid: the instantiation matrix is a
budget item that every design must price (a design that doubles compile time
twice is not shippable), and translation-unit sharding is a standing
precondition for growing it. That is the trade — **compile-time cost bounded and
visible, in exchange for run-time behaviour that is enumerable.** A project whose
central claims are numeric buys the second with the first.

## 4. Consequences a contributor should expect

- **A new architecture is a ratification event**, not a `-gencode` line. Adding
  `sm_XX` without its manifest fails the build.
- **A toolchain bump invalidates every manifest**, in both directions. The
  numbers are a property of the assembler as much as of the source, so they say
  nothing about a different assembler even if they got *better*. Re-ratify.
- **Re-ratification is a measurement, not an edit.** Regenerating a manifest by
  re-running the compile is the sanctioned move; hand-editing an entry to make a
  gate pass would be asserting a spill nobody measured, which is precisely the
  failure the manifest replaces.
- **Re-ratifying to clear a refusal launders the refusal into a baseline.** When
  the ratchet refuses a change, the change is what must move. The in-tree
  precedent is a measured optimization held back because 12 instantiations gained
  spill; the resolution is a re-scope of the change to the structural axes where
  it is clean, not a re-stamped manifest.
- **Anything that quietly reopens the input space is refused even when it is
  convenient.** The known instance: `TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES`
  escapes a compiler-cache/depfile collision, but it deletes header dependency edges,
  after which a stale object file passes every gate the project owns. Hard
  reject; a build-cache wrapper is admissible only with its own toolchain pin,
  which is R1 extended to the cache.

## 5. The input-closure rule

Every stamping, keying, or fingerprinting mechanism this design adds must enumerate,
explicitly, everything its key and its value depend on — not just what it *means* to
depend on. A mechanism that silently depends on more than it enumerates produces a
false verdict that reads as a true one, because the gate still runs and still prints
PASS or FAIL; it is answering a question nobody asked.

Two measured instances, both in this tree:

- **A symbol name is a key, and an anonymous namespace makes it depend on the checkout
  PATH.** A kernel in an anonymous namespace carries a mangled discriminator derived
  from the source directory — an input the ratification manifest never named, because
  "the source" was assumed to mean the file's *content* — so a build at another path
  is certifiable-looking and wrong. The fix is at the cause, not a path convention:
  every kernel lives in a named namespace (the ast-grep rule
  `tools/lint/rules/global_fn_in_anon_namespace.yml`), so the key is a function of
  content and toolchain only, the two things the manifest claims to certify.
- **A manifest is a value, and it depended on the codegen flag list.** The manifest
  pinned the assembler version and the calibration macros but not the `nvcc` flag
  list itself, so a codegen-affecting flag change left no fingerprint in anything a
  gate read — caught only because both the build and the ratification import one flag
  list in code, a property of the *source*, not of the *gate* (`tools/ratify.py`'s
  `check_flags`, "THE GAP THIS CLOSES"). Fixed by adding `toolchain.nvcc_flags` to the
  manifest and checking it on every ratification, refused exactly like a missing
  `defines` block.

Both instances have the same shape: a fingerprint whose *stated* input space (source
content, or source content plus assembler) was narrower than its *actual* input space
(source content plus checkout path; source content plus assembler plus flags). The
rule this teaches is checked at design time, not caught by a gate — the gate is what
reveals the gap after the fact, which is exactly the failure mode it is meant to
avoid repeating. Before landing a new stamp, key, or hash: write down everything it
is sensitive to, then verify nothing sensitive is missing from that list.

## 6. What this discipline is *not* claiming

It does not claim the kernel is optimal, or that the measured numbers are good.
It claims they are **the numbers**, that they were taken, and that they cannot
change without something failing. Every other document in this tree cites
measurements; this is the machinery that makes a citation mean something a decade
from now.
