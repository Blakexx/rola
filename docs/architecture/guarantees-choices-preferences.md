# Guarantees, choices, preferences

Every statement about this engine is one of three kinds, and confusing them is
the most expensive mistake a contributor can make. A guarantee argued down with a
benchmark, or a choice defended as though it were a guarantee, both waste the
same week.

| | Definition | What overturns it |
|---|---|---|
| **GUARANTEE** | A property the engine may never break. A change that breaks one produces **a different engine**, not a faster one. | A ruling from the project owner, after re-deriving everything that rests on it. Never a measurement. |
| **CHOICE** | A decision that was *measured*, recorded with its number, and is revisable by a better measurement at the governing regime. | A measurement, taken to the standard in §4. |
| **PREFERENCE** | Style and consistency. Load-bearing for readability and review, not for behaviour. | A conversation. Follow it meanwhile; do not re-litigate it inside an unrelated change. |

---

## 1. Guarantees

**Semantics**

- **The fp64 oracle is the specification.** `rola/ops/naive.py` is the artifact
  future maintainers trust when everything else has been rewritten. It and the
  kernel-vs-oracle quality-diff test are a **protected class**: exempt from test
  trimming, re-run under falsification whenever touched, and — critically —
  **independent by gate**: its own code path, no helpers shared with the kernel it
  judges. An optimized kernel validated against a *fresh* reference proves only
  self-consistency — the mechanism by which a shared-router bug passes a full
  suite, which is not hypothetical here.
- **The recurrence is affine in the input.** No state-dependent state update, ever.
  → [no-delta-rule.md](no-delta-rule.md).
- **Dead routes are exact zeros and zeros are inert.**
  → [self-masking-theorem.md](self-masking-theorem.md).
- **Sparsity comes from the activation's exact zeros only.** No epsilon floor, no
  straight-through estimator, no thresholded skip. A skip is exact by
  construction or it does not ship.
- **Membership is decided by the realized support bits and by nothing else.**
  A digit is live exactly when it is nonzero (`csrc/rola/src/facts/liveness_contract.cuh`'s
  `digit_live`), and the facts pass's words carry that bit to every consumer
  ([`facts/liveness_contract.md`](../internals/facts/liveness_contract.md)). An
  amplitude comparison in a kernel is only ever a *performance* skip on a value where
  inclusion is a no-op, and none decides membership.
- **Paging is bitwise invisible**, not approximately equivalent. An absent atom
  reads as zeros and is never stored back
  ([`paging/paging.md`](../internals/paging/paging.md)).

**Numerics**

- **State accumulators are fp32 or exact integer** — the fold's state, the mass, the readout's and the den's sums, the output's sum
  across CTAs, decay dials. → [precision.md](precision.md).
- **The amplitude is never logged.** The direct staged-factor lowering is required,
  not preferred: a product of factors is formed from the factors, never through a
  log-space sum, in every kernel.

**Build and execution**

- **Only measured binaries run**: pinned assembler, AOT-only with no PTX a driver
  could JIT, a ratified manifest per architecture, runtime refusal of anything
  else. → [closed-world-codegen.md](closed-world-codegen.md).
- **No fallbacks.** One comprehensive kernel path. A hard case is solved in the
  kernel, not routed around. A genuine hardware limit is documented as such; an
  unfinished implementation dressed as a limit is a defect.
- **Correctness-edge configurations are refused at the config validator, never
  patched in the kernel.** "The math is the math." The only sanctioned kernel
  branch is one that buys efficiency on a structural axis.
- **No known deficiency ships.** An agent or contributor who knows the right fix
  makes it before committing; one facing genuine ambiguity stops and asks. Review
  exists to catch the *unknowable*, and a review finding that the author could
  have self-caught is an author failure.

**Known non-guarantees, stated so nobody assumes them.** A stateful call is **not**
transactional, in either backing: a continuation advances its carried state IN PLACE,
so a launch that fails part way through leaves that state part advanced rather than as
it was. Recovering a pre-call state is `clone()` before the call
([`../internals/state.md`](../internals/state.md#the-dense-continuation)). The output
`y` is **not** bit-reproducible run to run: the carry's, the intra's and the decode step's
fan-ins add fp32 partials in arrival order, and bitwise run-to-run reproducibility is not worth a
cost (Blake, 2026-09-30). The **state** *is* byte-equal and is gated as such; every `y`
comparison is held to the fp32 fan-ins' reassociation bound
([`decode.md#determinism`](../internals/decode/decode.md#determinism)).

## 2. Choices — measured, and revisable

Each of these is a number, not a belief. The number is what a contributor must
beat.

| Choice | The measurement | Where it could move |
|---|---|---|

**A choice is only as good as the regime it was measured in**, and this record
contains its own cautionary tale: perf claims that scale with routing sparsity
are functions of a *fixture knob*. The benchmarked logit gain sits on the
steepest part of the curve, with candidate fraction spanning 0.96 → 0.09 over
gains 1…32 and no plateau. Until a trained model gives a real census, quote perf
claims with the gain; never present the sparse arm as
"the realistic regime" without that qualifier. This is the single largest
uncertainty in the performance record.

## 3. Preferences

- **Code carries structure; `docs/internals/` carries reasons.** At most a file
  header, a block before an important declaration, and a one-line hazard stub at a
  measured hazard. The measured basis: the state-of-the-art hot-file comment band
  is 10–27% and *falls* as code hardens; a 62% kernel header in this tree failed
  by **rot**, not by volume — three false statements in one comment block.
- **The oracle's style is auditability, at zero regard for efficiency.** Literal
  rendering of the math, explicit loops where clearest, fp64 throughout, trusted
  third-party primitives (`einsum`, `matmul`) over anything hand-fused.
  **Cleverness in the oracle is a defect even when it is correct**, because its
  entire value is verification by inspection.
- **No dead code.** Deletions go in **deletion-only commits** — never mixed with
  moves or renames, so a revert is surgical — each recorded in
  `docs/internals/DELETIONS.md`.
- **Linear, single-purpose commits on `master`.**
- **`csrc/rola` is formatted once, mechanically** (clang-format, pinned, Google
  base, 100 columns, comment reflow off), enforced by the commit gate and CI so that
  every later diff is content.
- **Names follow the architectural mental model** — the reduction-theorem
  vocabulary — rather than implementation history.
- **One API; we own it.** Transformation layers are the pipeline and are fine;
  translation layers that only rename or reshape are a disease. Raw configuration
  dictionaries belong to consumers, not to the library.

## 4. What each category demands of a contributor

**To change a guarantee**: stop and ask. Bring the list of everything that rests
on it — for the self-masking theorem that is three separate design conclusions
and a paging invariant. Expect the answer to be no.

**To change a choice**: bring a measurement that clears the bar the existing
number cleared.

- **At-scale means both axes.** `T` *and* `N`. A `T`-only sweep does not satisfy
  the rule, and a measurement that omits `N` is re-taken rather than accepted.
- **On real routing where the claim is about routing.** Synthetic fixtures draw
  read and write supports independently, which does not merely miss coincidence —
  it **manufactures anti-coincidence**. Measured: three producer cells give
  `mixed = 1.000000` over a million visits; two synthetic entmax cells give
  9.1–9.3%.
- **Process-alternated, clock-controlled, two rounds, distributions reported.**
  Light-tier probing is allowed and labelled; a light-tier decision is re-confirmed
  at full rigor before landing.
- **Through the gates, in the tree the manifest was ratified in.** Register/spill
  ratchet, bit-identity suites, the scoped test lines, census. A compile probe
  of a few shapes is a *screen*, not a ratchet — it cannot see a regression in
  the instantiations it did not compile.
- **With a gate that can fail.** Every new gate ships its must-fail
  demonstration. The three sanctioned mechanisms, all with in-tree precedent:
  in-test non-vacuity assertions (which catch vacuous fixtures — a fixture with
  0.64% read-only visits; a head-major fixture in which every cost ties, so the
  old and new keys coincide); recorded, re-runnable **mutation falsification**; and a
  coverage-vs-dispatch census join. No AST hashing, no source-string machinery.
- **Then three adversarial gates** — rules, invariants, re-validation — after the
  commit and before the merge. Never merge on a self-report.

**To change a preference**: just follow it. If it is wrong, raise it as its own
change.

## 5. The bar behind all of it

This repository is written as **ancestral code**: something others will read,
maintain and contribute to for years, possibly as the reference implementation of
its architecture. Every structural decision is judged by the reader a decade out
rather than the author this week. That is why the reasons live in this tree
rather than in a chat log, why the ratification discipline is framed as
reproducibility for future readers rather than as process, and why the oracle is
treated as the crown jewel rather than as a test.
