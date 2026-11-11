# The constitution

Blake's standing rulings: stable, changed only by Blake, and above every law. The laws -- the coordinator's procedural
and incident-derived rules -- are `docs/KERNEL_STANDARDS.md`; a law never contradicts a rule here. Each rule is stated
once, with every amendment folded in; the numbers are permanent, so a citation never breaks.

<a id="principles"></a>
## The principles

- CORRECTNESS INSTRUMENTS: the fp64 oracle is the truth, and bit-identity is the refactor instrument
  (`docs/testing.md`).
- EFFICIENCY IS CORRECTNESS: a design is built in its best form, never correct first and fast later.
- THE PERFORMANCE TARGETS (Blake, 2026-09-30): "at dense we are at the tensor floor, at sparse we are significantly
  below FA, down to basically the overhead floor which is ideally small"; decode the same, "such that our latency
  nearly falls exactly with sparsity"; the intra may be a constant cost at every density, as long as it is low. Every
  kernel is priced against these floors, never against FlashAttention parity.
- PARITY IS A FAILING GRADE (Blake, 2026-08-20): matching FlashAttention at sparse `N = L` while doing many times less
  useful work is a failure, never a result; it is reported as an unexplained loss with a ledger -- each resource's
  floor against its measurement, each loss classified geometry, algorithm or implementation and reconciled to the
  gap. Newer cards' asynchronous machinery is no explanation; the cause is named from receipts.
- LEAN ON EVERY AXIS, ON EVERY CARD: one kernel runs on every card, so nothing is wasted because this card does not bind
  on it -- instructions, each pipe, shared-memory wavefronts and bytes are each kept at what the design needs; a trade
  between them is made only when forced, and priced on every card.
- CLOSED WORLD: only measured binaries run (`docs/ratification.md`).
- ONE FORM PER MECHANISM: no fallbacks, no dead code.
- ONE STRATEGY, APPLIED UNIFORMLY (Blake, 2026-06-26: "we shouldnt be making any 'branching' to avoid edge cases. if we
  have a strategy, we should do it uniformly"): nothing branches the algorithm to dodge one configuration's cost; that
  cost is the configuration's own property, reported so the user can choose another (a cost flat routing pays is why a
  tree or square one exists).
- INTERFACE FIRST: a subsystem's whole interface and primitives land first and every caller moves onto them at once;
  a part not yet built raises a named error, never a silent fallback; no parallel old path is kept beside the new.
- NO WORKAROUNDS: a problem the kernel can solve is solved in the kernel, at its root -- never a band-aid, a capability
  shim, a padding hack or "for now" residue.
- SPARSITY IS THE ACTIVATION'S: a skip is an exact zero the routing activation (entmax, sparsemax) produced, never a
  threshold, an epsilon floor or a straight-through operator.
- BUILD THE WHOLE DESIGN, OR STOP AND NAME THE GAP.
- FOUNDATION, THEN A NEW BODY (Blake, 2026-08-29: "build a generic foundation, then a kernel body"): a new foundation is
  never grafted onto a body that needs replacing; its contracts, derivations and primitives land first, then the body
  is built bottom-up in its final form -- final mechanisms, the pipelined structure from the skeleton -- never as a
  prototype to be fixed later.
- THE ANCESTRAL-CODE STANDARD: code and docs are written for the reader a decade out.
- PORTABILITY: the repository references only itself (`docs/KERNEL_STANDARDS.md#s17`).
- ONE ATOMIC PRODUCT: the repository in published form is one self-contained kernel product, gated holistically at its
  current state; every test lives in one of three tiers, `tests/unit`, `tests/integration`, `tests/oracle`
  (`docs/testing.md`), plus `measure/`; no per-campaign test directories, and no campaign names in the tree -- names
  are roles.
- SELF-DOCUMENTING CODE: code is legible without comments. Comments are sparse -- a file header and one-to-three-line
  declaration blocks naming the purpose and the one invariant -- and state what the code does and why, in the present
  tense, in the library's own terms: never a past state, a superseded mechanism, a stage or campaign identifier, or a
  comparison to what was. Derivations and why-not-the-other-form prose live in `docs/internals/`, which mirrors
  `csrc/`.
- THE REPO IS ONE ATOMIC DESIGN: its docs are its whole current state and reference nothing outside it -- no log, plan
  or tracker -- save another design cited as the reason ours differs.
- DOCS ARE CURRENT TRUTH: a doc states what is; the past appears only as the rationale for the current shape. Removals
  are recorded in `docs/internals/DELETIONS.md` and git, and nowhere else.

<a id="r8"></a>
## R8. Design mechanisms that expand to all architectures

Every mechanism states its hopper (sm_90) and blackwell (sm_100) form at design time (wgmma and tcgen05 shared- and
tensor-memory operands, TMA box loads, clusters and distributed shared memory, vector RED); a mechanism that exists on
one architecture only is a workaround, not a design.

THIS LINE IS HOPPER'S AND BLACKWELL'S (Blake, 2026-11-09). The ampere family is the branch `ampere`, the frozen
reference: its design is no input to a hopper design ("the ampere design is just bad on hopper and not even in
consideration") and is never run on a hopper card as one. Its cards stay rows of the hardware table, tabulated and not
built. A kernel with no family schedule (the facts pass, decode, paging, entmax) may compile for sm_86 as it is
written for every card, so the development card runs it; no sm_86 schedule, atom, branch or plan is ever written to
make something run there ("never introduce a nonstandard sm86 form just to test something we wont even maintain or
use").

THE FORM IS NAMED PER FAMILY. The architectures fall into families -- `ampere` (sm_80 to sm_89), `hopper` (sm_90),
`blackwell` (sm_100) -- and a mechanism's form is named per family: an ATOM where only the instruction differs, the
family's SCHEDULE where the structure does (which warps do what, the pipeline's stages and barriers, the tiling). The
semantics, what the oracle defines, are one: every family's schedule shares them and is gated against the same
reference.

EVERY KERNEL ADAPTS TO ITS CARD. Every kernel is selected per product, compile-time specialized to it, and takes every
constant the product or the declaration decides from its plan; no kernel hard-codes one. A literal stays only where the
hardware fixes it (an MMA shape, the warp size), as the declaration's closed-world bound (refused by name past it), or
as an encoding.

ONE ENTRY A KERNEL, BLIND TO ITS IMPLEMENTATION. A kernel's caller sees one interface: the entry finds the device's card
from its tensors' device and selects the family's schedule and the atoms' instructions behind it; the caller passes no
hardware fact. HARDWARE FACTS LIVE IN THE KERNEL PATH: Python reads one only for a purpose the C++ path cannot serve (the
fp64 oracle, the measurement tools); shaping or launching a kernel call from a fact in Python moves into the entry.

NO FALLBACKS AT THE ATOM. A primitive may select different native instructions per architecture; it never selects a
wider-type emulation on a target architecture. An operation an architecture lacks is a declared design fact with its
cost on the card, never an `#if` branch. (An fp32 emulation of a bf16 multiply lived behind a passing oracle while the
native instruction existed all along.)

<a id="r9"></a>
## R9. Never branch on a fuzzy threshold

No dispatch, launch policy or in-kernel path keys on a measured statistic (sparsity fraction, density, live count,
"small L"). Branch on STRUCTURE the declaration fixes (tile counts, modes, plan axes); the kernel itself handles any
data distribution. A deterministic cost model per architecture choosing among exact forms is structure -- each form
gated by forcing it; a tuned cutoff that changes results is not. So is a bound the USER declares and the kernel
certifies (a declared precision met per block on published values, the reference modelling the same rule): Blake, "it's
still branching on a threshold either way, in the kind that we allow, which is a user-supplied threshold". A
data-dependent path whose criterion cannot be restated as such a bound stays out.

A structure branch fails loud at every layer, through one primitive: every CTA-uniform structure branch in device code
is a `uniform_switch<Set>(value, body)` (`common/structure_switch.cuh`), whose contract is the four layers -- a
generated case set with a count assert, compile-time exhaustiveness (the derivation's outputs within the set), the
seam's launch refusal reading the same set, and a `__trap()` default with a post-build coverage sweep. A hand-written
device `switch` on a geometry field is a lint finding.

<a id="r10"></a>
## R10. Design and code for all regimes at once

Dense, sparse, mixed, clustered, small and large L, small and large N. Never overfit a mechanism, a measurement or a
verdict to one cell: every design states its cost in each regime, every stage is probed at both ends of the
distribution, and a mechanism that wins one regime by losing another is a finding to rule on, not a win. Dense is the
worst case and a test of the kernel, never a regime to concede.

<a id="r11"></a>
## R11. The axis law

A compile-time axis is legitimate only as a deterministic function of the declaration the selector can pick;
everything else is derived or a constant. The compiled axes are the carve's plan, per side (the shipped set's key);
per-level widths and the leaf order are runtime addressing; the window and its token grain are constants.

<a id="r12"></a>
## R12. Shape compile-time, addressing runtime

Only the register shape -- the plan's rows, factors and channels a warp holds -- is compiled. Every shared-memory
tenant is launch-sized from the descriptor; residency is asserted at the seam as a refusal; no runtime trip count sits
in a register-shaped loop; shared memory is never max-sized.

<a id="r13"></a>
## R13. The state owns its format

A descriptor (D, per-level widths that are powers of two from 16, the canonical order, the page rectangle as the
trailing four canonical bits, d_v) is bound at the first call and checked after. Pages hold each leaf-channel as an s24
at its (leaf, 32-channel group)'s power-of-two exponent and the mass in fp32; a kernel converts at its own boundary
(the fold to fp32 in registers, which the readout reads as its bf16 operand). The warp sub-box's spans are compile-time
from the plan; prefill and decode share one admission law; N is a multiple of 16, and a ragged N is refused, never
padded.

TWO CONTRACTS, ONE TRANSLATOR. The API accepts any shape (any width from 2, any d_v, any L) and serves it by padding;
the kernel accepts only the descriptor's strict shape and its operand formats and refuses the rest -- it never pads or
masks. The seam is the only translator. Both boundaries are tested: API tests prove padded runs bit-identical, and
kernel-boundary tests call the launch surface with unpadded shapes and assert the refusal; a lint keeps logical-width
fields out of every kernel entry signature.

<a id="r14"></a>
## R14. Ownership grains nest on the page, nothing more

Decode's unit is the atom; prefill's box owns whole rows; a kernel consumes activity bits at its own grain and the
page table derives its own.

<a id="r15"></a>
## R15. The CTA is the SM

One resident CTA per SM with the warps the register file admits; warps partition the box by role inside it. Latency
hiding is the kernel's job (covers, split-phase edges): exposed latency is a coverage finding, never grounds to split
the CTA. One CTA per SM is the goal, not a measured choice -- a two-CTA launch is only a diagnostic of how far the body
is from the target.

<a id="r16"></a>
## R16. Ship without arm registration

A Python-package user never registers an arm. The shipped set is the carve plans `tools/manifests/shipped_set.json`
declares per architecture; an unknown declaration is admitted by a refusal-checked derivation, never by enumeration.

<a id="r17"></a>
## R17. One family, one form a component

Operands are the bf16 family (Blake, 2026-11-09). Every level's rows, either side, dense or sparse, are u8 unit-max
with the row's fp32 scale; the values are bf16 unit-max with an fp32 scale a token. An MMA's operands are bf16 (a u8
byte is exact in it) and its accumulation fp32 -- the fold, the readout and the den alike -- but for a window's pair
products of rows, which are integer MMAs of the bytes, exact (the data model locked 2026-11-09 has the intra so);
the output's sum is
fp32, in shared memory and across CTAs: no scale pass, no image, no switch of precision by card or by density. Each
component has one form across densities and cards, unless measurements show that no single form wins or ties
everywhere. The family's error is its own class, 0.16 to 0.20% a token against the fp64 oracle on the flagship cells
(measured), accepted on every card and recorded in `docs/architecture/precision.md`.

<a id="r18"></a>
## R18. One body per op family

Dense and sparse are plans of the one body, never a second body.

<a id="r19"></a>
## R19. Laws land with their stage

A ruling that changes a law is written into the standards in the same stage that implements it, never only on a
card; the milestone carries the drift guards (lints, docs, skills).

<a id="r20"></a>
## R20. The ownership unit is the CTA's box

The CTA's box is the carve's box (per-level extents and a channel extent), carved per side by its plan; the shape set
is the plans. A CTA loads and stores its box at entry and exit, so its read-side warps' union and its write-side
warps' union are the same leaves; the CTA's box equals the SM's at one CTA per SM, the goal (R15). Where the
accumulator lives (registers, tensor memory) changes how a warp touches its share, never who owns what.

<a id="r21"></a>
## R21. Every asynchronous operation states two edges

An asynchronous write -- TMA, `cp.async.bulk`, `wgmma`, `cp.async` -- has two hazards, and a wait
covers only one. Its COMPLETION edge (read after write) is the wait or barrier phase behind which the first read of
the new contents stands. Its ISSUE edge (write after read) is the barrier after the last read of the old contents by
any warp: a warp's wait orders that warp against its own copies, never against another warp's reads of the bytes it is
about to overwrite. Every shared tenant an asynchronous operation writes states both edges in its ledger, and a review
refuses a tenant that names only the wait. The two edges are a pipeline stage's empty and full barriers; where a
kernel has no stage ring (per-lane copies behind the CTA's own edges), the issue is placed past the edge that ends the
last read.

<a id="r22"></a>
## R22. Degrade smoothly, scale, partition into independent units

Blake: "my concern is any kernel design that 'spills' at some threshold rather than just cleanly degrading. How things
scale is important too, instead of just designing for one specific point. Kernels should ideally partition atomic,
independent work across ctas that join back together in clean ways. At least, normally unless there is some strong
reason not to. Everything is about tradeoffs"; "It's also ok to join via atomics, I still don't care too much about
determinism".
1. NO THRESHOLD CLIFFS. A kernel's cost rises along a curve with density, width, depth and N. A design that changes
   path or falls off a cliff when the data crosses a capacity (a shared-memory buffer sized for the sparse case, a held
   window that only fits narrow levels) needs a stated strong reason in its design block. A capacity the PLAN fixes (a
   box, a stage, a token block) is structure, not a threshold; a capacity the DATA can overrun is a threshold. Shared
   memory sets how many of a window's phases are co-active: as the area shrinks, fewer phases run at once (a fill
   overlapping only what has freed), then the phases' chunks shrink -- a continuum, never a form that fits or falls
   back (Blake: "as smem decreases, it just means less coactive and eventually smaller individual phases").
2. DESIGN FOR THE SCALING, NOT A POINT. Every mechanism states how its cost grows with N, D, density and the
   architecture; a cache that may or may not hold a working set is priced as if it does not, so the cache is upside,
   never a cliff in the model or the gate.
3. INDEPENDENT UNITS, CLEAN JOINS. Work is partitioned into atomic, independent units across CTAs that join by
   disjoint writes, reductions or atomics (run-to-run bitwise determinism is not required), or a stage boundary
   between kernels. A CTA waiting on another CTA's progress needs a strong reason.
4. EVERYTHING IS A TRADE: a faster form that breaks 1-3 is chosen only with the reason and its numbers written down.

<a id="r23"></a>
## R23. The operands are scaled blocks

Blake: "why cant it just take in keys with their shared scale and per-slot coefficients, queries with per-slot
coefficients and a shared scale too ... and just use them? ... our producers just natively do that form. if someone
doesnt use them, they use our factory which transofrms their input into the form the standard way you do any scaling";
"the gain is a thing the producer is aware of, not something that our formaat knows about ... ultaimtely qkv are stord
in the same scale format, and thats what the op uses".
1. q, k and v are each SCALED BLOCKS: a level's rows (or the values') unit-max in its storage FORM, and the row's max
   its fp32 scale. The forms are R17's, one a kind: a level's rows u8 -- every level, dense or sparse, a zero a zero
   and no floor under it -- and the values bf16. A routing entry is never negative (the den, the liveness words and
   the decay's shares need it); the factory refuses an entry its form does not hold.
2. The op is defined on the blocks' products alone, the same for every consumer and the fp64 oracle: no level is
   required to sum to 1, and nothing in the format says whether one does.
3. A producer's own quantities -- a learned gain, an activation that lands on the simplex -- are folded into the
   blocks by the producer, each block's scale formed inside the one job that holds its row, and are invisible to every
   kernel and to the oracle. A caller's own levels reach the format through the factory, which quantizes each against
   its max and nothing else.
4. A consumer that needs a scale-free quantity (the decay's per-leaf shares) computes it from the rows itself.
