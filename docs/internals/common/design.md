# `csrc/rola/src/common/design.cuh` — the design derivations

The ladder's derivation layer: staging depth from leftover SMEM
(`stages_for`), the CTA target from the register economy (`cta_target`),
residency clauses asserted against the capability row. Every derived quantity
is a pure function of `Caps` + the arm's compile-time geometry — nothing
queries the machine ambiently, which is what makes the census replayable.

## Layer 2a -- the derived selections

LAYER 2a -- THE DERIVED SELECTIONS.

WHAT THE DESIGN LOOKS LIKE ON THIS ARCHITECTURE.  Every entry here is
COMPUTED from `arch_caps.cuh`'s facts; none is looked up per architecture,
because a per-arch table of selections is an architecture branch in a costume
and the standing rule is to branch on the flag, never on the number.

The kernel reads THIS layer for its design and calls `ops.cuh` to execute it.
It never reads a capability fact, and it never sees an architecture number.

WHAT IS HERE AND WHAT IS NOT
----------------------------
  STATE_RESIDENCY   where the fold's accumulator lives while it is also the
                    readout's operand.  DERIVED from the residency facts.
  STAGES            the ring's depth.  DERIVED from the work one round trip
                    must be covered with and from the shared-memory budget,
                    capped by this kernel's live register set.  It was a free
                    axis; what ended it is that the optimum turned out not to
                    move in DEPTH but in TOKENS IN FLIGHT, which is a
                    derivation from the chunk size rather than a search.
  the CTA target    the second launch bound: the residency this arm can
                    actually reach, shared-memory ceiling CAPPED BY THE
                    REGISTER CEILING.  The cap is not optional -- without it
                    the shared-memory ceiling ran past the register file and
                    three arms spilled (measured).

NOT here, deliberately: the SHARED-MEMORY LAYOUT (pad-8 against an XOR
swizzle) and the CHUNK SIZE.  Both are legal on every tabulated architecture
and both have a measured optimum that MOVES -- the swizzle saved the dense
base 0.545 inst/mma at C=16 and cost the routed kernel 1.260 at the same arm,
a genuine sign flip.  They stay FREE AXES, chosen by measurement, and a
selection here would make every future measurement of them unfalsifiable.

THE ONE OBLIGATION.  A selection that is knowingly suboptimal -- one where the
table says a better implementation exists on this part and the baseline does
not build it -- must ANNOUNCE ITSELF at compile time.  A documented gap a
reader sees is a stated engineering decision; a SILENT slower path is the
forbidden thing.  `Announce` below is that mechanism, and `ops.cuh` wires the
sites, because whether an implementation is built is a fact about operations.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/common/design.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="announce"></a>
### `Announce`

A compile-time diagnostic that does NOT fail the build.  Instantiating
`Announce<true>` calls a deprecated constructor, which every supported
toolchain reports with the message attached.  Used for a gap that is a
stated decision rather than a defect; a gap that would produce WRONG or
UNBUILDABLE code is a `static_assert` instead, and reads NOT BUILT.

<a id="note-l36"></a>
### near line 36

WHERE THE STATE LIVES.  The algorithm's requirement is not an instruction:
it is that THE FOLD'S ACCUMULATOR IS REUSABLE AS THE READOUT'S OPERAND
WITHOUT A MEMORY ROUND TRIP.  Ampere and Hopper satisfy it in the register
file (Hopper via the `_RS` atom family); a `tcgen05` part satisfies it in
tensor memory, with no register-resident state at all.  Naming the
requirement after the Ampere instruction would REFUSE that part for having
the wrong storage class while meeting the actual requirement.

<a id="kbuiltresidencies"></a>
### `kBuiltResidencies`

THE RESIDENCIES THIS BASELINE BUILDS.  The one statement layer 2a needs from
layer 2b, and it is a statement about OPERATIONS rather than about hardware:
`ops.cuh`'s consistency section asserts that every residency named here is
served by an operation it actually implements, so the two cannot drift.

REGISTERS ONLY, deliberately.  A TMEM path is not built and a SHARED path
never will be (it is the round trip this design exists to avoid).  Selecting
an unbuilt residency is a compile error reading NOT BUILT -- the architecture
is admissible in structure and the path is simply unwritten -- which is the
same distinction the 32-thread MMA unit already draws.  Building a
speculative path for hardware we do not have would be an operation invented
before it has a caller.

<a id="kspecializationpays"></a>
### `kSpecializationPays`

WHETHER A UNIT MAY BE DEDICATED TO PRODUCTION.  Derived, and the derivation
is about REGISTERS rather than about warps.

Register allocation is UNIFORM PER THREAD across a kernel and is sized by the
MAXIMUM live set over the paths a thread takes, not their sum.  So a unit that
produces and then consumes reuses the producer's registers once they are dead
-- it pays `max(produce, consume)`, and a producer live set smaller than the
consumer's is genuinely free.  DEDICATING a unit to production pays the SAME
`max()`, because the allocation is still uniform: a producer unit cannot be
given a small budget while a consumer unit keeps a large one.  So
specialization buys nothing and costs the unit's threads, which in a
latency-bound regime are the hiding mechanism.

What changes that is a run-time reallocation instruction, by which one group
gives up registers another claims.  THAT is the capability this reads, and it
is why specialization is a DERIVATION rather than a preference.

<a id="kbuiltspecialization"></a>
### `kBuiltSpecialization`

NOT BUILT, and announced rather than refused.  Where the reallocation exists,
a specialized producer would be the better implementation and this baseline
does not write one; the unspecialized path is correct everywhere it compiles,
and every unit runs its own production pipelined ahead of its own consumption.

<a id="kstagesmin"></a>
### `kStagesMin`

THE RING'S DEPTH, DERIVED.  One clause: TAKE THE SHALLOWEST RING THAT REACHES
THIS ARM'S MAXIMUM RESIDENCY.  Depth is BOUGHT, never taken.

THE SWEEP IS WHAT REMOVED THE AXIS.  Measured warm, one source, four builds
differing only in the depth each arm is given, on both cells: every arm of
every chunk size lands within 1.2% across depths 2..6, and the SHALLOWEST is
best or tied on eleven of the twelve (arm, cell) pairs -- worst case +0.4%,
at C=16 padded on the 16-CTA cell.  The optimum does not move, so the depth
is not an axis.

WHY MINIMISING RATHER THAN MAXIMISING.  Shared memory is not free even at
fixed residency: it is carved out of the unified L1, so a ring deeper than
residency pays for reads the L1 would otherwise absorb.  Measured at C=64
padded, where every depth is one resident CTA: the deepest affordable ring
(S=4, 73,728 B) cost +3.2% at the 16-CTA cell and +5.8% at the filled cell
against S=2 (55,296 B).  In the other direction the clause PAYS: at C=64
swizzled, S=2 fits TWO CTAs where the hand-fixed S=3 fits one, worth -18.3%
at the filled cell.

A NOTE ON THE INSTRUMENT, because it nearly wrote a different rule here.  The
first arm timed in a cold process bears the clock ramp from 210 MHz and reads
~14% slow.  That artifact looked exactly like a coverage requirement -- deep
rings good at small chunks -- and a coverage clause was written, measured
against a warm instrument, and deleted.  Every number above is warm, behind a
discarded warm-up pass.

On every arm of every tabulated architecture this evaluates to `kStagesMin`,
because the residency ceiling can only fall as the ring deepens.  The search
is written in the general form anyway: it states the RULE rather than its
current answer, and it is what would notice an arm whose fixed buffers leave
the shallowest ring unaffordable.

<a id="note-l217"></a>
### near line 217

The residency reachable at ANY depth.  The ring can only shrink it, so this
is the shallowest LEGAL ring's residency; an arm whose shallowest ring does
not fit at all is refused by `smem_fits` at its own site rather than silently
deepened here.

<a id="satisfiableresidencies"></a>
### `satisfiable_residencies`

The residencies that MEET THE REQUIREMENT on this architecture, as a mask.
SHARED never appears: a state that has to go through shared memory HAS taken
the round trip, which is a different algorithm.

<a id="near-line-51"></a>
### near line 51

PREFERENCE ORDER, most preferred first.  TMEM is preferred where it exists
because the accumulator is already there and the register file stays free.

<a id="selectresidency"></a>
### `select_residency`

The best residency that BOTH meets the requirement and has an implementation.
Returns SHARED to mean "none", which is the value that can never be selected.

<a id="residencysuboptimal"></a>
### `residency_suboptimal`

True when a residency BETTER than the selected one meets the requirement on
this part and simply is not built.  This is the announcement's condition and
not an error: the built path is correct, it is merely not the best available.

<a id="kresidencysuboptimal"></a>
### `kResidencySuboptimal`

Announced, not refused: the built path is correct on such a part, it is
merely not the best available there.

<a id="ctasbyregfloor"></a>
### `ctas_by_reg_floor`

The largest per-lane register count is a table quantity; this is its inverse
-- the most CTAs whose budget still holds a live set of `reg_floor`.

<a id="ceilingwithoutsmem"></a>
### `ceiling_without_smem`

The residency ceiling that is NOT a function of this arm's shared memory:
registers, threads and the hard CTA cap.

<a id="ctatarget"></a>
### `cta_target`

THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach.  Shared
memory, registers, threads and the CTA cap, whichever binds first.

<a id="stagesbyresidency"></a>
### `stages_by_residency`

The SHALLOWEST ring that reaches `want` CTAs, searched upward from the
minimum: the point past which shared memory buys nothing and still costs L1.

<a id="stagesfor"></a>
### `stages_for`

The compiling target's answers, so a kernel writes a geometry and never a
capability row.

<a id="smemfits"></a>
### `smem_fits`

THE ONE HARD RESOURCE REFUSAL, from the table rather than from a literal: an
arm either fits this architecture's per-CTA shared-memory maximum or it is
refused.  No arm is silently shrunk to fit.

<a id="kcapsdigest"></a>
### `kCapsDigest`

The digest of the capability row THE DEVICE CODE WAS COMPILED AGAINST,
carried here so a kernel's build stamp names no layer-1 symbol.
