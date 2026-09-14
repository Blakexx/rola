# `carry/carry.cu` — the carry family's host seam

Mirrors `csrc/rola/src/carry/carry.cu`, `carry/carry_arm.cuh` and
`carry/carry_arm_abi.cuh`. The body has its own page, `carry_kernel.md`.

<a id="refusals"></a>
## The kernel boundary refuses; it never translates

KERNEL_STANDARDS' two-contracts addendum: the API above accepts any shape and serves it by
padding, and this boundary accepts only the descriptor's strict shape. `refuse_shape` and
`refuse_operand` are the whole of it — the page rectangle, the depth, every level width a
power of two at or above the page granule, a value width that is a power of two at or above
sixteen, a launch shape the family builds, and then every operand's dtype, contiguity and
exact extent. A carried state is refused unless its two planes are the SAME pointer, because
the exit sweep stores only the pages this call writes and two planes would silently drop the
rest.

The closed-world arch refusal is the first statement of every entry, ahead of any of this: a
device that was never measured outranks a shape that was never built.

The addressing block is derived here, once, by the same `derive_carry_geom` the host mirror
`carry_geometry` calls, and its refusal string is passed through verbatim.

**The carve order is a runtime input the surface cannot yet name.** The stream schedule makes
the per-side level order a launch input, and the mode word is what expresses it; the launch
surface carries no such field, so this entry derives with every level declared dense on both
sides. That is a lawful point of the derivation — the two sides then share one order — and
it is a shape question, not an arithmetic one: the carve decides which leaves a warp owns,
never what the fold computes.

<a id="launch"></a>
## The launch

One CTA per `(box, batch-head)`: `grid = (owners, BH)`, `block = warps_per_cta * 32`,
dynamic shared memory from the box plan's ledger. The opt-in above 48 KB is done once per
arm behind a function-local flag. The launch bound is `(threads, 1)` — one CTA per SM is the
design point on every architecture, not a measured outcome, and the launch table says the
same thing in the same words.

The parameter block reaches the kernel as a GRID CONSTANT. The body reads its fields through
a reference from every inlined stage, and without that qualifier a reference to a by-value
kernel parameter forces a per-thread local copy of the whole block — 816 bytes of stack
frame and two hundred local accesses, in the measurement that found it.

<a id="census"></a>
## The build stamp and the census

`carry_build_stamp` returns ONE number, the way every other family's stamp does — a mixed
radix over four facts a kernel THIS binary compiled read off the device: the page granule
with the warps per SM, the capability digest the device code compiled against, the state's
share of the register file, and the cluster size. A path or a content hash could be produced
by a stale extension; this could not, which is why every harness that has to know which
binary answered asks for exactly this.

`carry_census` is the separate question and returns one row per built arm: the register
count, the local size, the shared bytes, the maximum threads and the binary version from
`cudaFuncGetAttributes`, then the box plan's own derived numbers — the threads, the box's
leaves, its pages, the accumulator's register count, the state elements a warp holds, the
state registers per lane, the segment slot count and the generated member count. The first
group is what the compiler did; the second is what the design says it should have done, and
having both in one row is what makes a disagreement visible.

## One arm, one translation unit

`tools/gen_shards.py` emits one `.cu` per arm group from the declared shipped set; each
defines its arms only when this build's generated selection header names them, so an
arm-subset build is a FILE subset and the arms compile in parallel rather than serially
inside one front end. The dispatch translation unit here holds no arm body — it declares
the ABI and calls it — which is what keeps a subset build from instantiating an arm locally
and leaving the shards doing nothing.

<a id="carve-order"></a>
## The carve order on the launch surface

`carry_forward` takes the carve order as `2 * D` entries — the read side's levels ranked
innermost first, then the write side's — and hands them to the one derivation, which is
where the order has always belonged (`docs/internals/common/geom.md#carve-order`). It reaches
the kernel inside the parameter block's addressing, as each side's `level_at`/`rank`, and
every stage that addresses a leaf reads it from there.

A CALLER NEVER SPELLS IT. The seam derives it from the level modes it already carries — a
side's sparse levels rank outermost — and hands the block it derived to this surface, so the
order the kernel addresses by and the order the host checked against are one derivation and
not two. A call that declares no modes gets the canonical order.

<a id="schedule"></a>
## The schedule dials

`carry_forward` takes a two-entry `schedule` beside the carve order: `[read grain, fold
grain]`, in LEAVES -- the leaf range one stream keys on. The grains are the primitive
ledger's class 3, per side, and each is a LAUNCH parameter
that selects a body: one uniform switch per phase, inside the window loop and outside the
batch loop, with the grain's k-step count, stream count and pass structure as compile-time
facts of the body it selects (`carry_kernel.md#streams`).

The boundary refuses a grain that is not a power of two between the page rectangle and the
warp sub-box (a finer stream would divide the innermost digits below the vector width, R13).
The grain at the warp sub-box is the one-stream point of the family, the dense form, which
the bench's `box` schedule name selects.

`rola.ops.carry.select_schedule` picks the default order policy, the first-live-box sort
(`kOrderFirstBox`); the identity order is the sort's A/B. The stream-grain dials this
paragraph once described are gone with the frame they selected (`DELETIONS.md`).
