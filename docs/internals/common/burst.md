# `common/burst.cuh` — the burst: the one loop shape that issues MMAs

Mirrors `csrc/rola/src/common/burst.cuh`.

<a id="tiers"></a>
## Two tiers

A phase that issues tensor-core work is written in two tiers. The DECIDE tier is everything
data-dependent: which tiles, which boxes, which rows; a vote, a ballot, a barrier's poll, a
list built from liveness words. It runs at a boundary — a chunk's take, a tile's start — and
leaves a LIST and a COUNT behind. The BURST tier is a counted loop over that list with nothing
data-dependent in it: no vote, no wait, no branch on data, operands read at static offsets
from the list. Sparsity is a shorter list, never a skip inside a burst.

Why the split is structural and not a style: a warp issues in order and only one or two
HMMAs ahead of the pipe, so any chain placed after a burst — a load, a shuffle, a multiply
that feed the next unit — is exposed whole on a warp whose partner is not issuing
(`carry/calibration.md`, the fragment rows). ptxas schedules within a basic block and, at this
register pressure, sinks loads to their uses and regroups bursts; a vote or a branch between
two bursts ends the block. The burst tier gives it one block and a dependency it must honor.

<a id="run"></a>
## `Burst<Ops>::run(first, count, gather, mma)`

`Ops` is a unit's operand set, a struct of registers (the fold's `FragOperands`: A scaled a
box and the V tiles; a readout box's would be A scaled and the B tiles). `gather(i, ops)`
fills unit `i`'s set: loads, shuffles, packed multiplies, and any per-unit arithmetic that
consumes them at once (the readout's den products). `mma(ops, next)` issues the unit's MMAs
and ORDERS them after `next`'s gather through `after`: a `prmt` that returns an operand whole
but depends on a register the other set's gather wrote last, so ptxas must issue that gather
before the instruction that consumes the operand. Two hooks a burst is the pattern: an early
HMMA after the next set's loads, so those issue inside the burst's first half; a late one after
the next set's whole chain, so the multiplies issue inside its second half.

The loop, over units `first .. first + count - 1`: two sets alternate; the first unit is
gathered ahead; each pair gathers `i + 1`, bursts `i` after it, gathers `i + 2` when it
exists, bursts `i + 1` after that; an odd count bursts its last unit against the stale other
set (the hooks then wait for nothing). The count is uniform, so the guards are plain
branches, and a unit past the count is never gathered. A phase that must decide mid-way (the
fold polls a freed slot's fill halfway through a chunk) runs two bursts with the decision
between them: the pipeline restarts once, one exposed gather, and the decide tier stays out
of the loop.

<a id="users"></a>
## Its users, and the one that is not

The fold's chunk (`carry/carry_kernel.md#fold`): the decide tier is the walk into the ring; the
burst is the fragments over it, as two runs with the fill poll between them. Measured at
nl64k-dense: the fold 47.0K -> 43.7K a warp a window against the same loop written as pairs by
hand, a fragment pair 2,259 -> 2,147 cycles.

The readout's tile is NOT on it (2026-09-17). Tried three ways -- the mask's `i`-th set bit by
`__fns` in the gather (a software loop: the tile +40%), a four-bit list built by a serial scan
at the tile's start (+15%: the scan carried 4.4% of the launch's stall samples), a byte list
written by the lanes at once into the warp's idle ring (+7%) -- the box burst on the primitive
stays about 600 cycles a tile behind the rolled loop with den on the FMA pipe, with the hooks
at the first and fifth HMMA or the fifth and last. The box's burst is eight HMMAs to the
fragment's eighteen, so the pair's own work (two hooks, a guard, the list load) is a larger
share of it, and the rolled loop, at the shared pipe rate when paired, gives up less alone than
the model predicts. A fourth form (2026-09-18) settled which: the mask scanned in the gather
itself (`__ffs` and clear, in the order the primitive gathers, so no list is built), the
loads hooked to the third HMMA, den on the FMA pipe -- alone 68 -> 76% in the fitted model,
paired 100% both ways, and MEASURED +2.6% at nl64k-dense. Its census: the gather lines lost
29K samples and the HMMA lines gained 53K. Paired, the rolled loop was already at the pipe's
rate, its gather's waits covered by the partner's HMMAs; hiding them inside the warp bought
nothing, and the form's extra issue (161 instructions a box pair against 136) was paid in the
lockstep pair. So the readout is the documented exception to §24 (`ops::mma` outside a burst
functor, which §24 forbids elsewhere) for a measured reason: a burst on the primitive hides
latency, and the readout's latency is already hidden by its pair -- no lint enforces the
rule yet, so the exception is carried here in prose rather than as a checked exemption.
