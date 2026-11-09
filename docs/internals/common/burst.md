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

`Ops` is a unit's operand set, a struct of registers (the producer's k-loop: a k-block's chunks of
both operands). `gather(i, ops)`
fills unit `i`'s set: loads, shuffles, packed multiplies, and any per-unit arithmetic that
consumes them at once. `mma(ops, next)` issues the unit's MMAs
and ORDERS them after `next`'s gather through `after`: a `prmt` that returns an operand whole
but depends on a register the other set's gather wrote last, so ptxas must issue that gather
before the instruction that consumes the operand. Two hooks a burst is the pattern: an early
HMMA after the next set's loads, so those issue inside the burst's first half; a late one after
the next set's whole chain, so the multiplies issue inside its second half.

The loop, over units `first .. first + count - 1`: two sets alternate; the first unit is
gathered ahead; each pair gathers `i + 1`, bursts `i` after it, gathers `i + 2` when it
exists, bursts `i + 1` after that; an odd count bursts its last unit against the stale other
set (the hooks then wait for nothing). The count is uniform, so the guards are real branches:
the tail's two are one-trip loops (`ops::once`), because a plain `if` around a small burst (the
producer's one-tile, one-group k-block is two HMMAs) is if-converted into predicated HMMAs
(§20; measured on the producer, 2026-09-29), and a unit past the count is never gathered. A phase that must decide
mid-way runs two bursts with the decision between them: the pipeline restarts once, one exposed gather, and the
decide tier stays out
of the loop.

<a id="users"></a>
## Its users

The producer's k-loop (`produce/produce.cu`'s `k_loop`, [`produce.md#accumulate`](../produce/ampere/produce_kernel.md#accumulate)):
a k-block's chunks of both operands are one unit's set, gathered while the previous block's MMAs issue, through the
granular fork below. The carry's fold steps in its own straight-line block form (`docs/internals/carry/ampere/fold.md#chunk`).

<a id="fork"></a>
## The granular fork (2026-09-19)

`run(first, count, gather, mma, last)`: `gather(unit, set)` fills the first unit's set ahead of
the loop; `mma(cur, next, unit)` bursts `cur` while gathering `unit` into `next` IN PIECES, each
pinned into a window of the burst's HMMAs by two dependencies -- the piece's input `after` an
HMMA's accumulator, so it cannot issue before that HMMA completes (an HMMA's completion latency
is its own issue interval, 33 cycles: `calibration.md`), and its result hooked into a later HMMA's
operand, so it must issue before that one; `last(cur, next)` bursts the final unit with nothing
to gather. The loop runs while two units remain, then one `mma` and one `last` or one `last`.
Pinning both ends is what makes the schedule HMMA, piece, HMMA, piece: pinned by the hook alone,
ptxas computed the hook's value right behind the loads it waits on (the probe row read 36.3 an
HMMA alone against 33.0 with both ends pinned, 38.5 unpinned). What it costs: a permute an end,
about twenty-six instructions a fragment pair, paid in a lockstep pair (the two-warp hooked probe
row 66.1 against 64.9); what it buys is a lone warp at the pipe's rate through its own gather.
