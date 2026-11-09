# `common/atoms/warp.cuh` — the warp's votes and controls

Mirrors `csrc/rola/src/common/atoms/warp.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): lane votes and their readings, the warp-uniform index, the lane prefix and broadcast, and the controls ptxas cannot see through.

- <a id="kballotlanes"></a>`kBallotLanes`: the lockstep unit's width, deliberately not the MMA's unit: a warpgroup
  issues one MMA across 128 threads and still votes in lanes of 32, and conflating them reads a quarter of a vote.

- <a id="lane-ballot"></a>`lane_ballot(pred)`: one bit per lane, in lane order -- the unit's own agreement about its
  lanes' data, with no memory round trip and nothing ordered. Written as the intrinsic: the promise is the bit vector.

- <a id="bit-count"></a>`bit_count`, <a id="firstset"></a>`first_set`, <a id="lastset"></a>`last_set`: the readings
  of a vote that turn agreement into POSITIONS -- how many, and where first or last. An empty vector gives
  `kBallotLanes` for `first_set` and `-1` for `last_set`, so "nowhere" compares past every lane with no separate case.

- <a id="lane-prefix-incl"></a>`lane_prefix_incl<N>`: the unit's running total; `N` a power of two no wider than the
  unit, lanes at or above it the caller's to mask. Every lane executes every step, so the unit is converged at each
  `__shfl_up_sync`.

- <a id="nth-set-bit"></a>`nth_set_bit`: the n-th set bit of a word by a branch-free binary search on popcounts,
  about thirty instructions, against the intrinsic `__fns`, a software loop (no shipped caller).

- <a id="uniform-warp"></a>`uniform_warp`: the warp's index as a value ptxas knows to be warp-uniform (three ballots
  on the thread index's bits); `threadIdx.x >> 5` is per-thread to its analysis, and a loop bounded by it is divergent,
  after which every shuffle is fenced and outlined.

- <a id="opaque"></a>`opaque(v)`: a value made opaque at this use (a `mov` in an `asm volatile`), so nothing
  derived from it is computed before it or shared with another use. A value hoisted out of a window loop is held live
  across the MMA bursts, where registers are scarce (KERNEL_STANDARDS §32): a thread's index, a window's index or its
  parity passed through it makes each offset formed where it is used, a few instructions there, instead of a register
  held across the window.

- <a id="once"></a>`once(cond)`: `for (int n = ops::once(cond); n > 0; --n)` is a real branch where `if (cond)` on a
  warp-uniform value would be if-converted and every warp would issue the body predicated off (KERNEL_STANDARDS §20).
  The bound is hidden from ptxas by an empty `asm volatile`, so the loop cannot be flattened as 0-or-1 trips. Never
  around a step loop: there the guard is the plain vote -- `for (k = __any_sync(full, s < n) ? 1 : 0; k > 0; --k) do
  { step } while (__any_sync(full, ++s < n));` -- since `once`'s opaque bound hides the loop's trip structure from
  ptxas (measured on the carry's skeleton: branches inside the step, extra MMA sites, spills; the plain vote clean).
