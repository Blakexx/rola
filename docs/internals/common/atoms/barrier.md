# `common/atoms/barrier.cuh` — the barriers

Mirrors `csrc/rola/src/common/atoms/barrier.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): the CTA's and named rendezvous, the mbarrier's phases, the warp's edge and the publish fence.

- `rendezvous()`: every thread of the CTA arrives and all prior shared accesses are ordered before all subsequent
  ones; also the WAR fence that lets a ring slot be overwritten. <a id="rendezvous-group"></a>`rendezvous_group(id, n)`
  is the same restricted to `n` warp-aligned threads at named barrier `id` (id 0 is the CTA's and is not offered;
  sm_80/86/90 provide sixteen). `rendezvous_arrive(id, n)` arrives there without waiting: a producer's stores before
  it are visible to every thread past the barrier's `rendezvous_group` (PTX: an arrive synchronizes with the sync on
  the same barrier), so a role hands its output to another and runs on. <a id="rendezvous-if"></a>`rendezvous_if(cond)`
  is `rendezvous()` predicated on a CTA-uniform `cond` (PTX allows a predicated `bar.sync` where every thread
  evaluates the condition alike): one instruction, so where a barrier is needed only sometimes no branch splits the
  block (the carry's rendezvous on its step words' `reuse`, `carry_kernel.md#kernel`).

- <a id="mbarrier"></a>`mbar_init`, `mbar_arrive_when_landed`, `mbar_arrive`, `mbar_wait`: an `mbarrier` (sm_80+)
  completes a phase at `count` arrivals and flips its parity. A thread's `cp.async` copies arrive when they LAND
  (`cp.async.mbarrier.arrive.noinc`: the arrival is one of the counted ones); a thread's own arrive releases its
  prior shared stores; a wait acquires. `mbar_wait` is a `test_wait` spin, cheap when the phase is already complete,
  which is the case a design arranges. `mbar_test` asks once, for a loop that does other work while copies have not
  landed (its answer voted over the warp before it steers a branch). `mbar_wait_sleep` polls with a 64 ns
  `__nanosleep`: on sm_80/86 a wait is a polled shared word, and a tight poll takes the issue slots of the warps doing
  the work it waits on.

- <a id="warp-sync"></a>`warp_sync()`: `bar.warp.sync 0xffffffff`. `__syncwarp()` at a site the compiler cannot prove
  convergent lowers to a CALL into `__cuda_sm70_warpsync`; the instruction keeps the edge a single WARPSYNC (no
  shipped caller).

- <a id="publish-fence"></a>`publish_fence()`: the order between a unit's shared writes and the flag that publishes
  them, at BLOCK scope: producers and consumers that are warps of one CTA need no device-scope fence, which would
  order traffic no participant can observe (no shipped caller).
