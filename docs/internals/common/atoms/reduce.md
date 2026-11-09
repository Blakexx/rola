# `common/atoms/reduce.cuh` — the reductions

Mirrors `csrc/rola/src/common/atoms/reduce.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): contributions into an accumulator others also add into, in global or shared memory.

- <a id="fan-in-add"></a>`fan_in_add(p, x)`: ONE contribution into an accumulator other units also contribute to,
  fire and forget. Nothing is returned, and that is the substance: a returning atomic keeps the issuing thread
  waiting, and the point of a reduction is that nothing waits. No order across contributors is defined, so a caller
  that needs a reproducible summation order needs another operation. Stated as `red.global.add.f32`, because an
  `atomicAdd` with an unused result is only sometimes lowered to a reduction; the SASS gate checks the non-returning
  form.

- <a id="red-global"></a>`red_global_add_f32`: the same with the address converted to the global window first. A
  generic `atomicAdd` on a pinned base lowers to an address-space probe, a shared-memory CAS spin and a local fallback
  beside a RETURNING atomic, whose round trip the next branch waits on (measured: 30% of a dense window's stall
  samples on those branches).

- <a id="or-shared-u32"></a>`or_shared_u32`: one contribution into a shared bitset, fire and forget -- the fan-in's
  contract over the bitwise monoid (no shipped caller). <a id="redsharedaddu32"></a>`red_shared_add_u32`: the same
  over the counting monoid, an arrival count whose incrementer never learns the total; `red_shared_add_f32` over fp32.

- <a id="atom-shared-add"></a>`atom_shared_add_u32`: `atom.shared.add.u32` with the old value returned -- a shared
  counter's take. Stated shared, so ptxas emits no address-space probe.
