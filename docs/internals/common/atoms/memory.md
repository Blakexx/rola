# `common/atoms/memory.cuh` — the loads and stores

Mirrors `csrc/rola/src/common/atoms/memory.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): shared and global accesses addressed rather than indexed, plain and predicated in the instruction.

- <a id="storef32"></a>`store_f32`: a summand stored at fp32, unrounded, for a buffer whose consumer is a fan-in
  ([`reduce.md#fan-in-add`](reduce.md#fan-in-add)).

- <a id="store-shared-u16"></a>`store_shared_u16`, <a id="storesharedu32"></a>`store_shared_u32`,
  <a id="loadsharedu16"></a>`load_shared_u16`, `load_shared_f32`, <a id="store-shared-f32x2"></a>`store_shared_f32x2`,
  <a id="load-shared-v4"></a>`load_shared_v4`: shared accesses ADDRESSED rather than indexed. A swizzled or
  dynamically indexed address is not an affine function of an index, and `dst[k++]` through a typed pointer
  re-derives a widening multiply every iteration (measured: a loop of one `STS.U16` a live bit ran as `IMAD:114,
  IADD3:69` with no `STS` in its top opcodes); a 32-bit shared address advanced by the element size costs one add.
  <a id="store-shared-f32-if"></a>`store_shared_f32_if` is the fp32 store predicated in the instruction (`@p STS`):
  an `if (live) x[i] = v` inside a step compiles to a branch and a convergence barrier around the store (the den's
  result store on sm_80, 2026-11-01), which the SASS gate reds. `store_shared_u32_if`, `store_shared_u64_if` and `store_shared_f32x2_if` are
  the word and the pair the same way (the value-carve readout's panel adds past a tile's run).
  The pair and quad forms are aligned single accesses (`st.shared.v2.f32`, `ld.shared.v4.u32`); a quad every lane
  reads at one address is a broadcast. (`store_shared_u16`, `store_shared_u32`, `load_shared_u16`,
  `store_shared_f32x2`, `load_shared_v4` and `load_shared_f32`: no shipped caller)

<a id="load-shared-if"></a>`load_shared_u32_if`, `load_shared_u64_if` and `load_shared_f32_if` are the loads predicated
the same way (`@p LDS`, 0 where `live` is false). A read-modify-write whose store is predicated off predicates its
load too, so a dead lane reads no word a live lane of its warp writes (the carry's panel adds: a tile's rows past its
run alias its first row's slot).

- <a id="store-bf16"></a>`store_bf16`: one fp32 to one bf16, rounded as `pack_bf16x2` rounds each half, for a
  transposed accumulator whose adjacent pair is adjacent in TOKEN and so in different rows of a [token][d_v] buffer.
