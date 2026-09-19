#pragma once

// LAYER 2b -- THE BASELINE OPERATIONS: how each of layer 2a's (`design.cuh`)
// steps is EXECUTED, as operations with contracts, defined by WHAT THE
// ALGORITHM NEEDS -- no architecture named, no inline PTX above this header.
// See docs/internals/common/ops.md

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "design.cuh"

namespace denseref {
namespace ops {

//: Bytes moved by one `stage_tile`.  A FACT OF THE OPERATION, not of the -- see docs/internals/common/ops.md#kstagebytes
constexpr int kStageBytes = 16;

//: The MMA's shape, in the units the carve is expressed in.
constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;

//: THE BUILT `mma`'s UNIT: the threads that jointly issue one MMA and jointly -- see docs/internals/common/ops.md#kmmaunitthreads
constexpr int kMmaUnitThreads = 32;

//: The operand sources the built `mma` accepts.  A kernel whose accumulator -- see docs/internals/common/ops.md#kmmaacceptsregistera
constexpr bool kMmaAcceptsRegisterA = true;
constexpr bool kMmaAcceptsRegisterB = true;
constexpr bool kMmaAcceptsTmemA = false;

//: The built `stage_tile` moves `kStageBytes` per issue.  A bulk (TMA) form -- see docs/internals/common/ops.md#kstagingisbulk
constexpr bool kStagingIsBulk = false;

using SmemAddr = uint32_t;

__device__ __forceinline__ SmemAddr smem_addr(const void* p) {
  return static_cast<SmemAddr>(__cvta_generic_to_shared(p));
}

//: A GATHERED ROW'S ADDRESS: a base formed once, displaced by a 32-bit BYTE -- see docs/internals/common/ops.md#gather-ptr
__device__ __forceinline__ const void* gather_ptr(const void* base, uint32_t off) {
  const void* p;
  asm("{ .reg .u64 t; cvt.u64.u32 t, %2; add.u64 %0, %1, t; }" : "=l"(p) : "l"(base), "r"(off));
  __builtin_assume(__isGlobal(p));
  return p;
}

//: A BASE THE COMPILER MUST KEEP RATHER THAN RE-DERIVE.  The value is unchanged; -- see docs/internals/common/ops.md#t
template <class T>
__device__ __forceinline__ T* pin_address(T* p) {
  asm("" : "+l"(p));
  return p;
}

__device__ __forceinline__ void* gather_ptr(void* base, uint32_t off) {
  void* p;
  asm("{ .reg .u64 t; cvt.u64.u32 t, %2; add.u64 %0, %1, t; }" : "=l"(p) : "l"(base), "r"(off));
  __builtin_assume(__isGlobal(p));
  return p;
}

//: THE STATE PLANE'S ONE BASE TRANSLATION (docs/internals/paging/paging.md). -- see docs/internals/common/ops.md#page-slot
__device__ __forceinline__ int32_t page_slot(const int32_t* page_tbl, uint32_t atom) {
  if (page_tbl == nullptr) return (int32_t)atom;
  return *reinterpret_cast<const int32_t*>(gather_ptr(page_tbl, atom * (uint32_t)sizeof(int32_t)));
}

//: THE FACTS PASS' TWO ORTHOGONAL PER-ATOM BITS, and the one activity -- see docs/internals/common/ops.md#atom-activity
constexpr uint8_t kAtomWritten = 1u;
constexpr uint8_t kAtomRead = 2u;
constexpr uint8_t kAtomActive = kAtomWritten | kAtomRead;

__device__ __forceinline__ uint8_t atom_activity(const uint8_t* atom_bits, uint32_t atom) {
  if (atom_bits == nullptr) return kAtomActive;
  return *reinterpret_cast<const uint8_t*>(gather_ptr(atom_bits, atom));
}

// ------------------------------------------------------------------ staging

__device__ __forceinline__ void stage_tile(SmemAddr dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

//: THE SAME COPY WITH ITS SOURCE RUN PREDICATED: a dead quantum zero-fills and reads
//: nothing. -- see docs/internals/common/ops.md#stage-tile-if
__device__ __forceinline__ void stage_tile_if(SmemAddr dst, const void* src, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(live ? kStageBytes : 0));
}

//: THE SAME COPY AT A NARROWER QUANTUM. -- see docs/internals/common/ops.md#stage-run
template <int Bytes>
__device__ __forceinline__ void stage_run(SmemAddr dst, const void* src) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == kStageBytes,
                "an asynchronous run is 4, 8 or `kStageBytes` bytes");
  if constexpr (Bytes == kStageBytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes));
  }
}

//: THE SAME NARROWER COPY WITH ITS SOURCE RUN PREDICATED. -- see docs/internals/common/ops.md#stage-run-if
template <int Bytes>
__device__ __forceinline__ void stage_run_if(SmemAddr dst, const void* src, bool live) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == kStageBytes,
                "an asynchronous run is 4, 8 or `kStageBytes` bytes");
  if constexpr (Bytes == kStageBytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(live ? Bytes : 0));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst), "l"(src), "n"(Bytes),
                 "r"(live ? Bytes : 0));
  }
}

//: THE SAME COPY ISSUED BY THE LIVE LANES ALONE (the calibration rows' form). -- see
//: docs/internals/common/ops.md#stage-run-lanes
template <int Bytes>
__device__ __forceinline__ void stage_run_lanes(SmemAddr dst, const void* src, bool live) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == kStageBytes,
                "an asynchronous run is 4, 8 or `kStageBytes` bytes");
  if constexpr (Bytes == kStageBytes) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16;\n}\n" ::
            "r"(dst),
        "l"(src), "r"((int)live));
  } else {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.ca.shared.global [%0], [%1], %2;\n}\n" ::
            "r"(dst),
        "l"(src), "n"(Bytes), "r"((int)live));
  }
}

__device__ __forceinline__ void stage_commit() { asm volatile("cp.async.commit_group;\n"); }

//: A SHARED-MEMORY BARRIER OBJECT (`mbarrier`, sm_80+): `count` arrivals complete a phase;
//: an asynchronous copy arrives when it lands (`.noinc`: it is one of the counted
//: arrivals), a thread's arrive releases its prior shared stores, a wait acquires. A
//: slot's phase parity flips at each completion. -- see docs/internals/common/ops.md#mbarrier
__device__ __forceinline__ void mbar_init(SmemAddr a, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_when_landed(SmemAddr a) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_arrive(SmemAddr a) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_wait(SmemAddr a, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "MBAR_WAIT_%=:\n"
      "mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra MBAR_WAIT_%=;\n}\n" ::"r"(a),
      "r"(parity)
      : "memory");
}

//: the same, asked once: has the phase of parity `parity` completed? The interleaved
//: loop's streams test their barriers and do the other stream's work when not.
__device__ __forceinline__ bool mbar_test(SmemAddr a, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(a), "r"(parity)
      : "memory");
  return ok != 0u;
}

//: A SHARED COUNTER'S TAKE: the old value, the counter advanced by `v`; explicitly shared
//: so ptxas emits no address-space probe. -- see docs/internals/common/ops.md#atom-shared-add
__device__ __forceinline__ uint32_t atom_shared_add_u32(SmemAddr a, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(a), "r"(v) : "memory");
  return old;
}

//: THE n-TH SET BIT of a word (n from 0), branch free: a binary search on popcounts.
//: Undefined past the last set bit. -- see docs/internals/common/ops.md#nth-set-bit
__device__ __forceinline__ int nth_set_bit(uint32_t w, int n) {
  int pos = 0;
#pragma unroll
  for (int width = 16; width >= 1; width >>= 1) {
    const int c = __popc(w & ((1u << width) - 1u));
    const bool hi = n >= c;
    n -= hi ? c : 0;
    pos += hi ? width : 0;
    w = hi ? (w >> width) : w;
  }
  return pos;
}

template <int KeepInFlight>
__device__ __forceinline__ void stage_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(KeepInFlight));
}

__device__ __forceinline__ void rendezvous() { __syncthreads(); }

//: A ONE-TRIP LOOP BOUND ptxas cannot see through: `for (int n = once(cond); n > 0; --n)`
//: is a real branch where an `if (cond)` on a warp-uniform value would be if-converted
//: and issued predicated off by every warp. -- see docs/internals/common/ops.md#once
__device__ __forceinline__ int once(bool cond) {
  int n = cond ? 1 : 0;
  asm volatile("" : "+r"(n));
  return n;
}

//: THE SM'S CYCLE COUNTER, for the phase ledger. -- see docs/internals/common/ops.md#cycles
__device__ __forceinline__ long long cycles() {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
  return c;
}

//: THE WARP'S OWN EDGE, stated as the instruction: `__syncwarp()` where the compiler cannot -- see docs/internals/common/ops.md#warp-sync
//: prove convergence becomes a CALL into a helper; this is one WARPSYNC.
__device__ __forceinline__ void warp_sync() {
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
}

// ------------------------------------------------------------------- edges

//: `rendezvous_group(id, n)` -- a barrier over `n` threads that arrive at the -- see docs/internals/common/ops.md#rendezvous-group
__device__ __forceinline__ void rendezvous_group(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n));
}

//: THE LOCKSTEP UNIT'S WIDTH, and it is deliberately not `kMmaUnitThreads`. -- see docs/internals/common/ops.md#kballotlanes
constexpr int kBallotLanes = 32;

//: ONE BIT PER LANE, IN LANE ORDER. The unit's own agreement about a property -- see docs/internals/common/ops.md#lane-ballot
__device__ __forceinline__ uint32_t lane_ballot(bool pred) {
  return __ballot_sync(0xFFFFFFFFu, pred);
}

//: THE WARP'S INDEX AS A WARP-UNIFORM VALUE. -- see docs/internals/common/ops.md#uniform-warp
template <int Warps>
__device__ __forceinline__ int uniform_warp() {
  static_assert(Warps > 0 && (Warps & (Warps - 1)) == 0, "a CTA is a power of two of warps");
  constexpr int kBits = Warps == 1 ? 0 : (Warps == 2 ? 1 : (Warps == 4 ? 2 : 3));
  static_assert(Warps == (1 << kBits), "a CTA is one, two, four or eight warps");
  int w = 0;
#pragma unroll
  for (int b = 0; b < kBits; ++b)
    w |= (__popc(__ballot_sync(0xFFFFFFFFu, ((threadIdx.x >> (5 + b)) & 1u) != 0u)) >> 5) << b;
  return w;
}

//: THE LANES BEFORE THIS ONE, as the hardware's own register rather than a value the body
//: would have to carry: `__popc(vote & lanes_below())` is a lane's RANK among the voters.
__device__ __forceinline__ uint32_t lanes_below() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

//: THE TWO READINGS OF A VOTE. A bit vector answers "how many" and "where -- see docs/internals/common/ops.md#bit-count
__device__ __forceinline__ int bit_count(uint32_t x) { return __popc(x); }

//: `kBallotLanes` for an empty vector, so "nowhere" compares greater than every -- see docs/internals/common/ops.md#firstset
__device__ __forceinline__ int first_set(uint32_t x) {
  const int f = __ffs((int)x);
  return f ? f - 1 : kBallotLanes;
}

//: THE OTHER END OF THE SAME VOTE, for a caller that consumes its lanes from -- see docs/internals/common/ops.md#lastset
__device__ __forceinline__ int last_set(uint32_t x) { return 31 - __clz((int)x); }

//: THE UNIT'S OWN RUNNING TOTAL. `N` must be a power of two no wider than the -- see docs/internals/common/ops.md#lane-prefix-incl
template <int N>
__device__ __forceinline__ int lane_prefix_incl(int x) {
#pragma unroll
  for (int d = 1; d < N; d <<= 1) {
    const int up = __shfl_up_sync(0xFFFFFFFFu, x, d);
    if ((int)(threadIdx.x & (kBallotLanes - 1)) >= d) x += up;
  }
  return x;
}

//: one lane's value, read by all of them.
__device__ __forceinline__ int lane_broadcast(int x, int src) {
  return __shfl_sync(0xFFFFFFFFu, x, src);
}

// ----------------------------------------------------------------- operands

using Frag = uint32_t[4];

//: THE `memory` CLOBBER IS PART OF THE CONTRACT. -- see docs/internals/common/ops.md#load-frag
__device__ __forceinline__ void load_frag(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ void load_frag_t(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

//: THE SAME OVER TWO TILES: lanes 0-15 address them. -- see docs/internals/common/ops.md#load-frag
__device__ __forceinline__ void load_frag_t2(uint32_t (&r)[2], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a)
               : "memory");
}

//: AN 8x8 b16 TILE, TRANSPOSED WHERE IT ALREADY LIES. -- see docs/internals/common/ops.md#transpose-frag-b16
__device__ __forceinline__ uint32_t transpose_frag_b16(uint32_t x) {
  static_assert(arch::kCaps.reg_transpose_b16,
                "UNSUPPORTED ARCHITECTURE: without an in-register b16 tile "
                "transpose the adjoint's channel contraction has to route its "
                "operand through shared memory, which is a different algorithm "
                "and not a slower arm of this one");
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(x));
  return d;
}

//: THE ATOM IS CHOSEN BY SHAPE FAMILY, NOT BY NAME. -- see docs/internals/common/ops.md#mma
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

//: THE SAME ATOM INTO A FRESH ACCUMULATOR (C = 0): the readout's per-box partial.
__device__ __forceinline__ void mma_fresh(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f));
}

// -------------------------------------------------------- bf16 elementwise

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  static_assert(arch::kCaps.bf16_cvt_packed,
                "UNSUPPORTED ARCHITECTURE: without a packed fp32->bf16x2 convert "
                "the accumulator cannot become an operand in place");
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&p);
}

__device__ __forceinline__ float pair_lo(uint32_t p) { return __uint_as_float(p << 16); }

__device__ __forceinline__ float pair_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }

//: THE ACCUMULATOR'S TOP HALVES, TAKEN AS A bf16 PAIR. -- see docs/internals/common/ops.md#hi-bf16
__device__ __forceinline__ uint16_t hi_bf16(float v) {
  return (uint16_t)(__float_as_uint(v) >> 16);
}

__device__ __forceinline__ uint32_t hi_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x7632;"
      : "=r"(d)
      : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return d;
}

//: the two floats' LOW halves, packed the same way: the residue under the bf16 plane.
__device__ __forceinline__ uint32_t lo_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x5410;"
      : "=r"(d)
      : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return d;
}

//: a packed bf16 pair times a packed bf16 pair, elementwise, rounded once.
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  const __nv_bfloat162 p = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                   *reinterpret_cast<const __nv_bfloat162*>(&b));
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: a halfword into both halves of a packed pair.
__device__ __forceinline__ uint32_t splat_u16(uint32_t x) { return (x & 0xFFFFu) * 0x10001u; }

//: AN fp32 GLOBAL REDUCTION, fire-and-forget, the address stated GLOBAL: a generic
//: `atomicAdd` on a pinned base lowers to an address-space probe with shared and local
//: fallbacks and a returning `ATOM` whose round trip the next branch waits on.
//: -- see docs/internals/common/ops.md#red-global
__device__ __forceinline__ void red_global_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(__cvta_generic_to_global(p)), "f"(v)
               : "memory");
}

//: an fp32 shared-memory add, the CTA's own.
__device__ __forceinline__ void red_shared_add_f32(SmemAddr a, float v) {
  asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}

//: One bf16 into BOTH halves of a packed pair, so that a scalar can be an -- see docs/internals/common/ops.md#splat-bf16x2
__device__ __forceinline__ uint32_t splat_bf16x2(__nv_bfloat16 x) {
  const __nv_bfloat162 p = __bfloat162bfloat162(x);
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: the same splat over a raw bf16 BIT PATTERN. -- see docs/internals/common/ops.md#splat-bf16x2-bits
__device__ __forceinline__ uint32_t splat_bf16x2_bits(unsigned short x) {
  return ((uint32_t)x << 16) | (uint32_t)x;
}

//: one raw bf16 bit pattern as a float, by value, for the same reason.
__device__ __forceinline__ float bf16_bits_to_float(unsigned short x) {
  const uint32_t w = (uint32_t)x << 16;
  return *reinterpret_cast<const float*>(&w);
}

//: THE ARCHETYPAL SELECTING ENTRY. Every implementation satisfies ONE contract: -- see docs/internals/common/ops.md#mul-bf16x2
template <bool Simd = arch::kCaps.bf16_simd_arith, bool Fma = arch::kCaps.bf16_simd_fma>
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  static_assert(Simd <= arch::kCaps.bf16_simd_arith,
                "mul.rn.bf16x2 requires .target sm_90 or higher; this "
                "architecture's row says so and ptxas agrees");
  static_assert(Fma <= arch::kCaps.bf16_simd_fma,
                "fma.rn.bf16x2 requires .target sm_80 or higher; this "
                "architecture's row says so and ptxas agrees");
  if constexpr (Simd) {
    uint32_t d;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
  } else if constexpr (Fma) {
    //: the packed pair of bf16 NEGATIVE ZEROS, which is the sign-preserving -- see docs/internals/common/ops.md#knegzeropair
    constexpr uint32_t kNegZeroPair = 0x80008000u;
    uint32_t d;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(kNegZeroPair));
    return d;
  } else {
    static_assert(Simd || Fma,
                  "a packed bf16 product needs mul.rn.bf16x2 (sm_90) or "
                  "fma.rn.bf16x2 (sm_80); every tabulated row carries one");
    return 0u;
  }
}

// -------------------------------------------------------------------- stores

__device__ __forceinline__ void store_bf16x2(__nv_bfloat16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}

//: THE STATE'S LO HALVES, PACKED. The top 16 bits of an fp32 ARE its bf16 -- see docs/internals/common/ops.md#pack-lo
__device__ __forceinline__ uint32_t pack_lo(float a, float b) {
  return (__float_as_uint(a) & 0xFFFFu) | (__float_as_uint(b) << 16);
}

__device__ __forceinline__ float join_hi_lo(uint32_t hi, uint32_t lo, int k) {
  return __uint_as_float(k ? ((hi & 0xFFFF0000u) | (lo >> 16)) : ((hi << 16) | (lo & 0xFFFFu)));
}

//: a byte off shared memory, zero-extended (the fold ring's row).
__device__ __forceinline__ uint32_t load_shared_u8(SmemAddr a) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

//: A SEQUENTIAL SHARED STORE, ADDRESSED RATHER THAN INDEXED. -- see docs/internals/common/ops.md#store-shared-u16
__device__ __forceinline__ void store_shared_u16(SmemAddr a, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
}

//: A SHARED WORD ACCESS, ADDRESSED RATHER THAN INDEXED for the same reason -- see docs/internals/common/ops.md#storesharedu32
__device__ __forceinline__ void store_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: TWO fp32 AT ONCE, an 8-byte aligned pair. -- see docs/internals/common/ops.md#store-shared-f32x2
__device__ __forceinline__ void store_shared_f32x2(SmemAddr a, float x, float y) {
  asm volatile("st.shared.v2.f32 [%0], {%1,%2};" ::"r"(a), "f"(x), "f"(y) : "memory");
}

//: ONE 8-BYTE WORD PAIR from shared memory. -- see docs/internals/common/ops.md#load-shared-v4
__device__ __forceinline__ uint2 load_shared_u64(SmemAddr a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}

__device__ __forceinline__ uint4 load_shared_u128(SmemAddr a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}

//: ONE 16-BYTE QUANTUM from shared memory. -- see docs/internals/common/ops.md#load-shared-v4
__device__ __forceinline__ uint4 load_shared_v4(SmemAddr a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}

//: ONE CONTRIBUTION INTO A SHARED BITSET, FIRE AND FORGET -- `fan_in_add`'s -- see docs/internals/common/ops.md#or-shared-u32
__device__ __forceinline__ void or_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.or.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: THE SAME OVER THE COUNTING MONOID.  A stream's arrival count is incremented once per -- see docs/internals/common/ops.md#redsharedaddu32
__device__ __forceinline__ void red_shared_add_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: THE ORDER BETWEEN A UNIT'S SHARED WRITES AND THE FLAG THAT PUBLISHES THEM, -- see docs/internals/common/ops.md#publish-fence
__device__ __forceinline__ void publish_fence() { __threadfence_block(); }

//: the 16-byte quantum into shared memory, addressed.  The segment's dead rows are -- see docs/internals/common/ops.md#storesharedvec16
__device__ __forceinline__ void store_shared_vec16(SmemAddr a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

//: one amplitude out of a staged factor row, by ADDRESS.  Its index is a digit, so -- see docs/internals/common/ops.md#loadsharedu16
__device__ __forceinline__ void store_shared_u64(SmemAddr a, uint32_t lo, uint32_t hi) {
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(a), "r"(lo), "r"(hi) : "memory");
}

__device__ __forceinline__ float load_shared_f32(SmemAddr a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}

__device__ __forceinline__ unsigned short load_shared_u16(SmemAddr a) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t load_shared_u32(SmemAddr a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

//: One element, rounded to nearest-even exactly as `pack_bf16x2` rounds each of -- see docs/internals/common/ops.md#store-bf16
__device__ __forceinline__ void store_bf16(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

//: One fp32 element, stored where it will later be read as a summand rather than -- see docs/internals/common/ops.md#storef32
__device__ __forceinline__ void store_f32(float* p, float v) { *p = v; }

//: ONE CONTRIBUTION INTO A SHARED ACCUMULATOR, FIRE AND FORGET. -- see docs/internals/common/ops.md#fan-in-add
__device__ __forceinline__ void fan_in_add(float* p, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(x) : "memory");
}

//: The 16-byte quantum, synchronously. -- see docs/internals/common/ops.md#load-vec16
__device__ __forceinline__ uint4 load_vec16(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void store_vec16(void* p, const uint4& v) {
  *reinterpret_cast<uint4*>(p) = v;
}

//: The magnitude bits of a packed bf16 pair.  A run of bf16 is entirely zero -- see docs/internals/common/ops.md#kbf16magnitudemask
constexpr uint32_t kBf16MagnitudeMask = 0x7FFF7FFFu;

//: THE OR OF A CONTIGUOUS RUN, AT THE WIDEST ACCESS THAT COVERS IT. -- see docs/internals/common/ops.md#or-run
template <int N>
__device__ __forceinline__ uint32_t or_run(const __nv_bfloat16* p) {
  static_assert(N == 1 || N == 2 || N == 4 || (N >= 8 && N % 8 == 0),
                "a run is one access of 2, 4, 8 or 16 bytes, or a whole number "
                "of 16-byte accesses");
  if constexpr (N == 1) {
    return (uint32_t)*reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (N == 2) {
    return *reinterpret_cast<const uint32_t*>(p);
  } else if constexpr (N == 4) {
    const uint2 x = *reinterpret_cast<const uint2*>(p);
    return x.x | x.y;
  } else {
    uint32_t acc = 0;
    //: §6: `N` is this function's template parameter (renamed from lowercase `n` -- see docs/internals/common/ops.md#koctets
    constexpr int kOctets = N / 8;
#pragma unroll
    for (int i = 0; i < kOctets; ++i) {
      const uint4 x = *reinterpret_cast<const uint4*>(p + i * 8);
      acc |= x.x | x.y | x.z | x.w;
    }
    return acc;
  }
}

__device__ __forceinline__ void copy_shared_to_global_16b(void* dst, const void* src) {
  store_vec16(dst, load_vec16(src));
}

// 2a/2b CONSISTENCY: A SELECTION IS INVALID IF NO AVAILABLE OPERATION CAN SERVE IT
// (checked against the operation set, never "this architecture lacks instruction X").

//: `design.cuh` claims REGISTERS is a residency this baseline builds. -- see docs/internals/common/ops.md#static-assert
static_assert(!(design::kBuiltResidencies
                & design::residency_bit(design::StateResidency::REGISTERS))
                  || (kMmaAcceptsRegisterA && arch::kCaps.bf16_cvt_packed),
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency "
              "REGISTERS is selected, but no built `mma` accepts an operand from the "
              "issuing unit's registers (or no packed accumulator->operand conversion "
              "exists to feed it). Either build one or stop claiming the residency.");

static_assert(!(design::kBuiltResidencies & design::residency_bit(design::StateResidency::TMEM))
                  || kMmaAcceptsTmemA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency TMEM "
              "is claimed as built, but no built `mma` sources an operand from tensor "
              "memory.");

//: THE CARVE'S UNIT. Not a capability and not an architecture clause: `mma`,
//: `load_frag` and the d_v carve above are written for a 32-thread unit owning
//: a 16-row slice of the state ... -- see docs/internals/common/ops.md#static-assert-2
static_assert(arch::kCaps.mma_unit_threads == kMmaUnitThreads,
              "NOT BUILT FOR THIS ARCHITECTURE: no built implementation of "
              "`mma` and `load_frag` serves this architecture's MMA unit. The "
              "operations here are written for a 32-thread unit owning a 16-row "
              "slice of the state; a 128-thread warpgroup unit owns 64. Every "
              "structural clause holds -- this is an unwritten implementation, "
              "not a refused structure.");

//: THE ANNOUNCEMENT SITES. Each is a case where the capability row records a -- see docs/internals/common/ops.md#note-l562
inline constexpr design::Announce<arch::kCaps.async_bulk_copy && !kStagingIsBulk>
    kStagingAnnouncement{};

}  // namespace ops
}  // namespace denseref
