// csrc/rola/src/common/atoms/copy.cuh -- THE STAGED COPIES: global to shared without the issuing thread, their groups
// and waits, and the 16-byte quantum moved synchronously; a concern of the baseline operations (`common/ops.cuh`). see
// docs/internals/common/atoms/copy.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "common/atoms/address.cuh"

namespace denseref {
namespace ops {

//: Bytes moved by one `stage_tile`.  A FACT OF THE OPERATION, not of the -- see docs/internals/common/atoms/copy.md#kstagebytes
constexpr int kStageBytes = 16;

__device__ __forceinline__ void stage_tile(SmemAddr dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

//: THE SAME COPY WITH ITS SOURCE RUN PREDICATED: a dead quantum zero-fills and reads
//: nothing. -- see docs/internals/common/atoms/copy.md#stage-tile-if
__device__ __forceinline__ void stage_tile_if(SmemAddr dst, const void* src, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(live ? kStageBytes : 0));
}

//: THE COPY ISSUED ONLY WHEN `issue`, and zero-filled past its source where not `live`: a guard predicate on the
//: instruction (`@p LDGSTS`), never a branch, for a copy whose dead form must write nothing.
//: -- see docs/internals/common/atoms/copy.md#stage-tile-when
__device__ __forceinline__ void stage_tile_when(SmemAddr dst, const void* src, bool issue,
                                                bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n" ::
          "r"(dst),
      "l"(src), "r"(live ? kStageBytes : 0), "r"(static_cast<int>(issue)));
}

//: THE SAME COPY OF A RUN'S FIRST `bytes` (0 to 16), the rest zero-filled. -- see docs/internals/common/atoms/copy.md#stage-tile-n
__device__ __forceinline__ void stage_tile_n(SmemAddr dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

//: THE SAME COPY AT A NARROWER QUANTUM. -- see docs/internals/common/atoms/copy.md#stage-run
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

//: THE SAME NARROWER COPY WITH ITS SOURCE RUN PREDICATED. -- see docs/internals/common/atoms/copy.md#stage-run-if
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
//: docs/internals/common/atoms/copy.md#stage-run-lanes
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

template <int KeepInFlight>
__device__ __forceinline__ void stage_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(KeepInFlight));
}

//: the 16-byte quantum into shared memory, addressed.  The segment's dead rows are -- see docs/internals/common/atoms/copy.md#storesharedvec16
__device__ __forceinline__ void store_shared_vec16(SmemAddr a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

//: the 16-byte quantum back out, addressed: four words a lane.
__device__ __forceinline__ uint4 load_shared_vec16(SmemAddr a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}

//: The 16-byte quantum, synchronously. -- see docs/internals/common/atoms/copy.md#load-vec16
__device__ __forceinline__ uint4 load_vec16(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void store_vec16(void* p, const uint4& v) {
  *reinterpret_cast<uint4*>(p) = v;
}

__device__ __forceinline__ void copy_shared_to_global_16b(void* dst, const void* src) {
  store_vec16(dst, load_vec16(src));
}

}  // namespace ops
}  // namespace denseref
