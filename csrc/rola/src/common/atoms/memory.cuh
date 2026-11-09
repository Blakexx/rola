// csrc/rola/src/common/atoms/memory.cuh -- THE LOADS AND STORES: shared and global accesses addressed rather than
// indexed, plain and predicated in the instruction; a concern of the baseline operations (`common/ops.cuh`). see
// docs/internals/common/atoms/memory.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "common/atoms/address.cuh"
#include "common/atoms/math.cuh"

namespace denseref {
namespace ops {

__device__ __forceinline__ void store_bf16x2(__nv_bfloat16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}

//: a byte off shared memory, zero-extended (the fold ring's row).
__device__ __forceinline__ uint32_t load_shared_u8(SmemAddr a) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

//: a byte into shared memory, addressed (the fold's index maps).
__device__ __forceinline__ void store_shared_u8(SmemAddr a, uint8_t v) {
  asm volatile("st.shared.u8 [%0], %1;" ::"r"(a), "h"((uint16_t)v) : "memory");
}

//: A SEQUENTIAL SHARED STORE, ADDRESSED RATHER THAN INDEXED. -- see docs/internals/common/atoms/memory.md#store-shared-u16
__device__ __forceinline__ void store_shared_u16(SmemAddr a, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
}

//: A SHARED WORD ACCESS, ADDRESSED RATHER THAN INDEXED for the same reason -- see docs/internals/common/atoms/memory.md#storesharedu32
__device__ __forceinline__ void store_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ void store_shared_f32(SmemAddr a, float v) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}

//: THE SAME STORE PREDICATED on `live`: one `@p STS`, never a branch around it. -- see docs/internals/common/atoms/memory.md#store-shared-f32-if
__device__ __forceinline__ void store_shared_f32_if(SmemAddr a, float v, bool live) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p st.shared.f32 [%0], %1;\n}\n" ::"r"(a),
               "f"(v), "r"(static_cast<int>(live))
               : "memory");
}

//: A WORD, A PAIR AND AN fp32 LOADED where `live`, 0 elsewhere: one `@p LDS`, never a branch around it, so a lane
//: whose store is predicated off reads no word another lane writes. -- see docs/internals/common/atoms/memory.md#load-shared-if
__device__ __forceinline__ uint32_t load_shared_u32_if(SmemAddr a, bool live) {
  uint32_t v = 0u;
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p ld.shared.u32 %0, [%1];\n}\n"
               : "+r"(v)
               : "r"(a), "r"(static_cast<int>(live))
               : "memory");
  return v;
}

__device__ __forceinline__ uint2 load_shared_u64_if(SmemAddr a, bool live) {
  uint2 v = make_uint2(0u, 0u);
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p ld.shared.v2.u32 {%0, %1}, [%2];\n}\n"
               : "+r"(v.x), "+r"(v.y)
               : "r"(a), "r"(static_cast<int>(live))
               : "memory");
  return v;
}

__device__ __forceinline__ float load_shared_f32_if(SmemAddr a, bool live) {
  return __uint_as_float(load_shared_u32_if(a, live));
}

//: TWO fp32 AT ONCE, an 8-byte aligned pair. -- see docs/internals/common/atoms/memory.md#store-shared-f32x2
__device__ __forceinline__ void store_shared_f32x2(SmemAddr a, float x, float y) {
  asm volatile("st.shared.v2.f32 [%0], {%1,%2};" ::"r"(a), "f"(x), "f"(y) : "memory");
}

//: THE WORD AND THE PAIR PREDICATED on `live`, as `store_shared_f32_if`. -- see docs/internals/common/atoms/memory.md#store-shared-f32-if
__device__ __forceinline__ void store_shared_u32_if(SmemAddr a, uint32_t v, bool live) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p st.shared.u32 [%0], %1;\n}\n" ::"r"(a),
               "r"(v), "r"(static_cast<int>(live))
               : "memory");
}

__device__ __forceinline__ void store_shared_u64_if(SmemAddr a, uint32_t lo, uint32_t hi,
                                                    bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p st.shared.v2.u32 [%0], {%1, %2};\n}\n" ::"r"(a),
      "r"(lo), "r"(hi), "r"(static_cast<int>(live))
      : "memory");
}

__device__ __forceinline__ void store_shared_f32x2_if(SmemAddr a, float x, float y, bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p st.shared.v2.f32 [%0], {%1,%2};\n}\n" ::"r"(a),
      "f"(x), "f"(y), "r"(static_cast<int>(live))
      : "memory");
}

//: ONE 8-BYTE WORD PAIR from shared memory. -- see docs/internals/common/atoms/memory.md#load-shared-v4
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

//: ONE 16-BYTE QUANTUM from shared memory. -- see docs/internals/common/atoms/memory.md#load-shared-v4
__device__ __forceinline__ uint4 load_shared_v4(SmemAddr a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}

//: one amplitude out of a staged factor row, by ADDRESS.  Its index is a digit, so -- see docs/internals/common/atoms/memory.md#loadsharedu16
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

//: One element, rounded to nearest-even exactly as `pack_bf16x2` rounds each of -- see docs/internals/common/atoms/memory.md#store-bf16
__device__ __forceinline__ void store_bf16(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

//: One fp32 element, stored where it will later be read as a summand rather than -- see docs/internals/common/atoms/memory.md#storef32
__device__ __forceinline__ void store_f32(float* p, float v) { *p = v; }

}  // namespace ops
}  // namespace denseref
