// csrc/rola/src/common/atoms/reduce.cuh -- a concern of `common/ops.cuh`. see docs/internals/common/atoms/reduce.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "common/atoms/address.cuh"

namespace denseref {
namespace ops {

//: A SHARED COUNTER'S TAKE -- see docs/internals/common/atoms/reduce.md#atom-shared-add
__device__ __forceinline__ uint32_t atom_shared_add_u32(SmemAddr a, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(a), "r"(v) : "memory");
  return old;
}

//: AN fp32 GLOBAL REDUCTION, fire-and-forget, its address global -- see docs/internals/common/atoms/reduce.md#red-global
__device__ __forceinline__ void red_global_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(__cvta_generic_to_global(p)), "f"(v)
               : "memory");
}

//: an fp32 shared-memory add, the CTA's own.
__device__ __forceinline__ void red_shared_add_f32(SmemAddr a, float v) {
  asm volatile("red.shared.add.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}

//: ONE CONTRIBUTION INTO A SHARED BITSET, FIRE AND FORGET -- see docs/internals/common/atoms/reduce.md#or-shared-u32
__device__ __forceinline__ void or_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.or.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: THE SAME OVER THE COUNTING MONOID. -- see docs/internals/common/atoms/reduce.md#redsharedaddu32
__device__ __forceinline__ void red_shared_add_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: ONE CONTRIBUTION INTO A SHARED ACCUMULATOR, FIRE AND FORGET. -- see docs/internals/common/atoms/reduce.md#fan-in-add
__device__ __forceinline__ void fan_in_add(float* p, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(x) : "memory");
}

}  // namespace ops
}  // namespace denseref
