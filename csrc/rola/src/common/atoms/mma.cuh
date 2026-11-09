// csrc/rola/src/common/atoms/mma.cuh -- THE MMA ATOMS AND THEIR OPERANDS: the MMA's shape and unit, its fragment loads
// and the register transpose, and the bf16, f16 and int8 atoms; a concern of the baseline operations
// (`common/ops.cuh`). see docs/internals/common/atoms/mma.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "common/atoms/address.cuh"

namespace denseref {
namespace ops {

using Frag = uint32_t[4];

//: THE `memory` CLOBBER IS PART OF THE CONTRACT. -- see docs/internals/common/atoms/mma.md#load-frag
__device__ __forceinline__ void load_frag(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ void load_frag2(uint32_t (&r)[2], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ void load_frag_t(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

//: THE SAME OVER TWO TILES: lanes 0-15 address them. -- see docs/internals/common/atoms/mma.md#load-frag
__device__ __forceinline__ void load_frag_t2(uint32_t (&r)[2], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a)
               : "memory");
}

//: AN 8x8 b16 TILE, TRANSPOSED WHERE IT ALREADY LIES. -- see docs/internals/common/atoms/mma.md#transpose-frag-b16
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

//: THE ATOM IS CHOSEN BY SHAPE FAMILY, NOT BY NAME. -- see docs/internals/common/atoms/mma.md#mma
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

//: THE INT8 ATOM AT K = 32, u8 x u8 into s32 (A four registers, B two). -- see docs/internals/common/atoms/mma.md#imma-u8
__device__ __forceinline__ void imma_u8_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//: THE SAME INTO A FRESH ACCUMULATOR (C = 0), so the accumulator is not live before it.
__device__ __forceinline__ void imma_u8_k32_fresh(int32_t (&d)[4], const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

//: THE SAME, u8 A against s8 B: unsigned weights against signed values.
__device__ __forceinline__ void imma_u8s8_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                              uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//: the u8 x s8 atom into a fresh accumulator (C = 0).
__device__ __forceinline__ void imma_u8s8_k32_fresh(int32_t (&d)[4], const uint32_t (&a)[4],
                                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

//: THE fp16 ATOM, fp32 ACCUMULATE: acc += A(16x16) . B(16x8), f16 operands. -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                     uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//: THE fp16 ATOM, fp16 ACCUMULATE: acc += A(16x16) . B(16x8), f16 operands, the accumulator two f16 pairs (rows g and
//: g + 8, columns 2t and 2t + 1). -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//: THE SAME INTO A FRESH fp32 FRAGMENT (C = 0). -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma_fresh(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                           uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
}

//: THE fp16-ACCUMULATE ATOM INTO A FRESH FRAGMENT (C = 0). -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma_f16_fresh(uint32_t (&d)[2], const uint32_t (&a)[4],
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0u));
}

//: THE SAME AT K = 8: acc += A(16x8) . B(8x8), half the k16 atom's tensor work. -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

//: THE SAME AT K = 8 INTO AN fp16 ACCUMULATOR, two f16 pairs a lane. -- see docs/internals/common/atoms/mma.md#hmma
__device__ __forceinline__ void hmma_k8_f16(uint32_t (&d)[2], uint32_t a0, uint32_t a1,
                                            uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a0), "r"(a1), "r"(b));
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

}  // namespace ops
}  // namespace denseref
