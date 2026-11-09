// csrc/rola/src/common/atoms/clock.cuh -- THE CLOCK: the SM's cycle counter for the phase ledgers; a concern of the
// baseline operations (`common/ops.cuh`). see docs/internals/common/atoms/clock.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

namespace denseref {
namespace ops {

//: THE SM'S CYCLE COUNTER, for the phase ledger. -- see docs/internals/common/atoms/clock.md#cycles
__device__ __forceinline__ long long cycles() {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
  return c;
}

//: The counter, read only once `after` has arrived: a load placed first then issues before the read. -- see docs/internals/common/atoms/clock.md#cycles-after
__device__ __forceinline__ long long cycles_after(long long after) {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c) : "l"(after));
  return c;
}

}  // namespace ops
}  // namespace denseref
