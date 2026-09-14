// csrc/rola/src/carry/carry_arm_abi.cuh -- ONE ARM'S ABI: pointers and scalars only, so a
// per-arm translation unit never parses the torch headers the seam needs.
// See docs/internals/carry/carry.md
#pragma once

#include <cuda_runtime.h>

#include "carry/params.cuh"

namespace rola::carry {

//: One arm's census, as the device build reports it -- see docs/internals/carry/carry.md#census
struct ArmCensus {
  int regs;
  int spill_stores;
  int spill_loads;
  int local_bytes;
  int smem_bytes;
  int max_threads;
  int binary_version;
  int threads;
  int box_leaves;
  int atoms;
  int boxes;
  int state_elems;
  int read_bytes;
  int pool_bytes;
};

#define ROLA_CARRY_ARM_DECLARE(I_, D_, DV_, W_)                                      \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int owners, int bh,        \
                                     cudaStream_t stream);                            \
  ArmCensus carry_arm_census_##I_();

}  // namespace rola::carry
