// carry_arm_abi.cuh -- ONE ARM'S ABI, no torch headers. see docs/internals/carry/carry_arm_abi.md
#pragma once

#include <cuda_runtime.h>

#include "carry/carry.cuh"

namespace rola::carry {

//: ONE ARM'S FORM on the current device's card. -- see docs/internals/carry/carry_arm_abi.md#form
struct CarryForm {
  int smem_bytes, area, read_row, write_row, read_list_group, write_list_group;
};

//: ONE ARM'S CENSUS, as the device build reports it. -- see docs/internals/carry/carry_arm_abi.md#census
struct ArmCensus {
  int regs, local_bytes, max_threads, binary_version;
  CarryForm form;
};

#define ROLA_CARRY_ARM_DECLARE(I_, FR_, FF_, RR_, RF_, CH_, SS_)                                  \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream); \
  ArmCensus carry_arm_census_##I_();

}  // namespace rola::carry
