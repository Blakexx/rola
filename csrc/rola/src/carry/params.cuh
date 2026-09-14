// csrc/rola/src/carry/params.cuh -- THE PARAMETER BLOCK, handed to the kernel as ONE value.
// See docs/internals/carry/params.md
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "carry/layout.cuh"
#include "common/geom.cuh"

namespace rola::carry {

struct CarryParams {
  //: the addressing, derived on the host, and each side's layout (read, write).
  CarryGeomRT g;
  SideLayout lay[2];

  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* v;
  float* num;
  float* den;
  const float* state_in;
  float* state_out;
  const int32_t* page_table;
  const int32_t* liveness;
  const uint8_t* activity;

  int L;               //: tokens in the call
  int liveness_words;  //: ceil(L / 32), the class-1 block's word stride
  //: THE ORDER POLICY. -- carry_kernel.md#order
  int order;
  //: THE PHASE LEDGER, `[cta][warp][kPhases]` int64, or null. -- carry_kernel.md#phase-ledger
  long long* ledger;
};

//: THE PHASES the ledger counts; the head's two edges split its row. -- carry_kernel.md#phase-ledger
enum CarryPhase : int {
  kPhaseHead = 0,
  kPhaseReadout = 1,
  kPhaseFold = 2,
  kPhaseSnapshot = 3,
  kPhaseEdges = 4,
  kPhaseSweep = 5,
  kPhaseHeadWords = 6,
  kPhaseHeadScans = 7,
  kPhases = 8
};

//: THE ORDER POLICIES. -- carry_kernel.md#order
constexpr int kOrderFirstBox = 0;
constexpr int kOrderIdentity = 1;
constexpr int kOrders = 2;

}  // namespace rola::carry
