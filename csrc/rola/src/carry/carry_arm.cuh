// csrc/rola/src/carry/carry_arm.cuh -- THE LAUNCH of one arm: the grid the carve implies, the shared memory its plan
// sizes, and the census the device build reports. Included only by the generated per-arm translation units.
// see docs/internals/carry/carry_arm.md
#pragma once

#include "carry/carry_arm_abi.cuh"
#include "carry/ampere/carry_kernel.cuh"
#include "common/arm.cuh"
#include "common/select.cuh"

namespace rola::carry {

template <class P>
constexpr CarryForm form_of_plan() {
  return CarryForm{P::kSmemBytes, P::kArea,          P::kReadRow,
                   P::kWriteRow,  P::kReadListGroup, P::kWriteListGroup};
}

//: THE ARM'S LAUNCH at build `B`: the call's read tables at its plan, the plan pass, the kernel at the plan's shared
//: bytes. Its kernels exist in `B`'s image alone (`in_image`). -- see docs/internals/carry/carry_arm.md#plan
template <arch::Build B, class Arm>
inline cudaError_t carry_launch_on(const CarryParams& p0, int boxes, int bh, cudaStream_t stream) {
  if constexpr (in_image(B)) {
    using P = Plan<Arm, B>;
    constexpr int bytes = P::kSmemBytes;
    //: the call's read tables on the device's card, and an area that holds the least plan
    CarryParams p = p0;
    p.tables = tables_of<P::kWindow>(p0.tables.heads, bytes);
    if (!least_plan_fits<P>(p.tables.area, p.reduce)) return cudaErrorInvalidConfiguration;
    const int classes = (p.read_lists.groups > 0 ? p.read_lists.groups : 1)
                        * (p.write_lists.groups > 0 ? p.write_lists.groups : 1);
    window_plan_kernel<arch::BuildTag<B>, Arm>
        <<<dim3((unsigned)classes, (unsigned)bh), (unsigned)kPlanThreads, 0, stream>>>(p);
    const auto fn = carry_kernel<arch::BuildTag<B>, Arm>;
    static bool configured = false;
    if (!configured) {
      const cudaError_t e =
          cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      if (e != cudaSuccess) return e;
      configured = true;
    }
    fn<<<dim3((unsigned)boxes, (unsigned)bh), (unsigned)kThreads, bytes, stream>>>(p);
    return cudaGetLastError();
  } else {
    return cudaErrorInvalidDeviceFunction;
  }
}

//: THE ARM'S LAUNCH on the current device's card (`common/select.cuh`).
template <class Arm>
inline cudaError_t carry_launch(const CarryParams& p, int boxes, int bh, cudaStream_t stream) {
  cudaError_t status = cudaErrorInvalidDevice;
  build_switch(current_card(), [&](auto b) {
    status = carry_launch_on<decltype(b)::build, Arm>(p, boxes, bh, stream);
  });
  return status;
}

//: THE ARM'S CENSUS on the current device's card: its kernel's attributes and its plan's form.
template <class Arm>
inline ArmCensus carry_census() {
  ArmCensus out{};
  build_switch(current_card(), [&](auto b) {
    constexpr arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB)) {
      cudaFuncAttributes a{};
      cudaFuncGetAttributes(&a, carry_kernel<arch::BuildTag<kB>, Arm>);
      out = ArmCensus{(int)a.numRegs, (int)a.localSizeBytes, (int)a.maxThreadsPerBlock,
                      (int)a.binaryVersion, form_of_plan<Plan<Arm, kB>>()};
    }
  });
  return out;
}

#define ROLA_CARRY_ARM_DEFINE(I_, FR_, FF_, RR_, RF_, CH_, SS_)                                      \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream) { \
    return carry_launch<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, SS_>>(p, boxes, bh, stream);        \
  }                                                                                                  \
  ArmCensus carry_arm_census_##I_() { return carry_census<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, SS_>>(); }

}  // namespace rola::carry
