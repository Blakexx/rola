// csrc/rola/src/intra/ampere/intra_partial.cuh -- THE PARTIAL CALL'S LAUNCH, its arms in their own translation unit.
// see docs/internals/intra/ampere/intra_partial.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/hardware.cuh"
#include "intra/intra.cuh"

namespace rola::intra::detail {

template <::denseref::arch::Build B, int D, bool kF16>
cudaError_t launch_partial(
    const IntraParams& p, int pieces, int threads, int64_t smem, int64_t bh,
    cudaStream_t stream);  // intra_partial.md: the launch's error, never thrown

}  // namespace rola::intra::detail
