// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// See docs/internals/common/build_stamp.md

#include "common/torch_seam.cuh"

#include "common/build_stamp.cuh"
#include "csrc_stamp.inc"

namespace rola {

namespace stampdet {

__global__ void csrc_stamp_kernel(int64_t* out) { *out = (int64_t)ROLA_CSRC_STAMP; }

}  // namespace stampdet

int64_t csrc_build_stamp() {
  auto out = empty_cuda({1}, Dtype::Long);
  stampdet::csrc_stamp_kernel<<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  ROLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

}  // namespace rola
