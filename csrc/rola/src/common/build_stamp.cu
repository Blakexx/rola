// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// See docs/internals/common/build_stamp.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "common/build_stamp.cuh"
#include "csrc_stamp.inc"

namespace rola {

//: NOT an anonymous namespace: nvcc mangles such a kernel with a build-path hash.
namespace stampdet {

__global__ void csrc_stamp_kernel(int64_t* out) { *out = (int64_t)ROLA_CSRC_STAMP; }

}  // namespace stampdet

int64_t csrc_build_stamp() {
  auto out = at::empty({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  stampdet::csrc_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.cpu().item<int64_t>();
}

}  // namespace rola
