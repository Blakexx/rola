// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// See docs/internals/common/sm_clock.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace rola {

//: NOT an anonymous namespace: nvcc mangles such a kernel with a build-path hash.
namespace clockdet {

//: THE SM'S EFFECTIVE CLOCK, read off the device under load: every SM spins a dependent
//: chain for a fixed count of its own cycles and CTA 0 reports its cycles against the
//: global timer. The driver's reported clock is not this number on this host.
__global__ void sm_clock_kernel(long long spin, unsigned long long* out) {
  const long long c0 = clock64();
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  float x = (float)threadIdx.x;
  long long c1;
  do {
    x = fmaf(x, 1.0000001f, 0.5f);
    c1 = clock64();
  } while (c1 - c0 < spin);
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = (unsigned long long)(c1 - c0);
    out[1] = t1 - t0;
  }
  if (x == -1.0f) out[2] = 1;  // keeps the chain live; never taken
}

}  // namespace clockdet

double sm_clock_ghz(int64_t spin_cycles) {
  auto out = at::zeros({3}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  clockdet::sm_clock_kernel<<<sms, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (long long)spin_cycles, reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto h = out.cpu();
  const double cycles = (double)h[0].item<int64_t>(), ns = (double)h[1].item<int64_t>();
  return ns > 0 ? cycles / ns : 0.0;
}

}  // namespace rola
