// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// See docs/internals/common/sm_clock.md

#include "common/arch_runtime.cuh"
#include "common/torch_seam.cuh"

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
  check_arch_table();
  auto out = zeros_cuda({3}, Dtype::Long);
  const int sms = current_device_properties().multiProcessorCount;
  clockdet::sm_clock_kernel<<<sms, 128, 0, current_stream()>>>(
      (long long)spin_cycles,
      reinterpret_cast<unsigned long long*>(out.mutable_data_ptr<int64_t>()));
  ROLA_CUDA_LAUNCH_CHECK();
  const auto h = to_cpu(out);
  const double cycles = (double)h.const_data_ptr<int64_t>()[0],
               ns = (double)h.const_data_ptr<int64_t>()[1];
  return ns > 0 ? cycles / ns : 0.0;
}

}  // namespace rola
