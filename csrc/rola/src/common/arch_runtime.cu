// csrc/rola/src/common/arch_runtime.cu -- the runtime half of the closed-world
// arch rule (setup.py's R4), beside arch_caps.cuh's compile-time half. The single
// definition every launching entry calls first, once per device per process.
// See docs/internals/common/arch_runtime.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <mutex>
#include <set>

#include "common/arch_caps.cuh"
#include "common/arch_runtime.cuh"

namespace rola {

//: THE RUNTIME HALF OF THE CLOSED-WORLD ARCH RULE (setup.py's R4). -- see docs/internals/common/arch_runtime.md#check-arch-table
void check_arch_table() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  //: ONCE PER PROCESS PER DEVICE, on the first launch rather than at import: a
  //: capability query on every call would be wasted work once the device is known
  //: good, and the mutex-guarded set is what makes "once" true under concurrent
  //: launches from multiple host threads.
  static std::mutex mu;
  static std::set<int> admitted;
  {
    const std::lock_guard<std::mutex> hold(mu);
    if (admitted.count(dev)) return;
  }

  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int cc = prop.major * 100 + prop.minor * 10;
  TORCH_CHECK(denseref::arch::tabulated(cc),
              "the capability table has no row for compute capability ", prop.major, ".",
              prop.minor,
              "; add it to caps_of() rather than letting an operation guess which "
              "implementation it has. This binary is closed-world: it runs the "
              "architectures it was measured on and refuses the rest.");

  const denseref::arch::Caps want = denseref::arch::caps_of(cc);
  TORCH_CHECK(denseref::arch::supported(want), "UNSUPPORTED ARCHITECTURE at compute capability ",
              prop.major, ".", prop.minor,
              ": a REQUIRED capability clause of the ratified baseline is false in its own "
              "row. There is no partial support and no slower route to take.");

  int smem_per_sm = 0, smem_per_cta_max = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_cta_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(want.smem_per_sm == smem_per_sm && want.smem_per_cta_max == smem_per_cta_max
                  && want.regs_per_sm == prop.regsPerMultiprocessor
                  && want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor,
              "the capability table disagrees with the driver at compute capability ", prop.major,
              ".", prop.minor, ": table {smem/SM ", want.smem_per_sm, ", smem/CTA ",
              want.smem_per_cta_max, ", regs/SM ", want.regs_per_sm, ", threads/SM ",
              want.max_threads_per_sm, "} vs driver {smem/SM ", smem_per_sm, ", smem/CTA ",
              smem_per_cta_max, ", regs/SM ", prop.regsPerMultiprocessor, ", threads/SM ",
              prop.maxThreadsPerMultiProcessor,
              "}. Every launch shape compiled into this binary was derived from the "
              "table, so they are wrong for this device.");

  const std::lock_guard<std::mutex> hold(mu);
  admitted.insert(dev);
}

}  // namespace rola
