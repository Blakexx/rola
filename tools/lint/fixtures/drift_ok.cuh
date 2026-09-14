// The compliant twin of drift_bad.cuh, for tools/lint/drift_guards.py. Not compiled.
#pragma once

namespace fixture {

template <int D, int DV, int WARPS_PER_CTA>
__global__ void carry_kernel(float* out);

__device__ void read_page(void* pages) {
  __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(pages);
  (void)row;
}

__device__ int kind_bits(int kind) {
  int bits = 0;
  uniform_switch<4>(kind, [&](auto k) { bits = decltype(k)::value; });
  return bits;
}

inline void admit_descriptor(int depth) {
  if (depth > 4) return;
}

__device__ void carve(const CarryGeomRT& g) { (void)g.bc; }

}  // namespace fixture
