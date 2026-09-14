// A DELIBERATELY NON-COMPLIANT fixture for tools/lint/drift_guards.py.
// Every construct here is one the drift guards must report. Not compiled.
#pragma once

namespace fixture {

template <int D, int DV, int WARPS_PER_CTA, int B, int K>
__global__ void carry_kernel(float* out);

template <int D, int DV, int WARPS_PER_CTA>
__global__ void carry_split_kernel(float* out);

__device__ void read_page(void* pages) {
  float* row = reinterpret_cast<float*>(pages);
  (void)row;
}

__device__ int kind_bits(int kind) {
  switch (kind) {
    case 0:
      return 1;
    default:
      return 0;
  }
}

inline void admit_descriptor(int depth, bool is_decode) {
  if (is_decode && depth > 2) return;
}

__device__ void carve() {
  const auto set = sub_boxes(2, 256, 8);
  (void)set;
}

}  // namespace fixture
