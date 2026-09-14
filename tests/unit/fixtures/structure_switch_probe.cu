// tests/unit/fixtures/structure_switch_probe.cu -- the toy kernel that exercises
// `rola::uniform_switch`'s four layers, including the trapping default.
// Driven by tests/unit/test_structure_switch.py; not part of the extension.
#include <cstdio>
#include <cstring>

#include "common/structure_switch.cuh"

namespace {

constexpr int kNA = 5;

//: THE GENERATED SET this toy switches over: member `a`'s value is a function of `a`
//: alone, so a body that ran under the wrong member is visible in the output.
__host__ __device__ constexpr int member_value(int a) { return 1 << (2 * a); }

//: EVERY LANE WRITES ITS OWN WORD, so the driver can assert the branch was UNIFORM:
//: all 32 lanes of the warp ran the same member and differ only by their lane id.
__global__ void probe(const int* sel, int* out, int n) {
  for (int i = 0; i < n; ++i) {
    rola::uniform_switch<kNA>(sel[i], [&](auto Ac) {
      constexpr int a = decltype(Ac)::value;
      out[i * 32 + (int)threadIdx.x] = member_value(a) + (int)threadIdx.x;
    });
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool trap = argc > 1 && std::strcmp(argv[1], "trap") == 0;
  const int n = trap ? 1 : kNA;
  int* sel = nullptr;
  int* out = nullptr;
  cudaMallocManaged(&sel, sizeof(int) * n);
  cudaMallocManaged(&out, sizeof(int) * n * 32);
  for (int i = 0; i < n; ++i) sel[i] = trap ? kNA : i;
  for (int i = 0; i < n * 32; ++i) out[i] = -1;
  probe<<<1, 32>>>(sel, out, n);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("LAUNCH_ERROR %s\n", cudaGetErrorName(err));
    return 2;
  }
  for (int i = 0; i < n; ++i)
    for (int lane = 0; lane < 32; ++lane)
      std::printf("%d %d %d\n", sel[i], lane, out[i * 32 + lane]);
  return 0;
}
