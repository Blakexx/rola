// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "common/arch_runtime.cuh"
#include "intra/intra_kernel.cuh"

namespace rola {
namespace intra {

//: NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a -- see docs/internals/intra/intra.md#note-l12
namespace stampdet {

//: Recompiled with the translation unit, so a test that reads it back is -- see docs/internals/intra/intra.md#gstampprobe
__device__ int64_t g_stamp_probe;

//: THE HEAVIEST BUILT ARM's footprint and the launch ABI's own width: the pair -- see docs/internals/intra/intra.md#stampkernel
__global__ void stamp_kernel(int64_t* out) {
  *out = ((int64_t)(sizeof(detail::IntraSmem<2, kWidthFlat, kWindowWide>)) * 1000003
          + (int64_t)detail::threads_of(kWindowWide))
             * 1009
         + (int64_t)sizeof(detail::IntraParams);
  g_stamp_probe = *out;
}

}  // namespace stampdet

namespace {

void check_plane(const at::Tensor& t, const char* name, int64_t bh, int64_t L, int64_t width) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16, name,
              " must be a contiguous CUDA bf16 tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(0) == bh && t.size(1) == L && t.size(2) == width, name,
              " must be [BH, L, sum-of-widths]");
}

//: The frozen support word, in the producer's own storage: int32 bit patterns, one -- see docs/internals/intra/intra.md#checksupport
void check_support(const at::Tensor& t, const char* name, int64_t bh, int64_t L, int64_t width) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt, name,
              " must be a contiguous CUDA int32 tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(0) == bh && t.size(1) == (L + 31) / 32 && t.size(2) == width,
              name, " must be [BH, ceil(L/32), sum-of-widths]");
}

//: THE BUILT SET, as one macro so the dispatch and the arm list cannot drift -- -- see docs/internals/intra/intra.md#intraarmsx
#define INTRA_ARMS_X(X_)                     \
  X_(2, kWidthFlat, kWindowFlagship)         \
  X_(2, kWidthFlat, kWindowWide)             \
  X_(2, kWidthFlat, kWindowSmall)            \
  X_(3, kWidthDeep3, kWindowSmall)           \
  X_(4, kWidthDeep4, kWindowSmall)

//: The block's whole reader side is resident, which puts the footprint past the -- see docs/internals/intra/intra.md#near-line-64
template <int D, int B, int W>
void launch(const detail::IntraParams& p, int64_t bh, int64_t windows, cudaStream_t stream) {
  constexpr int kSmem = (int)sizeof(detail::IntraSmem<D, B, W>);
  //: ONCE per instantiation: the attribute call is a millisecond of host time on the launch
  //: path, paid every call when it sat here unguarded (2026-09-08).
  static bool configured = false;
  if (!configured) {
    const int optin = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
    TORCH_CHECK(kSmem <= optin, "the intra arm's ", kSmem,
                "-byte block footprint exceeds this device's ", optin, "-byte opt-in limit");
    C10_CUDA_CHECK(cudaFuncSetAttribute(detail::intra_kernel<D, B, W>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    configured = true;
  }
  const dim3 grid((unsigned)windows, (unsigned)bh, 1);
  detail::intra_kernel<D, B, W><<<grid, detail::threads_of(W), kSmem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, const at::Tensor& sread, const at::Tensor& swrite,
                   at::Tensor& o, at::Tensor& den, int64_t levels, int64_t level_width,
                   int64_t level_modes, int64_t window) {
  check_arch_table();
  const int64_t bh = pread.size(0);
  const int64_t L = pread.size(1);
  const int64_t width = levels * level_width;
  check_plane(pread, "pread", bh, L, width);
  check_plane(pwrite, "pwrite", bh, L, width);
  //: THE WORD AXIS IS 32 TOKENS, the producer's block (`entmax/factor.cu:236-239`), -- see docs/internals/intra/intra.md#checksupport-2
  check_support(sread, "sread", bh, L, width);
  check_support(swrite, "swrite", bh, L, width);
  TORCH_CHECK(gwrite.is_cuda() && gwrite.is_contiguous() && gwrite.scalar_type() == at::kBFloat16
                  && gwrite.dim() == 2 && gwrite.size(0) == bh && gwrite.size(1) == L,
              "gwrite must be a contiguous CUDA bf16 [BH, L] tensor");
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kBFloat16 && v.dim() == 4
                  && v.size(3) == kValueWidth,
              "v must be a contiguous CUDA bf16 [B, T, H, 64] tensor");
  const int64_t batch = v.size(0), tokens = v.size(1), heads = v.size(2);
  TORCH_CHECK(batch * heads == bh && tokens == L,
              "v's [B, T, H] must flatten to the plane's BH, L");
  TORCH_CHECK(window > 0 && L % window == 0, "L must be a whole number of ", window,
              "-token windows");
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.scalar_type() == at::kFloat && o.dim() == 3
                  && o.size(0) == bh && o.size(1) == L && o.size(2) == kValueWidth,
              "o must be a contiguous CUDA fp32 [BH, L, 64] tensor");
  TORCH_CHECK(den.is_cuda() && den.is_contiguous() && den.scalar_type() == at::kFloat
                  && den.dim() == 2 && den.size(0) == bh && den.size(1) == L,
              "den must be a contiguous CUDA fp32 [BH, L] tensor");
  TORCH_CHECK(level_modes >= 0 && level_modes < (1LL << (2 * levels)),
              "level_modes carries two bits per level and no more");

  const at::cuda::CUDAGuard guard(pread.device());
  detail::IntraParams p{
      reinterpret_cast<const __nv_bfloat16*>(pread.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(pwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      sread.data_ptr<int32_t>(),
      swrite.data_ptr<int32_t>(),
      o.data_ptr<float>(),
      den.data_ptr<float>(),
      (int)L,
      (int)width,
      (int)((L + 31) / 32),
      (int)heads,
      (int)tokens,
      (uint32_t)level_modes,
  };
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define INTRA_DISPATCH(D_, B_, W_)                                     \
  if (levels == (D_) && level_width == (B_) && window == (W_)) {       \
    launch<D_, B_, W_>(p, bh, L / (W_), stream);                       \
  } else
  INTRA_ARMS_X(INTRA_DISPATCH) { TORCH_CHECK(false, "unbuilt intra arm"); }
#undef INTRA_DISPATCH
}

std::vector<std::vector<int64_t>> intra_arms() {
  std::vector<std::vector<int64_t>> out;
#define INTRA_LIST(D_, B_, W_)                                                 \
  out.push_back({(int64_t)(D_), (int64_t)(B_), (int64_t)(W_),                  \
                 (int64_t)sizeof(detail::IntraSmem<D_, B_, W_>)});
  INTRA_ARMS_X(INTRA_LIST)
#undef INTRA_LIST
  return out;
}

int64_t intra_build_stamp() {
  auto out = at::empty({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  stampdet::stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.cpu().item<int64_t>();
}

}  // namespace intra
}  // namespace rola
