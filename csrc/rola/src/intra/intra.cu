// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0

#include "common/torch_seam.cuh"

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

void check_plane(const Tensor& t, const char* name, int64_t bh, int64_t L, int64_t width) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == Dtype::BFloat16, name,
                  " must be a contiguous CUDA bf16 tensor");
  STD_TORCH_CHECK(t.dim() == 3 && t.size(0) == bh && t.size(1) == L && t.size(2) == width, name,
                  " must be [BH, L, sum-of-widths]");
}

//: The frozen support word, in the producer's own storage: int32 bit patterns, one -- see docs/internals/intra/intra.md#checksupport
void check_support(const Tensor& t, const char* name, int64_t bh, int64_t L, int64_t width) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == Dtype::Int, name,
                  " must be a contiguous CUDA int32 tensor");
  STD_TORCH_CHECK(
      t.dim() == 3 && t.size(0) == bh && t.size(1) == (L + 31) / 32 && t.size(2) == width, name,
      " must be [BH, ceil(L/32), sum-of-widths]");
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
    const int optin = current_device_properties().sharedMemPerBlockOptin;
    STD_TORCH_CHECK(kSmem <= optin, "the intra arm's ", kSmem,
                    "-byte block footprint exceeds this device's ", optin, "-byte opt-in limit");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::intra_kernel<D, B, W>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    configured = true;
  }
  const dim3 grid((unsigned)windows, (unsigned)bh, 1);
  detail::intra_kernel<D, B, W><<<grid, detail::threads_of(W), kSmem, stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace

void intra_forward(const Tensor& pread, const Tensor& pwrite, const Tensor& gwrite, const Tensor& v,
                   const Tensor& sread, const Tensor& swrite, Tensor& o, Tensor& den,
                   int64_t levels, int64_t level_width, int64_t level_modes, int64_t window) {
  check_arch_table();
  const int64_t bh = pread.size(0);
  const int64_t L = pread.size(1);
  const int64_t width = levels * level_width;
  check_plane(pread, "pread", bh, L, width);
  check_plane(pwrite, "pwrite", bh, L, width);
  //: THE WORD AXIS IS 32 TOKENS, the producer's block (`entmax/factor.cu:236-239`), -- see docs/internals/intra/intra.md#checksupport-2
  check_support(sread, "sread", bh, L, width);
  check_support(swrite, "swrite", bh, L, width);
  STD_TORCH_CHECK(gwrite.is_cuda() && gwrite.is_contiguous()
                      && gwrite.scalar_type() == Dtype::BFloat16 && gwrite.dim() == 2
                      && gwrite.size(0) == bh && gwrite.size(1) == L,
                  "gwrite must be a contiguous CUDA bf16 [BH, L] tensor");
  STD_TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == Dtype::BFloat16
                      && v.dim() == 4 && v.size(3) == kValueWidth,
                  "v must be a contiguous CUDA bf16 [B, T, H, 64] tensor");
  const int64_t batch = v.size(0), tokens = v.size(1), heads = v.size(2);
  STD_TORCH_CHECK(batch * heads == bh && tokens == L,
                  "v's [B, T, H] must flatten to the plane's BH, L");
  STD_TORCH_CHECK(window > 0 && L % window == 0, "L must be a whole number of ", window,
                  "-token windows");
  STD_TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.scalar_type() == Dtype::Float
                      && o.dim() == 3 && o.size(0) == bh && o.size(1) == L
                      && o.size(2) == kValueWidth,
                  "o must be a contiguous CUDA fp32 [BH, L, 64] tensor");
  STD_TORCH_CHECK(den.is_cuda() && den.is_contiguous() && den.scalar_type() == Dtype::Float
                      && den.dim() == 2 && den.size(0) == bh && den.size(1) == L,
                  "den must be a contiguous CUDA fp32 [BH, L] tensor");
  STD_TORCH_CHECK(level_modes >= 0 && level_modes < (1LL << (2 * levels)),
                  "level_modes carries two bits per level and no more");

  const DeviceGuard guard(pread.get_device_index());
  detail::IntraParams p{
      reinterpret_cast<const __nv_bfloat16*>(pread.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(pwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      sread.mutable_data_ptr<int32_t>(),
      swrite.mutable_data_ptr<int32_t>(),
      o.mutable_data_ptr<float>(),
      den.mutable_data_ptr<float>(),
      (int)L,
      (int)width,
      (int)((L + 31) / 32),
      (int)heads,
      (int)tokens,
      (uint32_t)level_modes,
  };
  const cudaStream_t stream = current_stream();
#define INTRA_DISPATCH(D_, B_, W_)                                     \
  if (levels == (D_) && level_width == (B_) && window == (W_)) {       \
    launch<D_, B_, W_>(p, bh, L / (W_), stream);                       \
  } else
  INTRA_ARMS_X(INTRA_DISPATCH) { STD_TORCH_CHECK(false, "unbuilt intra arm"); }
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
  auto out = empty_cuda({1}, Dtype::Long);
  stampdet::stamp_kernel<<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  ROLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

}  // namespace intra
}  // namespace rola
