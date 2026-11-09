// csrc/rola/src/scale/scale.cu -- THE SCALE PASS'S TRANSLATION UNIT: the host entry that validates the bundle's
// planes and the facts pass's `(F, m)`, allocates its outputs at the card's build, picks each side's form by its planes'
// dtypes and launches the three kernels. see docs/internals/scale/scale.md
#include "scale/scale_api.cuh"

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "scale/scale_kernel.cuh"

namespace rola::scale {

namespace detail {

//: A LEVEL'S ROWS as the kernels read them: `[B, T, H, width]` u8 (a sparse level) or f16 (a dense one), digits
//: contiguous, each row and stride on 16 bytes (a lane's digits are one 8- or 16-byte load).
//: -- see docs/internals/scale/scale_api.md#refusals
Plane plane_of(const Tensor& t, int64_t B, int64_t T, int64_t H, int64_t width, bool& f16,
               const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && (t.scalar_type() == Dtype::Byte || t.scalar_type() == Dtype::Half),
                  name, " is a CUDA uint8 or float16 tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(
      t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H && t.size(3) == width,
      name, " is [B, T, H, width] = [", B, ", ", T, ", ", H, ", ", width, "], got ",
      shape(t.sizes()));
  const int64_t size = t.element_size();
  STD_TORCH_CHECK(t.stride(3) == 1 && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0
                      && (t.stride(0) * size) % 16 == 0 && (t.stride(1) * size) % 16 == 0
                      && (t.stride(2) * size) % 16 == 0,
                  name, "'s rows start and stride on 16 bytes, digits contiguous");
  f16 = t.scalar_type() == Dtype::Half;
  return Plane{static_cast<const uint8_t*>(t.const_data_ptr()), t.stride(0) * size,
               t.stride(1) * size, t.stride(2) * size};
}

template <class Bt, bool SF16, bool RF16>
void writes(const ScaleParams& p, int64_t bh, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.windows), static_cast<unsigned>(bh));
  scale_writes<Bt, SF16, RF16><<<grid, ScalePlan<Bt::build>::kThreads, 0, stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

template <class Bt, bool SF16, bool RF16>
void reads(const ScaleParams& p, int64_t bq, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.windows), static_cast<unsigned>(bq));
  scale_reads<Bt, SF16, RF16><<<grid, ScalePlan<Bt::build>::kThreads,
                                static_cast<size_t>(p.width_s) * sizeof(float), stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

//: THE CALL AT BUILD `B`: its window's grid, the mass workspace and the bytes allocated on `device`, the three
//: launches, each side's form by its planes' dtypes. -- see docs/internals/scale/scale_api.md#scale-forward
template <::denseref::arch::Build B>
Tensor run(ScaleParams p, const bool (&f16)[4], int64_t BH, int device) {
  using Sp = ScalePlan<B>;
  using Bt = ::denseref::arch::BuildTag<B>;
  const int64_t windows = (p.L + Sp::kWindow - 1) / Sp::kWindow;
  STD_TORCH_CHECK(BH * p.heads <= 65535 && windows < (int64_t{1} << 31),
                  "the grid is (windows, heads) within the launch limits");
  const Tensor mass = empty_cuda(Shape({BH, windows, p.width_s}), Dtype::Float, device);
  Tensor bytes = empty_cuda(Shape({BH * p.heads, windows * Sp::kWindow}), Dtype::Char, device);
  p.mass = mass.mutable_data_ptr<float>();
  p.bytes = static_cast<int8_t*>(bytes.mutable_data_ptr());
  p.windows = static_cast<int>(windows);
  p.grid = static_cast<int>(windows * Sp::kWindow);

  const DeviceGuard guard(device);
  const cudaStream_t stream = current_stream();
  const bool ws16 = f16[0], wr16 = f16[1], rs16 = f16[2], rr16 = f16[3];
  // R9-CASES: each side's levels' dtypes, u8 (sparse) or f16 (dense)
  if (ws16 && wr16) {
    writes<Bt, true, true>(p, BH, stream);
  } else if (ws16) {
    writes<Bt, true, false>(p, BH, stream);
  } else if (wr16) {
    writes<Bt, false, true>(p, BH, stream);
  } else {
    writes<Bt, false, false>(p, BH, stream);
  }

  const dim3 scan(static_cast<unsigned>((p.width_s + Sp::kThreads - 1) / Sp::kThreads),
                  static_cast<unsigned>(BH));
  scale_scan<Bt><<<scan, Sp::kThreads, 0, stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();

  const int64_t bq = BH * p.heads;
  if (rs16 && rr16) {
    reads<Bt, true, true>(p, bq, stream);
  } else if (rs16) {
    reads<Bt, true, false>(p, bq, stream);
  } else if (rr16) {
    reads<Bt, false, true>(p, bq, stream);
  } else {
    reads<Bt, false, false>(p, bq, stream);
  }
  return bytes;
}

}  // namespace detail

//: -- see docs/internals/scale/scale_api.md#scale-forward
Tensor scale_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows, Tensor fm,
                     std::vector<int64_t> widths, int64_t level_s, int64_t heads,
                     double inv_factor) {
  check_arch_table();
  STD_TORCH_CHECK(widths.size() == 2 && read_rows.size() == 2 && write_rows.size() == 2,
                  "the scale pass is built for depth two: level S and level R; got ", widths.size(),
                  " levels");
  STD_TORCH_CHECK(level_s == 0 || level_s == 1, "level S is level 0 or 1, got ", level_s);
  const int64_t ls = level_s, lr = 1 - level_s;
  const Tensor& w0 = write_rows[0];
  STD_TORCH_CHECK(w0.dim() == 4, "a level's rows are [B, T, H, width]");
  const int64_t B = w0.size(0), L = w0.size(1), Hw = w0.size(2), BH = B * Hw;
  STD_TORCH_CHECK(heads >= 1 && L > 0, "one or more query heads a write head and tokens, got ",
                  heads, " and ", L);
  STD_TORCH_CHECK(
      widths[ls] % kLaneDigits == 0 && widths[lr] % kLaneDigits == 0 && widths[ls] * 4 <= 48 * 1024,
      "each level's width is a multiple of ", kLaneDigits,
      " digits and level S's masses fit 48 KB; got ", widths[ls], " and ", widths[lr]);

  ScaleParams p{};
  bool ws16 = false, wr16 = false, rs16 = false, rr16 = false;
  p.write_s =
      detail::plane_of(write_rows[ls], B, L, Hw, widths[ls], ws16, "the write side's level S rows");
  p.write_r =
      detail::plane_of(write_rows[lr], B, L, Hw, widths[lr], wr16, "the write side's level R rows");
  p.read_s = detail::plane_of(read_rows[ls], B, L, Hw * heads, widths[ls], rs16,
                              "the read side's level S rows");
  p.read_r = detail::plane_of(read_rows[lr], B, L, Hw * heads, widths[lr], rr16,
                              "the read side's level R rows");
  STD_TORCH_CHECK(fm.is_cuda() && fm.scalar_type() == Dtype::Float && fm.is_contiguous()
                      && fm.dim() == 3 && fm.size(0) == BH && fm.size(1) == L && fm.size(2) == 2,
                  "fm is the facts pass's contiguous fp32 [BH, L, 2]");
  p.fm = static_cast<const float2*>(fm.const_data_ptr());
  p.inv_factor = static_cast<float>(inv_factor);
  p.L = static_cast<int>(L);
  p.heads = static_cast<int>(heads);
  p.write_heads = static_cast<int>(Hw);
  p.width_s = static_cast<int>(widths[ls]);
  p.width_r = static_cast<int>(widths[lr]);

  const int device = w0.get_device_index();
  const bool f16[4] = {ws16, wr16, rs16, rr16};
  Tensor out;
  build_switch(card_of(device), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB)) out = detail::run<kB>(p, f16, BH, device);
  });
  return out;
}

}  // namespace rola::scale
