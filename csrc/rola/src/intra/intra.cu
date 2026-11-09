// THE INTRA'S TRANSLATION UNIT -- the call's geometry (its levels' widths, forms, stages and footprint) on the device's
// card, the host entry that validates the bundle's views and launches the kernel, the phase-ledger bind and the build
// stamp. see docs/internals/intra/intra.md

#include "common/torch_seam.cuh"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/window.cuh"
#include "intra/intra_api.cuh"
#include "intra/ampere/intra_kernel.cuh"
#include "intra/ampere/intra_partial.cuh"

namespace rola::intra {

//: NOT an anonymous namespace: a kernel's mangled name must not carry the build path.
namespace stampdet {

//: THE BUILT KERNEL's block rows and fixed tenants and the operand block's size at build `Bt::build`, read back by a
//: test as the device-side build fact.
template <class Bt>
__global__ void stamp_kernel(int64_t* out) {
  using Wk = detail::Walk<Bt::build>;
  *out = (static_cast<int64_t>(Wk::kBlockRows) * 1000003
          + static_cast<int64_t>(sizeof(typename Wk::Smem)))
             * 1000003
         + static_cast<int64_t>(sizeof(IntraParams));
}

}  // namespace stampdet

namespace detail {

//: A BUILD'S WALK, read on the host at the device's build: the plan its device code compiles.
//: -- see docs/internals/intra/intra.md#block
struct Shape {
  int window, rows, warps, pieces, threads;
  int64_t tenants;
};

template <arch::Build B>
Shape shape_at() {
  using Wk = Walk<B>;
  return Shape{Wk::kWindow,  Wk::kBlockRows,
               Wk::kWarps,   Wk::Pieces::count(),
               Wk::kThreads, static_cast<int64_t>(sizeof(typename Wk::Smem))};
}

Shape shape_of(int device) {
  Shape out{};
  build_switch(card_of(device), [&](auto b) { out = shape_at<decltype(b)::build>(); });
  return out;
}

//: THE CALL'S GEOMETRY from its levels' widths and forms: each level's slabs and stages, the reader row's slots, pitch
//: and the footprint. -- see docs/internals/intra/intra.md#stages
struct Geometry {
  Stage stage[kMaxStages];
  int slab0[kMaxLevels], level_stage0[kMaxLevels], level_stages[kMaxLevels];
  uint32_t level_slabs[kMaxLevels];
  int stages, slabs, pitch;
  int64_t smem;
};

Geometry geometry(const std::vector<int64_t>& widths, const std::vector<int64_t>& forms,
                  const Shape& card) {
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels && static_cast<int>(forms.size()) == D,
                  "the intra reads one or two levels, a form each; got ", D);
  Geometry g{};
  int chunk = 0;
  for (int l = 0; l < D; ++l) {
    const int64_t w = widths[l];
    const int form = static_cast<int>(forms[l]);
    STD_TORCH_CHECK(w >= kDigitGranule && w <= kMaxLevelWidth && w % kDigitGranule == 0, "level ",
                    l, " is ", w, " digits wide; the intra reads a level of ", kDigitGranule,
                    " to ", kMaxLevelWidth, " digits in whole ", kDigitGranule, "-digit granules");
    STD_TORCH_CHECK(form >= 0 && form < kForms, "level ", l, "'s form is ", form);
    const int slabs = static_cast<int>((w + kSlabDigits - 1) / kSlabDigits);
    g.slab0[l] = g.slabs;
    g.level_slabs[l] = ((1u << slabs) - 1u) << g.slabs;
    const int digits = form_stage_digits(form);
    const int slot = digits * (form_read_f16(form) ? 2 : 1) / kChunk;
    g.level_stage0[l] = g.stages;
    for (int d0 = 0; d0 < w; d0 += digits) {
      const int n =
          static_cast<int>((w - d0 < digits ? w - d0 : digits) + kSlabDigits - 1) / kSlabDigits;
      g.stage[g.stages] = Stage{g.stages, l, d0, g.slabs + d0 / kSlabDigits, (1u << n) - 1u, chunk};
      ++g.stages;
      chunk += slot;
    }
    g.level_stages[l] = g.stages - g.level_stage0[l];
    g.slabs += slabs;
  }
  //: at least three stages' slots, and room for every warp's deposit staging tile
  const int staging = (card.warps * kStagingBytes + card.rows - 1) / card.rows;
  const int least = 3 * kStageBytes > staging ? 3 * kStageBytes : staging;
  const int bytes = ((chunk * kChunk > least ? chunk * kChunk : least) + kStageBytes - 1)
                    / kStageBytes * kStageBytes;
  g.pitch = bytes;
  g.smem = card.tenants + static_cast<int64_t>(card.rows) * g.pitch;
  return g;
}

//: a level's form from its rows' dtypes: u8 or f16 a side
int form_of(const Tensor& read, const Tensor& write, int level) {
  for (const Tensor* t : {&read, &write})
    STD_TORCH_CHECK(t->scalar_type() == Dtype::Byte || t->scalar_type() == Dtype::Half, "level ",
                    level, "'s rows are u8 (sparse) or f16 (dense), got ", t->scalar_type());
  return (read.scalar_type() == Dtype::Half ? 1 : 0) | (write.scalar_type() == Dtype::Half ? 2 : 0);
}

RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, int64_t width,
                 const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " is a CUDA tensor");
  STD_TORCH_CHECK(
      t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H && t.size(3) == width,
      name, " is [B, T, H, width] = [", B, ", ", T, ", ", H, ", ", width, "], got ",
      shape(t.sizes()));
  const int64_t bytes = t.element_size();
  STD_TORCH_CHECK(t.stride(3) == 1 && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kChunk == 0
                      && (t.stride(0) * bytes) % kChunk == 0 && (t.stride(1) * bytes) % kChunk == 0
                      && (t.stride(2) * bytes) % kChunk == 0,
                  name, "'s rows start and stride on 16 bytes, digits contiguous");
  STD_TORCH_CHECK(
      t.stride(1) > 0 && t.stride(1) * bytes <= INT32_MAX, name,
      "'s token stride fits 31 bits of bytes (a copy adds its rows by one 32-bit stride)");
  return RowView{t.const_data_ptr(), t.stride(0) * bytes, t.stride(1) * bytes, t.stride(2) * bytes};
}

//: THE DIGIT UNITS: a u8 digit enters an MMA as `d / 16` (fp16 accumulate), `d 2^-24` (fp32, and the floor trick's
//: writer) or itself (a u8 x u8 level's IMMA), its unit value `d / 255`;
//: level 0's Gram is lifted to digit units, and the rest of every level's scale rides `unit`.
//: -- see docs/internals/intra/ampere/intra_kernel.md#widening
void units(const std::vector<int64_t>& forms, bool f16, float& lift, float& unit) {
  const double digit = f16 ? 16.0 : 16777216.0;
  double u = 1.0;
  for (size_t l = 0; l < forms.size(); ++l) {
    const int u8_sides =
        !form_read_f16(static_cast<int>(forms[l])) + !form_write_f16(static_cast<int>(forms[l]));
    //: a u8 x u8 level's IMMA Gram is the digits' own integers; the floor trick's (an f16 reader, a u8 writer)
    //: carries `2^-24` from the digit and `2^15` from the reader; every other u8 side its widening's `digit`
    const int form = static_cast<int>(forms[l]);
    const double level_lift = form == kFormU8U8    ? 1.0
                              : form == kFormF16U8 ? 16777216.0 / 32768.0
                                                   : std::pow(digit, u8_sides);
    if (l == 0)
      lift = static_cast<float>(level_lift);
    else
      u *= level_lift;
    u /= std::pow(255.0, u8_sides);
  }
  unit = static_cast<float>(u);
}

//: a call of whole windows on the arm `kEven`, one whose last window is partial on the arm `!kEven`
template <arch::Build B, int D, bool kF16>
void launch(const IntraParams& p, const Shape& card, int64_t smem, int64_t bh, bool even,
            cudaStream_t stream) {
  //: the attribute call is host time on the launch path: raised only when a call's footprint passes the last one set,
  //: each kernel's own
  static int64_t configured[2] = {0, 0};
  if (!even) {
    const cudaError_t e =
        launch_partial<B, D, kF16>(p, card.pieces, card.threads, smem, bh, stream);
    ROLA_CUDA_CHECK(e);
    configured[0] = smem > configured[0] ? smem : configured[0];
    return;
  }
  if (smem > configured[1]) {
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(intra_kernel<arch::BuildTag<B>, D, kF16, true>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
    configured[1] = smem;
  }
  const dim3 grid(static_cast<unsigned>(p.windows * card.pieces), static_cast<unsigned>(bh));
  intra_kernel<arch::BuildTag<B>, D, kF16, true>
      <<<grid, static_cast<unsigned>(card.threads), static_cast<size_t>(smem), stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

//: THE BOUND PHASE LEDGER, or null.
long long* g_ledger = nullptr;

}  // namespace detail

//: -- see docs/internals/intra/intra_api.md#rola-intra-ledger-bind
void rola_intra_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 2
                      && ledger->size(0) == kLedgerWarps && ledger->size(1) == kIntraPhases
                      && ledger->is_contiguous(),
                  "the intra's phase ledger is a contiguous CUDA int64 tensor [", kLedgerWarps,
                  "][", kIntraPhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
}

//: -- see docs/internals/intra/intra_api.md#rola-intra-forward
void rola_intra_forward(std::vector<Tensor> read, std::vector<Tensor> write, Tensor values,
                        Tensor fm, Tensor unions, Tensor num, Tensor den,
                        std::vector<int64_t> widths, int64_t dense_read, int64_t dense_write,
                        int64_t accumulate) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels && static_cast<int>(read.size()) == D
                      && static_cast<int>(write.size()) == D,
                  "each side one row set a level, against the layout's ", D, " levels");
  STD_TORCH_CHECK(accumulate >= -1 && accumulate <= 1,
                  "the intra accumulates in fp32 (0), fp16 (1) or the device's rule (-1), got ",
                  accumulate);
  STD_TORCH_CHECK(read[0].is_cuda() && read[0].dim() == 4,
                  "a level's rows are CUDA [B, T, H, width]");
  const int device = read[0].get_device_index();
  const DeviceGuard guard(device);
  const detail::Shape card = detail::shape_of(device);
  const bool f16 = accumulate < 0 ? readout_f16(device) : accumulate == 1;
  const int64_t B = read[0].size(0), T = read[0].size(1), H = read[0].size(2);
  STD_TORCH_CHECK(write[0].dim() == 4, "a level's write rows are [B, T, H_w, width]");
  const int64_t Hw = write[0].size(2);
  STD_TORCH_CHECK(Hw >= 1 && H % Hw == 0, "the write heads ", Hw, " divide the query heads ", H);
  STD_TORCH_CHECK(T >= 1, "a call carries at least one token");
  std::vector<int64_t> row_widths(D), forms(D);
  for (int l = 0; l < D; ++l) {
    row_widths[l] = read[l].dim() == 4 ? read[l].size(3) : 0;
    STD_TORCH_CHECK(row_widths[l] <= widths[l], "level ", l, "'s rows are ", row_widths[l],
                    " digits wide, past the layout's ", widths[l]);
    forms[l] = detail::form_of(read[l], write[l], l);
  }
  const detail::Geometry g = detail::geometry(row_widths, forms, card);

  IntraParams p{};
  facts::LivenessLayout layout{D, {}, static_cast<int>(T)};
  for (int l = 0; l < D; ++l) layout.B[l] = static_cast<int>(widths[l]);
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::row_view(read[l], B, T, H, row_widths[l], "read[l]");
    p.write[l] = detail::row_view(write[l], B, T, Hw, row_widths[l], "write[l]");
    p.width[l] = static_cast<int>(row_widths[l]);
    p.form[l] = static_cast<int>(forms[l]);
    p.slab0[l] = g.slab0[l];
    p.level_slabs[l] = g.level_slabs[l];
    p.group0[l] = layout.group_base(l);
    p.level_stage0[l] = g.level_stage0[l];
    p.level_stages[l] = g.level_stages[l];
  }
  for (int s = 0; s < g.stages; ++s) p.stage[s] = g.stage[s];
  p.stages = g.stages;
  p.slabs = g.slabs;
  p.groups = layout.groups();
  p.pitch = g.pitch;
  STD_TORCH_CHECK(values.scalar_type() == Dtype::Half, "values are the f16 plane, got ",
                  values.scalar_type());
  p.values = detail::row_view(values, B, T, Hw, kValueWidth, "values");
  STD_TORCH_CHECK(
      fm.is_cuda() && fm.scalar_type() == Dtype::Float && fm.is_contiguous() && fm.dim() == 3
          && fm.size(0) == B * Hw && fm.size(1) == T && fm.size(2) == 2,
      "fm is the facts pass's contiguous fp32 [", B * Hw, ", ", T, ", 2] factor products");
  p.fm = reinterpret_cast<const float2*>(fm.const_data_ptr<float>());
  p.H = static_cast<int>(H);
  p.group = static_cast<int>(H / Hw);
  p.L = static_cast<int>(T);
  //: a partial last window runs its tiles before the call's end
  p.windows = static_cast<int>((T + card.window - 1) / card.window);
  detail::units(forms, f16, p.lift, p.unit);
  STD_TORCH_CHECK(
      dense_read >= 0 && dense_read < (1 << D) && dense_write >= 0 && dense_write < (1 << D),
      "the dense-level bitmasks name levels inside the depth");
  p.dense[0] = static_cast<uint32_t>(dense_read);
  p.dense[1] = static_cast<uint32_t>(dense_write);
  STD_TORCH_CHECK(unions.is_cuda() && unions.scalar_type() == Dtype::Int && unions.is_contiguous()
                      && unions.dim() == 4 && unions.size(0) == B * Hw
                      && unions.size(1) == H / Hw + 1 && unions.size(2) == layout.groups()
                      && unions.size(3) == layout.tiles(),
                  "unions is the facts pass's [", B * Hw, ", ", H / Hw + 1, ", ", layout.groups(),
                  ", ", layout.tiles(), "] int32 tile-union table");
  p.unions = reinterpret_cast<const uint32_t*>(unions.const_data_ptr<int32_t>());
  STD_TORCH_CHECK(num.is_cuda() && num.scalar_type() == Dtype::Float && num.is_contiguous()
                      && num.numel() == B * H * T * kValueWidth,
                  "num is the contiguous fp32 [BH, L, ", kValueWidth, "] output plane");
  STD_TORCH_CHECK(den.is_cuda() && den.scalar_type() == Dtype::Float && den.is_contiguous()
                      && den.numel() == B * H * T,
                  "den is the contiguous fp32 [BH, L] output plane");
  p.num = num.mutable_data_ptr<float>();
  p.den = den.mutable_data_ptr<float>();
  p.ledger = detail::g_ledger;

  const cudaStream_t stream = current_stream();
  const bool even = T % card.window == 0;
  build_switch(card_of(device), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB)) {
      if (D == 1)
        f16 ? detail::launch<kB, 1, true>(p, card, g.smem, B * H, even, stream)
            : detail::launch<kB, 1, false>(p, card, g.smem, B * H, even, stream);
      else
        f16 ? detail::launch<kB, 2, true>(p, card, g.smem, B * H, even, stream)
            : detail::launch<kB, 2, false>(p, card, g.smem, B * H, even, stream);
    }
  });
}

//: -- see docs/internals/intra/intra_api.md#intra-footprint
int64_t intra_footprint(std::vector<int64_t> widths, std::vector<int64_t> forms, int64_t device) {
  return detail::geometry(widths, forms, detail::shape_of(static_cast<int>(device))).smem;
}

//: -- see docs/internals/intra/intra_api.md#intra-block-rows
int64_t intra_block_rows(int64_t device) { return detail::shape_of(static_cast<int>(device)).rows; }

//: -- see docs/internals/intra/intra_api.md#intra-build-stamp
int64_t intra_build_stamp() {
  auto out = empty_cuda({1}, Dtype::Long);
  build_switch(current_card(), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB))
      stampdet::stamp_kernel<::denseref::arch::BuildTag<kB>>
          <<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  });
  ROLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

}  // namespace rola::intra
