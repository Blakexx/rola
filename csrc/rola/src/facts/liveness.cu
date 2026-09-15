// csrc/rola/src/facts/liveness.cu -- the liveness pass's host TU: the seam's refusals,
// the operand block it builds, and the one launch. No fold lives here -- every class-2
// grain is a host fold over what this returns (rola/engine/facts/liveness.py).
// See docs/internals/facts/liveness_api.md

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "facts/liveness.cuh"
#include "facts/liveness_api.cuh"

namespace rola::facts {

namespace {

//: THE SEAM'S TRANSLATION, REFUSED RATHER THAN GUESSED: the descriptor's widths and the
//: two sides' static bits in, one operand block out.
//: -- see docs/internals/facts/liveness_api.md#the-plan
PassPlan build_plan(const std::vector<int64_t>& widths, int L, int64_t dense_read,
                    const std::vector<int64_t>& mask_read, int64_t dense_write,
                    const std::vector<int64_t>& mask_write) {
  const int D = (int)widths.size();
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the routing depth is one to four levels, got ", D);
  STD_TORCH_CHECK(L >= 1, "a call carries at least one token, got L=", L);

  PassPlan plan{};
  plan.D = D;
  plan.rows = 0;
  for (int level = 0; level < D; ++level) {
    const int64_t width = widths[level];
    STD_TORCH_CHECK(width >= 16 && width <= 256 && (width & (width - 1)) == 0, "level ", level,
                    " is ", width,
                    " digits wide. B_l is a POWER OF TWO AT OR ABOVE 16 -- the state's format "
                    "descriptor's law (KERNEL_STANDARDS.md §R13), which is what makes every "
                    "level's width and row base a whole number of the sixteen-digit groups "
                    "the pass reads per lane. There is no narrow path and no padding here.");
    plan.B[level] = (int)width;
    plan.rows += (int)width;
  }
  plan.L = L;
  plan.words = token_words(L);
  plan.side_words = (long)plan.rows * plan.words;

  const int need = mask_words(plan.rows);
  STD_TORCH_CHECK(need <= kMaxMaskWords, "the digit mask needs ", need, " words; the block holds ",
                  kMaxMaskWords);
  const int64_t modes[kSides] = {dense_read, dense_write};
  const std::vector<int64_t>* masks[kSides] = {&mask_read, &mask_write};
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(modes[side] >= 0 && modes[side] < (1 << D), "side ", side,
                    "'s dense-level bitmask ", modes[side], " names a level outside the depth ", D);
    STD_TORCH_CHECK((int)masks[side]->size() == need, "side ", side, " carries ",
                    masks[side]->size(), " digit-mask words; ", plan.rows, " rows need ", need);
    plan.dense[side] = (uint32_t)modes[side];
    for (int w = 0; w < need; ++w) plan.mask[side][w] = (uint32_t)(*masks[side])[w];
  }
  return plan;
}

const uint16_t* bits_of(const Tensor& plane) {
  return reinterpret_cast<const uint16_t*>(plane.const_data_ptr());
}

}  // namespace

//: -- see docs/internals/facts/liveness_api.md#liveness-words
Tensor liveness_words(const Tensor& read_plane, const Tensor& write_plane,
                      std::vector<int64_t> widths, int64_t dense_read,
                      std::vector<int64_t> mask_read, int64_t dense_write,
                      std::vector<int64_t> mask_write) {
  check_arch_table();
  for (const Tensor& plane : {read_plane, write_plane}) {
    STD_TORCH_CHECK(plane.is_cuda(), "the pass reads device amplitude planes");
    STD_TORCH_CHECK(plane.scalar_type() == Dtype::BFloat16, "the pass reads bf16 amplitudes, got ",
                    plane.scalar_type());
    STD_TORCH_CHECK(plane.is_contiguous(), "the amplitude plane is read as packed rows");
    STD_TORCH_CHECK(plane.dim() >= 2, "an amplitude plane is [..., L, sum_l B_l], got ",
                    plane.dim(), " dimensions");
  }
  STD_TORCH_CHECK(read_plane.sizes() == write_plane.sizes(),
                  "the two sides describe the SAME call: ", shape(read_plane.sizes()), " vs ",
                  shape(write_plane.sizes()));

  const int64_t rows = read_plane.size(read_plane.dim() - 1);
  const int64_t L = read_plane.size(read_plane.dim() - 2);
  const int64_t BH = read_plane.numel() / (rows * L);
  const PassPlan plan = build_plan(widths, (int)L, dense_read, mask_read, dense_write, mask_write);
  STD_TORCH_CHECK(plan.rows == rows, "the packed amplitude row is ", rows, " columns wide and the ",
                  plan.D, " widths sum to ", plan.rows,
                  ": a row index IS the column it votes from");

  Tensor out = empty_cuda({BH, (int64_t)kSides, rows, plan.words}, Dtype::Int,
                          read_plane.get_device_index());
  launch_liveness(bits_of(read_plane), bits_of(write_plane),
                  reinterpret_cast<uint32_t*>(out.mutable_data_ptr<int32_t>()), plan, (int)BH,
                  current_stream());
  return out;
}

}  // namespace rola::facts
