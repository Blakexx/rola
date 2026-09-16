// csrc/rola/src/carry/carry_host.cuh -- THE CALL'S DERIVATION: the kernel boundary's
// refusals and the parameter block, one derivation shared by the launch and by the part
// harness (measure/harness/carry_parts), so a part runs on the call the kernel would.
// See docs/internals/carry/carry.md
#pragma once
#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

#include "carry/box.cuh"
#include "carry/params.cuh"
#include "common/arch_runtime.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"

namespace rola::carry {
namespace host {

//: THE PAGE RECTANGLE the descriptor may name -- one law, shared with decode.
constexpr int64_t kPageBits = kAtomLeavesBits;

//: THE KERNEL BOUNDARY'S SHAPE LAW (KERNEL_STANDARDS.md §R13): the strict descriptor and
//: nothing else. The seam above pads; this refuses. -- see docs/internals/carry/carry.md#refusals
inline void refuse_shape(const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                         int64_t warps_per_cta) {
  STD_TORCH_CHECK(page_bits == kPageBits, "the page rectangle is the trailing ", kPageBits,
                  " canonical leaf bits; this call names ", page_bits);
  STD_TORCH_CHECK(!widths.empty() && (int)widths.size() <= kGeomMaxLevels,
                  "the addressing block is built to depth ", kGeomMaxLevels, "; this call names ",
                  widths.size(), " levels");
  for (size_t l = 0; l < widths.size(); ++l)
    STD_TORCH_CHECK(widths[l] >= kAtomLeaves && (widths[l] & (widths[l] - 1)) == 0, "level ", l,
                    " is ", widths[l],
                    " digits wide. THE KERNEL DOES NOT PAD: B_l is a power of two at or above ",
                    kAtomLeaves, ".");
  STD_TORCH_CHECK(dv > 0 && dv % 16 == 0 && (dv & (dv - 1)) == 0,
                  "DV is a power of two at or above sixteen; this call names ", dv);
  STD_TORCH_CHECK(is_warps_per_cta((int)warps_per_cta), "warps_per_cta=", warps_per_cta,
                  " is not a launch shape this family builds");
}

//: THE SCHEDULE DIAL AS THE CALL SPELLS IT: the ORDER POLICY, one entry.
//: -- see docs/internals/carry/carry.md#schedule
inline void refuse_schedule(const std::vector<int64_t>& schedule) {
  STD_TORCH_CHECK(schedule.size() == 1,
                  "the schedule names the order policy -- [order] -- and this call passed ",
                  schedule.size(), " entries");
  STD_TORCH_CHECK(schedule[0] >= 0 && schedule[0] < kOrders,
                  "the order policy is 0 (first live box, dead last) or 1 (identity), got ",
                  schedule[0]);
}

//: THE CARVE ORDER AS THE CALL SPELLS IT: both sides' levels ranked, read side then write
//: side, rank 0 innermost. -- see docs/internals/carry/carry.md#carve-order
inline CarveOrder carve_order_of(const std::vector<int64_t>& carve_order, int D) {
  STD_TORCH_CHECK((int)carve_order.size() == 2 * D, "the carve order names each side's ", D,
                  " levels in rank order (read then "
                  "write), so it is ",
                  2 * D, " entries; this call names ", carve_order.size());
  CarveOrder o{};
  for (int s = 0; s < 2; ++s)
    for (int l = 0; l < kGeomMaxLevels; ++l)
      o.level_at[s][l] = l < D ? (int)carve_order[s * D + l] : l;
  return o;
}

inline void refuse_operand(const char* name, const Tensor& t, Dtype dtype, Shape want) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " is a contiguous CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " is ", dtype, ", got ", t.scalar_type());
  STD_TORCH_CHECK(t.sizes() == want, name, " is ", shape(want), ", got ", shape(t.sizes()));
}

//: A PAGED STATE: the pool a page table's slots index, `[pages, kAtomLeaves, dv + 1]` fp32, whose page count is the
//: caller's commitment and not a shape this call derives.
inline void refuse_pool(const char* name, const Tensor& t, int64_t dv) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " is a contiguous CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Float, name, " is ", Dtype::Float, ", got ",
                  t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 3 && t.size(0) >= 1 && t.size(1) == kAtomLeaves && t.size(2) == dv + 1,
                  name, " is a page pool [pages, ", kAtomLeaves, ", DV + 1] = [pages, ",
                  kAtomLeaves, ", ", dv + 1, "] with a page table, got ", shape(t.sizes()));
  STD_TORCH_CHECK(
      t.numel() * (int64_t)sizeof(float) < 0x100000000LL, name,
      " is larger than 4 GiB: a slot's byte offset is a 32-bit displacement off the pool's base");
}

}  // namespace host

//: THE CALL DERIVED: every refusal applied, the parameter block filled but for the ledger.
struct CarryCall {
  CarryParams p;
  int64_t BH;
};

inline CarryCall derive_carry_call(const Tensor& read, const Tensor& write, const Tensor& gain,
                                   const Tensor& v, Tensor& num, Tensor& den,
                                   const std::vector<int64_t>& widths, int64_t dv,
                                   int64_t page_bits, int64_t warps_per_cta,
                                   const std::vector<int64_t>& carve_order,
                                   const std::vector<int64_t>& schedule, const Tensor& liveness,
                                   const Tensor& activity, const std::optional<Tensor>& state_in,
                                   const std::optional<Tensor>& state_out,
                                   const std::optional<Tensor>& page_table) {
  check_arch_table();
  host::refuse_shape(widths, dv, page_bits, warps_per_cta);
  host::refuse_schedule(schedule);

  const int D = (int)widths.size();
  int64_t leaves = 1, wtot = 0;
  for (int l = 0; l < D; ++l) {
    leaves *= widths[l];
    wtot += widths[l];
  }
  const int64_t BH = read.size(0), L = read.size(1);
  const int64_t pages = leaves / kAtomLeaves;
  const int64_t words = (L + 31) / 32;

  host::refuse_operand("read", read, Dtype::BFloat16, {BH, L, wtot});
  host::refuse_operand("write", write, Dtype::BFloat16, {BH, L, wtot});
  host::refuse_operand("gain", gain, Dtype::BFloat16, {BH, L});
  host::refuse_operand("v", v, Dtype::BFloat16, {BH, L, dv});
  host::refuse_operand("num", num, Dtype::Float, {BH, L, dv});
  host::refuse_operand("den", den, Dtype::Float, {BH, L});
  host::refuse_operand("liveness", liveness, Dtype::Int, {BH, 2, wtot, words});
  host::refuse_operand("activity", activity, Dtype::Byte, {BH, pages});
  //: THE STATE'S SHAPE IS THE PAGE TABLE'S TO DECIDE. Without a table the slot IS the page, so the state is the whole
  //: plane, one page an atom of every stream. With one, the state is the POOL its slots index and its leading extent is
  //: a CAPACITY: the pages this call touches were committed before it ran (the pre-analysis reads the activity bits and
  //: allocates), so a pool holding fewer pages than the state has atoms is the point of paging, not an error.
  //: -- see docs/internals/state.md#four-shapes
  if (page_table) {
    host::refuse_operand("page_table", *page_table, Dtype::Int, {BH, pages});
    if (state_in) host::refuse_pool("state_in", *state_in, dv);
    if (state_out) host::refuse_pool("state_out", *state_out, dv);
  } else {
    const std::vector<int64_t> plane{BH, pages, (int64_t)kAtomLeaves, dv + 1};
    if (state_in) host::refuse_operand("state_in", *state_in, Dtype::Float, plane);
    if (state_out) host::refuse_operand("state_out", *state_out, Dtype::Float, plane);
  }
  STD_TORCH_CHECK(!state_in || !state_out || state_in->data_ptr() == state_out->data_ptr(),
                  "a carried state is advanced IN PLACE: the exit sweep stores only the pages this "
                  "call writes, so two planes would silently drop the rest");
  STD_TORCH_CHECK(L <= 0x7FFFFFFF && BH * L * wtot < 0x40000000,
                  "the packed routing plane's byte offsets are 32-bit displacements off one base");
  STD_TORCH_CHECK(L * dv < 0x20000000,
                  "the readout planes' byte offsets are 32-bit displacements off one base");

  const int BC = box_leaves((int)dv, (int)warps_per_cta);
  CarryParams p{};
  int w[kGeomMaxLevels] = {0, 0, 0, 0};
  for (int l = 0; l < D; ++l) w[l] = (int)widths[l];
  const char* err = derive_carry_geom(D, w, BC, (int)dv, host::carve_order_of(carve_order, D),
                                      (int)warps_per_cta, (int)warps_per_cta, p.g);
  STD_TORCH_CHECK(err == nullptr, "the carry addressing block refuses this call: ", err);
  p.lay[0] = make_side_layout(D, BC, p.g.r.rank, p.g.r.rank);
  p.lay[1] = make_side_layout(D, BC, p.g.w.rank, p.g.r.rank);

  p.pread = reinterpret_cast<const __nv_bfloat16*>(read.data_ptr());
  p.pwrite = reinterpret_cast<const __nv_bfloat16*>(write.data_ptr());
  p.gwrite = reinterpret_cast<const __nv_bfloat16*>(gain.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.num = num.mutable_data_ptr<float>();
  p.den = den.mutable_data_ptr<float>();
  p.state_in = state_in ? state_in->mutable_data_ptr<float>() : nullptr;
  p.state_out = state_out ? state_out->mutable_data_ptr<float>() : nullptr;
  p.page_table = page_table ? page_table->mutable_data_ptr<int32_t>() : nullptr;
  p.liveness = liveness.mutable_data_ptr<int32_t>();
  p.activity = activity.mutable_data_ptr<uint8_t>();
  p.L = (int)L;
  p.liveness_words = (int)words;
  p.order = (int)schedule[0];
  return CarryCall{p, BH};
}

}  // namespace rola::carry
