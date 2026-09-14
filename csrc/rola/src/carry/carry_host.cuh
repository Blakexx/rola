// csrc/rola/src/carry/carry_host.cuh -- THE CALL'S DERIVATION: the kernel boundary's
// refusals and the parameter block, one derivation shared by the launch and by the part
// harness (benchmarks/unit/carry_parts), so a part runs on the call the kernel would.
// See docs/internals/carry/carry.md
#pragma once
#include <torch/extension.h>

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
  TORCH_CHECK(page_bits == kPageBits, "the page rectangle is the trailing ", kPageBits,
              " canonical leaf bits; this call names ", page_bits);
  TORCH_CHECK(!widths.empty() && (int)widths.size() <= kGeomMaxLevels,
              "the addressing block is built to depth ", kGeomMaxLevels, "; this call names ",
              widths.size(), " levels");
  for (size_t l = 0; l < widths.size(); ++l)
    TORCH_CHECK(widths[l] >= kAtomLeaves && (widths[l] & (widths[l] - 1)) == 0, "level ", l, " is ",
                widths[l],
                " digits wide. THE KERNEL DOES NOT PAD: B_l is a power of two at or above ",
                kAtomLeaves, ".");
  TORCH_CHECK(dv > 0 && dv % 16 == 0 && (dv & (dv - 1)) == 0,
              "DV is a power of two at or above sixteen; this call names ", dv);
  TORCH_CHECK(is_warps_per_cta((int)warps_per_cta), "warps_per_cta=", warps_per_cta,
              " is not a launch shape this family builds");
}

//: THE SCHEDULE DIAL AS THE CALL SPELLS IT: the ORDER POLICY, one entry.
//: -- see docs/internals/carry/carry.md#schedule
inline void refuse_schedule(const std::vector<int64_t>& schedule) {
  TORCH_CHECK(schedule.size() == 1,
              "the schedule names the order policy -- [order] -- and this call passed ",
              schedule.size(), " entries");
  TORCH_CHECK(schedule[0] >= 0 && schedule[0] < kOrders,
              "the order policy is 0 (first live box, dead last) or 1 (identity), got ",
              schedule[0]);
}

//: THE CARVE ORDER AS THE CALL SPELLS IT: both sides' levels ranked, read side then write
//: side, rank 0 innermost. -- see docs/internals/carry/carry.md#carve-order
inline CarveOrder carve_order_of(const std::vector<int64_t>& carve_order, int D) {
  TORCH_CHECK((int)carve_order.size() == 2 * D, "the carve order names each side's ", D,
              " levels in rank order (read then "
              "write), so it is ",
              2 * D, " entries; this call names ", carve_order.size());
  CarveOrder o{};
  for (int s = 0; s < 2; ++s)
    for (int l = 0; l < kGeomMaxLevels; ++l)
      o.level_at[s][l] = l < D ? (int)carve_order[s * D + l] : l;
  return o;
}

inline void refuse_operand(const char* name, const at::Tensor& t, at::ScalarType dtype,
                           at::IntArrayRef want) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " is a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " is ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.sizes() == want, name, " is ", want, ", got ", t.sizes());
}

}  // namespace host

//: THE CALL DERIVED: every refusal applied, the parameter block filled but for the ledger.
struct CarryCall {
  CarryParams p;
  int64_t BH;
};

inline CarryCall derive_carry_call(
    const at::Tensor& read, const at::Tensor& write, const at::Tensor& gain, const at::Tensor& v,
    at::Tensor& num, at::Tensor& den, const std::vector<int64_t>& widths, int64_t dv,
    int64_t page_bits, int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
    const std::vector<int64_t>& schedule, const at::Tensor& liveness, const at::Tensor& activity,
    const c10::optional<at::Tensor>& state_in, const c10::optional<at::Tensor>& state_out,
    const c10::optional<at::Tensor>& page_table) {
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

  host::refuse_operand("read", read, at::kBFloat16, {BH, L, wtot});
  host::refuse_operand("write", write, at::kBFloat16, {BH, L, wtot});
  host::refuse_operand("gain", gain, at::kBFloat16, {BH, L});
  host::refuse_operand("v", v, at::kBFloat16, {BH, L, dv});
  host::refuse_operand("num", num, at::kFloat, {BH, L, dv});
  host::refuse_operand("den", den, at::kFloat, {BH, L});
  host::refuse_operand("liveness", liveness, at::kInt, {BH, 2, wtot, words});
  host::refuse_operand("activity", activity, at::kByte, {BH, pages});
  if (page_table) host::refuse_operand("page_table", *page_table, at::kInt, {BH, pages});
  const std::vector<int64_t> plane{BH, pages, (int64_t)kAtomLeaves, dv + 1};
  if (state_in) host::refuse_operand("state_in", *state_in, at::kFloat, plane);
  if (state_out) host::refuse_operand("state_out", *state_out, at::kFloat, plane);
  TORCH_CHECK(!state_in || !state_out || state_in->data_ptr() == state_out->data_ptr(),
              "a carried state is advanced IN PLACE: the exit sweep stores only the pages this "
              "call writes, so two planes would silently drop the rest");
  TORCH_CHECK(L <= 0x7FFFFFFF && BH * L * wtot < 0x40000000,
              "the packed routing plane's byte offsets are 32-bit displacements off one base");
  TORCH_CHECK(L * dv < 0x20000000,
              "the readout planes' byte offsets are 32-bit displacements off one base");

  const int BC = box_leaves((int)dv, (int)warps_per_cta);
  CarryParams p{};
  int w[kGeomMaxLevels] = {0, 0, 0, 0};
  for (int l = 0; l < D; ++l) w[l] = (int)widths[l];
  const char* err = derive_carry_geom(D, w, BC, (int)dv, host::carve_order_of(carve_order, D),
                                      (int)warps_per_cta, (int)warps_per_cta, p.g);
  TORCH_CHECK(err == nullptr, "the carry addressing block refuses this call: ", err);
  p.lay[0] = make_side_layout(D, BC, p.g.r.rank, p.g.r.rank);
  p.lay[1] = make_side_layout(D, BC, p.g.w.rank, p.g.r.rank);

  p.pread = reinterpret_cast<const __nv_bfloat16*>(read.data_ptr());
  p.pwrite = reinterpret_cast<const __nv_bfloat16*>(write.data_ptr());
  p.gwrite = reinterpret_cast<const __nv_bfloat16*>(gain.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.num = num.data_ptr<float>();
  p.den = den.data_ptr<float>();
  p.state_in = state_in ? state_in->data_ptr<float>() : nullptr;
  p.state_out = state_out ? state_out->data_ptr<float>() : nullptr;
  p.page_table = page_table ? page_table->data_ptr<int32_t>() : nullptr;
  p.liveness = liveness.data_ptr<int32_t>();
  p.activity = activity.data_ptr<uint8_t>();
  p.L = (int)L;
  p.liveness_words = (int)words;
  p.order = (int)schedule[0];
  return CarryCall{p, BH};
}

}  // namespace rola::carry
