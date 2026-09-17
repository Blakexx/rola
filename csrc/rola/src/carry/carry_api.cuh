// csrc/rola/src/carry/carry_api.cuh -- the carry family's HOST entries for the pybind TU.
// See docs/internals/carry/carry.md
#pragma once
#include <optional>

#include "common/torch_seam.cuh"

#include <cstdint>
#include <vector>

namespace rola::carry {

void carry_forward(const Tensor& read, const Tensor& write, const Tensor& gain, const Tensor& v,
                   Tensor& num, Tensor& den, const std::vector<int64_t>& widths, int64_t dv,
                   int64_t page_bits, int64_t warps_per_cta,
                   const std::vector<int64_t>& carve_order, const std::vector<int64_t>& schedule,
                   const Tensor& liveness, const Tensor& activity,
                   const std::optional<Tensor>& state_in, const std::optional<Tensor>& state_out,
                   const std::optional<Tensor>& page_table);

//: the phase ledger binding (a debug instrument; None unbinds). -- carry_kernel.md#phase-ledger
void carry_ledger_bind(const std::optional<Tensor>& ledger);
//: the phase trace binding, `[ctas][warps][cap]` (a debug instrument; None unbinds). -- carry_kernel.md#phase-trace
void carry_trace_bind(const std::optional<Tensor>& trace, int64_t warps_per_cta);
int64_t carry_build_stamp();
std::vector<std::vector<int64_t>> carry_census();
std::vector<std::vector<int64_t>> carry_arms();

}  // namespace rola::carry
