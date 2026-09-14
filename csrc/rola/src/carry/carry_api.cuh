// csrc/rola/src/carry/carry_api.cuh -- the carry family's HOST entries for the pybind TU.
// See docs/internals/carry/carry.md
#pragma once
#include <optional>

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

void carry_forward(const at::Tensor& read, const at::Tensor& write, const at::Tensor& gain,
                   const at::Tensor& v, at::Tensor& num, at::Tensor& den,
                   const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                   int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
                   const std::vector<int64_t>& schedule, const at::Tensor& liveness,
                   const at::Tensor& activity, const c10::optional<at::Tensor>& state_in,
                   const c10::optional<at::Tensor>& state_out,
                   const c10::optional<at::Tensor>& page_table);

//: the phase ledger binding (a debug instrument; None unbinds). -- carry_kernel.md#phase-ledger
void carry_ledger_bind(const std::optional<at::Tensor>& ledger);
int64_t carry_build_stamp();
std::vector<std::vector<int64_t>> carry_census();
std::vector<std::vector<int64_t>> carry_arms();

}  // namespace rola::carry
