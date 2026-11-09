// csrc/rola/src/carry/carry_api.cuh -- the carry family's HOST entries for the pybind TU.
// see docs/internals/carry/carry_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::carry {

//: ONE LAUNCH over a call's windows. -- see docs/internals/carry/carry_api.md#carry-forward
Tensor carry_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows, Tensor values,
                     std::vector<Tensor> write_lists, Tensor write_exp, Tensor factors,
                     Tensor factor_shift, std::vector<Tensor> read_lists, Tensor scale, Tensor num,
                     Tensor den, Tensor check, std::vector<int64_t> widths,
                     std::vector<int64_t> extents, int64_t channels, int64_t write_list,
                     std::vector<int64_t> write_operand, int64_t read_list,
                     std::vector<int64_t> read_operand, std::vector<int64_t> key,
                     int64_t accumulate, int64_t reduce, int64_t forced);

int64_t carry_readout_accumulate(int64_t device);  // carry_api.md#carry-readout-accumulate

void carry_ledger_bind(const std::optional<Tensor>& ledger);
void carry_probe_bind(const std::optional<Tensor>& probe, const std::optional<Tensor>& den);
void carry_capture_bind(const std::optional<Tensor>& out);  // carry_api.md#carry-capture-bind
int64_t carry_capture_bytes();
int64_t carry_build_stamp();
std::vector<std::vector<int64_t>> carry_census();
std::vector<std::vector<int64_t>> carry_arms();

}  // namespace rola::carry
