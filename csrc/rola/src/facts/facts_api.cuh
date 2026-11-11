// csrc/rola/src/facts/facts_api.cuh -- the facts pass's HOST entry for the pybind TU. See
// docs/internals/facts/facts_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::facts {

void rola_facts_ledger_bind(std::optional<Tensor> ledger);  // facts_api.md#rola-facts-ledger-bind

std::vector<Tensor> rola_facts_forward(
    std::vector<Tensor> read, std::vector<Tensor> write, std::vector<Tensor> scale_write,
    Tensor v_scale, std::vector<int64_t> widths, int64_t dense_read, std::vector<int64_t> mask_read,
    int64_t dense_write, std::vector<int64_t> mask_write, int64_t write_list_level,
    int64_t read_list_level, int64_t read_list_group,
    int64_t write_list_group);  // facts_api.md#rola-facts-forward

}  // namespace rola::facts
