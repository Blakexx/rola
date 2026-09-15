// csrc/rola/src/common/geom_api.cuh -- the addressing block's HOST entries for the
// pybind TU. See docs/internals/common/geom_api.md
#pragma once

#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

std::vector<int64_t> geometry(std::vector<int64_t> widths, int64_t level_modes, int64_t bc,
                              int64_t nsr, int64_t nsw,
                              int64_t dv);  // the block, flattened; geom_api.md#geometry

std::vector<std::vector<int64_t>> sub_box_set(
    int64_t depth, int64_t box_leaves,
    int64_t workers);  // the generated shape set; geom_api.md#sub-box-set

}  // namespace rola::carry
