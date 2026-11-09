// csrc/rola/src/common/geom_api.cuh -- the addressing block's HOST entries for the
// pybind TU. See docs/internals/common/geom_api.md
#pragma once

#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

struct
    CarryGeomRT;  //: the block, complete in `common/geom.cuh`; the pybind TU never needs it whole

//: the tall owner set's block for a kernel family -- geom_api.md#tall-block
CarryGeomRT tall_block(const std::vector<int64_t>& widths, int64_t level_modes);

std::vector<int64_t> geometry(std::vector<int64_t> widths, int64_t level_modes, int64_t bc,
                              int64_t nsr, int64_t nsw, int64_t dv, int64_t interleave,
                              int64_t atom_floor);  // the block, flattened; geom_api.md#geometry

std::vector<std::vector<int64_t>> sub_box_set(
    int64_t depth, int64_t box_leaves, int64_t workers,
    int64_t atom_floor);  // the generated shape set; geom_api.md#sub-box-set

}  // namespace rola::carry
