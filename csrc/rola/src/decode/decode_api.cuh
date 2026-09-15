#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS -- the ONLY decode header
// `csrc/rola/rola_api.cpp` includes; see docs/internals/decode/decode_api.md#note-l3

#include "common/torch_seam.cuh"

#include <cstdint>
#include <vector>

namespace rola {
namespace decode {

int64_t rola_decode_capacity();  // the bound surface; see decode_api.md#rola-decode-capacity
int64_t rola_decode_producer_width_mirror();

int64_t rola_decode_residency(bool decay, int64_t levels);  // see decode_api.md#roladecoderesidency

int64_t
rola_decode_build_stamp();  // device-side build fact; see decode_api.md#roladecodebuildstamp

std::vector<std::vector<int64_t>> rola_decode_arms();  // see decode_api.md#roladecodearms

//: THE WHOLE STEP IN ONE LAUNCH -- fold, factor, walk, AND ITS OWN ADMISSION -- see docs/internals/decode/decode_api.md#note-l43
Tensor rola_decode_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                           std::vector<int64_t> normalize, std::optional<Tensor> dials,
                           Tensor g_write, Tensor v, Tensor state, Tensor ws, Tensor ctr,
                           Tensor growth, Tensor growth_any, Tensor growth_ctr, Tensor done,
                           Tensor atom_bits, std::vector<int64_t> level_widths,
                           std::vector<int64_t> level_row_offsets, int64_t lattice_k,
                           int64_t lattice_m, int64_t n_split, double eps,
                           std::optional<Tensor> page_table, std::optional<Tensor> pool_slots,
                           std::optional<Tensor> pool_cursor, std::optional<Tensor> pool_map);

}  // namespace decode
}  // namespace rola
