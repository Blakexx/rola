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

int64_t
rola_decode_build_stamp();  // device-side build fact; see decode_api.md#roladecodebuildstamp

std::vector<std::vector<int64_t>> rola_decode_arms();  // see decode_api.md#roladecodearms

//: binds the phase ledger the next steps book into, or unbinds it; see decode_api.md#decode-ledger-bind
void rola_decode_ledger_bind(const std::optional<Tensor>& ledger);

//: the phase ledger's warps at the current card's build, what a ledger is sized by; see decode_api.md#decode-ledger-warps
int64_t rola_decode_ledger_warps();

//: THE WHOLE STEP IN ONE LAUNCH -- fold, factor, walk, AND ITS OWN ADMISSION -- see docs/internals/decode/decode_api.md#note-l43
Tensor rola_decode_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                           std::vector<Tensor> read_scale, std::vector<Tensor> write_scale,
                           int64_t floor_read, int64_t floor_write, std::optional<Tensor> dials,
                           Tensor v, Tensor v_scale, Tensor state, Tensor yacc, Tensor queue,
                           Tensor growth, Tensor growth_any, Tensor growth_ctr, Tensor done,
                           Tensor atom_bits, std::vector<int64_t> level_widths,
                           std::vector<int64_t> level_row_offsets, int64_t lattice_k,
                           int64_t lattice_m, bool write_dense, int64_t ctas, double eps,
                           std::optional<Tensor> page_table, std::optional<Tensor> pool_slots,
                           std::optional<Tensor> pool_cursor);

}  // namespace decode
}  // namespace rola
