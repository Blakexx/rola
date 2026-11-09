// csrc/rola/src/intra/intra_api.cuh -- the intra's HOST entries for the pybind TU. See
// docs/internals/intra/intra_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::intra {

void rola_intra_forward(std::vector<Tensor> read, std::vector<Tensor> write, Tensor values,
                        Tensor fm, Tensor unions, Tensor num, Tensor den,
                        std::vector<int64_t> widths, int64_t dense_read, int64_t dense_write,
                        int64_t accumulate);  // intra_api.md#rola-intra-forward

void rola_intra_ledger_bind(std::optional<Tensor> ledger);  // intra_api.md#rola-intra-ledger-bind

int64_t intra_footprint(std::vector<int64_t> widths, std::vector<int64_t> forms,
                        int64_t device);  // intra_api.md#intra-footprint

int64_t intra_block_rows(int64_t device);  // intra_api.md#intra-block-rows

int64_t intra_build_stamp();  // intra_api.md#intra-build-stamp

}  // namespace rola::intra
