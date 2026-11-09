// THE EXTENSION'S OPERATOR REGISTRATION -- the only TU that registers with torch (host-only, so a `.cpp` rather than
// recompiled per `-gencode` in a `.cu`). Through torch's STABLE ABI: every entry is an operator in the `rola` namespace
// (`torch.ops.rola.<entry>`), declared by schema and implemented by a boxed function, so the binary depends on torch's
// stable C shim and never on its C++ ABI. The module object `rola_cu13._C` is only the handle an import loads the library by.
// Full entry inventory, the schemas and the VMM owner's handle protocol: see docs/internals/rola_api.md

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/build_stamp.cuh"
#include "common/select.cuh"
#include "common/torch_seam.cuh"
#include "src/carry/carry_api.cuh"
#include "src/common/geom_api.cuh"
#include "src/decode/decode_api.cuh"
#include "src/entmax/entmax.cuh"
#include "src/entmax/factor.cuh"
#include "src/scale/scale_api.cuh"
#include "src/facts/facts_api.cuh"
#include "src/intra/intra_api.cuh"
#include "src/paging/vmm_owner.cuh"
#include "src/produce/produce_api.cuh"

using rola::check_arch_table;
using rola::Tensor;

namespace {

// THE VMM OWNERS, by handle: an operator cannot return a native object, so the owner lives here and the caller holds
// an integer. A tensor view of an owner keeps the owner alive past its release (VmmOwner::base).
std::mutex g_owners_mutex;
std::unordered_map<int64_t, std::shared_ptr<rola::paging::VmmOwner>> g_owners;
int64_t g_next_owner = 1;

std::shared_ptr<rola::paging::VmmOwner> owner_of(int64_t handle) {
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  const auto it = g_owners.find(handle);
  STD_TORCH_CHECK(it != g_owners.end(), "no RoLA VMM owner holds handle ", handle,
                  " (released, or never created)");
  return it->second;
}

int64_t vmm_create(int64_t device, int64_t dense_limit_pages, int64_t page_rows, int64_t page_cols,
                   int64_t target_chunk_bytes) {
  auto owner = rola::paging::create_vmm_owner((int)device, dense_limit_pages, page_rows, page_cols,
                                              target_chunk_bytes);
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  g_owners.emplace(g_next_owner, std::move(owner));
  return g_next_owner++;
}

void vmm_release(int64_t handle) {
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  g_owners.erase(handle);
}

Tensor vmm_base(int64_t handle) { return owner_of(handle)->base(); }

void vmm_grow(int64_t handle, int64_t required_pages) { owner_of(handle)->grow(required_pages); }

void vmm_rollback_to(int64_t handle, int64_t mapped_capacity_pages) {
  owner_of(handle)->rollback_to(mapped_capacity_pages);
}

void vmm_reset(int64_t handle) { owner_of(handle)->reset(); }

void vmm_close(int64_t handle) { owner_of(handle)->close(); }

//: closed, mapped_capacity_pages, dense_limit_pages, committed_bytes, virtual_bytes, allocation_granularity, chunk_pages
std::vector<int64_t> vmm_facts(int64_t handle) {
  const auto owner = owner_of(handle);
  return {(int64_t)owner->closed(),
          owner->mapped_capacity_pages(),
          owner->dense_limit_pages(),
          (int64_t)owner->committed_bytes(),
          (int64_t)owner->virtual_bytes(),
          (int64_t)owner->allocation_granularity(),
          owner->chunk_pages()};
}

std::tuple<int64_t, bool, int64_t, std::string> vmm_probe(int64_t device) {
  const auto probe = rola::paging::VmmOwner::probe((int)device);
  return {probe.device, probe.supported, (int64_t)probe.allocation_granularity, probe.reason};
}

//: A DEVICE'S CARD (`common/select.cuh`): its compute capability, product row (-1: none), build variant, family and W
std::vector<int64_t> device_card_op(int64_t device) {
  const rola::Card& c = rola::card_of(static_cast<int>(device));
  return {c.cc, c.product, c.build.variant, static_cast<int64_t>(c.traits.family), c.traits.window};
}

}  // namespace

STABLE_TORCH_LIBRARY(rola, m) {
  // ---- the intra: the within-window pairs over the bundle -------------------
  m.def(
      "rola_intra_forward(Tensor[] read, Tensor[] write, Tensor values, Tensor fm, Tensor unions, Tensor(a!) num, "
      "Tensor(b!) den, int[] widths, int dense_read, int dense_write, int accumulate) -> ()");
  m.def("rola_intra_ledger_bind(Tensor? ledger) -> ()");
  m.def("intra_build_stamp() -> int");
  m.def("intra_footprint(int[] widths, int[] forms, int device) -> int");
  m.def("intra_block_rows(int device) -> int");

  // ---- the CARRY LAUNCH SURFACE --------------------------------------------
  m.def(
      "carry_forward(Tensor[] read_rows, Tensor[] write_rows, Tensor values, Tensor[] write_lists, Tensor write_exp, "
      "Tensor factors, Tensor factor_shift, Tensor[] read_lists, "
      "Tensor scale, Tensor(a!) num, Tensor(d!) den, Tensor(b!) check, int[] widths, int[] extents, int channels, "
      "int write_list, int[] write_operand, int read_list, int[] read_operand, int[] key, int accumulate, int reduce, "
      "int forced) -> Tensor");
  m.def("carry_readout_accumulate(int device) -> int");
  m.def("device_card(int device) -> int[]");
  m.def(
      "segment_tokens(int budget, int state, int per_token, int batch, int length, int device) -> int");
  m.def("carry_ledger_bind(Tensor? ledger) -> ()");
  m.def("carry_probe_bind(Tensor? probe, Tensor? den) -> ()");
  m.def("carry_capture_bind(Tensor? out) -> ()");
  m.def("carry_capture_bytes() -> int");
  m.def("carry_build_stamp() -> int");
  m.def("carry_census() -> int[][]");
  m.def("carry_arms() -> int[][]");
  m.def("csrc_build_stamp() -> int");
  m.def("sm_clock_ghz(int spin_cycles=20000000) -> float");

  // ---- the addressing block ------------------------------------------------
  m.def(
      "carry_geometry(int[] widths, int level_modes, int bc, int nsr, int nsw, int dv, int interleave, int atom_floor) -> int[]");
  m.def("carry_sub_boxes(int depth, int box_leaves, int workers, int atom_floor) -> int[][]");

  // ---- the facts pass: the bundle's rows to the class-1 words and the lists --
  m.def(
      "rola_facts_forward(Tensor[] read, Tensor[] write, Tensor[] scale_write, Tensor v_scale, "
      "int[] widths, int dense_read, int[] mask_read, int dense_write, int[] mask_write, int write_list_level, "
      "int read_list_level, int read_row, int read_list_group, int write_list_group) -> Tensor[]");
  m.def("rola_facts_ledger_bind(Tensor? ledger) -> ()");

  // ---- the scale pass: each token's scale byte for the carry's f16 sum, the write-aligned bound --
  m.def(
      "scale_forward(Tensor[] read_rows, Tensor[] write_rows, Tensor fm, int[] widths, int level_s, int heads, "
      "float inv_factor) -> Tensor");

  // ---- the producer: one launch from the tokens to the bundle -------------
  m.def(
      "rola_produce_forward(Tensor x, Tensor route_W, Tensor? route_bias, Tensor gain_bias, Tensor v_W, Tensor jobs, "
      "Tensor(a!) amp_read, Tensor(b!) amp_write, Tensor(o!)? dense_read, Tensor(n!)? dense_write, Tensor(c!) scale_read, "
      "Tensor(d!) scale_write, Tensor(i!) v, Tensor(j!) v_scale, Tensor(l!)? logit_probe, Tensor(m!)? level_probe, "
      "Tensor shapes, int levels, int dense_read_mask, int dense_write_mask) -> ()");

  m.def("rola_produce_ledger_bind(Tensor? ledger) -> ()");

  // ---- the producer's solves -----------------------------------------------
  m.def(
      "entmax_union_forward(Tensor read_logits, Tensor write_logits, Tensor(a!) midpoint_values, "
      "Tensor(b!) support_words, Tensor(c!) read_values, Tensor(d!) write_values, int[] logit_offsets, "
      "int[] value_offsets, int[] midpoint_offsets, int[] support_offsets, int[] widths, int heads, float alpha) "
      "-> ()");
  m.def(
      "entmax_union_backward(Tensor read_logits, Tensor write_logits, Tensor midpoint_values, Tensor support_words, "
      "Tensor read_values, Tensor write_values, Tensor d_read, Tensor d_write, Tensor(a!) d_read_logits, "
      "Tensor(b!) d_write_logits, int[] logit_offsets, int[] value_offsets, int[] midpoint_offsets, "
      "int[] support_offsets, int[] widths, int heads, float alpha) -> ()");
  m.def(
      "entmax_factor_forward(Tensor logits, Tensor? mask, Tensor(a!) values, Tensor(b!)? values_second, "
      "Tensor(c!) support_words, int[] logit_offsets, int[] value_offsets, int[] support_offsets, int[] widths, "
      "int heads, float alpha) -> ()");
  m.def(
      "entmax_factor_backward(Tensor values, Tensor support_words, Tensor? mask, Tensor d_values, "
      "Tensor? d_values_second, Tensor(a!) d_logits, int[] logit_offsets, int[] value_offsets, "
      "int[] support_offsets, int[] widths, bool accumulate, int heads, float alpha) -> ()");
  m.def(
      "routing_softmax_forward(Tensor logits, Tensor(a!) values, Tensor(b!)? values_second, int[] logit_offsets, "
      "int[] value_offsets, int[] widths, int heads) -> ()");
  m.def(
      "routing_softmax_backward(Tensor values, Tensor d_values, Tensor? d_values_second, Tensor(a!) d_logits, "
      "int[] logit_offsets, int[] value_offsets, int[] widths, bool accumulate, int heads) -> ()");

  // ---- the T=1 decode path -------------------------------------------------
  m.def(
      "rola_decode_forward(Tensor[] read, Tensor[] write, Tensor[] read_scale, Tensor[] write_scale, "
      "int floor_read, int floor_write, Tensor? dials, Tensor v, Tensor v_scale, "
      "Tensor(a!) state, Tensor(b!) yacc, Tensor(c!) queue, Tensor(d!) growth, Tensor(e!) growth_any, "
      "Tensor(f!) growth_ctr, Tensor(g!) done, Tensor(h!) atom_bits, int[] level_widths, int[] level_row_offsets, "
      "int lattice_k, int lattice_m, bool write_dense, int ctas, float eps, Tensor(i!)? page_table=None, "
      "Tensor? pool_slots=None, Tensor(j!)? pool_cursor=None) -> Tensor");
  m.def("rola_decode_build_stamp() -> int");
  m.def("rola_decode_arms() -> int[][]");
  m.def("rola_decode_ledger_bind(Tensor? ledger) -> ()");
  m.def("rola_decode_ledger_warps() -> int");
  m.def("rola_decode_capacity() -> int");
  m.def("rola_decode_producer_width_mirror() -> int");

  // ---- the VMM owner, by handle --------------------------------------------
  m.def("vmm_probe(int device) -> (int, bool, int, str)");
  m.def(
      "vmm_create(int device, int dense_limit_pages, int page_rows, int page_cols, "
      "int target_chunk_bytes=67108864) -> int");
  m.def("vmm_release(int handle) -> ()");
  m.def("vmm_base(int handle) -> Tensor");
  m.def("vmm_grow(int handle, int required_pages) -> ()");
  m.def("vmm_rollback_to(int handle, int mapped_capacity_pages) -> ()");
  m.def("vmm_reset(int handle) -> ()");
  m.def("vmm_close(int handle) -> ()");
  m.def("vmm_facts(int handle) -> int[]");
}

STABLE_TORCH_LIBRARY_IMPL(rola, CompositeExplicitAutograd, m) {
  m.impl("rola_intra_forward", TORCH_BOX(&rola::intra::rola_intra_forward));
  m.impl("rola_intra_ledger_bind", TORCH_BOX(&rola::intra::rola_intra_ledger_bind));
  m.impl("intra_build_stamp", TORCH_BOX(&rola::intra::intra_build_stamp));
  m.impl("intra_footprint", TORCH_BOX(&rola::intra::intra_footprint));
  m.impl("intra_block_rows", TORCH_BOX(&rola::intra::intra_block_rows));
  m.impl("carry_forward", TORCH_BOX(&rola::carry::carry_forward));
  m.impl("scale_forward", TORCH_BOX(&rola::scale::scale_forward));
  m.impl("carry_readout_accumulate", TORCH_BOX(&rola::carry::carry_readout_accumulate));
  m.impl("device_card", TORCH_BOX(&device_card_op));
  m.impl("segment_tokens", TORCH_BOX(&rola::segment_tokens));

  m.impl("carry_ledger_bind", TORCH_BOX(&rola::carry::carry_ledger_bind));
  m.impl("carry_probe_bind", TORCH_BOX(&rola::carry::carry_probe_bind));
  m.impl("carry_capture_bind", TORCH_BOX(&rola::carry::carry_capture_bind));
  m.impl("carry_capture_bytes", TORCH_BOX(&rola::carry::carry_capture_bytes));
  m.impl("carry_build_stamp", TORCH_BOX(&rola::carry::carry_build_stamp));
  m.impl("carry_census", TORCH_BOX(&rola::carry::carry_census));
  m.impl("carry_arms", TORCH_BOX(&rola::carry::carry_arms));
  m.impl("csrc_build_stamp", TORCH_BOX(&rola::csrc_build_stamp));
  m.impl("sm_clock_ghz", TORCH_BOX(&rola::sm_clock_ghz));
  m.impl("carry_geometry", TORCH_BOX(&rola::carry::geometry));
  m.impl("carry_sub_boxes", TORCH_BOX(&rola::carry::sub_box_set));
  m.impl("rola_facts_forward", TORCH_BOX(&rola::facts::rola_facts_forward));
  m.impl("rola_facts_ledger_bind", TORCH_BOX(&rola::facts::rola_facts_ledger_bind));
  m.impl("rola_produce_forward", TORCH_BOX(&rola::produce::rola_produce_forward));
  m.impl("rola_produce_ledger_bind", TORCH_BOX(&rola::produce::rola_produce_ledger_bind));
  m.impl("entmax_union_forward", TORCH_BOX(&rola::entmax::union_forward));
  m.impl("entmax_union_backward", TORCH_BOX(&rola::entmax::union_backward));
  m.impl("entmax_factor_forward", TORCH_BOX(&rola::entmax::factor_forward));
  m.impl("entmax_factor_backward", TORCH_BOX(&rola::entmax::factor_backward));
  m.impl("routing_softmax_forward", TORCH_BOX(&rola::entmax::softmax_forward));
  m.impl("routing_softmax_backward", TORCH_BOX(&rola::entmax::softmax_backward));
  m.impl("rola_decode_forward", TORCH_BOX(&rola::decode::rola_decode_forward));
  m.impl("rola_decode_ledger_bind", TORCH_BOX(&rola::decode::rola_decode_ledger_bind));
  m.impl("rola_decode_ledger_warps", TORCH_BOX(&rola::decode::rola_decode_ledger_warps));
  m.impl("rola_decode_build_stamp", TORCH_BOX(&rola::decode::rola_decode_build_stamp));
  m.impl("rola_decode_arms", TORCH_BOX(&rola::decode::rola_decode_arms));
  m.impl("rola_decode_capacity", TORCH_BOX(&rola::decode::rola_decode_capacity));
  m.impl("rola_decode_producer_width_mirror",
         TORCH_BOX(&rola::decode::rola_decode_producer_width_mirror));
  m.impl("vmm_probe", TORCH_BOX(&vmm_probe));
  m.impl("vmm_create", TORCH_BOX(&vmm_create));
  m.impl("vmm_release", TORCH_BOX(&vmm_release));
  m.impl("vmm_base", TORCH_BOX(&vmm_base));
  m.impl("vmm_grow", TORCH_BOX(&vmm_grow));
  m.impl("vmm_rollback_to", TORCH_BOX(&vmm_rollback_to));
  m.impl("vmm_reset", TORCH_BOX(&vmm_reset));
  m.impl("vmm_close", TORCH_BOX(&vmm_close));
  m.impl("vmm_facts", TORCH_BOX(&vmm_facts));
}

// THE MODULE OBJECT: importing `rola_cu13._C` loads this library, whose static registrations above run on load. The module
// itself carries nothing, so it is built against Python's limited API and one binary serves every CPython from the
// limited API's floor (3.10, torch's build flag) on.
extern "C" PyObject* PyInit__C(void) {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                               "_C",
                               "rola's operators, registered as torch.ops.rola",
                               -1,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
  return PyModule_Create(&module);
}
