// THE EXTENSION'S OPERATOR REGISTRATION -- the only TU that registers with torch (host-only, so a `.cpp` rather than
// recompiled per `-gencode` in a `.cu`). Through torch's STABLE ABI: every entry is an operator in the `rola` namespace
// (`torch.ops.rola.<entry>`), declared by schema and implemented by a boxed function, so the binary depends on torch's
// stable C shim and never on its C++ ABI. The module object `rola._C` is only the handle an import loads the library by.
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
#include "common/torch_seam.cuh"
#include "src/carry/carry_api.cuh"
#include "src/common/geom_api.cuh"
#include "src/decode/decode_api.cuh"
#include "src/entmax/entmax.cuh"
#include "src/entmax/factor.cuh"
#include "src/facts/liveness_api.cuh"
#include "src/intra/intra.cuh"
#include "src/paging/vmm_owner.cuh"

using rola::check_arch_table;
using rola::Tensor;

namespace {

// THE ONE REASON the reverse entry can fail on this line, spelled once so a test can
// assert it and a caller cannot mistake it for a shape refusal.
constexpr const char* kCarryBackwardUnimplemented =
    "the carry reverse pass has no implementation on this line: this binary carries the "
    "forward body and the reverse LAUNCH SURFACE with its refusals only (see "
    "docs/ARCHITECTURE.md for the pass the surface is waiting for)";

// THE REVERSE SURFACE'S STUB, named so the closed-world arch refusal can be its first statement like every other
// entry's: the refusal that a device was never measured on outranks the refusal that the pass behind it does not exist.
void carry_backward_unimplemented(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const std::vector<int64_t>&,
                                  int64_t, int64_t, int64_t, const std::vector<int64_t>&,
                                  const Tensor&, const Tensor&, const std::optional<Tensor>&,
                                  const std::optional<Tensor>&, const std::optional<Tensor>&) {
  check_arch_table();
  STD_TORCH_CHECK(false, kCarryBackwardUnimplemented, " (entry: carry_backward)");
}

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

}  // namespace

STABLE_TORCH_LIBRARY(rola, m) {
  // ---- the standalone intra kernel ----------------------------------------
  m.def(
      "intra_forward(Tensor pread, Tensor pwrite, Tensor gwrite, Tensor v, Tensor sread, Tensor swrite, "
      "Tensor(a!) o, Tensor(b!) den, int levels, int level_width, int level_modes, int window) -> ()");
  m.def("intra_build_stamp() -> int");
  m.def("intra_arms() -> int[][]");

  // ---- the CARRY LAUNCH SURFACE --------------------------------------------
  m.def(
      "carry_forward(Tensor read, Tensor write, Tensor gain, Tensor v, Tensor(a!) num, Tensor(b!) den, int[] widths, "
      "int dv, int page_bits, int warps_per_cta, int[] carve_order, int[] schedule, Tensor liveness, "
      "Tensor activity, Tensor? state_in, Tensor(c!)? state_out, Tensor? page_table) -> ()");
  m.def(
      "carry_backward(Tensor read, Tensor write, Tensor gain, Tensor v, Tensor d_num, Tensor d_den, int[] widths, "
      "int dv, int page_bits, int warps_per_cta, int[] carve_order, Tensor liveness, Tensor activity, "
      "Tensor? state_in, Tensor? d_state_out, Tensor? page_table) -> ()");
  m.def("carry_ledger_bind(Tensor? ledger) -> ()");
  m.def("carry_build_stamp() -> int");
  m.def("carry_census() -> int[][]");
  m.def("carry_arms() -> int[][]");
  m.def("csrc_build_stamp() -> int");
  m.def("sm_clock_ghz(int spin_cycles=20000000) -> float");

  // ---- the addressing block and the ONE liveness pass ----------------------
  m.def("carry_geometry(int[] widths, int level_modes, int bc, int nsr, int nsw, int dv) -> int[]");
  m.def("carry_sub_boxes(int depth, int box_leaves, int workers) -> int[][]");
  m.def(
      "liveness_words(Tensor read_plane, Tensor write_plane, int[] widths, int dense_read, int[] mask_read, "
      "int dense_write, int[] mask_write) -> Tensor");

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
      "rola_decode_forward(Tensor[] read, Tensor[] write, int[] normalize, Tensor? dials, Tensor g_write, Tensor v, "
      "Tensor(a!) state, Tensor(b!) ws, Tensor(c!) ctr, Tensor(d!) growth, Tensor(e!) growth_any, "
      "Tensor(f!) growth_ctr, Tensor(g!) done, Tensor atom_bits, int[] level_widths, int[] level_row_offsets, "
      "int lattice_k, int lattice_m, int n_split, float eps, Tensor? page_table=None, Tensor(h!)? pool_slots=None, "
      "Tensor(i!)? pool_cursor=None, Tensor(j!)? pool_map=None) -> Tensor");
  m.def("rola_decode_residency(bool decay, int levels) -> int");
  m.def("rola_decode_build_stamp() -> int");
  m.def("rola_decode_arms() -> int[][]");
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
  m.impl("intra_forward", TORCH_BOX(&rola::intra::intra_forward));
  m.impl("intra_build_stamp", TORCH_BOX(&rola::intra::intra_build_stamp));
  m.impl("intra_arms", TORCH_BOX(&rola::intra::intra_arms));
  m.impl("carry_forward", TORCH_BOX(&rola::carry::carry_forward));
  m.impl("carry_backward", TORCH_BOX(&carry_backward_unimplemented));
  m.impl("carry_ledger_bind", TORCH_BOX(&rola::carry::carry_ledger_bind));
  m.impl("carry_build_stamp", TORCH_BOX(&rola::carry::carry_build_stamp));
  m.impl("carry_census", TORCH_BOX(&rola::carry::carry_census));
  m.impl("carry_arms", TORCH_BOX(&rola::carry::carry_arms));
  m.impl("csrc_build_stamp", TORCH_BOX(&rola::csrc_build_stamp));
  m.impl("sm_clock_ghz", TORCH_BOX(&rola::sm_clock_ghz));
  m.impl("carry_geometry", TORCH_BOX(&rola::carry::geometry));
  m.impl("carry_sub_boxes", TORCH_BOX(&rola::carry::sub_box_set));
  m.impl("liveness_words", TORCH_BOX(&rola::facts::liveness_words));
  m.impl("entmax_union_forward", TORCH_BOX(&rola::entmax::union_forward));
  m.impl("entmax_union_backward", TORCH_BOX(&rola::entmax::union_backward));
  m.impl("entmax_factor_forward", TORCH_BOX(&rola::entmax::factor_forward));
  m.impl("entmax_factor_backward", TORCH_BOX(&rola::entmax::factor_backward));
  m.impl("routing_softmax_forward", TORCH_BOX(&rola::entmax::softmax_forward));
  m.impl("routing_softmax_backward", TORCH_BOX(&rola::entmax::softmax_backward));
  m.impl("rola_decode_forward", TORCH_BOX(&rola::decode::rola_decode_forward));
  m.impl("rola_decode_residency", TORCH_BOX(&rola::decode::rola_decode_residency));
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

// THE MODULE OBJECT: importing `rola._C` loads this library, whose static registrations above run on load. The module
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
