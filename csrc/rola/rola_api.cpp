// THE EXTENSION'S PYTHON BINDING -- the only TU that knows pybind exists (host-only,
// so a `.cpp` rather than recompiled per `-gencode` in a `.cu`). ONE MODULE, NOT TWO:
// the chunk consumer, decode sibling and paging arena all register here, one contract.
// Full entry inventory and the summary pass's no-allocation contract:
// see docs/internals/rola_api.md

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "src/carry/carry_api.cuh"
#include "src/decode/decode_api.cuh"
#include "src/entmax/entmax.cuh"
#include "src/entmax/factor.cuh"
#include "src/paging/vmm_owner.cuh"
#include "common/arch_runtime.cuh"
#include "common/build_stamp.cuh"
#include "src/common/geom_api.cuh"
#include "src/facts/liveness_api.cuh"
#include "src/intra/intra.cuh"

namespace py = pybind11;
using rola::check_arch_table;

// THE ONE REASON the reverse entry can fail on this line, spelled once so a test can
// assert it and a caller cannot mistake it for a shape refusal.
static constexpr const char* kCarryBackwardUnimplemented =
    "the carry reverse pass has no implementation on this line: this binary carries the "
    "forward body and the reverse LAUNCH SURFACE with its refusals only (see "
    "docs/ARCHITECTURE.md for the pass the surface is waiting for)";

// THE REVERSE SURFACE'S STUB, named (rather than an inline lambda) so the closed-world arch
// refusal can be its first statement like every other entry's: the refusal that a device was
// never measured on outranks the refusal that the pass behind it does not exist yet.
static void carry_backward_unimplemented(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const std::vector<int64_t>&, int64_t, int64_t, int64_t,
                                         const std::vector<int64_t>&, const at::Tensor&,
                                         const at::Tensor&, const c10::optional<at::Tensor>&,
                                         const c10::optional<at::Tensor>&,
                                         const c10::optional<at::Tensor>&) {
  check_arch_table();
  TORCH_CHECK(false, kCarryBackwardUnimplemented, " (entry: carry_backward)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- the standalone intra kernel ----------------------------------------
  m.def("intra_forward", &rola::intra::intra_forward,
        "intra: the within-window causal term, REDuced into o/den", py::arg("pread"),
        py::arg("pwrite"), py::arg("gwrite"), py::arg("v"), py::arg("sread"), py::arg("swrite"),
        py::arg("o"), py::arg("den"), py::arg("levels"), py::arg("level_width"),
        py::arg("level_modes"), py::arg("window"));
  m.def("intra_build_stamp", &rola::intra::intra_build_stamp,
        "intra: a fact only the loaded binary can produce");
  m.def("intra_arms", &rola::intra::intra_arms, "intra: the built (levels, window) arms");

  // ---- the CARRY LAUNCH SURFACE --------------------------------------------
  m.def("carry_forward", &rola::carry::carry_forward,
        "the carry forward pass: the inter term's readout and the window fold, one kernel",
        py::arg("read"), py::arg("write"), py::arg("gain"), py::arg("v"), py::arg("num"),
        py::arg("den"), py::arg("widths"), py::arg("dv"), py::arg("page_bits"),
        py::arg("warps_per_cta"), py::arg("carve_order"), py::arg("schedule"), py::arg("liveness"),
        py::arg("activity"), py::arg("state_in"), py::arg("state_out"), py::arg("page_table"));
  m.def("carry_backward", &carry_backward_unimplemented,
        "the carry reverse launch surface -- CONTRACT ONLY, no implementation on this line",
        py::arg("read"), py::arg("write"), py::arg("gain"), py::arg("v"), py::arg("d_num"),
        py::arg("d_den"), py::arg("widths"), py::arg("dv"), py::arg("page_bits"),
        py::arg("warps_per_cta"), py::arg("carve_order"), py::arg("liveness"), py::arg("activity"),
        py::arg("state_in"), py::arg("d_state_out"), py::arg("page_table"));
  m.def("carry_ledger_bind", &rola::carry::carry_ledger_bind,
        "bind (or unbind with None) the carry phase ledger: a CUDA int64 [ctas][warps][phases] "
        "tensor the next launches add per-warp phase cycles into");
  m.def("carry_build_stamp", &rola::carry::carry_build_stamp,
        "the carry family's device build stamp -- a fact the loaded binary alone can state");
  m.def("carry_census", &rola::carry::carry_census,
        "one row per built carry arm: what the compiler did, then what the box plan says "
        "it should have done");
  m.def("carry_arms", &rola::carry::carry_arms,
        "the (D, DV, warps_per_cta) carry arms this binary carries");

  m.def("csrc_build_stamp", &rola::csrc_build_stamp,
        "the content hash of csrc/rola/src that this fatbin compiled against, read "
        "back off the device -- the fact a stale binary cannot produce");
  m.def("sm_clock_ghz", &rola::sm_clock_ghz,
        "the SM's effective clock in GHz under a spin of the given cycles on every SM, read "
        "off the device: the state a measurement was taken in",
        py::arg("spin_cycles") = 20000000);

  // ---- the addressing block, a foundation feature -------------------------
  m.def("carry_geometry", &rola::carry::geometry,
        "the addressing block, derived once and flattened -- the SHIPPED derivation, "
        "which is what a model test must measure",
        py::arg("widths"), py::arg("level_modes"), py::arg("bc"), py::arg("nsr"), py::arg("nsw"),
        py::arg("dv"));
  m.def("carry_sub_boxes", &rola::carry::sub_box_set,
        "the GENERATED set of warp sub-box shapes a side selects its member from", py::arg("depth"),
        py::arg("box_leaves"), py::arg("workers"));

  // ---- the ONE liveness pass, a foundation feature ------------------------
  m.def("liveness_words", &rola::facts::liveness_words,
        "the ONE geometry-independent liveness pass -- the class-1 words, one bit per "
        "(side, token, level, digit); every consumer's grain is a host fold over these",
        py::arg("read_plane"), py::arg("write_plane"), py::arg("widths"), py::arg("dense_read"),
        py::arg("mask_read"), py::arg("dense_write"), py::arg("mask_write"));

  // ---- the producer's union entmax solve ----------------------------------
  m.def("entmax_union_forward", &rola::entmax::union_forward,
        "union entmax forward: midpoint values, read/write routes, packed support");
  m.def("entmax_union_backward", &rola::entmax::union_backward,
        "union entmax VJP into read/write logits");

  // ---- the producer's independent (per-side) solves ------------------------
  //: One entry per direction per family, each covering as many LEVELS as the caller -- see docs/internals/rola_api.md#near-line-113
  m.def("entmax_factor_forward", &rola::entmax::factor_forward,
        "independent entmax forward over one or more levels: routes and packed support");
  m.def("entmax_factor_backward", &rola::entmax::factor_backward,
        "independent entmax VJP into the packed logits");
  m.def("routing_softmax_forward", &rola::entmax::softmax_forward,
        "softmax routing forward over one or more levels");
  m.def("routing_softmax_backward", &rola::entmax::softmax_backward,
        "softmax routing VJP into the packed logits");

  // ---- the T=1 decode path -------------------------------------------------
  //: A SIBLING of the chunk consumer, not a specialization of it: shared produce -- see docs/internals/rola_api.md#pybind11-module-note-l126
  m.def(
      "rola_decode_forward", &rola::decode::rola_decode_forward,
      "RoLA T=1 decode step: operand fold + lattice factor walk + token-stationary "
      "gather-GEMV + the per-batch-head residency verdict, in one kernel.",
      py::arg("read"), py::arg("write"), py::arg("normalize"), py::arg("dials"), py::arg("g_write"),
      py::arg("v"), py::arg("state"), py::arg("ws"), py::arg("ctr"),
      //: The verdict's buffers: the per-batch-head answer, its reduction, the reducing -- see docs/internals/rola_api.md#near-line-133
      py::arg("growth"), py::arg("growth_any"), py::arg("growth_ctr"), py::arg("done"),
      py::arg("atom_bits"), py::arg("level_widths"), py::arg("level_row_offsets"),
      //: THE LEAF ORDER, as data: the `(k, m)` box the state plane is written in.
      py::arg("lattice_k"), py::arg("lattice_m"), py::arg("n_split"), py::arg("eps"),
      //: `page_table` resolves each touched ATOM to its slot; `None` is the dense -- see docs/internals/rola_api.md#near-line-140
      py::arg("page_table") = c10::optional<at::Tensor>(),
      //: THE SLACK POOL, all three or none: the slots the host pre-committed and -- see docs/internals/rola_api.md#near-line-143
      py::arg("pool_slots") = c10::optional<at::Tensor>(),
      py::arg("pool_cursor") = c10::optional<at::Tensor>(),
      py::arg("pool_map") = c10::optional<at::Tensor>());
  //: THE DEVICE-SIDE BUILD STAMP: a fact only the loaded binary can produce, which is the -- see docs/internals/rola_api.md#near-line-149
  m.def("rola_decode_residency", &rola::decode::rola_decode_residency,
        "the decode step's declared CTAs per SM for this arm, read off the launch bound",
        py::arg("decay"), py::arg("levels"));
  m.def("rola_decode_build_stamp", &rola::decode::rola_decode_build_stamp,
        "the decode step's own footprint, read back off a kernel this translation unit "
        "compiles");
  m.def("rola_decode_arms", &rola::decode::rola_decode_arms,
        "the decode arms this binary carries, [d_v, D, decay] per row; a subset means "
        "an iteration build (ROLA_DECODE_ARMS), which is not shippable");
  m.def("rola_decode_capacity", &rola::decode::rola_decode_capacity,
        "the decode path's maximum level width; must be >= the producer's "
        "MAX_BRANCH_WIDTH (tests/unit/test_decode_capacity.py pins the pair)");
  m.def("rola_decode_producer_width_mirror", &rola::decode::rola_decode_producer_width_mirror,
        "the C++ mirror of the producer's MAX_BRANCH_WIDTH; pinned EQUAL to the Python "
        "constant so a stale mirror cannot silently satisfy the capacity static_assert");

  // ---- the VMM owner ------------------------------------------------------
  m.def(
      "rola_vmm_probe",
      [](int device) {
        const auto probe = rola::paging::VmmOwner::probe(device);
        py::dict result;
        result["device"] = probe.device;
        result["supported"] = probe.supported;
        result["allocation_granularity"] = probe.allocation_granularity;
        result["reason"] = probe.reason;
        return result;
      },
      py::arg("device"));

  py::class_<rola::paging::VmmOwner, std::shared_ptr<rola::paging::VmmOwner>>(m, "RoLAVmmOwner")
      .def("base", &rola::paging::VmmOwner::base)
      .def("grow", &rola::paging::VmmOwner::grow, py::arg("required_pages"))
      .def("rollback_to", &rola::paging::VmmOwner::rollback_to, py::arg("mapped_capacity_pages"))
      .def("reset", &rola::paging::VmmOwner::reset)
      .def("close", &rola::paging::VmmOwner::close)
      .def_property_readonly("closed", &rola::paging::VmmOwner::closed)
      .def_property_readonly("mapped_capacity_pages",
                             &rola::paging::VmmOwner::mapped_capacity_pages)
      .def_property_readonly("dense_limit_pages", &rola::paging::VmmOwner::dense_limit_pages)
      .def_property_readonly("committed_bytes", &rola::paging::VmmOwner::committed_bytes)
      .def_property_readonly("virtual_bytes", &rola::paging::VmmOwner::virtual_bytes)
      .def_property_readonly("allocation_granularity",
                             &rola::paging::VmmOwner::allocation_granularity)
      .def_property_readonly("chunk_pages", &rola::paging::VmmOwner::chunk_pages)
      .def_property_readonly("backing_kind", &rola::paging::VmmOwner::backing_kind);

  //: The page geometry is the ARENA's -- `page_rows x page_cols` fp32, which is -- see docs/internals/rola_api.md#pybind11-module-note-l203
  m.def("rola_vmm_create", &rola::paging::create_vmm_owner, py::arg("device"),
        py::arg("dense_limit_pages"), py::arg("page_rows"), py::arg("page_cols"),
        py::arg("target_chunk_bytes") = 64ll * 1024ll * 1024ll);
}
