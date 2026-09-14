#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs
// share; see docs/internals/decode/decode.md#note-l3

#include <cstdint>

#include "common/geom.cuh"

namespace rola {
namespace decode {

using rola::carry::kAtomLeaves;  // page granule from the addressing block; see decode.md#kmaxlevels

constexpr int kMaxLevels = 4;  // 1 <= D <= 4; see decode.md#kmaxlevels-2

constexpr int kProducerMaxBranchWidth = 256;  // a GATE; see decode.md#kproducermaxbranchwidth
constexpr int kDecodeMaxLevelWidth = 256;
static_assert(kDecodeMaxLevelWidth >= kProducerMaxBranchWidth,
              "the decode path's level-width capacity is below the producer's "
              "MAX_BRANCH_WIDTH: a topology the producer can emit would over-read "
              "the staged amplitude row");

constexpr int kDecodeThreads = 1024;  // one fat CTA per SM; see decode.md#kdecodethreads
constexpr int kDecodeWarps = kDecodeThreads / 32;

constexpr int kUnitBlockWarps = 8;  // NOT the warp count; decode.md#kunitblockwarps
constexpr int kUnitBlockWarpsBits = 3;
static_assert(1 << kUnitBlockWarpsBits == kUnitBlockWarps,
              "kUnitBlockWarpsBits must be log2(kUnitBlockWarps)");
static_assert(kUnitBlockWarps <= kDecodeWarps && kDecodeWarps % kUnitBlockWarps == 0,
              "a CTA's warps are a whole number of unit blocks");
constexpr int kUnitRepeats = kDecodeWarps / kUnitBlockWarps;

constexpr int kDecodeMaxSpan = 32;  // refused at the host entry; decode.md#kdecodemaxspan

constexpr int kDecodeResidentThreads = 1024;  // CTA/SM residency; decode.md#kdecoderesidentthreads

__host__ __device__ constexpr int decode_step_blocks_per_sm(bool decay, int levels) {
  (void)decay;
  (void)levels;
  return kDecodeResidentThreads / kDecodeThreads;
}

constexpr int kAtomShift =
    rola::carry::geom_ilog2(kAtomLeaves);  // shift+mask; decode.md#katomshift
constexpr int kAtomMask = kAtomLeaves - 1;
static_assert(1 << kAtomShift == kAtomLeaves, "kAtomShift must be log2(kAtomLeaves)");

struct Span {  // one operand's address, as the producer left it; decode.md#span
  const void* base;
  int64_t sb;
  int64_t sh;
  int64_t sw;
};

struct DecodeParams {     // every pointer/extent one decode step needs; see decode.md#decodeparams
  Span read[kMaxLevels];  // raw operands, addressed not copied; decode.md#decodeparams-note-l152
  Span write[kMaxLevels];
  Span g_write;
  Span v;
  int normalize[kMaxLevels];  // per READ level: still carries its own mass?; decode.md#near-line-98
  int levels_bf16;  // storage type per operand family, 1=bfloat16 0=float; see decode.md#levelsbf16
  int g_bf16;
  int v_bf16;

  int level_width[kMaxLevels];       // B_l
  int level_amp_offset[kMaxLevels];  // into the flat [sum_l b_l] SMEM staging
  int level_row_offset[kMaxLevels];  // into the flattened per-head dial array

  //: THE (k, m) BOX, `BoxPlan`'s fields as runtime scalars under `BoxPlan`'s -- see docs/internals/decode/decode.md#decodeparams-note-l174
  int lat_k;
  int lat_s[kMaxLevels];
  int lat_g[kMaxLevels];
  int lat_gsuf[kMaxLevels];       // prod_{l' > l} lat_g[l']
  int lat_sbits[kMaxLevels];      // log2 lat_s[l]: a digit splits into (owner, run) by shifts
  int lat_run_shift[kMaxLevels];  // BoxPlan::run_shift(l)
  int lat_local_bits;             // BoxPlan::kLocalBits = log2 BC

  int lat_inner_bits;  // the walk's unit; log2 s_{D-1}; see decode.md#decodeparams-note-l190
  int lat_outer_bits;  // lat_local_bits - lat_inner_bits: the unit's radix inside an owner
  int lat_oshift[kMaxLevels];  // r_l's shift inside the unit index; see decode.md#near-line-129
  int mask_off[kMaxLevels];    // digit masks' word layout; see decode.md#near-line-132
  int mask_words[kMaxLevels];
  int mask_total;
  int omask_off[kMaxLevels];  // owner bitmask word layout; decode.md#decodeparams-note-l209
  int omask_words[kMaxLevels];
  int omask_total;
  int olist_off[kMaxLevels];  // compacted bitmask lists; decode.md#decodeparams-note-l221
  int olist_total;
  int dlist_off[kMaxLevels];  // write side's live-digit list, outer only; decode.md#near-line-150
  int dlist_total;
  int atoms_per_unit_shift;  // atom+unit reconciled, PAGED only; decode.md#decodeparams-note-l233
  int lat_units_per_atom_shift;

  const float* dials;  // [H, total_rows] or nullptr

  float* state;       // DENSE [BH,N,cols] or PAGED [slots,kAtomLeaves,cols]; see decode.md#state
  int32_t* page_tbl;  // [BH, atoms_per_bh], -1=ABSENT, null=dense; decode.md#decodeparams-note-l250
  int32_t* growth;    // [BH] int32, per-batch-head verdict; see decode.md#growth-2
  int32_t* growth_any;  // [1] int32, OR of growth over batch-heads; see decode.md#growthany
  int32_t* growth_ctr;
  int32_t* done;    // [BH] int32, done flags; see decode.md#decodeparams-note-l265
  bool* atom_bits;  // [BH, atoms_per_bh] bool, this step's write set; see decode.md#atombits

  const int32_t* pool_slots;  // pre-zeroed by the host; decode.md#decodeparams-note-l275
  int32_t* pool_cursor;
  int32_t* pool_map;
  int pool_cap;
  float* y;  // [BH, d_v] -- the FUSED ratio readout, num/(den + eps)

  float* ws;     // [BH, n_split, cols] -- split-K workspace; see decode.md#ws
  int32_t* ctr;  // [BH]

  int64_t N;
  int atoms_per_bh;  // ceil(N / kAtomLeaves): the PAGE count per bh, both backings.
  int BH;
  int H;
  int D;
  int n_split;
  int total_rows;  // sum_l b_l
  float eps;
};

//: `ceil(n / d)` for non-negative ints, used on host and device.
__host__ __device__ __forceinline__ int ceil_div_int(int n, int d) { return (n + d - 1) / d; }

//: EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, -- see docs/internals/decode/decode.md#extract-bits-2
__device__ __forceinline__ uint32_t extract_bits(const uint32_t* src, int src_words, int start,
                                                 int count) {
  const int w = start >> 5;
  const int sh = start & 31;
  const uint32_t lo = (w < src_words) ? src[w] : 0u;
  const uint32_t hi = ((w + 1) < src_words) ? src[w + 1] : 0u;
  //: HAZARD shift-32-ub -- docs/internals/decode/decode.md#shift-32-ub
  const uint32_t v = (sh == 0) ? lo : ((lo >> sh) | (hi << (32 - sh)));
  return (count >= 32) ? v : (v & ((1u << count) - 1u));
}

// ---------------------------------------------------------------------------
// The step's entry points. `decode.cu` owns the step and the host entry;
// `rola_api.cpp` binds only the host entry.
// ---------------------------------------------------------------------------

size_t decode_step_smem_bytes(int total_rows, int d_v, int cols, bool decay, int mask_total,
                              int omask_total, int olist_total, int dlist_total);
void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget);

}  // namespace decode
}  // namespace rola
