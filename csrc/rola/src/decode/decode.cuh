#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs
// share; see docs/internals/decode/decode.md#note-l3

#include <cstdint>

#include "common/geom.cuh"
#include "common/hardware.cuh"
#include "common/select.cuh"

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

//: THE WALK'S REGISTERS A THREAD: a warp holds a quarter of an atom's leaves a lane group in flight. -- see docs/internals/decode/decode.md#decode-threads
constexpr int kDecodeRegsPerThread = 128;

//: THE STEP'S PLAN at build `B`: ONE CTA an SM, of the threads the card's registers hold at `kDecodeRegsPerThread`.
//: -- see docs/internals/decode/decode.md#declared-residency
template <::denseref::arch::Build B>
struct DecodePlan {
  static constexpr int kThreads =
      ::denseref::arch::traits_of(B).caps.regs_per_sm / kDecodeRegsPerThread;
  static constexpr int kWarps = kThreads / 32;
  static constexpr int kBlocksPerSm = 1;
  static_assert(kThreads % 32 == 0 && kThreads <= 1024,
                "the step's CTA is whole warps, at most 1024 threads");
  static_assert((2 * kMaxLevels * kDecodeMaxLevelWidth) % kThreads == 0,
                "the digit pass gives every thread the same count of both sides' digit slots");
  static_assert(kWarps >= 3 * kMaxLevels - 1,
                "the prologue's list and row-sum jobs are a warp each");
  static_assert(kThreads >= 64, "a thread a value channel, at the widest value head");
};

constexpr int kUnitBlockWarps = 8;  // NOT the warp count; decode.md#kunitblockwarps
constexpr int kUnitBlockWarpsBits = 3;
static_assert(1 << kUnitBlockWarpsBits == kUnitBlockWarps,
              "kUnitBlockWarpsBits must be log2(kUnitBlockWarps)");

constexpr int kDecodeMaxSpan = 32;  // refused at the host entry; decode.md#kdecodemaxspan

//: `expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as -- see docs/internals/decode/decode.md#kdecodemaxexponent
constexpr float kDecodeMaxExponent = 88.0f;

//: THE LEAF'S LANES: a leaf row is `DV / 8` lanes of 8 channels, so a warp walks `32 / (DV / 8)` units at once. -- see docs/internals/decode/decode.md#leaf-lanes
template <int DV>
struct LeafLanes {
  static constexpr int kLanes = DV / 8;
  static constexpr int kGroups = 32 / kLanes;
  static_assert(DV % 32 == 0 && kLanes <= 32, "a leaf row is whole 32-channel groups of 4 lanes");
};

//: THE STEP'S PHASES, as the phase ledger books them; see decode.md#phase-ledger
enum DecodePhase : int { kPhFold, kPhFactor, kPhAdmit, kPhWalk, kPhCombine, kDecodePhases };

//: THE SMALLEST UNIT SHARE worth a CTA's prologue (its fold and factor tables): the grid's supply bound.
//: -- see docs/internals/decode/decode.md#the-split-policy
constexpr int kDecodeUnitsPerCta = 16;

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
  Span read[kMaxLevels];  // unit-max coefficient rows, u8; see decode.md#decodeparams
  Span write[kMaxLevels];
  Span read_scale[kMaxLevels];  // fp32 row maxima, [B, 1, H]
  Span write_scale[kMaxLevels];
  Span v;           // the value, bf16 against the token's largest channel
  Span v_scale;     // the token's fp32 value scale
  int write_atoms;  // the write term walks whole atoms; decode.md#whole-atoms

  int level_width[kMaxLevels];       // B_l
  int level_amp_offset[kMaxLevels];  // into the flat [sum_l b_l] SMEM staging
  int level_row_offset[kMaxLevels];  // into the flattened per-head dial array

  //: THE (k, m) BOX, the lattice's fields as runtime scalars under `rola/ops/lattice.py`'s -- see docs/internals/decode/decode.md#decodeparams-note-l174
  int lat_k;
  int lat_s[kMaxLevels];
  int lat_g[kMaxLevels];
  int lat_gsuf[kMaxLevels];       // prod_{l' > l} lat_g[l']
  int lat_sbits[kMaxLevels];      // log2 lat_s[l]: a digit splits into (owner, run) by shifts
  int lat_run_shift[kMaxLevels];  // lattice run_shift(l)
  int lat_local_bits;             // lattice local bits = log2 BC

  int lat_inner_bits;  // the walk's unit; log2 s_{D-1}; see decode.md#decodeparams-note-l190
  int lat_outer_bits;  // lat_local_bits - lat_inner_bits: the unit's radix inside an owner
  int lat_oshift[kMaxLevels];  // r_l's shift inside the unit index; see decode.md#near-line-129
  int mask_off[kMaxLevels];    // digit masks' word layout; see decode.md#near-line-132
  int mask_words[kMaxLevels];
  int mask_total;
  int olist_off[kMaxLevels];  // compacted bitmask lists; decode.md#decodeparams-note-l221
  int olist_total;
  int dlist_off[kMaxLevels];  // write side's live-digit list, outer only; decode.md#near-line-150
  int dlist_total;
  int atoms_per_unit_shift;  // atom+unit reconciled, PAGED only; decode.md#decodeparams-note-l233
  int lat_units_per_atom_shift;

  const float* dials;  // [H, total_rows] or nullptr

  uint8_t*
      state;  // DENSE [BH, atoms, page bytes] or PAGED [slots, page bytes]; see decode.md#state
  int32_t* page_tbl;  // [BH, atoms_per_bh], -1=ABSENT, null=dense; decode.md#decodeparams-note-l250
  int32_t* growth;    // [BH] int32, per-batch-head verdict; see decode.md#growth-2
  int32_t* growth_any;  // [1] int32, OR of growth over batch-heads; see decode.md#growthany
  int32_t* growth_ctr;
  int32_t* done;    // [BH] int32, done flags; see decode.md#decodeparams-note-l265
  bool* atom_bits;  // [BH, atoms_per_bh] bool, this step's write set; see decode.md#atombits

  const int32_t* pool_slots;  // pre-zeroed by the host; decode.md#decodeparams-note-l275
  int32_t* pool_cursor;
  int pool_cap;
  float* y;  // [BH, d_v] -- the FUSED ratio readout, num/(den + eps)

  float* yacc;  // [BH, cols] -- a head's readout sums, zeroed by its finisher; see decode.md#queue
  int32_t* queue;  // [4 BH + 2] -- the work queue's counters; see decode.md#queue
  long long*
      ledger;  // [ctas][the plan's warps][kDecodePhases] cycles, or null; see decode.md#phase-ledger

  int64_t N;
  int atoms_per_bh;  // ceil(N / kAtomLeaves): the PAGE count per bh, both backings.
  int BH;
  int H;
  int D;
  int ctas;        // the resident grid; see decode.md#queue
  int total_rows;  // sum_l b_l
  float eps;
};

//: THE WORK QUEUE'S ADMISSION WORDS: a head's published state (ready, needy) and a visiting CTA's -- see docs/internals/decode/decode.md#queue-admission
constexpr int kQueueReady = 2;
constexpr int kQueueNeedy = 4;
constexpr int kQueueOwner = 8;
constexpr int kQueueFollower = 16;
constexpr int kQueueDecided = 32;

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

size_t decode_step_smem_bytes(int warps, int total_rows, int d_v, bool decay, int mask_total,
                              int olist_total, int dlist_total);
void launch_decode_step(const Card& card, const DecodeParams& p, int d_v, bool decay,
                        int64_t smem_budget);

namespace detail {
//: THE STEP'S PLAN at a card's build, for the host: its warps and its residency (`DecodePlan`).
struct StepPlan {
  int warps, blocks_per_sm;
};

StepPlan step_plan(const Card& card);
int step_warps(const Card& card);
}  // namespace detail

}  // namespace decode
}  // namespace rola
