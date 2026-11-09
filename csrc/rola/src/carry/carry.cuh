// csrc/rola/src/carry/carry.cuh -- THE CARRY's constants and operand block: a CTA a box of leaves, window by window
// its state read out to the window's readers and the window's writes folded in. see docs/internals/carry/carry.md
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "common/window.cuh"

namespace rola::carry {

//: THE CTA AND THE LISTS; the window is the build's (its plan's). -- see docs/internals/carry/carry.md#constants
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
//: A LIST'S BOUNDARIES, every 64th union slot (`list_at`). -- see docs/internals/carry/carry.md#lists
constexpr int kListBound = 64;

__host__ __device__ constexpr int list_at(int w) { return w / kListBound; }

constexpr int kMaxLevels = 4;
//: THE BOX'S CAPS (an arm's own box is its plan's). -- see docs/internals/carry/ampere/plan.md#box
constexpr int kMaxExtent = 32, kStateValues = 32768;
constexpr int kPreRows = kMaxExtent + 1;

//: A WINDOW'S PLAN RECORD at `heads` query heads a write head. -- see docs/internals/carry/ampere/schedule.md#plan
__host__ __device__ constexpr int max_steps(int heads, int w) {
  return list_at(w) * heads + list_at(w) + 1;
}

__host__ __device__ constexpr int plan_words(int heads, int w) {
  return (1 + (max_steps(heads, w) + 1) / 2 + 3) / 4 * 4;
}

//: A SLICE's most bytes: a token's operand row over one level, f16.
constexpr int kSliceBytes = 2 * kMaxExtent;

//: THE LISTS' ADDRESSES, the facts pass's to emit and the carry's to read: a gap slot's token, a pool row's swizzled
//: chunk. -- see docs/internals/carry/carry.md#lists
constexpr uint16_t kGap = 0x8000;

__host__ __device__ __forceinline__ constexpr uint32_t pool_chunk(int row, uint32_t s, uint32_t k) {
  return row == 128 ? k ^ (s & 7u) : (k & ~3u) | ((k ^ (s >> 1)) & 3u);
}

//: A UNIT BYTE: the least `u` with `1.02 S 2^shift <= 2^(u + 15)`, int8. -- see docs/internals/scale/scale.md
constexpr int kUnitByteNone = -100;

__device__ __forceinline__ int unit_byte(float s, int shift) {
  const uint32_t bits = __float_as_uint(1.02f * s);
  const int e = static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
  const int u = min(max(e + shift - 15, -128), 127);
  return s > 0.0f ? u : kUnitByteNone;
}

//: A PLANE where the bundle holds it: a token's row at `base + b sb + h sh + t st` bytes.
struct Plane {
  const uint8_t* base;
  int64_t sb, st, sh;
};

//: ONE SIDE OF THE CARVE: its list level (-1 for none) and operand levels. -- see docs/internals/carry/carry.md#sides
struct Side {
  int list, operands;
  int operand[kMaxLevels], elem[kMaxLevels];
  Plane rows[kMaxLevels];
};

//: ONE SIDE'S LISTS, the facts pass's, merged over `heads` query heads' rows. -- see docs/internals/carry/carry.md#lists
struct SideLists {
  const uint16_t* union_tok;
  const int32_t* union_count;
  const uint2* entries;
  const uint16_t* pre;
  const int32_t* totals;
  int width, groups, heads;
};

//: THE CALL'S READ TABLES' offsets and its area. -- see docs/internals/carry/ampere/plan.md#read-tables
struct ReadTables {
  int heads, plan_words;
  int union_at, pre_at, scale_at, plans_at, area_at, area;
};

//: THE CROSS-CTA SUM's dtype. -- see docs/internals/carry/carry_api.md#carry-reduce
enum Reduce : int { kReduceF16 = 0, kReduceF32 = 1 };

//: THE OPERAND BLOCK, a grid constant. -- see docs/internals/carry/carry.md#params
struct CarryParams {
  int D, H, L, windows, dv, channels;
  int widths[kMaxLevels], extents[kMaxLevels], stride[kMaxLevels];
  Side write, read;
  Plane values;
  SideLists write_lists, read_lists;
  const int16_t* write_exp;
  //: a write side with no list level: each token's (F, m) and each window's shifts of them, the facts pass's
  const float2* factors;
  const int16_t* factor_shift;
  const int8_t* scale;  // the f16 sum's scale bytes: carry_api.md#carry-reduce
  int reduce;           // `Reduce`
  int readout_f16;  // the readout's MMAs accumulate in fp16 (1) or fp32 (0): carry_api.md#carry-readout-accumulate
  int forced;  // a forced plan form, -1 none: carry_api.md#carry-forward
  uint32_t* num;
  uint32_t*
      plans;  // the plan pass's records, [BH][windows][box classes][plan words]: schedule.md#pass
  ReadTables tables;  // the read tables' offsets and the area, the launch's: plan.md#read-tables
  float* check;
  float* probe;
  float* probe_den;
  //: each query head's token's den, fp32 `[BH heads, L]`, added into by every CTA holding its leaves (`handoff.md#den`)
  float* den;
  long long* ledger;
};

//: A LAUNCH CAPTURED, its operand block and grid. -- see docs/internals/carry/carry_api.md#carry-capture-bind
struct CarryCapture {
  int64_t bytes, boxes, bh;
  CarryParams p;
};

//: THE PHASE LEDGER's phases. -- see docs/internals/carry/ampere/carry_kernel.md#phases
enum CarryPhase : int {
  kPhaseLanded,
  kPhaseMaxima,
  kPhaseStore,
  kPhaseReadout,
  kPhaseDrain,
  kPhaseFill,
  kPhaseFold,
  kPhaseWait,
  kCarryPhases
};

}  // namespace rola::carry
