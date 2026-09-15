// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- hand CUDA, CUB warp primitives.
// Design of record: docs/internals/entmax/factor.md.
// Gates: `tests/integration/test_production_levels.py` (fp64 oracle),
// `tests/unit/test_entmax_production.py`, `tests/oracle/test_entmax_cuda_gates.py`.
//
// Four kernels, two families, one launch shape: the exact alpha=1.5/2.0 threshold solve
// (forward + VJP) and the softmax pair. They share `entmax.cuh`'s warp-row vocabulary
// with the union solve next door, and the entmax forward IS the union forward's phases
// B/C/D fed one logit tensor instead of a midpoint.
//
// ONE LAUNCH ACROSS LEVELS: `blockIdx.z` selects a LevelTable entry, so a plan's levels
// that share a width class, an alpha and an output dtype run in ONE launch with
// bit-identical arithmetic -- docs/internals/entmax/factor.md#batched-levels

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "dispatch_switch.cuh"
#include "entmax.cuh"
#include "factor.cuh"

namespace rola {
namespace entmax {
namespace detail {

// ---------------------------------------------------------------------------
// FORWARD -- the threshold-after-sort solve
// ---------------------------------------------------------------------------
template <int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void factor_forward_kernel(
    const LogitT* __restrict__ logits, const bool* __restrict__ mask, OutT* __restrict__ values,
    OutT* __restrict__ values_second, int32_t* __restrict__ support_words, int tokens, int heads,
    LevelTable table, Strides sz, Strides smask, Strides sv, Strides sv2, Strides ssup) {
  constexpr int W_PAD = LW * IPT;
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;

  using WarpScanP = cub::WarpScan<Pair, LW>;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedI = cub::WarpReduce<int, LW>;
  using WarpRedU = cub::WarpReduce<unsigned long long, LW>;

  //: HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias -- see docs/internals/entmax/factor.md#sort-bytes
  constexpr int SORT_BYTES = RowSort<LW, IPT>::kSlotBytes;
  constexpr int STAGE_BYTES = W_PAD * (int)sizeof(uint32_t);
  constexpr int RAW_SLOT = SORT_BYTES > STAGE_BYTES ? SORT_BYTES : STAGE_BYTES;
  constexpr int SLOT = (RAW_SLOT + 15) & ~15;
  __shared__ __align__(16) char smem[BLOCK_WARPS * SLOT];
  static_assert(sizeof(typename WarpScanP::TempStorage) <= 4, "WarpScan must be shfl-only");
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedI::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedU::TempStorage) <= 4, "WarpReduce must be shfl-only");

  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int logical = threadIdx.x / LW;
  const int logical_in_warp = (threadIdx.x % WARP_SIZE) / LW;
  typename WarpScanP::TempStorage scan_store;
  typename WarpRedF::TempStorage red_store;
  typename WarpRedI::TempStorage redi_store;
  typename WarpRedU::TempStorage redu_store;

  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t support_base = table.support_off[level];

  uint32_t part[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) part[i] = 0u;

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    // ---- PHASE A: load striped (coalesced); the candidate mask is per ELEMENT ----
    float z[IPT];
    uint32_t valid_mask = 0u;
    int local_valid = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool in_width = col < width;
      const bool valid =
          in_width && (mask == nullptr || mask[addr_split(smask, sb, sh, token, logit_base + col)]);
      if (valid) {
        valid_mask |= (1u << i);
        ++local_valid;
      }
      z[i] = valid ? load_value(logits, addr_split(sz, sb, sh, token, logit_base + col)) : 0.0f;
    }
    //: A masked row can be EMPTY, so the member count is a per-row reduction and not -- see docs/internals/entmax/factor.md#validcount
    int valid_count = width;
    if (mask != nullptr) {
      valid_count = WarpRedI(redi_store).Reduce(local_valid, Sum());
      valid_count = __shfl_sync(lmask, valid_count, 0, LW);
    }
    const bool has_valid = valid_count > 0;

    // ---- PHASE B: row max/min, sentinel-pad, sort descending ----
    float local_max = -INFINITY, local_min = INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((valid_mask >> i) & 1u) {
        local_max = fmaxf(local_max, z[i]);
        local_min = fminf(local_min, z[i]);
      }
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);
    row_max = has_valid ? row_max : 0.0f;
    float row_min = WarpRedF(red_store).Reduce(local_min, Min());
    row_min = __shfl_sync(lmask, row_min, 0, LW);

    //: A finite row-relative sentinel keeps padded lanes strictly below every real -- see docs/internals/entmax/factor.md#sentinel
    const float sentinel = has_valid ? (row_min - row_max) - 1.0f : -1.0f;
    float keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool valid = (valid_mask >> i) & 1u;
      z[i] = valid ? z[i] - row_max : 0.0f;
      keys[i] = valid ? z[i] : sentinel;
    }
    RowSort<LW, IPT>::sort(keys, lane, lmask, smem + warp * SLOT, logical_in_warp);

    // ---- PHASE C: tau. Every full-width intermediate COLLAPSES TO A SCALAR. ----
    Pair local_sum{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      local_sum.p += keys[i];
      local_sum.q += keys[i] * keys[i];
    }
    Pair carry;
    WarpScanP(scan_store).ExclusiveScan(local_sum, carry, Pair{0.0f, 0.0f}, PairAdd());

    int best_col = -1;
    float best_tau = 0.0f;
    float run_p = carry.p, run_q = carry.q;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      run_p += keys[i];
      run_q += keys[i] * keys[i];
      const int rank = lane * IPT + i;
      const float k = (float)(rank + 1);
      float tau_c;
      bool cand;
      if (ALPHA15) {
        const float disc = run_p * run_p - k * (run_q - 4.0f);
        tau_c = (run_p - sqrtf(fmaxf(disc, 0.0f))) / k;
        cand = (rank < valid_count) && (disc >= 0.0f) && (keys[i] > tau_c);
      } else {
        tau_c = (run_p - 1.0f) / k;
        cand = (rank < valid_count) && (keys[i] > tau_c);
      }
      if (cand) {
        best_col = rank;
        best_tau = tau_c;
      }
    }
    //: ONE reduction selects the greatest valid k AND carries its tau: the column in the -- see docs/internals/entmax/factor.md#packedkey
    unsigned long long packed_key = ((unsigned long long)(uint32_t)(best_col + 1) << 32)
                                    | (unsigned long long)__float_as_uint(best_tau);
    packed_key = WarpRedU(redu_store).Reduce(packed_key, Max());
    packed_key = __shfl_sync(lmask, packed_key, 0, LW);
    const int chosen = (int)(uint32_t)(packed_key >> 32) - 1;
    const float tau = chosen >= 0 ? __uint_as_float((uint32_t)(packed_key & 0xffffffffu)) : 0.0f;

    // ---- PHASE D: amplitudes, both outputs, and the support bit ----
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = ((valid_mask >> i) & 1u) && (z[i] > tau);
      float base = (ALPHA15 ? 0.5f : 1.0f) * (z[i] - tau);
      base = sup ? fmaxf(base, 0.0f) : 0.0f;
      const float p = ALPHA15 ? base * base : base;
      const int64_t off = addr(sv, stream, token, value_base + col);
      store_value(values, off, p);
      //: The second output is the SAME solve, never a second one: a `shares_solve` level -- see docs/internals/entmax/factor.md#near-line-200
      if (values_second != nullptr)
        store_value(values_second, addr(sv2, stream, token, value_base + col), p);
      //: HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary -- see docs/internals/entmax/factor.md#near-line-204
      if (sup && bf16_nonzero(p)) part[i] |= (1u << (token & 31));
    }
  }

  // ---- EPILOGUE: pack the 32-token support word. ONE barrier. ----
  //: The butterfly runs over exactly the bits at or above LW, which is what separates -- see docs/internals/entmax/factor.md#near-line-212
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    for (int d = LW; d < WARP_SIZE; d <<= 1) part[i] |= __shfl_xor_sync(0xffffffffu, part[i], d);
  }
  uint32_t* stage = reinterpret_cast<uint32_t*>(smem + warp * SLOT);
  if (logical_in_warp == 0) {
#pragma unroll
    for (int i = 0; i < IPT; ++i) stage[lane + i * LW] = part[i];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll 1
    for (int col = lane; col < width; col += LW) {
      uint32_t acc = 0u;
#pragma unroll
      for (int w = 0; w < BLOCK_WARPS; ++w)
        acc |= reinterpret_cast<const uint32_t*>(smem + w * SLOT)[col];
      support_words[stream * ssup.stream + (int64_t)word * ssup.token
                    + (support_base + col) * ssup.col] = (int32_t)acc;
    }
  }
}

// ---------------------------------------------------------------------------
// BACKWARD -- the alpha-entmax rank-1 Jacobian action.
// see docs/internals/entmax/factor.md#launch-bounds
template <int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void factor_backward_kernel(
    const OutT* __restrict__ values, const int32_t* __restrict__ support_words,
    const bool* __restrict__ mask, const OutT* __restrict__ d_values,
    const OutT* __restrict__ d_values_second, LogitT* __restrict__ d_logits, int tokens, int heads,
    bool accumulate, LevelTable table, Strides sv, Strides ssup, Strides smask, Strides sdv,
    Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedP = cub::WarpReduce<Pair, LW>;
  static_assert(sizeof(typename WarpRedP::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedP::TempStorage redp_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-2
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t support_base = table.support_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    uint32_t live_mask = 0u;
    Pair partial{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid =
          (col < width)
          && (mask == nullptr || mask[addr_split(smask, sb, sh, token, logit_base + col)]);
      const uint32_t wword =
          (col < width) ? (uint32_t)support_words[stream * ssup.stream + (int64_t)word * ssup.token
                                                  + (support_base + col) * ssup.col]
                        : 0u;
      const bool live = valid && ((wword >> (token & 31)) & 1u);
      if (live) live_mask |= (1u << i);
      const float v = live ? load_value(values, addr(sv, stream, token, value_base + col)) : 0.0f;
      const float s = live ? (ALPHA15 ? sqrtf(v) : 1.0f) : 0.0f;
      float dp = live ? load_value(d_values, addr(sdv, stream, token, value_base + col)) : 0.0f;
      if (live && d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      partial.p += s * dp;
      partial.q += s;
    }
    Pair red = WarpRedP(redp_store).Reduce(partial, PairAdd());
    float num = __shfl_sync(lmask, red.p, 0, LW);
    float den = __shfl_sync(lmask, red.q, 0, LW);
    den = den > 0.0f ? den : 1.0f;
    const float shift = num / den;

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool live = (live_mask >> i) & 1u;
      float dz = 0.0f;
      if (live) {
        const float v = load_value(values, addr(sv, stream, token, value_base + col));
        const float s = ALPHA15 ? sqrtf(v) : 1.0f;
        float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
        if (d_values_second != nullptr)
          dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
        dz = s * (dp - shift);
      }
      const int64_t out = addr_split(sdz, sb, sh, token, logit_base + col);
      if (accumulate)
        add_value(d_logits, out, dz);
      else
        store_value(d_logits, out, dz);
    }
  }
}

// ---------------------------------------------------------------------------
// SOFTMAX -- the full-support level (`dense_routing`). No threshold, no support word.
// ---------------------------------------------------------------------------
template <int LW, int IPT, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void softmax_forward_kernel(
    const LogitT* __restrict__ logits, OutT* __restrict__ values, OutT* __restrict__ values_second,
    int tokens, int heads, LevelTable table, Strides sz, Strides sv, Strides sv2) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-3
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    float e[IPT];
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      e[i] = (col < width) ? load_value(logits, addr_split(sz, sb, sh, token, logit_base + col))
                           : -INFINITY;
      local_max = fmaxf(local_max, e[i]);
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      e[i] = (lane + i * LW < width) ? expf(e[i] - row_max) : 0.0f;
      local_sum += e[i];
    }
    float den = WarpRedF(red_store).Reduce(local_sum, Sum());
    den = __shfl_sync(lmask, den, 0, LW);

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = e[i] / den;
      store_value(values, addr(sv, stream, token, value_base + col), p);
      if (values_second != nullptr)
        store_value(values_second, addr(sv2, stream, token, value_base + col), p);
    }
  }
}

template <int LW, int IPT, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void softmax_backward_kernel(
    const OutT* __restrict__ values, const OutT* __restrict__ d_values,
    const OutT* __restrict__ d_values_second, LogitT* __restrict__ d_logits, int tokens, int heads,
    bool accumulate, LevelTable table, Strides sv, Strides sdv, Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-4
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    float local_dot = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = load_value(values, addr(sv, stream, token, value_base + col));
      float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
      if (d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      local_dot += p * dp;
    }
    float dot = WarpRedF(red_store).Reduce(local_dot, Sum());
    dot = __shfl_sync(lmask, dot, 0, LW);

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = load_value(values, addr(sv, stream, token, value_base + col));
      float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
      if (d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      const float dz = p * (dp - dot);
      const int64_t out = addr_split(sdz, sb, sh, token, logit_base + col);
      if (accumulate)
        add_value(d_logits, out, dz);
      else
        store_value(d_logits, out, dz);
    }
  }
}

// ---------------------------------------------------------------------------
// HOST DISPATCH -- the width CLASS is chosen here, so the kernel branches on nothing.
// The level table's entries are the caller's grouping: same class, same alpha, same
// dtype -- docs/internals/entmax/factor.md#batched-levels
// ---------------------------------------------------------------------------
template <typename LogitT, typename OutT>
void launch_factor_forward(const Tensor& logits, const std::optional<Tensor>& mask, Tensor& values,
                           const std::optional<Tensor>& values_second, Tensor& support,
                           const LevelTable& table, int levels, int klass, int heads,
                           double alpha) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams
  const int n_streams = logits.size(0) * heads, tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "independent entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(alpha == 1.5, [&](auto A15) {
          factor_forward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, LogitT, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  (const LogitT*)logits.data_ptr(), ptr_or_null<const bool>(mask),
                  (OutT*)values.data_ptr(), ptr_or_null<OutT>(values_second),
                  support.mutable_data_ptr<int32_t>(), tokens, heads, table,
                  five_logits(logits, heads), five_or_zero(mask), five(values),
                  five_or_zero(values_second), five(support));
        });
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

template <typename LogitT, typename OutT>
void launch_factor_backward(const Tensor& values, const Tensor& support,
                            const std::optional<Tensor>& mask, const Tensor& d_values,
                            const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                            const LevelTable& table, int levels, int klass, int heads,
                            bool accumulate, double alpha) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-2
  const int n_streams = d_logits.size(0) * heads, tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "independent entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(alpha == 1.5, [&](auto A15) {
          factor_backward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, LogitT, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  (const OutT*)values.data_ptr(), support.mutable_data_ptr<int32_t>(),
                  ptr_or_null<const bool>(mask), (const OutT*)d_values.data_ptr(),
                  ptr_or_null<const OutT>(d_values_second), (LogitT*)d_logits.data_ptr(), tokens,
                  heads, accumulate, table, five(values), five(support), five_or_zero(mask),
                  five(d_values), five_or_zero(d_values_second), five_logits(d_logits, heads));
        });
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

template <typename LogitT, typename OutT>
void launch_softmax_forward(const Tensor& logits, Tensor& values,
                            const std::optional<Tensor>& values_second, const LevelTable& table,
                            int levels, int klass, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-3
  const int n_streams = logits.size(0) * heads, tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "softmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        softmax_forward_kernel<kLaneWidth, kItemsPerThread, LogitT, OutT>
            <<<grid, BLOCK_THREADS, 0, stream>>>(
                (const LogitT*)logits.data_ptr(), (OutT*)values.data_ptr(),
                ptr_or_null<OutT>(values_second), tokens, heads, table, five_logits(logits, heads),
                five(values), five_or_zero(values_second));
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

template <typename LogitT, typename OutT>
void launch_softmax_backward(const Tensor& values, const Tensor& d_values,
                             const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                             const LevelTable& table, int levels, int klass, int heads,
                             bool accumulate) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-4
  const int n_streams = d_logits.size(0) * heads, tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "softmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        softmax_backward_kernel<kLaneWidth, kItemsPerThread, LogitT, OutT>
            <<<grid, BLOCK_THREADS, 0, stream>>>(
                (const OutT*)values.data_ptr(), (const OutT*)d_values.data_ptr(),
                ptr_or_null<const OutT>(d_values_second), (LogitT*)d_logits.data_ptr(), tokens,
                heads, accumulate, table, five(values), five(d_values),
                five_or_zero(d_values_second), five_logits(d_logits, heads));
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

void check_values(const Tensor& values, const std::optional<Tensor>& second) {
  STD_TORCH_CHECK(values.scalar_type() == Dtype::Float || values.scalar_type() == Dtype::BFloat16,
                  "route values must be fp32 or bf16");
  STD_TORCH_CHECK(!second.has_value() || second->scalar_type() == values.scalar_type(),
                  "both route value outputs must share a dtype");
}

}  // namespace detail

void factor_forward(const Tensor& logits, const std::optional<Tensor>& mask, Tensor& values,
                    const std::optional<Tensor>& values_second, Tensor& support_words,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(logits);
  STD_TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "independent entmax supports alpha 1.5 / 2.0 only");
  STD_TORCH_CHECK(support_offsets.size() == widths.size(),
                  "the entmax forward writes a support word per level");
  detail::check_values(values, values_second);
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(logits.get_device_index());
  detail::logit_value_switch(logits.scalar_type(), values.scalar_type(), [&](auto L, auto V) {
    detail::launch_factor_forward<decltype(L), decltype(V)>(
        logits, mask, values, values_second, support_words, table, (int)widths.size(), klass,
        (int)heads, alpha);
  });
}

void factor_backward(const Tensor& values, const Tensor& support_words,
                     const std::optional<Tensor>& mask, const Tensor& d_values,
                     const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, bool accumulate, int64_t heads,
                     double alpha) {
  check_arch_table();
  detail::check_logits(d_logits);
  STD_TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "independent entmax supports alpha 1.5 / 2.0 only");
  STD_TORCH_CHECK(support_offsets.size() == widths.size(),
                  "the entmax backward reads a support word per level");
  detail::check_values(values, d_values_second);
  STD_TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
                  "cotangents must share the route values' dtype");
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(d_logits.get_device_index());
  detail::logit_value_switch(d_logits.scalar_type(), values.scalar_type(), [&](auto L, auto V) {
    detail::launch_factor_backward<decltype(L), decltype(V)>(
        values, support_words, mask, d_values, d_values_second, d_logits, table, (int)widths.size(),
        klass, (int)heads, accumulate, alpha);
  });
}

void softmax_forward(const Tensor& logits, Tensor& values,
                     const std::optional<Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                     int64_t heads) {
  check_arch_table();
  detail::check_logits(logits);
  detail::check_values(values, values_second);
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(logits.get_device_index());
  detail::logit_value_switch(logits.scalar_type(), values.scalar_type(), [&](auto L, auto V) {
    detail::launch_softmax_forward<decltype(L), decltype(V)>(logits, values, values_second, table,
                                                             (int)widths.size(), klass, (int)heads);
  });
}

void softmax_backward(const Tensor& values, const Tensor& d_values,
                      const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      bool accumulate, int64_t heads) {
  check_arch_table();
  detail::check_logits(d_logits);
  detail::check_values(values, d_values_second);
  STD_TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
                  "cotangents must share the route values' dtype");
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(d_logits.get_device_index());
  detail::logit_value_switch(d_logits.scalar_type(), values.scalar_type(), [&](auto L, auto V) {
    detail::launch_softmax_backward<decltype(L), decltype(V)>(values, d_values, d_values_second,
                                                              d_logits, table, (int)widths.size(),
                                                              klass, (int)heads, accumulate);
  });
}

}  // namespace entmax
}  // namespace rola
