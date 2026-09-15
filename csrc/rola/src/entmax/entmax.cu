// THE PRODUCER'S UNION ENTMAX SOLVE -- hand CUDA, CUB warp primitives.
// Design of record: docs/internals/entmax/entmax.md.
// Gates: `tests/unit/test_entmax_cuda_gates.py` (fp64 oracle only; the Triton kernel is
// a debugging diagnostic and is never asserted against).
//
// ONE ROW PER LOGICAL WARP: every collective runs along one token's row, so all of them
// are warp-scope and shuffle-based. Three facts follow, and each is load-bearing rather
// than incidental -- register demand is O(items-per-lane) and NOT O(CTA tile); the solve
// uses ZERO shared memory beyond the sort scratch, because every `cub::WarpScan` /
// `WarpReduce` TempStorage here resolves to a measured 1-byte pure-shfl specialization;
// and there is exactly ONE `__syncthreads` in each kernel, in the epilogue.
//
// The Triton history that forced hand CUDA, the phase-by-phase walk, the backward's
// Jacobian derivation and its measured spill trades: docs/internals/entmax/entmax.md

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "dispatch_switch.cuh"
#include "entmax.cuh"

namespace rola {
namespace entmax {
namespace detail {

//: The overflow-free read-write difference: two opposite-sign finite extrema -- see docs/internals/entmax/entmax.md#overflow-free-delta-2
__device__ __forceinline__ float overflow_free_delta(float r, float w, bool* saturated) {
  const float kMax = 3.4028234663852886e38f;
  const float kHalfMax = 1.7014117331926443e38f;
  bool rneg = r < 0.0f;
  bool opposite = rneg != (w < 0.0f);
  float rmag = fabsf(r), wmag = fabsf(w);
  float hi = fmaxf(rmag, wmag), lo = fminf(rmag, wmag);
  float room = kMax - hi;
  float mag = hi + fminf(lo, room);
  float opp = rneg ? -mag : mag;
  float same = (opposite ? 0.0f : r) - (opposite ? 0.0f : w);
  if (saturated) *saturated = opposite && (hi > kHalfMax) && (lo > room);
  return opposite ? opp : same;
}

//: log(sigmoid(-delta)) == -softplus(delta), finite for every finite input. `log1pf` -- see docs/internals/entmax/entmax.md#logwritegate
__device__ __forceinline__ float log_write_gate(float delta) {
  return -(fmaxf(delta, 0.0f) + log1pf(expf(-fabsf(delta))));
}

// ---------------------------------------------------------------------------
// FORWARD
// ---------------------------------------------------------------------------
template <int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void union_forward_kernel(
    const LogitT* __restrict__ read_logits, const LogitT* __restrict__ write_logits,
    float* __restrict__ midpoint_values, int32_t* __restrict__ support_words,
    OutT* __restrict__ read_values, OutT* __restrict__ write_values, int tokens, int heads,
    LevelTable table, Strides sr, Strides sw, Strides smid, Strides sread, Strides swrite,
    Strides ssup) {
  constexpr int W_PAD = LW * IPT;
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;

  using WarpScanP = cub::WarpScan<Pair, LW>;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedU = cub::WarpReduce<unsigned long long, LW>;

  //: HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias -- see docs/internals/entmax/entmax.md#sort-bytes
  constexpr int SORT_BYTES = RowSort<LW, IPT>::kSlotBytes;
  constexpr int STAGE_BYTES = W_PAD * (int)sizeof(uint32_t);
  constexpr int RAW_SLOT = SORT_BYTES > STAGE_BYTES ? SORT_BYTES : STAGE_BYTES;
  constexpr int SLOT = (RAW_SLOT + 15) & ~15;
  __shared__ __align__(16) char smem[BLOCK_WARPS * SLOT];
  //: These three TempStorages are per-thread rather than __shared__, which is -- see docs/internals/entmax/entmax.md#static-assert
  static_assert(sizeof(typename WarpScanP::TempStorage) <= 4, "WarpScan must be shfl-only");
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedU::TempStorage) <= 4, "WarpReduce must be shfl-only");

  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int logical = threadIdx.x / LW;  // logical warp id within the block
  const int logical_in_warp = (threadIdx.x % WARP_SIZE) / LW;
  typename WarpScanP::TempStorage scan_store;
  typename WarpRedF::TempStorage red_store;
  typename WarpRedU::TempStorage redu_store;

  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/entmax.md#stream32
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t midpoint_base = table.midpoint_off[level];
  const int64_t support_base = table.support_off[level];

  uint32_t part[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) part[i] = 0u;

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    //: Skipping a row past the end is EXACTLY equivalent, not merely close: -- see docs/internals/entmax/entmax.md#continue
    if (local >= WORD_TOKENS || token >= tokens) continue;

    // ---- PHASE A: load striped (coalesced), fold, RETIRE both logit tiles ----
    float midpoint[IPT], delta[IPT];
    uint32_t support_mask = 0u, saturated_mask = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const float r =
          valid ? load_value(read_logits, addr_split(sr, sb, sh, token, logit_base + col)) : 0.0f;
      const float w =
          valid ? load_value(write_logits, addr_split(sw, sb, sh, token, logit_base + col)) : 0.0f;
      midpoint[i] = 0.5f * r + 0.5f * w;
      bool sat = false;
      delta[i] = overflow_free_delta(r, w, &sat);
      if (sat) saturated_mask |= (1u << i);
    }

    // ---- PHASE B: row max/min, sentinel-pad, sort descending ----
    float local_max = -INFINITY, local_min = INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (lane + i * LW < width) {
        local_max = fmaxf(local_max, midpoint[i]);
        local_min = fminf(local_min, midpoint[i]);
      }
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);
    float row_min = WarpRedF(red_store).Reduce(local_min, Min());
    row_min = __shfl_sync(lmask, row_min, 0, LW);

    float keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool valid = lane + i * LW < width;
      midpoint[i] = valid ? midpoint[i] - row_max : 0.0f;
      //: A finite row-relative sentinel keeps padded lanes strictly below every real -- see docs/internals/entmax/entmax.md#near-line-161
      keys[i] = valid ? midpoint[i] : (row_min - row_max) - 1.0f;
    }
    //: A sort is a PERMUTATION, so striped input is fine and the blocked output is what -- see docs/internals/entmax/entmax.md#near-line-165
    RowSort<LW, IPT>::sort(keys, lane, lmask, smem + warp * SLOT, logical_in_warp);

    // ---- PHASE C: tau. prefix/prefix_sq/tau_candidates/candidate/chosen_col all
    //      COLLAPSE TO SCALARS -- none is ever materialized full width. ----
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
        cand = (rank < width) && (disc >= 0.0f) && (keys[i] > tau_c);
      } else {
        tau_c = (run_p - 1.0f) / k;
        cand = (rank < width) && (keys[i] > tau_c);
      }
      if (cand) {
        best_col = rank;
        best_tau = tau_c;
      }
    }
    //: ONE reduction selects the greatest valid k AND carries its tau: the column in the -- see docs/internals/entmax/entmax.md#packedkey
    unsigned long long packed_key = ((unsigned long long)(uint32_t)(best_col + 1) << 32)
                                    | (unsigned long long)__float_as_uint(best_tau);
    packed_key = WarpRedU(redu_store).Reduce(packed_key, Max());
    packed_key = __shfl_sync(lmask, packed_key, 0, LW);
    const int chosen = (int)(uint32_t)(packed_key >> 32) - 1;
    const float tau = chosen >= 0 ? __uint_as_float((uint32_t)(packed_key & 0xffffffffu)) : 0.0f;

    // ---- PHASE D: values in place over `midpoint`; `read` emitted HERE, while `delta`
    //      is still the logit difference and before phase E overwrites it. ----
    float values[IPT];
    uint32_t read_live_mask = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool sup = (col < width) && (midpoint[i] > tau);
      if (sup) support_mask |= (1u << i);
      float base = (ALPHA15 ? 0.5f : 1.0f) * (midpoint[i] - tau);
      base = sup ? fmaxf(base, 0.0f) : 0.0f;
      values[i] = ALPHA15 ? base * base : base;
      if (col < width) {
        midpoint_values[addr(smid, stream, token, midpoint_base + col)] = values[i];
        const float read = values[i] * (1.0f / (1.0f + expf(-delta[i])));
        store_value(read_values, addr(sread, stream, token, value_base + col), read);
        if (bf16_nonzero(read)) read_live_mask |= (1u << i);
      }
    }

    // ---- PHASE E: log-weights IN PLACE over `delta`, then the write normalization ----
    float local_lw = -INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool sup = (support_mask >> i) & 1u;
      delta[i] = sup ? logf(values[i]) + log_write_gate(delta[i]) : -INFINITY;
      if (sup) local_lw = fmaxf(local_lw, delta[i]);
    }
    float max_lw = WarpRedF(red_store).Reduce(local_lw, Max());
    max_lw = __shfl_sync(lmask, max_lw, 0, LW);
    const bool has_mass = max_lw > -INFINITY;
    const float safe_max = has_mass ? max_lw : 0.0f;

    float scaled[IPT];
    float local_den = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool sup = (support_mask >> i) & 1u;
      scaled[i] = (sup && has_mass) ? expf(delta[i] - safe_max) : 0.0f;
      local_den += scaled[i];
    }
    float den = WarpRedF(red_store).Reduce(local_den, Sum());
    den = __shfl_sync(lmask, den, 0, LW);
    const bool has_weight = den > 0.0f;
    const float safe_den = has_weight ? den : 1.0f;

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = (support_mask >> i) & 1u;
      const float write = (sup && has_weight) ? scaled[i] / safe_den : 0.0f;
      store_value(write_values, addr(swrite, stream, token, value_base + col), write);
      //: HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary -- see docs/internals/entmax/entmax.md#near-line-267
      if (sup && (((read_live_mask >> i) & 1u) || bf16_nonzero(write)))
        part[i] |= (1u << (token & 31));
    }
  }

  // ---- EPILOGUE: pack the 32-token support word. ONE barrier. ----
  //: The butterfly runs over exactly the bits at or above LW, which is what separates -- see docs/internals/entmax/entmax.md#near-line-275
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
// BACKWARD -- the alpha-entmax Jacobian action through the union split.
// see docs/internals/entmax/entmax.md#launch-bounds
template <int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void union_backward_kernel(
    const LogitT* __restrict__ read_logits, const LogitT* __restrict__ write_logits,
    const float* __restrict__ midpoint_values, const int32_t* __restrict__ support_words,
    const OutT* __restrict__ read_values, const OutT* __restrict__ write_values,
    const OutT* __restrict__ d_read, const OutT* __restrict__ d_write,
    LogitT* __restrict__ d_read_logits, LogitT* __restrict__ d_write_logits, int tokens, int heads,
    LevelTable table, Strides sr, Strides sw, Strides smid, Strides srv, Strides swv, Strides sdr,
    Strides sdw, Strides sdrl, Strides sdwl, Strides ssup) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedP = cub::WarpReduce<Pair, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedP::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;
  typename WarpRedP::TempStorage redp_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/entmax.md#stream32-2
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t midpoint_base = table.midpoint_off[level];
  const int64_t support_base = table.support_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    uint32_t support_mask = 0u, saturated_mask = 0u;
    // ---- the write-side rank-1: sum(d_write * write_values) ----
    //: This pass carries NOTHING into the next one but two scalars (`local_dot` and the -- see docs/internals/entmax/entmax.md#localdot
    float local_dot = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const uint32_t wword =
          valid ? (uint32_t)support_words[stream * ssup.stream + (int64_t)word * ssup.token
                                          + (support_base + col) * ssup.col]
                : 0u;
      if (valid && ((wword >> (token & 31)) & 1u)) support_mask |= (1u << i);
      const float wv =
          valid ? load_value(write_values, addr(swv, stream, token, value_base + col)) : 0.0f;
      const float dw =
          valid ? load_value(d_write, addr(sdw, stream, token, value_base + col)) : 0.0f;
      local_dot += dw * wv;
    }
    float dot = WarpRedF(red_store).Reduce(local_dot, Sum());
    dot = __shfl_sync(lmask, dot, 0, LW);

    //: `s` is NOT kept: it is recomputed from a re-read `midpoint_values` in the final -- see docs/internals/entmax/entmax.md#near-line-368
    float d_delta[IPT], d_mid[IPT];
    Pair partial{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const bool sup = (support_mask >> i) & 1u;
      const float wv =
          valid ? load_value(write_values, addr(swv, stream, token, value_base + col)) : 0.0f;
      const float dw =
          valid ? load_value(d_write, addr(sdw, stream, token, value_base + col)) : 0.0f;
      const float dlw = wv * (dw - dot);
      const float r =
          valid ? load_value(read_logits, addr_split(sr, sb, sh, token, logit_base + col)) : 0.0f;
      const float w =
          valid ? load_value(write_logits, addr_split(sw, sb, sh, token, logit_base + col)) : 0.0f;
      bool sat = false;
      const float delta = overflow_free_delta(r, w, &sat);
      if (sat) saturated_mask |= (1u << i);
      const float read_gate = 1.0f / (1.0f + expf(-delta));
      const float write_gate = 1.0f / (1.0f + expf(delta));
      const float mid =
          valid ? midpoint_values[addr(smid, stream, token, midpoint_base + col)] : 0.0f;
      const float dr =
          valid ? load_value(d_read, addr(sdr, stream, token, value_base + col)) : 0.0f;
      const float rv =
          valid ? load_value(read_values, addr(srv, stream, token, value_base + col)) : 0.0f;
      //: the VJP divides by the amplitude, which is exactly 0 at a TIE: substitute 1 and -- see docs/internals/entmax/entmax.md#safe
      const float safe = (sup && mid != 0.0f) ? mid : 1.0f;
      d_mid[i] = dr * read_gate + (sup ? dlw / safe : 0.0f);
      d_delta[i] = (dr * rv * write_gate - dlw * read_gate) * (sat ? 0.0f : 1.0f);
      //: the entmax rank-1's two sums accumulate HERE, so `s` never becomes an array.
      const float si = sup ? (ALPHA15 ? sqrtf(mid) : 1.0f) : 0.0f;
      partial.p += si * d_mid[i];
      partial.q += si;
    }

    // ---- the entmax rank-1: (numerator, denominator) in ONE fused reduction ----
    Pair red = WarpRedP(redp_store).Reduce(partial, PairAdd());
    float num = __shfl_sync(lmask, red.p, 0, LW);
    float den = __shfl_sync(lmask, red.q, 0, LW);
    den = den > 0.0f ? den : 1.0f;

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = (support_mask >> i) & 1u;
      const float mid = midpoint_values[addr(smid, stream, token, midpoint_base + col)];
      const float si = sup ? (ALPHA15 ? sqrtf(mid) : 1.0f) : 0.0f;
      const float d_ml = si * (d_mid[i] - num / den);
      store_value(d_read_logits, addr_split(sdrl, sb, sh, token, logit_base + col),
                  0.5f * d_ml + d_delta[i]);
      store_value(d_write_logits, addr_split(sdwl, sb, sh, token, logit_base + col),
                  0.5f * d_ml - d_delta[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// HOST DISPATCH -- the width CLASS is chosen here, so the kernel branches on
// nothing. One launch per union level. The (w_pad -> LW, IPT) table and why the launch
// count is unchanged from the Triton lowering: docs/internals/entmax/entmax.md#launch-shape
// ---------------------------------------------------------------------------
template <typename LogitT, typename OutT>
void launch_forward(const Tensor& rl, const Tensor& wl, Tensor& mid, Tensor& sup, Tensor& rv,
                    Tensor& wv, const LevelTable& table, int levels, int w_pad, int heads,
                    double alpha) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/entmax.md#nstreams
  const int n_streams = rl.size(0) * heads, tokens = rl.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  const bool a15 = alpha == 1.5;

  //: THE LAUNCH IS WRITTEN ONCE, and the `(lane width, items per thread)` pair is -- see docs/internals/entmax/entmax.md#kwpad
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      w_pad, "union entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(a15, [&](auto A15) {
          union_forward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, LogitT, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  (const LogitT*)rl.data_ptr(), (const LogitT*)wl.data_ptr(),
                  mid.mutable_data_ptr<float>(), sup.mutable_data_ptr<int32_t>(),
                  (OutT*)rv.data_ptr(), (OutT*)wv.data_ptr(), tokens, heads, table,
                  five_logits(rl, heads), five_logits(wl, heads), five(mid), five(rv), five(wv),
                  five(sup));
        });
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

template <typename LogitT, typename OutT>
void launch_backward(const Tensor& rl, const Tensor& wl, const Tensor& mid, const Tensor& sup,
                     const Tensor& rv, const Tensor& wv, const Tensor& dr, const Tensor& dw,
                     Tensor& drl, Tensor& dwl, const LevelTable& table, int levels, int w_pad,
                     int heads, double alpha) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/entmax.md#nstreams-2
  const int n_streams = drl.size(0) * heads, tokens = drl.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  const bool a15 = alpha == 1.5;

  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      w_pad, "union entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(a15, [&](auto A15) {
          union_backward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, LogitT, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  (const LogitT*)rl.data_ptr(), (const LogitT*)wl.data_ptr(),
                  mid.mutable_data_ptr<float>(), sup.mutable_data_ptr<int32_t>(),
                  (const OutT*)rv.data_ptr(), (const OutT*)wv.data_ptr(),
                  (const OutT*)dr.data_ptr(), (const OutT*)dw.data_ptr(), (LogitT*)drl.data_ptr(),
                  (LogitT*)dwl.data_ptr(), tokens, heads, table, five_logits(rl, heads),
                  five_logits(wl, heads), five(mid), five(rv), five(wv), five(dr), five(dw),
                  five_logits(drl, heads), five_logits(dwl, heads), five(sup));
        });
      });
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace detail

void union_forward(const Tensor& read_logits, const Tensor& write_logits, Tensor& midpoint_values,
                   Tensor& support_words, Tensor& read_values, Tensor& write_values,
                   const std::vector<int64_t>& logit_offsets,
                   const std::vector<int64_t>& value_offsets,
                   const std::vector<int64_t>& midpoint_offsets,
                   const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                   int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(read_logits);
  STD_TORCH_CHECK(read_logits.scalar_type() == write_logits.scalar_type(),
                  "read/write routing logits must share a dtype");
  STD_TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "union entmax supports alpha 1.5 / 2.0 only");
  STD_TORCH_CHECK(read_values.scalar_type() == write_values.scalar_type(),
                  "read/write route values must share a dtype");
  const auto table =
      detail::build_table(logit_offsets, value_offsets, support_offsets, widths, midpoint_offsets);
  const int levels = (int)widths.size(), w_pad = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(read_logits.get_device_index());
  detail::logit_value_switch(
      read_logits.scalar_type(), read_values.scalar_type(), [&](auto L, auto V) {
        detail::launch_forward<decltype(L), decltype(V)>(read_logits, write_logits, midpoint_values,
                                                         support_words, read_values, write_values,
                                                         table, levels, w_pad, (int)heads, alpha);
      });
}

void union_backward(const Tensor& read_logits, const Tensor& write_logits,
                    const Tensor& midpoint_values, const Tensor& support_words,
                    const Tensor& read_values, const Tensor& write_values, const Tensor& d_read,
                    const Tensor& d_write, Tensor& d_read_logits, Tensor& d_write_logits,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& midpoint_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(read_logits);
  STD_TORCH_CHECK(read_logits.scalar_type() == d_read_logits.scalar_type(),
                  "the union VJP writes logit gradients in the logits' own dtype");
  STD_TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "union entmax supports alpha 1.5 / 2.0 only");
  const auto table =
      detail::build_table(logit_offsets, value_offsets, support_offsets, widths, midpoint_offsets);
  const int levels = (int)widths.size(), w_pad = detail::padded_width((int)widths[0]);
  const DeviceGuard guard(read_logits.get_device_index());
  detail::logit_value_switch(read_logits.scalar_type(), read_values.scalar_type(),
                             [&](auto L, auto V) {
                               detail::launch_backward<decltype(L), decltype(V)>(
                                   read_logits, write_logits, midpoint_values, support_words,
                                   read_values, write_values, d_read, d_write, d_read_logits,
                                   d_write_logits, table, levels, w_pad, (int)heads, alpha);
                             });
}

}  // namespace entmax
}  // namespace rola
