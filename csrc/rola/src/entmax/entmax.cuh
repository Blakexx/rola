// THE PRODUCER'S UNION ENTMAX SOLVE -- host declarations. The implementation, both
// kernels and the host dispatch, is `entmax.cu`.
//
// THE STRUCTURAL IDEA: ONE ROW PER (LOGICAL) WARP. Every collective in the solve runs
// along one token's row, so all of them are warp-scope and shuffle-based -- which is
// what makes register demand O(items-per-lane), the warp count a free occupancy knob,
// and the shared-memory need zero.
//
// THE SUPPORT WORD LAYOUT IS FROZEN (the per-leaf write bitmap has nine downstream
// consumers): one CTA owns exactly the 32 tokens of one support word.
//
// Why this is CUDA and not Triton, the measured occupancy history, and the whole
// solve: docs/internals/entmax/entmax.md
//
// The device-side block below is the warp-row vocabulary the union solve and the
// independent solves (`factor.cu`) both speak; the host declarations follow it.
#pragma once

#include <cstdint>
#include <vector>

#include "common/torch_seam.cuh"

#ifdef __CUDACC__
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda_bf16.h>

#include "common/hardware.cuh"
#endif

namespace rola {
namespace entmax {

#ifdef __CUDACC__
//: `rola_api.cpp` includes this header and is compiled by the HOST compiler, so every -- see docs/internals/entmax/entmax.md#near-line-33
namespace detail {

//: FROZEN (nine downstream consumers): one CTA owns exactly the 32 tokens of one -- see docs/internals/entmax/entmax.md#wordtokens
constexpr int WORD_TOKENS = 32;
constexpr int BLOCK_WARPS = 4;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
//: THE SOLVES' REGISTER TARGET a thread: 64, and the backward's a width class -- 72 where a thread holds 8 items' two
//: terms across its reduction. -- see docs/internals/entmax/entmax.md#residency
constexpr int kForwardRegs = 64;

constexpr int backward_regs(int ipt) { return ipt == 8 ? 72 : kForwardRegs; }

//: THE SOLVES' PLAN at build `B`: the blocks an SM their register targets fit in the card's registers, each launch
//: bound's residency. -- see docs/internals/entmax/entmax.md#residency
template <::denseref::arch::Build B>
struct EntmaxPlan {
  static constexpr int kRegsPerSm = ::denseref::arch::traits_of(B).caps.regs_per_sm;
  static constexpr int kMinBlocks = kRegsPerSm / (BLOCK_THREADS * kForwardRegs);

  static constexpr int backward_min_blocks(int ipt) {
    return kRegsPerSm / (BLOCK_THREADS * backward_regs(ipt));
  }
};

//: `(stream, stream2, token, col, tile, lane)` -- ONE address expression serves -- see docs/internals/entmax/entmax.md#strides-2
struct Strides {
  int64_t stream, stream2, token, col, tile, lane;
};

//: THE FLAT ADDRESS, unchanged: ONE stream register, and every term after it -- see docs/internals/entmax/entmax.md#addr
__device__ __forceinline__ int64_t addr(const Strides& s, int64_t stream, int64_t tok,
                                        int64_t col) {
  return stream * s.stream + tok * s.token + (tok >> 5) * s.tile + col * s.col
         + (tok & 31) * s.lane;
}

//: THE SPLIT ADDRESS, for the LOGIT plane alone. `b` and `h` are separate axes -- see docs/internals/entmax/entmax.md#addr-split
__device__ __forceinline__ int64_t addr_split(const Strides& s, int64_t sb, int64_t sh, int64_t tok,
                                              int64_t col) {
  return sb * s.stream + sh * s.stream2 + tok * s.token + col * s.col;
}

template <typename T>
__device__ __forceinline__ void store_value(T* p, int64_t off, float v);

template <>
__device__ __forceinline__ void store_value<float>(float* p, int64_t off, float v) {
  p[off] = v;
}

template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(__nv_bfloat16* p, int64_t off, float v) {
  p[off] = __float2bfloat16(v);
}

//: The ACCUMULATING store, for the tied backward's second contribution. bf16 device -- see docs/internals/entmax/entmax.md#near-line-86
template <typename T>
__device__ __forceinline__ void add_value(T* p, int64_t off, float v);

template <>
__device__ __forceinline__ void add_value<float>(float* p, int64_t off, float v) {
  atomicAdd(p + off, v);
}

template <>
__device__ __forceinline__ void add_value<__nv_bfloat16>(__nv_bfloat16* p, int64_t off, float v) {
  atomicAdd(p + off, __float2bfloat16(v));
}

template <typename T>
__device__ __forceinline__ float load_value(const T* p, int64_t off);

template <>
__device__ __forceinline__ float load_value<float>(const float* p, int64_t off) {
  return p[off];
}

template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(const __nv_bfloat16* p, int64_t off) {
  return __bfloat162float(p[off]);
}

struct Descending {
  __device__ __forceinline__ bool operator()(float a, float b) const { return a > b; }
};

//: THE ROW SORT. Phase B needs one token's row in descending order; `RowSort` -- see docs/internals/entmax/entmax.md#rowsort
template <int LW, int IPT>
struct RowSort {
  using Merge = cub::WarpMergeSort<float, IPT, LW>;
  static constexpr bool kInRegisters = IPT <= 4;
  //: Shared bytes per PHYSICAL warp. Zero for the register network, which is the whole -- see docs/internals/entmax/entmax.md#kslotbytes
  static constexpr int kSlotBytes =
      kInRegisters ? 0 : (WARP_SIZE / LW) * (int)sizeof(typename Merge::TempStorage);

  //: The element's rank is the BLOCKED index `lane*IPT + i`, which is the layout -- see docs/internals/entmax/entmax.md#sort
  __device__ __forceinline__ static void sort(float (&keys)[IPT], int lane, unsigned lmask,
                                              char* slot, int logical_in_warp) {
    if constexpr (!kInRegisters) {
      Merge(reinterpret_cast<typename Merge::TempStorage*>(slot)[logical_in_warp])
          .Sort(keys, Descending());
      return;
    }
    constexpr int N = LW * IPT;
    const int base = lane * IPT;
#pragma unroll
    for (int k = 2; k <= N; k <<= 1) {
#pragma unroll
      for (int j = k >> 1; j >= IPT; j >>= 1) {
        const bool low = (base & j) == 0;
        const bool desc = (base & k) == 0;
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
          const float other = __shfl_xor_sync(lmask, keys[i], j / IPT, LW);
          const bool swap = low ? (desc == (keys[i] < other)) : (desc == (other < keys[i]));
          keys[i] = swap ? other : keys[i];
        }
      }
#pragma unroll
      for (int j = (k >> 1) < IPT ? (k >> 1) : (IPT >> 1); j > 0; j >>= 1) {
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
          if ((i & j) != 0) continue;
          const int p = i | j;
          const bool desc = ((base + i) & k) == 0;
          const bool swap = desc == (keys[i] < keys[p]);
          const float a = swap ? keys[p] : keys[i];
          const float b = swap ? keys[i] : keys[p];
          keys[i] = a;
          keys[p] = b;
        }
      }
    }
  }
};

struct Pair {
  float p, q;
};

struct PairAdd {
  __device__ __forceinline__ Pair operator()(const Pair& a, const Pair& b) const {
    return {a.p + b.p, a.q + b.q};
  }
};

//: THE ROW REDUCTIONS, owned here with the bodies CCCL 2.x shipped as `cub::Max`/`Min`/`Sum` -- see docs/internals/entmax/entmax.md#reductions
struct Max {
  template <typename T, typename U>
  __device__ __forceinline__ typename ::cuda::std::common_type<T, U>::type operator()(T&& t,
                                                                                      U&& u) const {
    return ((u) > (t)) ? (u) : (t);
  }
};

struct Min {
  template <typename T, typename U>
  __device__ __forceinline__ typename ::cuda::std::common_type<T, U>::type operator()(T&& t,
                                                                                      U&& u) const {
    return ((u) < (t)) ? (u) : (t);
  }
};

struct Sum {
  template <typename T, typename U>
  __device__ __forceinline__ auto operator()(T&& t, U&& u) const
      -> decltype(::cuda::std::forward<T>(t) + ::cuda::std::forward<U>(u)) {
    return ::cuda::std::forward<T>(t) + ::cuda::std::forward<U>(u);
  }
};

__device__ __forceinline__ bool bf16_nonzero(float v) {
  return __bfloat162float(__float2bfloat16(v)) != 0.0f;
}

//: HAZARD logical-warp-mask -- docs/internals/entmax/entmax.md#logical-warp-mask -- see docs/internals/entmax/entmax.md#near-line-183
template <int LW>
__device__ __forceinline__ unsigned logical_warp_mask() {
  return (LW == WARP_SIZE) ? 0xffffffffu
                           : (((1u << LW) - 1u) << (((threadIdx.x % WARP_SIZE) / LW) * LW));
}

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

//: THE ALPHA-ENTMAX THRESHOLD (alpha 1.5 or 2) of one row held striped: `valid` marks its `valid_count` members,
//: shifted in place by the row max; tau is relative to that shift. -- see docs/internals/entmax/entmax.md#threshold
template <int LW, int IPT, bool ALPHA15>
__device__ __forceinline__ float threshold(float (&z)[IPT], uint32_t valid, int valid_count,
                                           int lane, unsigned lmask, char* slot,
                                           int logical_in_warp) {
  using WarpScanP = cub::WarpScan<Pair, LW>;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedU = cub::WarpReduce<unsigned long long, LW>;
  static_assert(sizeof(typename WarpScanP::TempStorage) <= 4, "WarpScan must be shfl-only");
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedU::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpScanP::TempStorage scan_store;
  typename WarpRedF::TempStorage red_store;
  typename WarpRedU::TempStorage redu_store;
  const bool has_valid = valid_count > 0;

  float local_max = -INFINITY, local_min = INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((valid >> i) & 1u) {
      local_max = fmaxf(local_max, z[i]);
      local_min = fminf(local_min, z[i]);
    }
  }
  float row_max = WarpRedF(red_store).Reduce(local_max, Max());
  row_max = __shfl_sync(lmask, row_max, 0, LW);
  row_max = has_valid ? row_max : 0.0f;
  float row_min = WarpRedF(red_store).Reduce(local_min, Min());
  row_min = __shfl_sync(lmask, row_min, 0, LW);

  //: A finite row-relative sentinel keeps padded lanes strictly below every real -- see docs/internals/entmax/entmax.md#near-line-161
  const float sentinel = has_valid ? (row_min - row_max) - 1.0f : -1.0f;
  float keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool member = (valid >> i) & 1u;
    z[i] = member ? z[i] - row_max : 0.0f;
    keys[i] = member ? z[i] : sentinel;
  }
  //: A sort is a PERMUTATION, so striped input is fine and the blocked output is what -- see docs/internals/entmax/entmax.md#near-line-165
  RowSort<LW, IPT>::sort(keys, lane, lmask, slot, logical_in_warp);

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
  //: ONE reduction selects the greatest valid k AND carries its tau: the column in the -- see docs/internals/entmax/entmax.md#packedkey
  unsigned long long packed_key = ((unsigned long long)(uint32_t)(best_col + 1) << 32)
                                  | (unsigned long long)__float_as_uint(best_tau);
  packed_key = WarpRedU(redu_store).Reduce(packed_key, Max());
  packed_key = __shfl_sync(lmask, packed_key, 0, LW);
  const int chosen = (int)(uint32_t)(packed_key >> 32) - 1;
  return chosen >= 0 ? __uint_as_float((uint32_t)(packed_key & 0xffffffffu)) : 0.0f;
}

//: THE UNION'S WRITE SIDE: `delta` in, the logit difference; out, the write weights -- the support's log-values plus
//: the write gate, normalized over the row. -- see docs/internals/entmax/entmax.md#union-write
//: `kFast`: the logarithms, exponentials and quotient on the special-function unit.
template <int LW, int IPT, bool kFast = false>
__device__ __forceinline__ void union_write(const float (&values)[IPT], float (&delta)[IPT],
                                            uint32_t support, unsigned lmask) {
  using WarpRedF = cub::WarpReduce<float, LW>;
  typename WarpRedF::TempStorage red_store;
  float local_lw = -INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    const float gate = kFast ? -(fmaxf(delta[i], 0.0f) + __logf(1.0f + __expf(-fabsf(delta[i]))))
                             : log_write_gate(delta[i]);
    delta[i] = sup ? (kFast ? __logf(values[i]) : logf(values[i])) + gate : -INFINITY;
    if (sup) local_lw = fmaxf(local_lw, delta[i]);
  }
  float max_lw = WarpRedF(red_store).Reduce(local_lw, Max());
  max_lw = __shfl_sync(lmask, max_lw, 0, LW);
  const bool has_mass = max_lw > -INFINITY;
  const float safe_max = has_mass ? max_lw : 0.0f;
  float local_den = 0.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    const float shifted = delta[i] - safe_max;
    delta[i] = (sup && has_mass) ? (kFast ? __expf(shifted) : expf(shifted)) : 0.0f;
    local_den += delta[i];
  }
  float den = WarpRedF(red_store).Reduce(local_den, Sum());
  den = __shfl_sync(lmask, den, 0, LW);
  const bool has_weight = den > 0.0f;
  const float safe_den = has_weight ? den : 1.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    delta[i] =
        (sup && has_weight) ? (kFast ? __fdividef(delta[i], safe_den) : delta[i] / safe_den) : 0.0f;
  }
}

//: THE PER-LAUNCH LEVEL TABLE. It travels in the kernel's PARAMETER space (448 -- see docs/internals/entmax/entmax.md#max-batched-levels
constexpr int MAX_BATCHED_LEVELS = 16;

struct LevelTable {
  int64_t logit_off[MAX_BATCHED_LEVELS];
  int64_t value_off[MAX_BATCHED_LEVELS];
  int64_t support_off[MAX_BATCHED_LEVELS];
  //: The union solve's midpoint plane is packed over the UNION levels alone, so it does -- see docs/internals/entmax/entmax.md#near-line-202
  int64_t midpoint_off[MAX_BATCHED_LEVELS];
  int width[MAX_BATCHED_LEVELS];
};

//: The tuple a tensor's own RANK implies: rank-4 is the tile-major arena (no -- see docs/internals/entmax/entmax.md#five
inline Strides five(const Tensor& t) {
  if (t.dim() == 4) return {t.stride(0), 0, 0, t.stride(2), t.stride(1), t.stride(3)};
  return {t.stride(0), 0, t.stride(1), t.stride(2), 0, 0};
}

//: THE LOGIT PLANE, in whichever of its two layouts the caller holds it: token- -- see docs/internals/entmax/entmax.md#five-logits
inline Strides five_logits(const Tensor& t, int heads) {
  STD_TORCH_CHECK(heads >= 1, "the stream split needs heads >= 1; 1 is one head");
  if (t.dim() == 4) {
    STD_TORCH_CHECK(t.size(2) == heads, "a token-major logit plane's head axis must be `heads`");
    return {t.stride(0), t.stride(2), t.stride(1), t.stride(3), 0, 0};
  }
  STD_TORCH_CHECK(heads == 1, "a flat [BH, T, W] logit plane declares no split (heads == 1)");
  return {t.stride(0), 0, t.stride(1), t.stride(2), 0, 0};
}

inline Strides five_or_zero(const std::optional<Tensor>& t) {
  if (!t.has_value()) return {0, 0, 0, 0, 0, 0};
  return five(*t);
}

template <typename T>
T* ptr_or_null(const std::optional<Tensor>& t) {
  return t.has_value() ? (T*)t->data_ptr() : nullptr;
}

//: The padded width is what selects the `(lane width, items per thread)` pair, and every -- see docs/internals/entmax/entmax.md#paddedwidth
inline int padded_width(int width) {
  int w = 1;
  while (w < width) w <<= 1;
  return w < 2 ? 2 : w;
}

inline LevelTable build_table(const std::vector<int64_t>& logit_off,
                              const std::vector<int64_t>& value_off,
                              const std::vector<int64_t>& support_off,
                              const std::vector<int64_t>& widths,
                              const std::vector<int64_t>& midpoint_off = {}) {
  const int n = (int)widths.size();
  STD_TORCH_CHECK(n > 0 && n <= MAX_BATCHED_LEVELS, "batched level count must be in [1, ",
                  MAX_BATCHED_LEVELS, "], got ", n);
  STD_TORCH_CHECK((int)logit_off.size() == n && (int)value_off.size() == n,
                  "level offset tables must match the width table");
  STD_TORCH_CHECK(support_off.empty() || (int)support_off.size() == n,
                  "support offset table must match the width table");
  const int klass = padded_width((int)widths[0]);
  LevelTable table{};
  for (int i = 0; i < n; ++i) {
    STD_TORCH_CHECK(padded_width((int)widths[i]) == klass,
                    "a batched solve must be one width class; level ", i, " has width ", widths[i],
                    " against class ", klass);
    table.logit_off[i] = logit_off[i];
    table.value_off[i] = value_off[i];
    table.support_off[i] = support_off.empty() ? 0 : support_off[i];
    table.midpoint_off[i] = midpoint_off.empty() ? value_off[i] : midpoint_off[i];
    table.width[i] = (int)widths[i];
  }
  return table;
}

}  // namespace detail
#endif  // __CUDACC__

//: TWO DTYPE AXES, ONE CALL: the LOGIT plane's and the route values'. -- see docs/internals/entmax/entmax.md#check-logits
#ifdef __CUDACC__
namespace detail {
//: THE LOGIT PLANE IS bf16 ON THE SHIPPED PATH -- the router GEMM runs bf16 operands on -- see docs/internals/entmax/entmax.md#checklogits
inline void check_logits(const Tensor& logits) {
  STD_TORCH_CHECK(logits.is_cuda(), "routing logits must be CUDA");
  STD_TORCH_CHECK(logits.scalar_type() == Dtype::Float || logits.scalar_type() == Dtype::BFloat16,
                  "routing logits must be fp32 or bf16");
}

template <typename Body>
inline void logit_value_switch(Dtype logit, Dtype value, Body&& body) {
  STD_TORCH_CHECK(logit == Dtype::Float || logit == Dtype::BFloat16,
                  "routing logits must be fp32 or bf16");
  STD_TORCH_CHECK(value == Dtype::Float || value == Dtype::BFloat16,
                  "route values must be fp32 or bf16");
  if (logit == Dtype::Float) {
    if (value == Dtype::Float)
      body(float{}, float{});
    else
      body(float{}, __nv_bfloat16{});
  } else {
    if (value == Dtype::Float)
      body(__nv_bfloat16{}, float{});
    else
      body(__nv_bfloat16{}, __nv_bfloat16{});
  }
}
}  // namespace detail
#endif  // __CUDACC__

//: ONE LAUNCH, MANY LEVELS: the entries take BASE tensors plus per-level COLUMN -- see docs/internals/entmax/entmax.md#union-forward
void union_forward(const Tensor& read_logits, const Tensor& write_logits, Tensor& midpoint_values,
                   Tensor& support_words, Tensor& read_values, Tensor& write_values,
                   const std::vector<int64_t>& logit_offsets,
                   const std::vector<int64_t>& value_offsets,
                   const std::vector<int64_t>& midpoint_offsets,
                   const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                   int64_t heads, double alpha);

void union_backward(const Tensor& read_logits, const Tensor& write_logits,
                    const Tensor& midpoint_values, const Tensor& support_words,
                    const Tensor& read_values, const Tensor& write_values, const Tensor& d_read,
                    const Tensor& d_write, Tensor& d_read_logits, Tensor& d_write_logits,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& midpoint_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha);

}  // namespace entmax
}  // namespace rola
