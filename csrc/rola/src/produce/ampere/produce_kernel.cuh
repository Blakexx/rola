// csrc/rola/src/produce/ampere/produce_kernel.cuh -- THE PRODUCER'S KERNEL as it runs on ampere, composed of its
// components. see docs/internals/produce/ampere/produce_kernel.md
#pragma once

#include "common/burst.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/static_for.cuh"
#include "common/structure_switch.cuh"
#include "entmax/entmax.cuh"
#include "facts/liveness_contract.cuh"
#include "produce/produce.cuh"

namespace rola::produce::detail {

namespace ops = denseref::ops;
namespace row_solve = rola::entmax::detail;
using Clock = PhaseClock<kProducePhases>;

//: A (token, head)'s slot in the head-major outputs `[B, H, T]`: the flat token split into its batch and step.
__device__ __forceinline__ int64_t slot_of(const ProduceParams& p, int token, int head) {
  const int b =
      p.T == 1 ? token
               : static_cast<int>(__umulhi(static_cast<uint32_t>(token), p.t_mul) >> p.t_shift);
  return (static_cast<int64_t>(b) * p.H + head) * p.T + (token - b * p.T);
}

//: THE BLOCK'S JOB: its control from its shape word (the parameters', read at the block's index: uniform to ptxas), its
//: addressing from the table. -- see docs/internals/produce/ampere/produce_kernel.md#jobs
__device__ __forceinline__ Job load_job(const ProduceParams& p, int index) {
  //: the word read fresh at each call (`opaque`), so no k-loop holds it
  const int shape = p.shapes[ops::opaque(index)] & 0xffff;
  const int32_t* row = p.jobs + static_cast<int64_t>(index) * kJobFields;
  return Job{(shape >> 5) & 1, shape & 7,        (shape >> 3) & 3,   (shape >> 6) & 1,
             (shape >> 7) & 1, row[kFieldHead],  row[kFieldHeads],   row[kFieldWidth],
             row[kFieldIn0],   row[kFieldIn1],   row[kFieldOutRead], row[kFieldOutWrite],
             row[kFieldLevel], (shape >> 8) & 7, (shape >> 11) & 1,  (shape >> 12) & 1,
             (shape >> 13) & 1};
}

//: A job's projection columns: a route job's row (both rows of a union's), or a value job's heads' channels.
__device__ __forceinline__ int job_columns(const ProduceParams& p, const Job& job) {
  if (job.kind == kJobValue) return job.heads * p.d_v;
  return job.in1 >= 0 ? 2 * job.width : job.width;
}

//: A ONE-ROW JOB'S ACCUMULATOR COLUMN `a` AS ITS JOB COLUMN, or -1 (`columns` the job's, `job_columns`): a narrow
//: union's write row starts at its width class's tile offset (`32 << rows`), so a column's two logits share a lane.
//: It reads no parameter: a lambda holding the parameters' address would copy them to local memory.
//: -- see docs/internals/produce/ampere/produce_kernel.md#form
__device__ __forceinline__ int job_column(const Job& job, int columns, int a) {
  if (job.kind == kJobRoute && job.in1 >= 0) {
    const int offset = 32 << job.rows;
    return a < job.width                           ? a
           : a >= offset && a < offset + job.width ? job.width + (a - offset)
                                                   : -1;
  }
  return a < columns ? a : -1;
}

__device__ __forceinline__ int route_column(const Job& job, int j) {
  return j < job.width ? job.in0 + j : job.in1 + (j - job.width);
}

//: THE WEIGHT ROW of job column `j`: its hidden run, contiguous.
__device__ __forceinline__ const __nv_bfloat16* weight_row(const ProduceParams& p, const Job& job,
                                                           int j) {
  if (job.kind == kJobValue) return p.v_W + static_cast<int64_t>(job.head * p.d_v + j) * p.v_row;
  return p.route_W + job.head * p.w_head + static_cast<int64_t>(route_column(job, j)) * p.w_col;
}

//: The job's bias row, or null: a value job has none, and a route job only when the routing declares one.
__device__ __forceinline__ const float* bias_row(const ProduceParams& p, const Job& job) {
  return job.kind == kJobRoute && p.route_bias != nullptr ? p.route_bias + job.head * p.C : nullptr;
}

//: Column `j`'s bias: a predicated load, never a branch around one.
__device__ __forceinline__ float column_bias(const float* bias, const Job& job, int j, bool live) {
  return bias != nullptr && live ? __ldg(bias + route_column(job, j)) : 0.0f;
}

//: A STAGE ROW'S 16-BYTE CHUNK `c`: a row is `S` k-steps (32 S bytes), its chunk at `c` XOR the row's swizzle --
//: eight rows at one chunk in eight bank groups at every depth. -- see docs/internals/produce/ampere/produce_kernel.md#stage
template <int S>
__device__ __forceinline__ uint32_t chunk_at(int row, int c) {
  const int swizzle = S == 1 ? (row >> 2) & 1 : S == 2 ? (row >> 1) & 3 : row & 7;
  return static_cast<uint32_t>(row * 32 * S + ((c ^ swizzle) << 4));
}

//: A JOB'S FORM: its columns in one warp's accumulators (a one-row job, a narrow union's both rows, a value job's
//: heads), or a wide union's halves across a warp pair. -- see docs/internals/produce/ampere/produce_kernel.md#form
__device__ __forceinline__ bool pair_form(const Job& job) { return job.pair != 0; }

//: THE JOB'S WEIGHT BASE, uniform a block: its head's routing columns or its heads' value channels.
__device__ __forceinline__ const __nv_bfloat16* weight_base(const ProduceParams& p,
                                                            const Job& job) {
  return job.kind == kJobValue ? p.v_W + static_cast<int64_t>(job.head) * p.d_v * p.v_row
                               : p.route_W + job.head * p.w_head;
}

//: A THREAD'S COPY SOURCES, formed once a job as 32-bit element offsets (from the job's weight base and from x; the
//: host bounds both): its `QW` weight chunks' and `QX` x chunks' runs, the k offset added a stage, and one mask of
//: what issues. A column past the job's is never issued (only dead accumulators read it).
template <int QW, int QX>
struct Sources {
  uint32_t w[QW], x[QX];
  //: bit q < QW: weight chunk q issues; bit QW + q: x chunk q issues; bit QW + QX + q: its token is live
  uint32_t mask;
};

template <int S, int QW, int QX, class Column>
__device__ __forceinline__ Sources<QW, QX> sources(const ProduceParams& p, const Job& job,
                                                   Column column, int token0, int x_rows) {
  constexpr int kChunks = 2 * S;
  const int t = static_cast<int>(threadIdx.x);
  Sources<QW, QX> s;
  s.mask = 0u;
  rola::static_for<QW>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, col = column(u / kChunks);
    const bool issue = col >= 0;
    const int64_t run = !issue ? 0
                        : job.kind == kJobValue
                            ? static_cast<int64_t>(col) * p.v_row
                            : static_cast<int64_t>(route_column(job, col)) * p.w_col;
    s.w[q] = static_cast<uint32_t>(run) + 8u * static_cast<uint32_t>(u % kChunks);
    s.mask |= (issue ? 1u : 0u) << q;
  });
  rola::static_for<QX>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, row = u / kChunks, token = token0 + row;
    const bool issue = row < x_rows, live = issue && token < p.BT;
    s.x[q] = static_cast<uint32_t>(live ? token : 0) * static_cast<uint32_t>(p.hidden)
             + 8u * static_cast<uint32_t>(u % kChunks);
    s.mask |= (issue ? 1u : 0u) << (QW + q) | (live ? 1u : 0u) << (QW + QX + q);
  });
  return s;
}

//: ONE STAGE OF `S` K-STEPS: the job's columns' 16 S hidden and the stage's x rows', one copy group of every thread; a
//: chunk past the hidden size zero-fills. `issue` false copies nothing (the loop's last stage).
template <int S, int XOff, int QW, int QX>
__device__ __forceinline__ void stage_step(const ProduceParams& p, const __nv_bfloat16* base,
                                           const Sources<QW, QX>& s, int k, uint32_t stage,
                                           bool issue) {
  constexpr int kChunks = 2 * S;
  const int t = static_cast<int>(threadIdx.x);
  rola::static_for<QW>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, c = u % kChunks;
    const bool live = k + 8 * c < p.hidden;
    ops::stage_tile_when(stage + chunk_at<S>(u / kChunks, c),
                         base + (s.w[q] + (live ? static_cast<uint32_t>(k) : 0u)),
                         issue && ((s.mask >> q) & 1u), live);
  });
  rola::static_for<QX>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, c = u % kChunks;
    const bool live = k + 8 * c < p.hidden && ((s.mask >> (QW + QX + q)) & 1u);
    ops::stage_tile_when(stage + XOff + chunk_at<S>(u / kChunks, c),
                         p.x + (s.x[q] + (live ? static_cast<uint32_t>(k) : 0u)),
                         issue && ((s.mask >> (QW + q)) & 1u), live);
  });
  ops::stage_commit();
}

//: ONE STAGE'S MMAs of a warp, each of its `S` k-steps: B its 16 tokens' 16 hidden (`ldmatrix.x4`, both n-tiles' two
//: k halves), then each of its `M` column tiles' A (`ldmatrix.x4`) into both n-tiles' MMAs, the tile's stage rows from
//: `row0(i)`. -- see docs/internals/produce/ampere/produce_kernel.md#accumulate
//: @burst
template <int M, int S, int XOff, class Row0>
__device__ __forceinline__ void mma_stage(uint32_t stage, int token_row, int lane, Row0 row0,
                                          float (&acc)[kAccTiles][kTokenTiles][4]) {
  //: the stage's (k-step, column tile) pairs in order, each one's fragments loaded a pair ahead of its MMAs
  constexpr int kPairs = S * M;
  auto b_at = [&](int ks) {
    return stage + XOff
           + chunk_at<S>(token_row + (lane & 7) + 8 * (lane >> 4), 2 * ks + ((lane >> 3) & 1));
  };
  auto a_at = [&](int j) {
    return stage
           + chunk_at<S>(row0(j % M) + (lane & 7) + 8 * ((lane >> 3) & 1),
                         2 * (j / M) + (lane >> 4));
  };
  uint32_t a[2][4], b[2][4];
  ops::load_frag(b[0], b_at(0));
  ops::load_frag(a[0], a_at(0));
  rola::static_for<kPairs>([&](auto J) {
    constexpr int j = decltype(J)::value, ks = j / M, i = j % M;
    if constexpr (j + 1 < kPairs) {
      if constexpr ((j + 1) % M == 0) ops::load_frag(b[((j + 1) / M) & 1], b_at((j + 1) / M));
      ops::load_frag(a[(j + 1) & 1], a_at(j + 1));
    }
    ops::mma(acc[i][0], a[j & 1], b[ks & 1]);
    ops::mma(acc[i][1], a[j & 1], b[ks & 1] + 2);
  });
}

//: THE K-LOOP: two stages, one copy group and one CTA barrier a stage (KERNEL_STANDARDS 4's sm_80 form) -- the stage
//: landed and the other released at the barrier, its refill issued then, a stage ahead of its MMAs.
//: -- see docs/internals/produce/ampere/produce_kernel.md#accumulate
template <int M, int S, int XOff, int QW, int QX, class Row0>
__device__ __forceinline__ void k_loop(const ProduceParams& p, const __nv_bfloat16* base,
                                       const Sources<QW, QX>& s, uint32_t stages, int token_row,
                                       Row0 row0, float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int steps = (p.hidden + kKStep * S - 1) / (kKStep * S);
  //: a column past the job's is never copied: its rows zeroed in both stages once, so its accumulators keep their seed
  rola::static_for<QW>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = static_cast<int>(threadIdx.x) + q * kProduceThreads;
    const uint32_t at = chunk_at<S>(u / (2 * S), u % (2 * S));
    if (!((s.mask >> q) & 1u)) {
      ops::store_shared_vec16(stages + at, uint4{0u, 0u, 0u, 0u});
      ops::store_shared_vec16(stages + kStageSlot + at, uint4{0u, 0u, 0u, 0u});
    }
  });
  stage_step<S, XOff>(p, base, s, 0, stages, true);
#pragma unroll 1
  for (int step = 0; step < steps; ++step) {
    ops::stage_wait<0>();
    __syncthreads();
    const uint32_t here = stages + static_cast<uint32_t>((step & 1) * kStageSlot);
    stage_step<S, XOff>(p, base, s, (step + 1) * kKStep * S,
                        stages + static_cast<uint32_t>(((step + 1) & 1) * kStageSlot),
                        step + 1 < steps);
    mma_stage<M, S, XOff>(here, token_row, lane, row0, acc);
  }
}

//: THE ACCUMULATORS SEEDED with the routing bias, a lane's columns `col(i, h)` of tile `i`'s rows `g` and `g + 8` (0
//: where none): a predicated load a column, before the k-loop, where no accumulator is live yet. A route job's dead
//: columns seed at -inf and stay there (their weights are zeros), so no pass of the solve tests membership.
//: -- see docs/internals/produce/ampere/produce_kernel.md#solve
template <class Col>
__device__ __forceinline__ void seed_bias(const ProduceParams& p, const Job& job, int columns,
                                          Col col, float (&acc)[kAccTiles][kTokenTiles][4]) {
  const float* bias = job.kind == kJobRoute ? bias_row(p, job) : nullptr;
  const float dead = job.kind == kJobRoute ? -INFINITY : 0.0f;
  rola::static_for<kAccTiles>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<2>([&](auto H) {
      constexpr int h = decltype(H)::value;
      const int j = col(i, h);
      const bool live = j >= 0 && j < columns;
      const float b = live ? column_bias(bias, job, j, true) : dead;
      rola::static_for<kTokenTiles>([&](auto N) {
        constexpr int n = decltype(N)::value;
        acc[i][n][2 * h] = b;
        acc[i][n][2 * h + 1] = b;
      });
    });
  });
}

//: THE JOB'S LOGITS from the accumulators, for its gate: a lane's tile `i` rows at columns `col(i, h)`, tokens
//: `token(n, t)` -- written only when the probe is asked for.
template <class Col, class Token>
__device__ __forceinline__ void probe_logits(const ProduceParams& p, int job_index, int columns,
                                             Col col, Token token,
                                             const float (&acc)[kAccTiles][kTokenTiles][4]) {
  if (p.logit_probe == nullptr) return;
  float* out = p.logit_probe + static_cast<int64_t>(job_index) * p.BT * kMaxColumns;
  rola::static_for<kAccTiles>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<kTokenTiles * 4>([&](auto E) {
      constexpr int n = decltype(E)::value / 4, e = decltype(E)::value % 4;
      const int j = col(i, e >> 1), tk = token(n, e & 1);
      if (j >= 0 && j < columns && tk < p.BT)
        out[static_cast<int64_t>(tk) * kMaxColumns + j] = acc[i][n][e];
    });
  });
}

//: A value the whole warp holds, reduced by a butterfly: every lane ends with the same bits.
template <class Op>
__device__ __forceinline__ float warp_all(float v, Op op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

//: A ROW'S VALUE REDUCED over the 8 lanes holding it (those sharing `lane & 3`): three shuffles, every lane the same
//: bits. -- see docs/internals/produce/ampere/produce_kernel.md#solve
template <class Op>
__device__ __forceinline__ float row_all(float v, Op op) {
#pragma unroll
  for (int offset = 4; offset < 32; offset <<= 1)
    v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

//: A WIDE UNION'S ROW SPANS A WARP PAIR (the pair form: warp `2 j + h` holds columns `128 h ..` of both its rows), so a
//: row's reductions combine the pair's halves: each warp's row values through an exchange in shared memory, one pair
//: barrier a round, its two buffers alternating so no barrier guards a buffer's reuse. `on` false for every other
//: job, whose reductions are the warp's own. -- see docs/internals/produce/ampere/produce_kernel.md#wide-union
constexpr int kPairBarrier = 1;
constexpr int kExchangeValues = 4;
constexpr int kExchangeRound = 2 * 4 * kExchangeValues;
constexpr int kExchangeBytes = kProduceWarps / 2 * 2 * kExchangeRound * 4;

struct Pair {
  float* slots;  // the pair's [2 rounds][2 halves][4 rows][kExchangeValues]
  int half, bar, round;
  bool on;

  //: each value `v[j]` (already reduced over the warp's half) combined with the partner warp's, both warps the same
  //: bits (the ops commute)
  template <int N, class Op>
  __device__ __forceinline__ void combine(float (&v)[N], int lane, Op op) {
    static_assert(N <= kExchangeValues, "an exchange's values fit its slot");
    if (!on) return;
    float* const buf = slots + (round & 1) * kExchangeRound;
    float* const mine = buf + (half * 4 + (lane & 3)) * kExchangeValues;
    const float* const theirs = buf + ((1 - half) * 4 + (lane & 3)) * kExchangeValues;
    if ((lane >> 2) == 0) {
      rola::static_for<N>([&](auto J) { mine[decltype(J)::value] = v[decltype(J)::value]; });
    }
    ops::rendezvous_group(bar, 64);
    rola::static_for<N>([&](auto J) {
      v[decltype(J)::value] = op(v[decltype(J)::value], theirs[decltype(J)::value]);
    });
    ++round;
  }

  template <class Op>
  __device__ __forceinline__ float combine1(float v, int lane, Op op) {
    float a[1] = {v};
    combine<1>(a, lane, op);
    return a[0];
  }
};

//: A LANE'S TWO TOKEN ROWS of one n-tile: slot `S` its token `2 t + S`, value `V` the column tile `V >> 1`'s row half
//: `V & 1` -- an accumulator, named. -- see docs/internals/produce/ampere/produce_kernel.md#solve
template <int S, int V>
__device__ __forceinline__ float& row_value(float (&acc)[kAccTiles][4]) {
  return acc[V >> 1][2 * (V & 1) + S];
}

template <int S, int V>
__device__ __forceinline__ float row_value(const float (&acc)[kAccTiles][4]) {
  return acc[V >> 1][2 * (V & 1) + S];
}

constexpr int kSlots = 2;

//: A LANE'S `N` VALUES `B ..` OF SLOT `S` REDUCED PAIRWISE: log2 N dependent steps, where a running fold is N.
template <int S, int B, int N, class Op>
__device__ __forceinline__ float row_tree(const float (&acc)[kAccTiles][4], Op op) {
  if constexpr (N == 1)
    return row_value<S, B>(acc);
  else
    return op(row_tree<S, B, N / 2>(acc, op), row_tree<S, B + N / 2, N - N / 2>(acc, op));
}

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct AddOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

struct MinOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a, b); }
};

//: A VALUE'S SUMS PAST `lo`: where `z > lo`, its count, value and square added -- one compare and three predicated
//: adds, never a select (an ALU-pipe instruction, at half the FP32 rate). -- see docs/internals/produce/ampere/produce_kernel.md#threshold
__device__ __forceinline__ void sums_past(float z, float lo, float& k, float& P, float& Q) {
  asm("{\n .reg .pred p;\n setp.gt.f32 p, %3, %4;\n @p add.rn.f32 %0, %0, 0f3F800000;\n"
      " @p add.rn.f32 %1, %1, %3;\n @p fma.rn.f32 %2, %3, %3, %2;\n}"
      : "+f"(k), "+f"(P), "+f"(Q)
      : "f"(z), "f"(lo));
}

//: One where `z >= tau`: one compare, one predicated add.
__device__ __forceinline__ void count_at(float z, float tau, float& n) {
  asm("{\n .reg .pred p;\n setp.ge.f32 p, %1, %2;\n @p add.rn.f32 %0, %0, 0f3F800000;\n}"
      : "+f"(n)
      : "f"(z), "f"(tau));
}

//: SOFTMAX on a lane's two rows of `M` column tiles, in place (a dead column at -inf, its level 0); `top` each row's
//: largest level, its own quotient. -- see docs/internals/produce/ampere/produce_kernel.md#solve
template <int M>
__device__ __forceinline__ void softmax_rows(float (&acc)[kAccTiles][4], float (&top)[kSlots]) {
  constexpr float kLog2e = 1.4426950408889634f;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const float m = row_all(row_tree<s, 0, 2 * M>(acc, MaxOp{}), MaxOp{}), ml = m * kLog2e;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      row_value<s, v>(acc) = ops::exp2_approx(fmaf(row_value<s, v>(acc), kLog2e, -ml));
    });
    const float sum = row_all(row_tree<s, 0, 2 * M>(acc, AddOp{}), AddOp{});
    const float inv = __fdividef(1.0f, sum);
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      row_value<s, v>(acc) *= inv;
    });
    top[s] = inv;
  });
}

//: THE ALPHA-ENTMAX LEVELS (alpha 1.5 where `a15`, else 2: uniform, the shape word's) on a lane's two rows, in place,
//: by Newton from the left: each row shifted by its max (a dead column stays -inf, below every threshold); `f(tau) = sum
//: g(z - tau) - 1` convex and decreasing from `tau = -2` (alpha 2: `-1`), so a step never passes the root and the closed
//: form on the support `{z > lo}` is the root once no member lies below it. One vote over the warp's 8 rows ends the
//: loop (a settled row's further steps keep its threshold). `top` each row's largest level.
//: -- see docs/internals/produce/ampere/produce_kernel.md#threshold
template <int M>
__device__ __forceinline__ void entmax_rows(float (&acc)[kAccTiles][4], bool a15,
                                            float (&top)[kSlots], Pair& pr, int lane,
                                            Clock& clock) {
  float lo[kSlots], tau[kSlots];
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const float m =
        pr.combine1(row_all(row_tree<s, 0, 2 * M>(acc, MaxOp{}), MaxOp{}), lane, MaxOp{});
    rola::static_for<2 * M>([&](auto V) { row_value<s, decltype(V)::value>(acc) -= m; });
    lo[s] = a15 ? -2.0f : -1.0f;
    tau[s] = 0.0f;
  });

#pragma unroll 1
  for (int step = 0; step < kNewtonSteps; ++step) {
    //: settled: no member in (lo, tau) -- where tau > lo, the members at or past tau are all the support counted
    bool settled = true;
    rola::static_for<kSlots>([&](auto Sl) {
      constexpr int s = decltype(Sl)::value;
      float k = 0.0f, P = 0.0f, Q = 0.0f;
      rola::static_for<2 * M>(
          [&](auto V) { sums_past(row_value<s, decltype(V)::value>(acc), lo[s], k, P, Q); });
      const float here = k;
      float kpq[3] = {row_all(k, AddOp{}), row_all(P, AddOp{}), row_all(Q, AddOp{})};
      pr.combine<3>(kpq, lane, AddOp{});
      k = kpq[0], P = kpq[1], Q = kpq[2];
      tau[s] = a15 ? __fdividef(P - ops::sqrt_approx(fmaxf(P * P - k * (Q - 4.0f), 0.0f)), k)
                   : __fdividef(P - 1.0f, k);
      float past = 0.0f;
      rola::static_for<2 * M>(
          [&](auto V) { count_at(row_value<s, decltype(V)::value>(acc), tau[s], past); });
      settled = settled && (tau[s] <= lo[s] || past == here);
      lo[s] = a15 ? lo[s]
                        + __fdividef((Q - 2.0f * P * lo[s] + k * lo[s] * lo[s]) * 0.25f - 1.0f,
                                     0.5f * (P - k * lo[s]))
                  : tau[s];
    });
    //: the pair's two votes one, so both warps leave the loop together -- voted again, so ptxas sees the exit
    //: warp-uniform (a value read from shared memory is not provably so)
    const bool warp_settled = __all_sync(0xffffffffu, settled);
    const float pair_settled = pr.combine1(warp_settled ? 1.0f : 0.0f, lane, MinOp{});
    if (__all_sync(0xffffffffu, pair_settled > 0.5f)) break;
  }
  clock.lap(kPhaseThreshold);

  //: the level `relu(h (z - tau))`, squared at alpha 1.5: `base * fma(base, w1, w0)` is `base^2` at (1, 0) and `base`
  //: at (0, 1), both exactly, so no select a value
  const float h = a15 ? 0.5f : 1.0f, w1 = a15 ? 1.0f : 0.0f, w0 = a15 ? 0.0f : 1.0f;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const float ht = h * tau[s];
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const float base = fmaxf(fmaf(h, row_value<s, v>(acc), -ht), 0.0f);
      row_value<s, v>(acc) = base * fmaf(base, w1, w0);
    });
    const float peak = fmaxf(-ht, 0.0f);
    top[s] = peak * fmaf(peak, w1, w0);
  });
}

//: A NARROW UNION'S SPLIT on a lane's two rows, its read logits at values `v` and its write logits at `v + 2 M` (the
//: class's tile offset), in two halves around the row's one entmax: `union_midpoints` leaves each column's midpoint
//: at `v` and its logits' overflow-free difference at `v + 2 M`; past the entmax `union_sides` forms the read side,
//: the level times the sigmoid of the difference, and the write side, the level times the write gate normalized over
//: the support in the log domain -- the reference's `_torch_union_split`. `top_write` is the argmax's own quotient
//: `1 / den`, `top_read` the read side's max. -- see docs/internals/produce/ampere/produce_kernel.md#union
template <int M>
__device__ __forceinline__ void union_midpoints(float (&acc)[kAccTiles][4]) {
  static_assert(4 * M <= 2 * kAccTiles, "a narrow union's two rows fit a warp's accumulators");
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const float r = row_value<s, v>(acc), w = row_value<s, v + 2 * M>(acc);
      const float z = 0.5f * r + 0.5f * w;
      row_value<s, v>(acc) = z;
      //: a dead column (both logits -inf) takes a zero difference, its level 0 on both sides
      row_value<s, v + 2 * M>(acc) =
          z == -INFINITY ? 0.0f : row_solve::overflow_free_delta(r, w, nullptr);
    });
  });
}

template <int M>
__device__ __forceinline__ void union_sides(float (&acc)[kAccTiles][4], float (&top_read)[kSlots],
                                            float (&top_write)[kSlots], Pair& pr, int lane) {
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    //: the write's log weights over the support (a supported level is positive), the read side in place; the
    //: special-function forms (`__logf`, `__expf`: within ~1e-6 at these ranges, under the solve's bound)
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const float level = row_value<s, v>(acc), d = row_value<s, v + 2 * M>(acc);
      const float gate = -(fmaxf(d, 0.0f) + __logf(1.0f + __expf(-fabsf(d))));
      row_value<s, v + 2 * M>(acc) = level > 0.0f ? __logf(level) + gate : -INFINITY;
      row_value<s, v>(acc) = __fdividef(level, 1.0f + __expf(-d));
    });
    const float m =
        pr.combine1(row_all(row_tree<s, 2 * M, 2 * M>(acc, MaxOp{}), MaxOp{}), lane, MaxOp{});
    const float shift = m > -INFINITY ? m : 0.0f;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      row_value<s, v + 2 * M>(acc) = __expf(row_value<s, v + 2 * M>(acc) - shift);
    });
    const float den =
        pr.combine1(row_all(row_tree<s, 2 * M, 2 * M>(acc, AddOp{}), AddOp{}), lane, AddOp{});
    const float inv = den > 0.0f ? __fdividef(1.0f, den) : 0.0f;
    rola::static_for<2 * M>([&](auto V) { row_value<s, decltype(V)::value + 2 * M>(acc) *= inv; });
    top_write[s] = inv;
    top_read[s] = pr.combine1(row_all(row_tree<s, 0, 2 * M>(acc, MaxOp{}), MaxOp{}), lane, MaxOp{});
  });
}

//: THE JOB'S SOLVED LEVELS from the accumulators, for its gate: the read side at `[0, w)` from values `v`, the write
//: side at `[w, 2w)` from values `v + W` (a one-activation row's `W` 0, its one level both sides') -- written only
//: when the probe is asked for.
template <int W, class Token>
__device__ __forceinline__ void probe_levels(const ProduceParams& p, int job_index, int width,
                                             int col0, int lane, Token token,
                                             const float (&acc)[kAccTiles][4]) {
  if (p.level_probe == nullptr) return;
  float* out = p.level_probe + static_cast<int64_t>(job_index) * p.BT * kMaxColumns;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const int tk = token(s);
    rola::static_for<2 * kAccTiles - W>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const int j = col0 + 8 * v + (lane >> 2);
      if (j < width && tk < p.BT) {
        out[static_cast<int64_t>(tk) * kMaxColumns + j] = row_value<s, v>(acc);
        out[static_cast<int64_t>(tk) * kMaxColumns + width + j] = row_value<s, v + W>(acc);
      }
    });
  });
}

//: A COEFFICIENT IN ITS STORED FORM against the row's max, by `unit` (the row's reciprocal max at the plane's full
//: scale, one quotient a row; never the IEEE divide's slow-path call): a u8 digit is ONE FFMA -- `level unit + 1.5
//: 2^23` holds the nearest-even integer in its low mantissa bits, the byte the store keeps -- where `F2I` is a
//: conversion-pipe instruction; a dense level's f16 at 1 with `floor` taken. -- see docs/internals/produce/ampere/produce_kernel.md#bundle
constexpr float kDigitRound = 12582912.0f;

__device__ __forceinline__ void store_coef(uint8_t* at, float level, float unit, float) {
  *at = static_cast<uint8_t>(__float_as_uint(fmaf(level, unit, kDigitRound)));
}

__device__ __forceinline__ void store_coef(__half* at, float level, float unit, float floor) {
  *at = __float2half_rn(fmaxf(level * unit, floor));
}

__device__ __forceinline__ void store_zero(uint8_t* at) { *at = uint8_t{0}; }

__device__ __forceinline__ void store_zero(__half* at) { *at = __float2half_rn(0.0f); }

//: A WARP'S STAGED ROWS: one token slot's 4 rows of one side (a lane quad's tokens), a row the widest slot plus 16
//: bytes (four rows' bytes in distinct banks), past the parked n-tiles in the stages the k-loop released.
//: -- see docs/internals/produce/ampere/produce_kernel.md#bundle
constexpr int kStagedRowBytes = kMaxWidth * 2 + 16;
constexpr int kStagedWarpBytes = 4 * kStagedRowBytes;
//: A WARP'S PARKED N-TILE: its second n-tile's accumulators, 16 bytes a lane a column tile, lane-major.
constexpr int kParkedWarpBytes = kAccTiles * 32 * 16;
static_assert(kProduceWarps * (kParkedWarpBytes + kStagedWarpBytes) + kExchangeBytes
                  <= kProduceSmem,
              "the parked n-tiles, the staged rows and the pairs' exchange fit the stages");

//: One lane's 16-byte chunk of a hidden run, or zeros past the run's end (and no load).
__device__ __forceinline__ uint4 chunk(const __nv_bfloat16* run, int k, bool live) {
  return live ? __ldg(reinterpret_cast<const uint4*>(run + k)) : make_uint4(0u, 0u, 0u, 0u);
}

//: THE WRITE GAIN'S LOGIT of one token, a lane quad its row: each lane every fourth 16-byte chunk of the gain column
//: and the token's row (the loop uniform, a chunk past the hidden size zero), the quad's sum by two shuffles, plus both
//: biases.
__device__ __forceinline__ float gain_logit(const ProduceParams& p, const Job& job, int token,
                                            bool live, int quad) {
  const int column = p.C - p.G;
  const __nv_bfloat16* w = p.route_W + job.head * p.w_head + static_cast<int64_t>(column) * p.w_col;
  const __nv_bfloat16* xr = p.x + static_cast<int64_t>(live ? token : 0) * p.hidden;
  float dot = 0.0f;
  //: eight chunk pairs in flight (256 hidden a trip), where one at a time waited out each pair's load
#pragma unroll 8
  for (int k0 = 0; k0 < p.hidden; k0 += 32) {
    const int k = k0 + 8 * quad;
    const uint4 wc = chunk(w, k, live && k < p.hidden);
    const uint4 xc = chunk(xr, k, live && k < p.hidden);
    const uint32_t wv[4] = {wc.x, wc.y, wc.z, wc.w};
    const uint32_t xv[4] = {xc.x, xc.y, xc.z, xc.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      dot += ops::pair_lo(wv[j]) * ops::pair_lo(xv[j]) + ops::pair_hi(wv[j]) * ops::pair_hi(xv[j]);
    }
  }
  dot += __shfl_xor_sync(0xffffffffu, dot, 1);
  dot += __shfl_xor_sync(0xffffffffu, dot, 2);
  const float bias = p.route_bias != nullptr ? p.route_bias[job.head * p.C + column] : 0.0f;
  return dot + bias + p.gain_bias[job.head * p.G];
}

//: THE WRITE GAIN INTO WRITE LEVEL 0'S SCALE over a warp's 16 tokens from `first`: the scales this warp's bundle
//: stored, each times its token's `softplus(logit + bias)` -- a lane quad a token, 8 a pass -- run past the tile loop,
//: whose accumulators are dead by then. The gain is the producer's, folded into a scale; no job waits on another.
//: -- see docs/internals/produce/ampere/produce_kernel.md#gain
__device__ void apply_gain(const ProduceParams& p, const Job& job, int first, int lane) {
  __syncwarp();
  const int quad = lane & 3, row = lane >> 2;
#pragma unroll 1
  for (int pass = 0; pass < kWarpTokens / 8; ++pass) {
    const int token = first + 8 * pass + row;
    const bool here = token < p.BT;
    const float logit = gain_logit(p, job, token, here, quad);
    if (here && quad == 0)
      p.scale_write[slot_of(p, token, job.head) * p.D] *=
          logit > 20.0f ? logit : log1pf(expf(logit));
  }
}

//: ONE SIDE OF A WARP'S ROWS IN THE BUNDLE'S FORM, one token slot at a time (`Out`: u8, or f16 for a dense level, its
//: floor taken in a live row): each row's coefficients against its max `top` and its slot's tail zeros staged, the max
//: as the level's scale, then the rows' slots stored 16 bytes a lane. `tile0` the n-tile's first token.
//: -- see docs/internals/produce/ampere/produce_kernel.md#bundle
template <int M, class Out>
__device__ __forceinline__ void bundle_side(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, const float (&acc)[kAccTiles][4],
                                            const float (&top)[kSlots], Out* plane, int plane_row,
                                            int out_col, float* scale, bool floored,
                                            uint8_t* staged, Pair& pr, int col0) {
  constexpr int kGrain = 32 / static_cast<int>(sizeof(Out));
  constexpr int kRowElems = kStagedRowBytes / static_cast<int>(sizeof(Out));
  //: the warp's columns of the row, `col0 ..` (a wide union's half; the whole row at 0), and of its slot
  const int row_slot = (job.width + kGrain - 1) / kGrain * kGrain, g = lane >> 2;
  const int width = min(max(job.width - col0, 0), 16 * M),
            slot = min(max(row_slot - col0, 0), 16 * M);
  const int chunks = slot * static_cast<int>(sizeof(Out)) / 16;
  Out* rows = reinterpret_cast<Out*>(staged) + (lane & 3) * kRowElems;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    constexpr float kFull = sizeof(Out) == 1 ? 255.0f : 1.0f;
    const float m = top[s], unit = __fdividef(kFull, fmaxf(m, 1e-30f));
    const float floor = floored && m > 0.0f ? facts::kDenseFloor : 0.0f;
    //: every value staged, a dead column's too: its level is 0 (a 0 digit), and a staged column past the slot is
    //: never stored
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      store_coef(rows + 8 * v + g, row_value<s, v>(acc), unit, floor);
    });
    //: an f16 row's floor lands on its dead columns too: the slot's tail past the width zeroed (under 16 a row)
    if constexpr (sizeof(Out) == 2) {
      if (slot > width) {
        __syncwarp();
        Out* tail = reinterpret_cast<Out*>(staged) + (lane >> 3) * kRowElems;
#pragma unroll 1
        for (int c = width + (lane & 7); c < slot; c += 8) store_zero(tail + c);
      }
    }
    const int token = tile0 + 2 * (lane & 3) + s;
    if (g == 0 && token < p.BT && pr.half == 0) {
      const int64_t at = slot_of(p, token, job.head) * p.D + job.level;
      scale[at] = m;
    }
    __syncwarp();

    //: 8 lanes a row, a lane's chunks every eighth: one slot address a row
    {
      const int r = lane >> 3, tk = tile0 + 2 * r + s;
      if (tk < p.BT) {
        uint8_t* row = reinterpret_cast<uint8_t*>(
            plane + slot_of(p, tk, job.head) * static_cast<int64_t>(plane_row) + out_col + col0);
#pragma unroll 1
        for (int chunk = lane & 7; chunk < chunks; chunk += 8)
          *reinterpret_cast<uint4*>(row + 16 * chunk) =
              *reinterpret_cast<const uint4*>(staged + r * kStagedRowBytes + 16 * chunk);
      }
    }
    //: the next slot's staging overwrites these rows
    __syncwarp();
  });
}

//: A ONE-ROW JOB'S ROWS IN THE BUNDLE: each side it writes, its plane the side's dense f16 or sparse u8 (the shape
//: word's bits). The read side's fold is carried (its scale the max); the write side's applied. The sides are a rolled
//: loop over runtime operands, so the bundle's body is two copies (a u8 and an f16 one), never four.
template <int M>
__device__ __forceinline__ void bundle_rows(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, float (&acc)[kAccTiles][4],
                                            const float (&top_read)[kSlots],
                                            const float (&top_write)[kSlots], uint8_t* staged,
                                            Pair& pr, int col0) {
#pragma unroll 1
  for (int side = 0; side < 2; ++side) {
    if (!(side == 0 ? job.reads : job.writes)) continue;
    //: a narrow union's write row comes down to the read row's values once the read side is stored
    if constexpr (4 * M <= 2 * kAccTiles) {
      if (side == 1 && job.solve >= kUnion15) {
        rola::static_for<kSlots>([&](auto Sl) {
          constexpr int sl = decltype(Sl)::value;
          rola::static_for<2 * M>([&](auto V) {
            constexpr int v = decltype(V)::value;
            row_value<sl, v>(acc) = row_value<sl, v + 2 * M>(acc);
          });
        });
      }
    }
    float top[kSlots];
    rola::static_for<kSlots>([&](auto Sl) {
      top[decltype(Sl)::value] =
          side == 0 ? top_read[decltype(Sl)::value] : top_write[decltype(Sl)::value];
    });
    const int out_col = side == 0 ? job.out_read : job.out_write;
    float* scale = side == 0 ? p.scale_read : p.scale_write;
    if (side == 0 ? job.dense_read : job.dense_write)
      bundle_side<M>(p, job, tile0, lane, acc, top, p.dense[side], p.dense_rows[side], out_col,
                     scale, true, staged, pr, col0);
    else
      bundle_side<M>(p, job, tile0, lane, acc, top, side == 0 ? p.amp_read : p.amp_write,
                     side == 0 ? p.rows_read : p.rows_write, out_col, scale, false, staged, pr,
                     col0);
  }
}

//: THE SOLVE of one n-tile of a one-row route job's rows on the accumulators, and their bundle, in its row class of
//: `M` column tiles: its activation, the level probe, the bundle. -- see docs/internals/produce/ampere/produce_kernel.md#solve
template <int M>
__device__ __forceinline__ void solve_class(const ProduceParams& p, const Job& job, int job_index,
                                            int tile0, int lane, float (&acc)[kAccTiles][4],
                                            uint8_t* staged, Pair& pr, int col0, Clock& clock) {
  float top[kSlots], top_write[kSlots];
  auto token = [&](int s) { return tile0 + 2 * (lane & 3) + s; };
  //: a narrow union is its midpoints' entmax, split (a wide one, 256 columns a row, is the pair form's)
  constexpr bool kNarrow = 4 * M <= 2 * kAccTiles;
  const bool split = job.solve >= kUnion15;
  if constexpr (kNarrow) {
    if (split) union_midpoints<M>(acc);
  }
  if (job.solve == kSoftmax)
    softmax_rows<M>(acc, top);
  else
    entmax_rows<M>(acc, job.solve == kEntmax15 || job.solve == kUnion15, top, pr, lane, clock);
  rola::static_for<kSlots>(
      [&](auto Sl) { top_write[decltype(Sl)::value] = top[decltype(Sl)::value]; });
  if constexpr (kNarrow) {
    if (split) union_sides<M>(acc, top, top_write, pr, lane);
  }
  clock.lap(kPhaseLevels);
  if (split)
    probe_levels<kNarrow ? 2 * M : 0>(p, job_index, job.width, col0, lane, token, acc);
  else
    probe_levels<0>(p, job_index, job.width, col0, lane, token, acc);
  bundle_rows<M>(p, job, tile0, lane, acc, top, top_write, staged, pr, col0);
  clock.lap(kPhaseBundle);
}

//: THE SOLVE of one n-tile, its row class the shape word's (warp-uniform to ptxas). The class's body is a
//: force-inlined function, the switch's lambda one call to it: a lambda's own body past the inliner's threshold is
//: outlined, and an outlined body proves no shuffle converged. -- see docs/internals/produce/ampere/produce_kernel.md#solve
__device__ __forceinline__ void solve_tile(const ProduceParams& p, const Job& job, int job_index,
                                           int tile0, int lane, float (&acc)[kAccTiles][4],
                                           uint8_t* staged, Clock& clock) {
  rola::uniform_switch<4>(job.rows, [&](auto R) __attribute__((always_inline)) {
    Pair alone{nullptr, 0, 0, 0, false};
    solve_class<2 << decltype(R)::value>(p, job, job_index, tile0, lane, acc, staged, alone, 0,
                                         clock);
  });
}

//: ONE N-TILE OF A VALUE JOB'S ROWS, its heads `C = d_v / 8` of a lane's values each: per (token, head) the channel max
//: |v| as the scale and each channel against it rounded once to f16. -- see docs/internals/produce/ampere/produce_kernel.md#value-rows
template <int M, int C>
__device__ __forceinline__ void value_class(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, const float (&acc)[kAccTiles][4]) {
  const int g = lane >> 2;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const int token = tile0 + 2 * (lane & 3) + s;
    rola::static_for<2 * M / C>([&](auto Hh) {
      constexpr int h = decltype(Hh)::value;
      float m = 0.0f;
      rola::static_for<C>(
          [&](auto K) { m = fmaxf(m, fabsf(row_value<s, h * C + decltype(K)::value>(acc))); });
      m = row_all(m, [](float a, float b) { return fmaxf(a, b); });
      const float inv = __fdividef(1.0f, fmaxf(m, 1e-30f));
      if (h < job.heads && token < p.BT) {
        const int64_t slot = slot_of(p, token, job.head + h);
        rola::static_for<C>([&](auto K) {
          constexpr int k = decltype(K)::value;
          p.v[slot * p.d_v + 8 * k + g] = __float2half_rn(row_value<s, h * C + k>(acc) * inv);
        });
        if (g == 0) p.v_scale[slot] = m;
      }
    });
  });
}

//: ONE N-TILE OF A VALUE JOB'S ROWS, its column class and head size the shape word's (warp-uniform to ptxas).
__device__ __forceinline__ void value_tile(const ProduceParams& p, const Job& job, int tile0,
                                           int lane, const float (&acc)[kAccTiles][4]) {
  rola::uniform_switch<3>(job.tiles - 2, [&](auto F) __attribute__((always_inline)) {
    constexpr int M = 4 << decltype(F)::value;
    rola::uniform_switch<4>(job.rows, [&](auto R) __attribute__((always_inline)) {
      constexpr int C = 4 << decltype(R)::value;
      if constexpr (2 * M >= C) value_class<M, C>(p, job, tile0, lane, acc);
    });
  });
}

//: A ONE-ROW JOB'S PROJECTION: every warp the job's columns for its 16 tokens, in the job's class of `M` column tiles
//: and `S` k-steps a stage (the stage's bytes held near level). -- see docs/internals/produce/ampere/produce_kernel.md#form
__device__ __forceinline__ void project_one_row(const ProduceParams& p, int job_index, int token0,
                                                uint32_t stages, int warp, int lane,
                                                float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int g = lane >> 2;
  {
    const Job job = load_job(p, job_index);
    const int columns = job_columns(p, job);
    seed_bias(
        p, job, columns, [&](int i, int h) { return job_column(job, columns, 16 * i + g + 8 * h); },
        acc);
    const __nv_bfloat16* base = weight_base(p, job);
    rola::uniform_switch<3>(job.tiles - 2, [&](auto F) {
      //: the class's stage: `32 / M` k-steps, at most four, so its weights are 16 KB (8 KB at 64 columns), 64 MMAs a
      //: warp between barriers at 128 and 256 columns, and its x rows follow them
      constexpr int M = 4 << decltype(F)::value, S = 2 * kAccTiles / M < 4 ? 2 * kAccTiles / M : 4,
                    kXOff = 16 * M * 32 * S;
      constexpr int QW = 16 * M * 2 * S / kProduceThreads,
                    QX = kTokenTile * 2 * S / kProduceThreads;
      const Sources<QW, QX> s = sources<S, QW, QX>(
          p, job, [&](int a) { return job_column(job, columns, a); }, token0, kTokenTile);
      k_loop<M, S, kXOff>(
          p, base, s, stages, kWarpTokens * warp, [](int i) { return 16 * i; }, acc);
    });
  }
  //: the job's fields read again past the k-loop, never held across it
  const Job job = load_job(p, job_index);
  const int columns = job_columns(p, job);
  auto token = [&](int n, int e) {
    return token0 + kWarpTokens * warp + 8 * n + 2 * (lane & 3) + e;
  };
  probe_logits(
      p, job_index, columns,
      [&](int i, int h) { return job_column(job, columns, 16 * i + g + 8 * h); }, token, acc);
}

//: A WIDE UNION'S PROJECTION, one 32-token pass: a warp pair a token group, warp `2j + h` columns `128 h ..` of
//: both rows, eight tiles a row, one k-step a stage (its 512 columns fill it). -- see docs/internals/produce/ampere/produce_kernel.md#form
__device__ __forceinline__ void project_pair(const ProduceParams& p, int job_index, int token0,
                                             uint32_t stages, int warp, int lane,
                                             float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int pair = warp >> 1, half = warp & 1, g = lane >> 2;
  auto col = [&](int i, int h, int width) {
    const int c = 128 * half + 16 * (i & 7) + g + 8 * h;
    return c < width ? (i < 8 ? c : width + c) : -1;
  };
  {
    const Job job = load_job(p, job_index);
    const int columns = job_columns(p, job), width = job.width;
    seed_bias(p, job, columns, [&](int i, int h) { return col(i, h, width); }, acc);
    const __nv_bfloat16* base = weight_base(p, job);
    constexpr int QW = kMaxColumns * 2 / kProduceThreads, QX = 1;
    const Sources<QW, QX> s = sources<1, QW, QX>(
        p, job, [&](int c) { return c < columns ? c : -1; }, token0, kTokenTile / 2);
    k_loop<kAccTiles, 1, kMaxColumns * 32>(
        p, base, s, stages, kWarpTokens * pair,
        [&](int i) { return (i < 8 ? 0 : width) + 128 * half + 16 * (i & 7); }, acc);
  }
  //: the job's fields read again past the k-loop, never held across it
  const Job job = load_job(p, job_index);
  const int columns = job_columns(p, job), width = job.width;
  auto token = [&](int n, int e) {
    return token0 + kWarpTokens * pair + 8 * n + 2 * (lane & 3) + e;
  };
  probe_logits(p, job_index, columns, [&](int i, int h) { return col(i, h, width); }, token, acc);
}

//: THE GRID: token tiles fastest, so the blocks resident together share one job's weights in L2; two blocks an SM, so
//: one's solve runs beside the other's MMAs. The parameters are `__grid_constant__`: a body's lambdas hold their
//: address, which otherwise copies them whole to local memory. -- see docs/internals/produce/ampere/produce_kernel.md#grid
__global__ __launch_bounds__(kProduceThreads, 2) void produce_kernel(
    const __grid_constant__ ProduceParams p) {
  const int job_index = static_cast<int>(blockIdx.y), tile = static_cast<int>(blockIdx.x);
  const int token0 = tile * kTokenTile;
  extern __shared__ __align__(128) uint8_t stage_bytes[];
  const uint32_t stages = ops::smem_addr(stage_bytes);
  __shared__ long long s_clock[kProduceWarps][Clock::kRow];
  //: the warp's index as ptxas knows it uniform (`uniform_warp`): a uniform value held across the k-loop at the register
  //: cap can live outside the vector registers
  const int warp = ops::uniform_warp<kProduceWarps>(), lane = static_cast<int>(threadIdx.x) & 31;
  Clock clock(&s_clock[warp][0], lane, p.ledger != nullptr);
  clock.lap(kPhaseJob);
  if (pair_form(load_job(p, job_index))) {
    //: a pass's 32 tokens a k-loop; the barrier between passes releases the stages the last step read. The bound is
    //: tested against the parameters' token count, so no pass count is held across the k-loop.
    //: each pass solved on its accumulators before the next projects: the warp pair's halves of each row, its
    //: reductions through the pair's exchange (past the parked and staged rows)
    const int pair = warp >> 1, half = warp & 1;
    Pair pr{reinterpret_cast<float*>(stage_bytes
                                     + kProduceWarps * (kParkedWarpBytes + kStagedWarpBytes))
                + pair * 2 * kExchangeRound,
            half, kPairBarrier + pair, 0, true};
#pragma unroll 1
    for (int pass = 0; pass < 2 && token0 + pass * (kTokenTile / 2) < p.BT; ++pass) {
      float acc[kAccTiles][kTokenTiles][4];
      project_pair(p, job_index, token0 + pass * (kTokenTile / 2), stages, warp, lane, acc);
      //: every warp past its last stage before any parks or stages there
      __syncthreads();
      clock.lap(kPhaseProject);
      const Job job = load_job(p, job_index);
      uint8_t* parked = stage_bytes + warp * kParkedWarpBytes;
      uint8_t* staged = stage_bytes + kProduceWarps * kParkedWarpBytes + warp * kStagedWarpBytes;
      float tile_acc[kAccTiles][4];
      rola::static_for<kAccTiles>([&](auto I) {
        constexpr int i = decltype(I)::value;
        *reinterpret_cast<float4*>(parked + (i * 32 + lane) * 16) =
            make_float4(acc[i][1][0], acc[i][1][1], acc[i][1][2], acc[i][1][3]);
        rola::static_for<4>(
            [&](auto E) { tile_acc[i][decltype(E)::value] = acc[i][0][decltype(E)::value]; });
      });
#pragma unroll 1
      for (int n = 0; n < kTokenTiles; ++n) {
        if (n > 0) {
          rola::static_for<kAccTiles>([&](auto I) {
            constexpr int i = decltype(I)::value;
            const float4 v = *reinterpret_cast<const float4*>(parked + (i * 32 + lane) * 16);
            tile_acc[i][0] = v.x, tile_acc[i][1] = v.y, tile_acc[i][2] = v.z, tile_acc[i][3] = v.w;
          });
        }
        solve_class<kAccTiles / 2>(p, job, job_index,
                                   token0 + pass * (kTokenTile / 2) + kWarpTokens * pair + 8 * n,
                                   lane, tile_acc, staged, pr, 128 * half, clock);
      }
      //: the next pass's stages overwrite the parked and staged rows
      __syncthreads();
    }
    const Job job = load_job(p, job_index);
    if (half == 0 && job.kind == kJobRoute && job.writes && job.level == 0 && p.G > 0) {
#pragma unroll 1
      for (int pass = 0; pass < 2 && token0 + pass * (kTokenTile / 2) < p.BT; ++pass)
        apply_gain(p, job, token0 + pass * (kTokenTile / 2) + kWarpTokens * pair, lane);
    }
  } else {
    float acc[kAccTiles][kTokenTiles][4];
    project_one_row(p, job_index, token0, stages, warp, lane, acc);
    //: every warp past its last stage before any parks or stages there
    __syncthreads();
    clock.lap(kPhaseProject);
    const Job job = load_job(p, job_index);

    {
      //: the second n-tile parked, so the solve or the values hold one n-tile's 64 accumulators
      uint8_t* parked = stage_bytes + warp * kParkedWarpBytes;
      uint8_t* staged = stage_bytes + kProduceWarps * kParkedWarpBytes + warp * kStagedWarpBytes;
      float tile[kAccTiles][4];
      rola::static_for<kAccTiles>([&](auto I) {
        constexpr int i = decltype(I)::value;
        *reinterpret_cast<float4*>(parked + (i * 32 + lane) * 16) =
            make_float4(acc[i][1][0], acc[i][1][1], acc[i][1][2], acc[i][1][3]);
        rola::static_for<4>(
            [&](auto E) { tile[i][decltype(E)::value] = acc[i][0][decltype(E)::value]; });
      });
      //: the n-tiles one rolled loop, so the solve's and the bundle's bodies are one copy, not one an n-tile
#pragma unroll 1
      for (int n = 0; n < kTokenTiles; ++n) {
        if (n > 0) {
          rola::static_for<kAccTiles>([&](auto I) {
            constexpr int i = decltype(I)::value;
            const float4 v = *reinterpret_cast<const float4*>(parked + (i * 32 + lane) * 16);
            tile[i][0] = v.x, tile[i][1] = v.y, tile[i][2] = v.z, tile[i][3] = v.w;
          });
        }
        const int tile0 = token0 + kWarpTokens * warp + 8 * n;
        if (job.kind == kJobRoute)
          solve_tile(p, job, job_index, tile0, lane, tile, staged, clock);
        else
          value_tile(p, job, tile0, lane, tile);
      }
      if (job.kind == kJobRoute && job.writes && job.level == 0 && p.G > 0)
        apply_gain(p, job, token0 + kWarpTokens * warp, lane);
    }
  }

  //: a route job's gain pass is its bundle's
  clock.lap(load_job(p, job_index).kind == kJobRoute ? kPhaseBundle : kPhaseValues);
  clock.publish(p.ledger, job_index, kProduceWarps, warp);
}

}  // namespace rola::produce::detail
