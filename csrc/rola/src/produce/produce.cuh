#pragma once

// THE PRODUCER: one launch from a stretch of tokens to the bundle, every width and kind a runtime field of the job
// table the layer's declared routing compiles into. see docs/internals/produce/produce.md

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace rola {
namespace produce {

constexpr int kProduceThreads = 128;
constexpr int kProduceWarps = kProduceThreads / 32;
//: A block's tokens: a warp's sixteen, two MMA n-tiles. -- see docs/internals/produce/ampere/produce_kernel.md#accumulate
constexpr int kTokenTiles = 2;
constexpr int kWarpTokens = 8 * kTokenTiles;
constexpr int kTokenTile = kProduceWarps * kWarpTokens;
//: One MMA k-step of hidden a stage.
constexpr int kKStep = 16;
//: A solve row's widest; a job holds two (a union's) or a value job's heads.
constexpr int kMaxWidth = 256;
constexpr int kMaxColumns = 2 * kMaxWidth;
//: A warp's accumulators: 16 column tiles of its 16 tokens. -- see docs/internals/produce/ampere/produce_kernel.md#form
constexpr int kAccTiles = 16;
constexpr int kWarpColumns = kAccTiles * 16;
//: A stage: at most 16 KB of weights and 8 KB of x rows, two in flight. -- see docs/internals/produce/ampere/produce_kernel.md#stage
constexpr int kStageWeights = kMaxColumns * kKStep * 2;
constexpr int kStageX = kTokenTile * 4 * kKStep * 2;
constexpr int kStageSlot = kStageWeights + kStageX;
constexpr int kStages = 2;
constexpr int kProduceSmem = kStages * kStageSlot;
//: The threshold's Newton steps at most: rows of every measured distribution settle in 8 or fewer.
constexpr int kNewtonSteps = 16;

//: THE JOB TABLE, one row of int32 per job: its addressing. Its control rides the parameters as a shape word.
//: -- see docs/internals/produce/ampere/produce_kernel.md#jobs
enum JobField : int {
  kFieldHead,
  kFieldHeads,
  kFieldWidth,
  kFieldIn0,
  kFieldIn1,
  kFieldOutRead,
  kFieldOutWrite,
  kFieldLevel,
  kJobFields
};

enum JobKind : int { kJobRoute = 0, kJobValue = 1 };

//: A launch's jobs at most: one shape word each in the parameter space.
constexpr int kMaxJobs = 1024;

//: THE SOLVES a route job runs on its row: the closed registry's kerneled activations.
enum SolveKind : int { kSoftmax = 0, kEntmax15 = 1, kEntmax20 = 2, kUnion15 = 3, kUnion20 = 4 };

//: THE PHASE LEDGER's phases, a warp's cycles each, booked by job over the tiles. -- see docs/internals/produce/ampere/produce_kernel.md#phase-ledger
enum ProducePhase : int {
  kPhaseJob,
  kPhaseProject,
  kPhaseThreshold,
  kPhaseLevels,
  kPhaseBundle,
  kPhaseValues,
  kProducePhases
};

//: A JOB, its table row and its shape word decoded. -- see docs/internals/produce/ampere/produce_kernel.md#jobs
struct Job {
  int kind, solve, rows, reads, writes, head, heads, width, in0, in1, out_read, out_write, level,
      tiles, pair, dense_read, dense_write;
};

//: THE OPERANDS, as addresses: every tensor the host entry validated, with the strides it read off them.
struct ProduceParams {
  const __nv_bfloat16* x;        // [BT, hidden]
  const __nv_bfloat16* route_W;  // [H, hidden, C], (h, k, c) at h * w_head + c * w_col + k
  int64_t w_head, w_col;
  const float* route_bias;   // [H, C] or null
  const float* gain_bias;    // [H, G]
  const __nv_bfloat16* v_W;  // [H * d_v, hidden], (c, k) at c * v_row + k
  int64_t v_row;
  const int32_t* jobs;  // [n_jobs, kJobFields]

  uint8_t*
      amp_read;  // [B, H, T, rows_read], u8: the sparse read levels; every output head-major (produce.md#bundle)
  uint8_t* amp_write;  // [B, H, T, rows_write], u8: the sparse write levels
  __half* dense
      [2];  // [B, H, T, dense_rows[side]], f16: a side's dense levels, floored, or null (produce.md#dense)
  float* scale_read;  // [B, H, T, D]
  float* scale_write;
  __half* v;           // [B, H, T, d_v], f16 against the token's largest channel
  float* v_scale;      // [B, H, T]
  float* logit_probe;  // [jobs, BT, kMaxColumns] or null: every job's projection, for its gate
  float* level_probe;  // the same, a route job's solved levels: read at [0, w), write at [w, 2w)
  long long* ledger;   // [jobs, warps, phases] or null
  int16_t shapes[kMaxJobs];  // a job's shape word (produce.md#jobs), by block index

  int BT, T, hidden, H, C, G, rows_read, rows_write, D, d_v;
  //: T's fast division (`slot_of`): a token's sequence `umulhi(token, t_mul) >> t_shift`, below 2^31
  uint32_t t_mul;
  int t_shift;
  int dense_rows[2];
  int dense_mask
      [2];  // a side's dense levels (read 0, write 1), a bit a level: their rows go to `dense[side]`
};

}  // namespace produce
}  // namespace rola
