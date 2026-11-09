// measure/lab/experiments/read_group.cu -- QUESTION: when a warp's read digits are taken M at a time (M = 1, 2, 4), a
// group's tiles the 8-token tiles of its digits' union with each digit's coefficient, the M digits summed in one exit,
// what does a group-tile cost -- the coefficient on each digit's read-row operand (OPERAND) or on each digit's fp32
// outputs (OUTPUT) -- and what does the readout cost a window, its group-tiles and their MMAs against M = 1's tiles?
// METHOD: `read_stream`'s staged readout (a ring of 8 stages a warp, a stage a tile's 8 read rows and its record, one
// record stream a warp across windows and passes, pads not copied) over its three cells (`read_group.inputs.py`);
// a group-tile loads its rows once, runs 4 MMAs a digit, and leaves its outputs in one exit -- 8 conversions, 4 packs,
// 4 shared atomic adds; OPERAND scales the rows by each digit's coefficients (4 bf16x2 FMAs a digit) into one
// accumulator, OUTPUT keeps an accumulator a digit and folds its coefficients into the exit's conversions (an FFMA a
// digit an output); UNIFIED (ruled 2026-10-18, one form for every M) is OUTPUT with each position's panel field and
// unit and each lane's two tokens' M coefficients as f16 pairs in one load, a factor the coefficient times its
// digit's held scale and its token's unit (the FMA pipe, so a zero coefficient stays zero). Its designed counts a
// group-tile: the consumer ~29 + 8 M (OPERAND), ~29 + 12 M (OUTPUT), ~36 + 16 M (UNIFIED), the producer ~20. A
// sample is the CTA's span a window; each form's drained panel is checked word for word -- the same panel at every
// M, since only the sums' grouping changes.
// see docs/lab/read_group.md
#include <cstdlib>

#include "read_image.cuh"

namespace read_group {

using read_image::kPanelPitch;
using read_image::kPlaneBytes;
using read_image::kPlanes;
using read_image::kSlots;
using read_image::kThreads;
using read_image::kWarps;
using read_image::kWords;

constexpr int kGroups = 8, kTiles = 2, kSliceGroups = 8, kRow = 64, kPasses = 2, kStaged = 8;
constexpr int kRing = kPlanes * kPlaneBytes;

//: THE SCALINGS: OUTPUT, OPERAND, UNIFIED
enum Mode : int { kOutput = 0, kOperand = 1, kUnified = 2 };

//: A GROUP-TILE'S RECORD (`read_group.inputs.py`): its exit part (64 bytes; UNIFIED 32), its coefficients (OUTPUT
//: 8 (M - 1) words, OPERAND 8 M, UNIFIED 4 M), then 8 tokens; a stage its 8 rows and the record; the layout past the
//: ring, the warps' digit scales (UNIFIED's, 4 floats a warp) last
template <int M, int MODE>
struct Rec {
  static constexpr int kExitWords = MODE == kUnified ? 8 : 16;
  static constexpr int kCoefWords = MODE == kOperand ? 8 * M : MODE == kOutput ? 8 * (M - 1) : 4 * M;
  static constexpr int kEntries = 4 * (kExitWords + kCoefWords), kBytes = kEntries + 32, kStage = 8 * kRow + kBytes;
  static constexpr int kPads = kRing + kWarps * kStaged * kStage, kPanel = kPads + 128 + 8 * kPanelPitch * 4;
  static constexpr int kScales = kPanel + kSlots * kPanelPitch * 4, kSmem = kScales + 16 * kWarps;
  static_assert(kBytes % 16 == 0 && kBytes / 16 <= 32, "a record is copied 16 bytes a lane");
};

//: THE INPUTS on the device: a box-window's passes `[windows][8]`, a group's tiles `[windows][2][8][8 warps][4]`, the
//: read rows, each CTA's drained panel
struct In {
  const uint16_t* passes;
  const uint16_t* tiles;
  const uint8_t* rows;
  uint32_t* panel_out;
  int windows;
};

struct Box {
  int slice, read;
};

__device__ __forceinline__ Box box_of() {
  const int b = static_cast<int>(blockIdx.x);
  return Box{(b / kTiles) % kSliceGroups, b / (kTiles * kSliceGroups)};
}

//: A LANE'S CONSTANTS for tokens on N (`read_stream`'s), its pad base in this layout
__device__ __forceinline__ read_image::Lane lane_of(uint32_t sm, int warp, int lane, int pads) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t plane = sm + (warp % kPlanes) * kPlaneBytes;
  const int srow = 8 * (lane >> 4) + r, sc = q & 1;
  const auto h = [](auto v) { return static_cast<decltype(v)>(lab::held(static_cast<uint32_t>(v))); };
  return {h(r), h(2 * t), h(2 * t + 1), h(16u * q), h(sm + pads + 4u * g + 32u * t),
          h(plane + read_image::in_plane(srow, sc)), h(plane + read_image::in_plane(srow, 2 + sc)), h(8u * r)};
}

__device__ __forceinline__ void copy16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_groups() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}

__device__ __forceinline__ const uint8_t* held64(const uint8_t* v) {
  asm volatile("" : "+l"(v));
  return v;
}

__device__ __forceinline__ const uint8_t* wide(const uint8_t* base, uint32_t a, uint32_t b) {
  const uint8_t* r;
  asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(base));
  return r;
}

//: a token's row copied unless it is a pad (its top bit set): the slot keeps a finite row its zero coefficients cancel
__device__ __forceinline__ void copy_row(uint32_t dst, const void* src, uint32_t token) {
  asm volatile("{ .reg .pred p; setp.ge.s32 p, %2, 0; @p cp.async.cg.shared.global [%0], [%1], 16; }" ::"r"(dst),
               "l"(src), "r"(token)
               : "memory");
}

__device__ __forceinline__ void copy16_if(uint32_t dst, const void* src, uint32_t live) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.cg.shared.global [%0], [%1], 16; }" ::"r"(dst),
               "l"(src), "r"(live)
               : "memory");
}

//: two bf16 products, exact here: `a x b + (-0)`
__device__ __forceinline__ uint32_t bf16_mul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

//: a lane's M coefficient words at `at`: one load
template <int M>
__device__ __forceinline__ void coef_words(uint32_t (&c)[M], uint32_t at) {
  if constexpr (M == 1) {
    c[0] = lds32(at);
  } else if constexpr (M == 2) {
    const uint2 v = read_image::lds64(at);
    c[0] = v.x;
    c[1] = v.y;
  } else {
    const uint4 v = lds128(at);
    c[0] = v.x;
    c[1] = v.y;
    c[2] = v.z;
    c[3] = v.w;
  }
}

//: A GROUP-TILE: 8 union tokens on N -- their rows (B) by one `ldmatrix.x4`, 4 MMAs against each of the group's digits'
//: states (`f[FIRST .. FIRST + M - 1]`), each lane's two tokens' outputs summed over the digits and added at their
//: panel rows once. OPERAND: the lane's B column (token g) scaled by each digit's coefficient, one accumulator; OUTPUT:
//: an accumulator a digit, its coefficients in the exit's conversions (the first digit's in the exit part)
template <int M, int MODE, int FIRST>
__device__ __forceinline__ void group_tile(uint32_t slot, uint32_t row_q, uint32_t ent, uint32_t coef, uint32_t pad,
                                           const uint32_t (&f)[4][16], const float (&sd)[4]) {
  uint32_t b[4];
  lab::ldsm(b, slot + row_q);
  if constexpr (MODE == kOperand) {
    uint32_t cw[M];
    coef_words<M>(cw, slot + coef);
    float acc[2][4] = {};
#pragma unroll
    for (int d = 0; d < M; ++d) {
      uint32_t bd[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) bd[k] = bf16_mul2(b[k], cw[d]);
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint32_t a[4] = {f[FIRST + d][(m * 2 + kb) * 4], f[FIRST + d][(m * 2 + kb) * 4 + 1],
                                 f[FIRST + d][(m * 2 + kb) * 4 + 2], f[FIRST + d][(m * 2 + kb) * 4 + 3]};
          lab::mma_bf16_f32(acc[m], a, bd[2 * kb], bd[2 * kb + 1]);
        }
    }
    const uint4 e = lds128(slot + ent);
    const float c0 = __uint_as_float(e.x), c1 = __uint_as_float(e.z);
    const uint32_t p0 = pad + e.y, p1 = pad + e.w;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      read_image::panel_red(p0 + 32 * m, read_image::s16_exact(acc[m][0], acc[m][2], c0));
      read_image::panel_red(p1 + 32 * m, read_image::s16_exact(acc[m][1], acc[m][3], c1));
    }
  } else {
    float acc[M][2][4] = {};
#pragma unroll
    for (int d = 0; d < M; ++d)
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint32_t a[4] = {f[FIRST + d][(m * 2 + kb) * 4], f[FIRST + d][(m * 2 + kb) * 4 + 1],
                                 f[FIRST + d][(m * 2 + kb) * 4 + 2], f[FIRST + d][(m * 2 + kb) * 4 + 3]};
          lab::mma_bf16_f32(acc[d][m], a, b[2 * kb], b[2 * kb + 1]);
        }
    float c0[M], c1[M];
    uint32_t p0, p1;
    if constexpr (MODE == kUnified) {
      //: each position's field and unit; each digit's f16 pair; a factor the coefficient, its digit's scale and
      //: its token's 2^-unit
      const uint2 e = read_image::lds64(slot + ent);
      uint32_t cw[M];
      coef_words<M>(cw, slot + coef);
      const float u0 = __uint_as_float((127u - (e.x & 0xffu)) << 23);
      const float u1 = __uint_as_float((127u - (e.y & 0xffu)) << 23);
#pragma unroll
      for (int d = 0; d < M; ++d) {
        const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&cw[d]));
        c0[d] = c.x * (sd[FIRST + d] * u0);
        c1[d] = c.y * (sd[FIRST + d] * u1);
      }
      p0 = pad + (e.x >> 16);
      p1 = pad + (e.y >> 16);
    } else {
      const uint4 e = lds128(slot + ent);
      c0[0] = __uint_as_float(e.x);
      c1[0] = __uint_as_float(e.z);
#pragma unroll
      for (int d = 1; d < M; ++d) {
        const uint2 v = read_image::lds64(slot + coef + 8 * (d - 1));
        c0[d] = __uint_as_float(v.x);
        c1[d] = __uint_as_float(v.y);
      }
      p0 = pad + e.y;
      p1 = pad + e.w;
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      float l0 = fmaf(acc[0][m][0], c0[0], 12582912.0f), h0 = fmaf(acc[0][m][2], c0[0], 12563648.0f);
      float l1 = fmaf(acc[0][m][1], c1[0], 12582912.0f), h1 = fmaf(acc[0][m][3], c1[0], 12563648.0f);
#pragma unroll
      for (int d = 1; d < M; ++d) {
        l0 = fmaf(acc[d][m][0], c0[d], l0);
        h0 = fmaf(acc[d][m][2], c0[d], h0);
        l1 = fmaf(acc[d][m][1], c1[d], l1);
        h1 = fmaf(acc[d][m][3], c1[d], h1);
      }
      read_image::panel_red(p0 + 32 * m, (__float_as_uint(h0) << 16) + __float_as_uint(l0));
      read_image::panel_red(p1 + 32 * m, (__float_as_uint(h1) << 16) + __float_as_uint(l1));
    }
  }
}

//: THE GROUPED READOUT, STAGED: each warp's ring of 8 stages, its stages one record stream across windows and passes
//: (`staged_m{M}_{mode}.bin`), the producer reading a stage's tokens from the slot it refills and copying its rows and
//: record in one burst; the consumer each pass's every group's first tile in one block (the warp's 4 digits' states
//: loaded), then each group's rest, the panel drained a pass. `SET` names the dataset's instantiation
template <int M, int MODE, int SET>
__global__ void __launch_bounds__(kThreads, 1) grouped(In in, const uint8_t* records, int stream_records,
                                                       long long* t) {
  using R = Rec<M, MODE>;
  constexpr int G = 4 / M;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, r = lane >> 2, c = lane & 3;
  const Box x = box_of();
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < R::kSmem / 16; k += kThreads) {
    const uint32_t v = k < kRing / 16 ? 0x3d80bd80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  //: the warps' digit scales, 2^sx with sx = (digit >> 3) & 1: a warp's digit slot i's (i & 1)
  if (threadIdx.x < 4 * kWarps)
    reinterpret_cast<float*>(smem + R::kScales)[threadIdx.x] = (threadIdx.x & 1) ? 2.0f : 1.0f;
  __syncthreads();
  const read_image::Lane lc = lane_of(sm, warp, lane, R::kPads);
  const uint32_t ring = lab::held(sm + kRing + warp * kStaged * R::kStage);
  const uint8_t* const stream = records + (static_cast<size_t>(x.read) * kWarps + warp) * stream_records * R::kBytes;
  const uint8_t* const rec_lane = held64(stream + kStaged * R::kBytes + 16 * lane);
  const uint8_t* const rows_c = held64(in.rows + x.slice * kRow + 16 * c);
  const uint32_t row_c = lab::held(r * kRow + 16 * (c ^ ((r >> 1) & 3))), rec_dst = lab::held(8 * kRow + 16 * lane);
  const uint32_t rec_live = lab::held(lane < R::kBytes / 16 ? 1u : 0u);
  const uint32_t tok_at = lab::held(8 * kRow + R::kEntries + 4 * r);
  const auto slot_of = [&](uint32_t s) { return ring + (s & (kStaged - 1)) * R::kStage; };
  //: stages 0-7's tokens into their slots (records 0-7's last 32 bytes, two lanes a slot)
  if (lane < 16)
    copy16(slot_of(lane >> 1) + 8 * kRow + R::kEntries + 16 * (lane & 1),
           stream + (lane >> 1) * R::kBytes + R::kEntries + 16 * (lane & 1));
  commit();
  wait_groups<0>();
  __syncwarp();
  uint32_t ps = 0;
  const uint8_t* prec = rec_lane;
  const auto issue = [&] {
    const uint32_t slot = slot_of(ps), tk = lds32(slot + tok_at);
    copy_row(slot + row_c, wide(rows_c, tk, 8 * kRow), tk);
    copy16_if(slot + rec_dst, prec, rec_live);
    commit();
    prec += R::kBytes;
    ++ps;
  };
  const long long t0 = clock64();
#pragma unroll 1
  for (int q = 0; q < kStaged - 1; ++q) issue();
  const auto passes_of = [&](int w) {
    return w < W ? static_cast<int>(in.passes[static_cast<size_t>(w) * kGroups + x.read]) : 0;
  };
  const auto tiles_of = [&](int w, int p) {
    return w < W && lane < G
               ? static_cast<int>(in.tiles[(((static_cast<size_t>(w) * kPasses + p) * kGroups + x.read) * kWarps +
                                            warp) * 4 + lane])
               : 0;
  };
  int np = passes_of(0), np_next = passes_of(1), ct0 = tiles_of(0, 0), ct1 = tiles_of(0, 1);
  int ct0_next = tiles_of(1, 0), ct1_next = tiles_of(1, 1);
  uint32_t cs = 0;
  const uint32_t row_q = lab::held((lane & 7) * kRow + 16 * ((((lane & 7) >> 1) & 3) ^ (lane >> 3)));
  const uint32_t ent = lab::held(8 * kRow + (MODE == kUnified ? 8 : 16) * (lane & 3));
  const uint32_t coef = lab::held(8 * kRow + (MODE == kOperand   ? 64 + 4 * M * (lane >> 2)
                                              : MODE == kOutput ? 64 + 8 * (M - 1) * (lane & 3)
                                                                : 32 + 4 * M * (lane & 3)));
  const uint32_t cell = lab::held(sm + R::kPanel + ((32 * warp + (lane >> 4)) * kPanelPitch + (lane & 15)) * 4);
  const auto drain = [&](int p) {
    uint32_t* target = in.panel_out + ((static_cast<size_t>(blockIdx.x) * kPasses + p) * kSlots + 32 * warp +
                                       (lane >> 4)) * kWords + (lane & 15);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const uint32_t v = lds32(cell + k * 2 * kPanelPitch * 4);
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(cell + k * 2 * kPanelPitch * 4), "r"(0u) : "memory");
      asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target + k * 2 * kWords), "r"(v) : "memory");
    }
  };
#pragma unroll 1
  for (int w = 0; w < W; ++w) {
#pragma unroll 1
    for (int p = 0; p < np; ++p) {
      const int ct = p == 0 ? ct0 : ct1;
      //: every group's first tile, one block: its G slots held, the ring's other slots ahead
      issue();
      wait_groups<kStaged - G>();
      __syncwarp();
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) read_image::state_in<1>(f[i], lc, read_image::plane_off(i));
      //: the warp's digits' scales, loaded each pass as a window's exponents are
      float sd[4];
      {
        const uint4 v = lds128(sm + R::kScales + 16 * warp);
        sd[0] = __uint_as_float(v.x);
        sd[1] = __uint_as_float(v.y);
        sd[2] = __uint_as_float(v.z);
        sd[3] = __uint_as_float(v.w);
      }
#pragma unroll
      for (int k = 0; k < G; ++k) {
        if (k == 0) group_tile<M, MODE, 0>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
        if constexpr (G > 1) {
          if (k == 1) group_tile<M, MODE, M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
        }
        if constexpr (G > 2) {
          if (k == 2) group_tile<M, MODE, 2 * M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
          if (k == 3) group_tile<M, MODE, 3 * M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
        }
        ++cs;
      }
#pragma unroll 1
      for (int q = 0; q < G - 1; ++q) issue();
      //: each group's rest, a tile waiting for its stage
#pragma unroll
      for (int k = 0; k < G; ++k) {
        const int n = __shfl_sync(0xffffffffu, ct, k) - 1;
#pragma unroll 1
        for (int s = 0; s < n; ++s) {
          issue();
          wait_groups<kStaged - 1>();
          __syncwarp();
          if (k == 0) group_tile<M, MODE, 0>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
          if constexpr (G > 1) {
            if (k == 1) group_tile<M, MODE, M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
          }
          if constexpr (G > 2) {
            if (k == 2) group_tile<M, MODE, 2 * M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
            if (k == 3) group_tile<M, MODE, 3 * M>(slot_of(cs), row_q, ent, coef, lc.pad, f, sd);
          }
          ++cs;
        }
      }
      __syncthreads();
      drain(p);
      __syncthreads();
    }
    np = np_next;
    np_next = passes_of(w + 2);
    ct0 = ct0_next;
    ct1 = ct1_next;
    ct0_next = tiles_of(w + 2, 0);
    ct1_next = tiles_of(w + 2, 1);
  }
  wait_groups<0>();
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
}

//: a dataset's inputs
struct Host {
  std::vector<uint16_t> passes, tiles[3];
  std::vector<uint8_t> rows, staged[8];
  std::vector<uint32_t> panel;
  int ctas = 0, windows = 0, stream_records[8] = {};
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "read_group: %s is missing\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
  return v;
}

template <class T>
std::vector<T> as(const std::vector<uint8_t>& b) {
  return std::vector<T>(reinterpret_cast<const T*>(b.data()), reinterpret_cast<const T*>(b.data() + b.size()));
}

//: the forms' order in `Host::staged` and their files
const char* kForm[8] = {"m1_output", "m2_output", "m2_operand", "m4_output", "m4_operand",
                        "m1_unified", "m2_unified", "m4_unified"};

Host load(const std::string& dir) {
  Host h;
  const std::vector<int> head = as<int>(file(dir + "/head.bin"));
  h.ctas = head[1];
  h.windows = head[2];
  h.passes = as<uint16_t>(file(dir + "/passes.bin"));
  h.rows = file(dir + "/rows.bin");
  h.panel = as<uint32_t>(file(dir + "/panel.bin"));
  const char* tiles[3] = {"/tiles_m1.bin", "/tiles_m2.bin", "/tiles_m4.bin"};
  for (int k = 0; k < 3; ++k) h.tiles[k] = as<uint16_t>(file(dir + tiles[k]));
  for (int k = 0; k < 8; ++k) {
    h.staged[k] = file(dir + "/staged_" + kForm[k] + ".bin");
    h.stream_records[k] = *reinterpret_cast<const int*>(h.staged[k].data());
    h.staged[k].erase(h.staged[k].begin(), h.staged[k].begin() + 4);
  }
  return h;
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

//: the device copies of a dataset: its inputs a form (the tiles by M) and each form's records
struct Device {
  In in[3];
  const uint8_t* staged[8];
};

template <int M, int MODE, int SET>
void run(const lab::Bench& b, const std::string& variant, const Device& dev, const Host& h, int form) {
  using R = Rec<M, MODE>;
  const std::string name = lab::kernel("read_group::grouped", {M, MODE, SET});
  b.expect(name, "LDGSTS", ">=", 2);
  b.expect(name, "HMMA.16816.F32.BF16", ">=", 4 * M);
  if (MODE == kOperand) b.expect(name, "HFMA2.BF16", ">=", 4 * M);
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a calibration of the grouped readout's cost for the carve model; its designed counts in the header");
  if (b.expect_only) return;
  const In& in = dev.in[M == 1 ? 0 : M == 2 ? 1 : 2];
  LAB_CHECK(cudaFuncSetAttribute(grouped<M, MODE, SET>, cudaFuncAttributeMaxDynamicSharedMemorySize, R::kSmem));
  const auto launch = [&] {
    grouped<M, MODE, SET><<<h.ctas, kThreads, R::kSmem>>>(in, dev.staged[form], h.stream_records[form], b.times);
  };
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps, h.ctas) / h.windows; });
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(h.panel.size());
  LAB_CHECK(cudaMemcpy(got.data(), in.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.panel[i];
  b.check(variant, bad == 0, std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                                 " drained panel words exact (each CTA's, summed over the windows)");
  b.record(variant, "cycles a window", s);
}

template <int SET>
void cell(const lab::Bench& b, const char* label, const Device& dev, const Host& h) {
  const std::string l = label;
  run<1, 0, SET>(b, l + "M 1 (each digit its tiles)", dev, h, 0);
  run<2, 0, SET>(b, l + "M 2, output scaling", dev, h, 1);
  run<2, 1, SET>(b, l + "M 2, operand scaling", dev, h, 2);
  run<4, 0, SET>(b, l + "M 4, output scaling", dev, h, 3);
  run<4, 1, SET>(b, l + "M 4, operand scaling", dev, h, 4);
  run<1, 2, SET>(b, l + "unified, M 1", dev, h, 5);
  run<2, 2, SET>(b, l + "unified, M 2", dev, h, 6);
  run<4, 2, SET>(b, l + "unified, M 4", dev, h, 7);
}

}  // namespace read_group

int main(int argc, char** argv) {
  using namespace read_group;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Host h[3];
  Device dev[3] = {};
  const char* sub[3] = {"", "/k16", "/c16"};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_group: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    for (int k = 0; k < 3; ++k) {
      h[k] = load(dir + sub[k]);
      if (h[k].ctas > b.sms) {
        std::fprintf(stderr, "read_group: %d boxes need as many SMs, the device has %d\n", h[k].ctas, b.sms);
        return 1;
      }
      uint32_t* out = nullptr;
      LAB_CHECK(cudaMalloc(&out, h[k].panel.size() * 4));
      const uint16_t* passes = up(h[k].passes);
      const uint8_t* rows = up(h[k].rows);
      for (int m = 0; m < 3; ++m) dev[k].in[m] = In{passes, up(h[k].tiles[m]), rows, out, h[k].windows};
      for (int f = 0; f < 8; ++f) dev[k].staged[f] = up(h[k].staged[f]);
    }
  }
  cell<0>(b, "alt-k4: ", dev[0], h[0]);
  cell<1>(b, "alt-k16: ", dev[1], h[1]);
  cell<2>(b, "cohort-k16: ", dev[2], h[2]);
}
