// measure/lab/experiments/row_stream.cu -- QUESTION: when the fold's rows are streamed a list at a time into each
// warp's ring (merge width 1) instead of staged once a window, how much of their copies hides under the fold, at what
// ring depth, and what do the copies cost? METHOD: the carry's fold step over real nl64k-alt-k4 write lists
// (`row_stream.inputs.py`): 80 CTAs, box `b` of one head over 64 windows, its rows a token's 32-byte slice of its slice
// group and 64-byte values of its channel tile -- shared between the boxes as the carry's are, so L2 serves what one
// box fetched to the others; 8 warps a CTA each fold its 4 lists `w + 8 j` into 128 fp32 registers, 16-token steps
// while more than 8 entries remain, then an 8-token tail. RESIDENT: each window's union rows copied once, a CTA
// barrier, the fold, a barrier. STREAMED: each warp's ring of `D` stages of 8 rows, copied ahead across lists and
// windows from its window's stage record (a pre-pass's plan) staged a window ahead, or with its lookups at their use;
// `CA` the copies allocated in L1 (`cp.async.ca`) or not (`.cg`, the carry's); bounds: each form's fold alone and
// copies alone. A sample is the CTA's span a window; every forming variant's state is checked against the mirror's.
// see docs/internals/tools/lab.md#row-stream
#include <cuda_fp16.h>

#include <type_traits>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/static_for.cuh"
#include "lab.cuh"

namespace row_stream {

constexpr int kWarps = 8, kThreads = 32 * kWarps, kWindow = 512, kLists = 32, kRow = 128, kGroups = 8, kTiles = 2;
//: a streamed stage: 8 rows, then its 8 entries' coefficients (f16) in a 128-byte pad, so every stage keeps the
//: rows' 32-byte phase (a copy's destination off its source's phase doubles its L2 sectors)
constexpr int kStage = 8 * kRow + 128, kSlices = 256, kValues = 128;

//: THE INPUTS on the device: per window and write group a list's entries `n[32]`, an entry's token `tok[32][512]`, the
//: union's slots `cnt` and a slot's token `uni[512]`; a token's slices `[windows][512][256 B]` and values
//: `[windows][512][128 B]`; per window, write group and warp its stage record (`record_bytes` each)
struct In {
  const uint16_t* n;
  const uint16_t* tok;
  const uint16_t* cnt;
  const uint16_t* uni;
  const uint8_t* slices;
  const uint8_t* values;
  const uint8_t* rec;
  float* out;
  int windows, record_bytes;
};

//: THIS CTA'S BOX (the carry's order): its channel tile, write group and slice group
struct Box {
  int tile, group, slice;
};

__device__ __forceinline__ Box box_of() {
  const int b = static_cast<int>(blockIdx.x);
  return Box{b % kTiles, (b / kTiles) % kGroups, b / (kTiles * kGroups)};
}

//: chunk `c` (0 .. 7) of window `w` token `t`'s row: slice chunks 0, 1 (2, 3 zero-filled), value chunks 4 .. 7
__device__ __forceinline__ const uint8_t* chunk_src(const In& in, const Box& x, int w, int t, int c) {
  const size_t wt = static_cast<size_t>(w) * kWindow + t;
  return c < 4 ? in.slices + wt * kSlices + 32 * x.slice + 16 * (c & 1)
               : in.values + wt * kValues + 64 * x.tile + 16 * (c - 4);
}

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

//: a 16-byte copy, zero-filled where `live` is false; `CA` allocates in L1
template <int CA>
__device__ __forceinline__ void copy16(uint32_t dst, const void* src, bool live) {
  const int bytes = live ? 16 : 0;
  if constexpr (CA) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  }
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_groups() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_t(uint32_t (&x)[4], uint32_t at) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "r"(at));
}

__device__ __forceinline__ void mma16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
  return *reinterpret_cast<const uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

//: c for two entries of a step, 1 where live and 0 past the list
__device__ __forceinline__ uint32_t live2(int e, int n) {
  return (e < n ? 0x3c00u : 0u) | ((e + 1 < n ? 0x3c00u : 0u) << 16);
}

//: A STEP of 16 tokens (`Steps16`) or 8 into list `J`'s accumulators, its entries `s ..` of `n`; `row(i)` the shared
//: address of entry `i`'s row and `key(i)` its chunks' swizzle (chunk `k` of a row at `(k ^ key) * 16`)
template <int Steps16, class Row, class Key>
__device__ __forceinline__ void step(float (&acc)[2][4][4], int s, int n, Row row, Key key, int lane) {
  const int t = lane & 3, tile = lane >> 3, r = lane & 7;
  if constexpr (Steps16) {
    const uint32_t c01 = live2(s + 2 * t, n), c89 = live2(s + 8 + 2 * t, n);
    uint32_t a[2][4];
    rola::static_for<2>([&](auto M) {
      const int e = (tile >= 2 ? 8 : 0) + r, k = 2 * M + (tile & 1);
      ldsm_t(a[M], row(e) + 16u * static_cast<uint32_t>((k ^ key(e)) & 7));
      a[M][0] = hmul2(a[M][0], c01);
      a[M][1] = hmul2(a[M][1], c01);
      a[M][2] = hmul2(a[M][2], c89);
      a[M][3] = hmul2(a[M][3], c89);
    });
    uint32_t b[4][2];
    rola::static_for<2>([&](auto H) {
      const int e = (tile & 1 ? 8 : 0) + r, k = 4 + 2 * H + (tile >> 1);
      uint32_t x[4];
      ldsm_t(x, row(e) + 16u * static_cast<uint32_t>((k ^ key(e)) & 7));
      b[2 * H][0] = x[0];
      b[2 * H][1] = x[1];
      b[2 * H + 1][0] = x[2];
      b[2 * H + 1][1] = x[3];
    });
    rola::static_for<2>([&](auto M) {
      rola::static_for<4>([&](auto N) { mma16(acc[M][N], a[M], b[N][0], b[N][1]); });
    });
  } else {
    const uint32_t c01 = live2(s + 2 * t, n);
    uint32_t a[4], b[4];
    ldsm_t(a, row(r) + 16u * static_cast<uint32_t>((tile ^ key(r)) & 7));
    ldsm_t(b, row(r) + 16u * static_cast<uint32_t>(((4 + tile) ^ key(r)) & 7));
#pragma unroll
    for (int k = 0; k < 4; ++k) a[k] = hmul2(a[k], c01);
    rola::static_for<2>([&](auto M) {
      rola::static_for<4>([&](auto N) { mma8(acc[M][N], a[2 * M], a[2 * M + 1], b[N]); });
    });
  }
}

//: a list's steps over its `n` entries (fold.cuh's `steps`): 16 tokens while more than 8 remain, then 8; `prep(two)`
//: before each step and `post(two)` after it (`two`: a 16-token step, two stages), `row(s, i)` entry `s + i`'s row;
//: `Fold` 0 runs the preps and posts alone (the copies' bound)
template <int Fold, class Prep, class Post, class Row, class Key>
__device__ __forceinline__ void steps(float (&acc)[2][4][4], int n, Prep prep, Post post, Row row, Key key, int lane) {
  int s = 0;
  for (int k = __any_sync(0xffffffffu, n - s > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      prep(std::true_type{});
      if constexpr (Fold) step<1>(acc, s, n, [&](int i) { return row(s, i); }, [&](int i) { return key(s, i); }, lane);
      post(std::true_type{});
      s += 16;
    } while (__any_sync(0xffffffffu, n - s > 8));
  for (int k = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; k > 0; --k) {
    prep(std::false_type{});
    if constexpr (Fold) step<0>(acc, s, n, [&](int i) { return row(s, i); }, [&](int i) { return key(s, i); }, lane);
    post(std::false_type{});
  }
}

//: THE STATE OUT, each element at its (list, digit, channel)
__device__ __forceinline__ void state_out(float* out, const float (&acc)[4][2][4][4], int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
  rola::static_for<4>([&](auto J) {
    rola::static_for<2>([&](auto M) {
      rola::static_for<4>([&](auto N) {
        rola::static_for<4>([&](auto I) {
          const int list = warp + kWarps * J, digit = 16 * M + g + 8 * (I >> 1), ch = 8 * N + 2 * t + (I & 1);
          out[(static_cast<size_t>(blockIdx.x) * kLists + list) * 1024 + digit * 32 + ch] = acc[J][M][N][I];
        });
      });
    });
  });
}

__device__ __forceinline__ void span_out(long long* t, long long t0, int warp, int lane) {
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
}

//: RESIDENT: each window's union rows copied once into the pool (slot `i` its row, chunks swizzled by `i & 7`) and
//: each list's entries as slots into shared memory (the carry stages its entries so), the fold, a barrier before the
//: next window's copies; `PART` as the streamed kernel's
template <int CA, int PART>
__global__ void __launch_bounds__(kThreads, 1) resident(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint16_t* slot_of = reinterpret_cast<uint16_t*>(sm);
  uint16_t* ent = slot_of + kWindow;
  uint8_t* pool = sm + 2 * kWindow + 2 * kLists * kWindow;
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const Box x = box_of();
  float acc[4][2][4][4] = {};
  __syncthreads();
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < in.windows; ++w) {
    const size_t bw = static_cast<size_t>(w) * kGroups + x.group;
    const int u = in.cnt[bw];
    const uint16_t* uni = in.uni + bw * kWindow;
#pragma unroll 1
    for (int k = static_cast<int>(threadIdx.x); k < 8 * u; k += kThreads) {
      const int i = k >> 3, c = k & 7;
      if constexpr (PART != 1)
        copy16<CA>(smem(pool) + i * kRow + 16 * ((c ^ i) & 7), chunk_src(in, x, w, uni[i], c), c < 2 || c >= 4);
    }
    commit();
#pragma unroll 1
    for (int i = static_cast<int>(threadIdx.x); i < u; i += kThreads) slot_of[uni[i]] = static_cast<uint16_t>(i);
    __syncthreads();
    int n[4];
    rola::static_for<4>([&](auto J) {
      const int l = warp + kWarps * J;
      n[J] = in.n[bw * kLists + l];
      const uint16_t* tok = in.tok + (bw * kLists + l) * kWindow;
#pragma unroll 1
      for (int e = lane; e < n[J]; e += 32) ent[l * kWindow + e] = slot_of[tok[e]];
    });
    wait_groups<0>();
    __syncthreads();
    rola::static_for<4>([&](auto J) {
      const uint16_t* le = ent + (warp + kWarps * J) * kWindow;
      const int nj = n[J];
      steps<PART != 2>(
          acc[J], nj, [](auto) {}, [](auto) {},
          [&](int s, int i) { return smem(pool) + kRow * static_cast<uint32_t>(le[min(s + i, max(nj - 1, 0))]); },
          [&](int s, int i) { return static_cast<int>(le[min(s + i, max(nj - 1, 0))]); }, lane);
    });
    __syncthreads();
  }
  span_out(t, t0, warp, lane);
  state_out(in.out, acc, warp, lane);
}

//: STREAMED: each warp's ring of `D` stages of 8 rows (`D >= 2`: a 16-token step holds two), its stages its window
//: records' in order (its lists `w + 8 j`, each list's entries 8 a stage), each slot refilled as its step ends,
//: across lists and windows. `AHEAD`: the producer reads a stage's tokens from its window's record, copied to shared
//: memory two windows ahead, and the consumer its lists' entries a window ahead into registers; else each from global
//: memory at its use. `PART` 1 the fold alone (no row copies, the ring's stale rows), 2 the copies alone, 4 the fold
//: alone with no producer. `SET` names the
//: dataset's instantiation, so the profiler keeps each dataset's counters apart. A record lands with the stage group it
//: was issued in (a window's entry waits on every group but the newest D - 1, or on every group where fewer were
//: committed since); `MB` 1 on its buffer's mbarrier instead (every lane's copies, `cp.async.mbarrier.arrive`)
template <int D, int CA, int PART, int AHEAD, int SET = 0, int MB = 0>
__global__ void __launch_bounds__(kThreads, 1) streamed(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, C = in.record_bytes, r = lane >> 2,
            c = lane & 3;
  const Box x = box_of();
  //: the lane's constants held in registers (opaque, so ptxas keeps them rather than re-forming them each step)
  const uint32_t ring = lab::held(smem(sm) + warp * D * kStage), ring_last = lab::held(ring + (D - 1) * kStage);
  const auto next_slot = [&](uint32_t slot) { return slot == ring_last ? ring : slot + kStage; };
  uint16_t* const recs = reinterpret_cast<uint16_t*>(sm + kWarps * D * kStage + warp * 3 * C);
  const uint32_t bars = smem(sm) + kWarps * (D * kStage + 3 * C) + warp * 24;
  float acc[4][2][4][4] = {};
  const auto list_n = [&](int w, int j) {
    return w < W ? static_cast<int>(in.n[(static_cast<size_t>(w) * kGroups + x.group) * kLists + warp + kWarps * j])
                 : 0;
  };
  const auto record = [&](int w) {
    return in.rec + ((static_cast<size_t>(w) * kGroups + x.group) * kWarps + warp) * C;
  };
  //: window `w`'s record into its buffer `w % 3` (fetched two windows ahead), a lane a 16-byte chunk; with `MB` every
  //: lane's arrival on the buffer's mbarrier once its copies land
  const auto fetch = [&](int w) {
#pragma unroll 1
    for (int q = lane; q < (w < W ? C / 16 : 0); q += 32)
      copy16<CA>(smem(recs) + (w % 3) * C + 16 * q, record(w) + 16 * q, true);
    if constexpr (MB) {
      if (w < W) lab::mbar_arrive_when_landed(bars + 8 * (w % 3));
    }
  };
  //: window `w`'s record: its buffer (`AHEAD`) or global memory
  const auto rec_of = [&](int w) -> const uint16_t* {
    return AHEAD ? recs + (w % 3) * (C / 2) : reinterpret_cast<const uint16_t*>(record(w));
  };
  //: THE PRODUCER'S CURSOR: the window and stage it copies next and the window's stages; its running addresses -- the
  //: lane's token in the window's record, the ring slot it fills next, the window's first token; the groups committed
  //: and the groups carrying the next two windows' records
  int pw = 0, ps = 0, groups = 0, carrier = 0, carrier_next = 0;
  if constexpr (AHEAD && MB) {
    if (lane == 0)
      for (int k = 0; k < 3; ++k) lab::mbar_init(bars + 8 * k, 32);
    __syncwarp();
    fetch(0);
    lab::mbar_wait(bars, 0);
    fetch(1);
    fetch(2);
  } else if constexpr (AHEAD) {
    fetch(0);
    commit();
    wait_groups<0>();
    __syncwarp();
    fetch(1);
    fetch(2);
  }
  const uint32_t row_s = lab::held(r * kRow + 16 * ((c ^ r) & 7));
  const uint32_t row_v = lab::held(r * kRow + 16 * (((c + 4) ^ r) & 7));
  const uint8_t* const slice_c = in.slices + 32 * x.slice + 16 * (c & 1);
  const uint8_t* const values_c = in.values + 64 * x.tile + 16 * c;
  const bool slice_live = c < 2 || lane == 2;
  const uint16_t* pe = rec_of(0) + 8 + r;
  const uint8_t* pcg = record(0) + 32;
  uint32_t pslot = ring;
  const uint8_t* slice_w = slice_c;
  const uint8_t* values_w = values_c;
  int S = rec_of(0)[0];
  //: the cursor to the next window holding stages, its record landed first (a wait on every group but the newest D - 1,
  //: or every group where fewer have been committed since)
  const auto enter = [&] {
#pragma unroll 1
    do {
      ++pw;
      ps = 0;
      if constexpr (AHEAD && MB) {
        if (pw < W) lab::mbar_wait(bars + 8 * (pw % 3), (pw / 3) & 1);
        fetch(pw + 2);
      } else if constexpr (AHEAD) {
        if (pw < W) {
          if (groups - 1 - carrier < D - 1) {
            commit();
            ++groups;
            wait_groups<0>();
          } else {
            wait_groups<D - 1>();
          }
          __syncwarp();
        }
        fetch(pw + 2);
        carrier = carrier_next;
        carrier_next = groups;
      }
      pe = rec_of(pw) + 8 + r;
      pcg = record(pw) + 32;
      slice_w = slice_c + static_cast<size_t>(pw) * kWindow * kSlices;
      values_w = values_c + static_cast<size_t>(pw) * kWindow * kValues;
      S = pw < W ? rec_of(pw)[0] : 0;
    } while ((pw < W) & (S == 0));
  };
  if (S == 0) enter();
  const auto issue = [&] {
    if (pw < W) {
      const uint32_t tk = *pe;
      if constexpr (PART != 1) {
        //: the slice's chunks (lanes `c` 0, 1); lane 2 the stage's coefficients in the same instruction; the slice
        //: slots' last 32 bytes, zeroed once, rewritten with zeros by the rest
        const bool cs = lane == 2;
        copy16<CA>(cs ? pslot + 8 * kRow : pslot + row_s, cs ? pcg : slice_w + tk * kSlices, slice_live);
        copy16<CA>(pslot + row_v, values_w + tk * kValues, true);
      } else {
        if (lane == 2) copy16<CA>(pslot + 8 * kRow, pcg, true);
      }
      pe += 16;
      pcg += 32;
      if (++ps >= S) enter();
    }
    pslot = next_slot(pslot);
    commit();
    ++groups;
  };
  //: the warp's ring zeroed once: a slice slot's last 32 bytes stay zero
  for (int k = lane; k < D * kStage / 16; k += 32)
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(ring + 16 * k), "r"(0) : "memory");
  __syncwarp();
  const long long t0 = clock64();
  if constexpr (PART != 4) {
#pragma unroll 1
    for (int q = 0; q < D; ++q) issue();
  }
  //: THE CONSUMER: the ring slot of its next stage, the entries of its window's and the next window's lists (lane `j`
  //: list `j`'s); a lane's row and swizzle offsets in a stage held for both step forms, its stages' coefficients read
  //: from the stage. A step waits for its stages, every stage but the D - its stages ahead, and refills its slots.
  int cn = list_n(0, lane & 3), cn_next = list_n(1, lane & 3);
  uint32_t cslot = ring;
  const int tile = lane >> 3, rr = lane & 7;
  const auto off = [&](int k) { return lab::held(static_cast<uint32_t>(rr * kRow + 16 * ((k ^ rr) & 7))); };
  const uint32_t a0 = off(tile & 1), a1 = off(2 + (tile & 1)), b0 = off(4 + (tile >> 1)), b1 = off(6 + (tile >> 1));
  const uint32_t e8a = off(tile), e8b = off(4 + tile), coff = lab::held(8 * kRow + 4 * (lane & 3));
  const bool hi_a = lab::held(tile >= 2 ? 1u : 0u) != 0, hi_b = lab::held(tile & 1) != 0;
  constexpr bool kFold = PART < 2 || PART == 4;
  //: a list's steps over its `n` entries: 16 tokens while more than 8 remain (A's rows from the stage `hi_a` picks, B's
  //: from `hi_b`'s), then 8
  const auto fold_list = [&](float (&ac)[2][4][4], int n) {
    int s = 0;
    for (int k = __any_sync(0xffffffffu, n - s > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
      do {
        const uint32_t second = next_slot(cslot);
        if constexpr (PART != 4) {
          wait_groups<D - 2>();
          __syncwarp();
        }
        if constexpr (kFold) {
          const uint32_t sa = hi_a ? second : cslot, sb = hi_b ? second : cslot;
          uint32_t a[2][4], x[4], y[4];
          ldsm_t(a[0], sa + a0);
          ldsm_t(a[1], sa + a1);
          ldsm_t(x, sb + b0);
          ldsm_t(y, sb + b1);
          const uint32_t c01 = lds32(cslot + coff), c89 = lds32(second + coff);
          rola::static_for<2>([&](auto M) {
            a[M][0] = hmul2(a[M][0], c01);
            a[M][1] = hmul2(a[M][1], c01);
            a[M][2] = hmul2(a[M][2], c89);
            a[M][3] = hmul2(a[M][3], c89);
            mma16(ac[M][0], a[M], x[0], x[1]);
            mma16(ac[M][1], a[M], x[2], x[3]);
            mma16(ac[M][2], a[M], y[0], y[1]);
            mma16(ac[M][3], a[M], y[2], y[3]);
          });
        }
        cslot = next_slot(second);
        if constexpr (PART != 4) {
          issue();
          issue();
        }
        s += 16;
      } while (__any_sync(0xffffffffu, n - s > 8));
    for (int k = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; k > 0; --k) {
      if constexpr (PART != 4) {
        wait_groups<D - 1>();
        __syncwarp();
      }
      if constexpr (kFold) {
        uint32_t a[4], b[4];
        ldsm_t(a, cslot + e8a);
        ldsm_t(b, cslot + e8b);
        const uint32_t c01 = lds32(cslot + coff);
#pragma unroll
        for (int q = 0; q < 4; ++q) a[q] = hmul2(a[q], c01);
        rola::static_for<2>([&](auto M) {
          rola::static_for<4>([&](auto N) { mma8(ac[M][N], a[2 * M], a[2 * M + 1], b[N]); });
        });
      }
      cslot = next_slot(cslot);
      if constexpr (PART != 4) issue();
    }
  };
#pragma unroll 1
  for (int w = 0; w < W; ++w) {
    rola::static_for<4>([&](auto J) { fold_list(acc[J], AHEAD ? __shfl_sync(0xffffffffu, cn, J) : list_n(w, J)); });
    cn = cn_next;
    cn_next = list_n(w + 2, lane & 3);
  }
  wait_groups<0>();
  span_out(t, t0, warp, lane);
  state_out(in.out, acc, warp, lane);
}

//: a pointer HELD in a register (`lab::held`'s 64-bit form), and `base + a * b` as one wide multiply-add on it
__device__ __forceinline__ const uint8_t* held64(const uint8_t* v) {
  asm volatile("" : "+l"(v));
  return v;
}

__device__ __forceinline__ const uint8_t* wide(const uint8_t* base, uint32_t a, uint32_t b) {
  const uint8_t* r;
  asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(base));
  return r;
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}

//: a token's 16-byte copy; where `token` is a pad (its top bit set) zeros read from nothing, or (`SKIP`) no copy -- the
//: slot keeps an earlier stage's finite row, which a pad's zero coefficient cancels
template <int SKIP>
__device__ __forceinline__ void copy16_token(uint32_t dst, const void* src, uint32_t token) {
  if constexpr (SKIP) {
    asm volatile("{ .reg .pred p; setp.ge.s32 p, %2, 0; @p cp.async.cg.shared.global [%0], [%1], 16; }" ::"r"(dst),
                 "l"(src), "r"(token)
                 : "memory");
  } else {
    asm volatile("{ .reg .pred p; setp.lt.s32 p, %2, 0; cp.async.cg.shared.global [%0], [%1], 16, p; }" ::"r"(dst),
                 "l"(src), "r"(token)
                 : "memory");
  }
}

//: a 16-byte copy issued only where `live`
__device__ __forceinline__ void copy16_if(uint32_t dst, const void* src, uint32_t live) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.cg.shared.global [%0], [%1], 16; }" ::"r"(dst),
               "l"(src), "r"(live)
               : "memory");
}

//: THE STAGED STREAM, the stream's floor form: each warp's ring of 4 stages of 2,144 bytes -- a step's 16 rows (a
//: tail's first 8), its 16 coefficients lane-ready, the 16 tokens of the stage 4 on -- its stages one record stream
//: across windows and digits (`staged.bin`), so the producer reads a stage's tokens from the slot it refills, copies
//: its rows (a pad's as zeros, read from nothing) and its record in one burst, and checks no window; a step reads its
//: coefficients from its stage. `PART` 0, 1 the fold with the producer and no row copies, 2 the copies alone
constexpr int kStaged = 4, kStagedStage = 16 * kRow + 96, kStagedRecord = 96;

template <int PART, int SET, int SKIP = 0>
__global__ void __launch_bounds__(kThreads, 1) staged(In in, const uint8_t* records, int stream_records,
                                                      long long* t) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, q = lane >> 2, c = lane & 3;
  const Box x = box_of();
  const uint32_t ring = lab::held(smem(sm) + warp * kStaged * kStagedStage);
  //: the ring zeroed once: a slice slot's last 32 bytes stay zero
  for (int k = lane; k < kStaged * kStagedStage / 16; k += 32)
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(ring + 16 * k), "r"(0) : "memory");
  __syncwarp();
  //: the lane's constants held in registers: its 16 bytes of each record (lanes 0-5); its copies -- values rows q and
  //: q + 8 chunk c, slice row q + 8 (c >> 1) chunk c & 1 -- their sources' bases and destinations in a stage
  const uint8_t* const stream =
      records + (static_cast<size_t>(x.group) * kWarps + warp) * stream_records * kStagedRecord;
  const uint8_t* const rec_lane = held64(stream + kStaged * kStagedRecord + 16 * lane);
  const uint8_t* const slices_c = held64(in.slices + 32 * x.slice + 16 * (c & 1));
  const uint8_t* const values_c = held64(in.values + 64 * x.tile + 16 * c);
  const int qs = q + 8 * (c >> 1);
  const uint32_t to_s = lab::held(qs * kRow + 16 * (((c & 1) ^ qs) & 7));
  const uint32_t to_v = lab::held(q * kRow + 16 * (((4 + c) ^ q) & 7)), hi_s = lab::held(c >> 1);
  const uint32_t rec_dst = lab::held(16 * kRow + 16 * lane), rec_live = lab::held(lane < 6 ? 1u : 0u);
  const uint32_t tok_at = lab::held(16 * kRow + 32 + 8 * q);
  const auto slot_of = [&](uint32_t s) { return ring + (s & (kStaged - 1)) * kStagedStage; };
  //: stages 0-3's tokens into their slots (records 0-3's last 64 bytes, four lanes a slot)
  if (lane < 16)
    copy16<0>(slot_of(lane >> 2) + 16 * kRow + 32 + 16 * (lane & 3),
              stream + (lane >> 2) * kStagedRecord + 32 + 16 * (lane & 3), true);
  commit();
  wait_groups<0>();
  __syncwarp();
  //: THE PRODUCER: the next stage into `slot` -- its tokens from the slot, then its rows and its record -- a group
  //: each; a step refills the slot the step before it read. A pad's token is masked to a valid row where its copy
  //: writes zeros; where it is not copied its address is unused
  constexpr uint32_t kMask = SKIP ? 0xffffffffu : 0x7fffffffu;
  const uint8_t* prec = rec_lane;
  const auto issue = [&](uint32_t slot) {
    const uint2 tk = lds64(slot + tok_at);
    if constexpr (PART != 1) {
      const uint32_t ts = hi_s ? tk.y : tk.x;
      copy16_token<SKIP>(slot + to_s, wide(slices_c, ts & kMask, 256), ts);
      copy16_token<SKIP>(slot + to_v, wide(values_c, tk.x & kMask, kValues), tk.x);
      copy16_token<SKIP>(slot + to_v + 8 * kRow, wide(values_c, tk.y & kMask, kValues), tk.y);
    }
    copy16_if(slot + rec_dst, prec, rec_live);
    commit();
    prec += kStagedRecord;
  };
  float acc[4][2][4][4] = {};
  const long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < kStaged - 1; ++k) issue(slot_of(k));
  //: THE CONSUMER: its digits' entries a window ahead (lane `j` digit `w + 8 j`'s), the stage `cs` it reads, a lane's
  //: row and swizzle offsets in a stage held (rows 8-15 at 8 rows on), a step's coefficients from its stage
  const auto digit_n = [&](int w, int j) {
    return w < W ? static_cast<int>(in.n[(static_cast<size_t>(w) * kGroups + x.group) * kLists + warp + kWarps * j])
                 : 0;
  };
  int cn = digit_n(0, lane & 3), cn_next = digit_n(1, lane & 3);
  uint32_t cs = 0, last = slot_of(kStaged - 1);
  const int tile = lane >> 3, rr = lane & 7, t4 = lane & 3;
  const auto off = [&](int k, int hi) {
    return lab::held(static_cast<uint32_t>((rr + 8 * hi) * kRow + 16 * ((k ^ rr) & 7)));
  };
  const uint32_t a0 = off(tile & 1, tile >= 2), a1 = off(2 + (tile & 1), tile >= 2);
  const uint32_t b0 = off(4 + (tile >> 1), tile & 1), b1 = off(6 + (tile >> 1), tile & 1);
  const uint32_t e8a = off(tile, 0), e8b = off(4 + tile, 0), coff = lab::held(16 * kRow + 8 * t4);
  constexpr bool kFold = PART != 2;
  const auto fold_digit = [&](float (&ac)[2][4][4], int n) {
    int s = 0;
    for (int k = __any_sync(0xffffffffu, n - s > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
      do {
        issue(last);
        wait_groups<kStaged - 1>();
        __syncwarp();
        const uint32_t slot = slot_of(cs);
        if constexpr (kFold) {
          uint32_t a[2][4], bx[4], by[4];
          ldsm_t(a[0], slot + a0);
          ldsm_t(a[1], slot + a1);
          ldsm_t(bx, slot + b0);
          ldsm_t(by, slot + b1);
          const uint2 cc = lds64(slot + coff);
          rola::static_for<2>([&](auto M) {
            a[M][0] = hmul2(a[M][0], cc.x);
            a[M][1] = hmul2(a[M][1], cc.x);
            a[M][2] = hmul2(a[M][2], cc.y);
            a[M][3] = hmul2(a[M][3], cc.y);
            mma16(ac[M][0], a[M], bx[0], bx[1]);
            mma16(ac[M][1], a[M], bx[2], bx[3]);
            mma16(ac[M][2], a[M], by[0], by[1]);
            mma16(ac[M][3], a[M], by[2], by[3]);
          });
        }
        last = slot;
        ++cs;
        s += 16;
      } while (__any_sync(0xffffffffu, n - s > 8));
    for (int k = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; k > 0; --k) {
      issue(last);
      wait_groups<kStaged - 1>();
      __syncwarp();
      const uint32_t slot = slot_of(cs);
      if constexpr (kFold) {
        uint32_t a[4], b[4];
        ldsm_t(a, slot + e8a);
        ldsm_t(b, slot + e8b);
        const uint32_t c01 = lds32(slot + coff);
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = hmul2(a[i], c01);
        rola::static_for<2>([&](auto M) {
          rola::static_for<4>([&](auto N) { mma8(ac[M][N], a[2 * M], a[2 * M + 1], b[N]); });
        });
      }
      last = slot;
      ++cs;
    }
  };
#pragma unroll 1
  for (int w = 0; w < W; ++w) {
    rola::static_for<4>([&](auto J) { fold_digit(acc[J], __shfl_sync(0xffffffffu, cn, J)); });
    cn = cn_next;
    cn_next = digit_n(w + 2, lane & 3);
  }
  wait_groups<0>();
  span_out(t, t0, warp, lane);
  state_out(in.out, acc, warp, lane);
}

//: the inputs from `--inputs`
struct Host {
  std::vector<uint16_t> n, tok, cnt, uni;
  std::vector<uint8_t> slices, values, rec, staged;
  std::vector<float> state;
  int ctas = 0, windows = 0, record_bytes = 0, stream_records = 0;
};

std::vector<uint8_t> file(const std::string& path, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

Host load(const std::string& dir) {
  Host h;
  std::ifstream f(dir + "/lists.bin", std::ios::binary);
  int head[4];
  f.read(reinterpret_cast<char*>(head), sizeof(head));
  h.ctas = head[1];
  h.windows = head[2];
  const size_t wg = static_cast<size_t>(h.windows) * kGroups;
  h.n.resize(wg * kLists);
  h.tok.resize(wg * kLists * kWindow);
  h.cnt.resize(wg);
  h.uni.resize(wg * kWindow);
  for (size_t i = 0; i < wg; ++i) {
    f.read(reinterpret_cast<char*>(&h.n[i * kLists]), kLists * 2);
    f.read(reinterpret_cast<char*>(&h.tok[i * kLists * kWindow]), kLists * kWindow * 2);
    f.read(reinterpret_cast<char*>(&h.cnt[i]), 2);
    f.read(reinterpret_cast<char*>(&h.uni[i * kWindow]), kWindow * 2);
  }
  const size_t tokens = static_cast<size_t>(h.windows) * kWindow;
  h.slices = file(dir + "/slices.bin", tokens * kSlices);
  h.values = file(dir + "/values.bin", tokens * kValues);
  std::ifstream rf(dir + "/records.bin", std::ios::binary);
  int rh[2];
  rf.read(reinterpret_cast<char*>(rh), sizeof(rh));
  h.record_bytes = rh[1];
  h.rec.resize(static_cast<size_t>(h.windows) * kGroups * kWarps * h.record_bytes);
  rf.read(reinterpret_cast<char*>(h.rec.data()), h.rec.size());
  h.state.resize(static_cast<size_t>(h.ctas) * kLists * 1024);
  std::ifstream(dir + "/state.bin", std::ios::binary)
      .read(reinterpret_cast<char*>(h.state.data()), h.state.size() * 4);
  std::ifstream sf(dir + "/staged.bin", std::ios::binary | std::ios::ate);
  const size_t staged_bytes = static_cast<size_t>(sf.tellg()) - 4;
  sf.seekg(0);
  sf.read(reinterpret_cast<char*>(&h.stream_records), 4);
  h.staged.resize(staged_bytes);
  sf.read(reinterpret_cast<char*>(h.staged.data()), static_cast<std::streamsize>(staged_bytes));
  return h;
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

//: `part` as the kernels' `PART`: 1, 2 or 4 (a part alone, a bound) form no state, so go unchecked; `keeps`: the step
//: signatures a form keeps -- `kBranches` the streamed producer's cursor inside the fold's step loop, `kGlobal` its
//: lookups from global memory at their use (or, with no producer, the window loop's list entries, the loop ptxas then
//: takes for the innermost)
constexpr int kBranches = 1, kGlobal = 2;

template <class Kernel>
void run(const lab::Bench& b, const std::string& kname, const std::string& variant, Kernel kernel, int smem_bytes,
         const In& in, const Host& h, int ctas, int part, int keeps) {
  b.expect(kname, "LDGSTS", ">=", part == 1 ? 0 : 1);
  if (keeps & kBranches) b.expect(kname, "BSSY", ">=", 0);
  if (keeps & kGlobal) b.expect(kname, "STEP_GLOBAL", ">=", 0);
  const bool folds = part == 0 || part == 1 || part == 4;
  b.expect(kname, "HMMA", folds ? ">=" : "==", folds ? 16 : 0);
  b.exempt(kname, "a copy-overlap probe: the carry's fold step over streamed or staged rows, its cost the question");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  const lab::Samples s = b.variant([&] { kernel<<<ctas, kThreads, smem_bytes>>>(in, b.times); },
                                   [&] { return b.span(kWarps, ctas) / h.windows; });
  if (part) {
    b.unchecked(variant, folds ? "a bound: the fold over stale rows, no state formed"
                                   : "a bound: copies and their waits alone, no state formed");
  } else {
    std::vector<float> out(static_cast<size_t>(ctas) * kLists * 1024);
    LAB_CHECK(cudaMemcpy(out.data(), in.out, out.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < out.size(); ++i) bad += out[i] != h.state[i];
    b.check(variant, bad == 0, std::to_string(bad) + " of " + std::to_string(out.size()) + " state values differ");
  }
  b.record(variant, "cycles a window", s);
}

//: the staged form: its records with the inputs; `part` as its `PART`
template <int PART, int SET, int SKIP = 0>
void run_staged(const lab::Bench& b, const std::string& variant, const In& in, const Host& h, const uint8_t* records,
                int ctas) {
  const std::string name = lab::kernel("staged", {PART, SET, SKIP});
  const bool folds = PART != 2;
  b.expect(name, "LDGSTS", ">=", PART == 1 ? 1 : 4);
  b.expect(name, "BSSY", ">=", 0);
  b.expect(name, "HMMA", folds ? ">=" : "==", folds ? 16 : 0);
  b.exempt(name, "a copy-overlap probe: the carry's fold step over staged rows, its cost the question");
  if (b.expect_only) return;
  const int bytes = kWarps * kStaged * kStagedStage;
  LAB_CHECK(cudaFuncSetAttribute(staged<PART, SET, SKIP>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] { staged<PART, SET, SKIP><<<ctas, kThreads, bytes>>>(in, records, h.stream_records, b.times); },
      [&] { return b.span(kWarps, ctas) / h.windows; });
  if (PART) {
    b.unchecked(variant, folds ? "a bound: the fold over stale rows, no state formed"
                               : "a bound: copies and their waits alone, no state formed");
  } else {
    std::vector<float> out(static_cast<size_t>(ctas) * kLists * 1024);
    LAB_CHECK(cudaMemcpy(out.data(), in.out, out.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < out.size(); ++i) bad += out[i] != h.state[i];
    b.check(variant, bad == 0, std::to_string(bad) + " of " + std::to_string(out.size()) + " state values differ");
  }
  b.record(variant, "cycles a window", s);
}

}  // namespace row_stream

int main(int argc, char** argv) {
  using namespace row_stream;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  //: each dataset's inputs (`row_stream.inputs.py`'s `DATASETS`): nl64k-alt-k4, flagship-alt-k16 in `k16/` and
  //: flagship-cohort-k16 in `c16/`
  Host h, h16, hc;
  In in{}, in16{}, inc{};
  const uint8_t *st = nullptr, *st16 = nullptr, *stc = nullptr;
  const auto device = [&](const std::string& d, Host& host, In& to) {
    host = load(d);
    if (host.ctas > b.sms) {
      std::fprintf(stderr, "row_stream: %d boxes need as many SMs, the device has %d\n", host.ctas, b.sms);
      std::exit(1);
    }
    float* out = nullptr;
    LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(host.ctas) * kLists * 1024 * 4));
    to = In{up(host.n), up(host.tok), up(host.cnt), up(host.uni), up(host.slices), up(host.values), up(host.rec),
            out, host.windows, host.record_bytes};
  };
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "row_stream: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    device(dir, h, in);
    device(dir + "/k16", h16, in16);
    device(dir + "/c16", hc, inc);
    st = up(h.staged);
    st16 = up(h16.staged);
    stc = up(hc.staged);
  }
  const int ctas = h.ctas, pool = 2 * kWindow + 2 * kLists * kWindow + kWindow * kRow;
  const auto res = [&](auto kernel, std::initializer_list<int> args, const char* variant, int part) {
    run(b, lab::kernel("resident", args), variant, kernel, pool, in, h, ctas, part, 0);
  };
  const auto on = [&](const Host& hs, const In& is, auto kernel, std::initializer_list<int> args,
                      const std::string& variant, int d, int part, int n) {
    run(b, lab::kernel("streamed", args), variant, kernel, kWarps * (d * kStage + 3 * hs.record_bytes) + 192, is, hs, n,
        part, kBranches | (*(args.begin() + 3) && part != 4 ? 0 : kGlobal));
  };
  const auto str = [&](auto kernel, std::initializer_list<int> args, const char* variant, int d, int part, int n) {
    on(h, in, kernel, args, variant, d, part, n);
  };
  res(resident<0, 0>, {0, 0}, "resident (union staged, barrier, fold)", 0);
  str(streamed<2, 0, 0, 1>, {2, 0, 0, 1, 0, 0}, "streamed, ring 2 (a 16-token step's own stages)", 2, 0, ctas);
  str(streamed<3, 0, 0, 1>, {3, 0, 0, 1, 0, 0}, "streamed, ring 3", 3, 0, ctas);
  str(streamed<4, 0, 0, 1>, {4, 0, 0, 1, 0, 0}, "streamed, ring 4", 4, 0, ctas);
  str(streamed<6, 0, 0, 1>, {6, 0, 0, 1, 0, 0}, "streamed, ring 6", 6, 0, ctas);
  str(streamed<4, 0, 0, 0>, {4, 0, 0, 0, 0, 0}, "streamed, ring 4, lookups at their use", 4, 0, ctas);
  str(streamed<4, 0, 1, 1>, {4, 0, 1, 1, 0, 0}, "bound: streamed, the fold alone (ring 4)", 4, 1, ctas);
  str(streamed<4, 0, 4, 1>, {4, 0, 4, 1, 0, 0}, "bound: streamed, the fold alone, no producer", 4, 4, ctas);
  str(streamed<4, 0, 0, 1, 0, 1>, {4, 0, 0, 1, 0, 1}, "streamed, ring 4, records on mbarriers", 4, 0, ctas);
  str(streamed<4, 0, 1, 1, 0, 1>, {4, 0, 1, 1, 0, 1}, "bound: the fold alone, records on mbarriers", 4, 1, ctas);
  str(streamed<4, 0, 2, 1>, {4, 0, 2, 1, 0, 0}, "bound: streamed, the copies alone (ring 4)", 4, 2, ctas);
  str(streamed<4, 0, 2, 1>, {4, 0, 2, 1, 0, 0}, "bound: streamed, the copies alone, 40 boxes", 4, 2, ctas / 2);
  str(streamed<4, 0, 2, 0>, {4, 0, 2, 0, 0, 0}, "bound: streamed, the copies alone, lookups at their use", 4, 2, ctas);
  res(resident<0, 1>, {0, 1}, "bound: resident, the fold alone (no copies)", 1);
  res(resident<0, 2>, {0, 2}, "bound: resident, the copies alone (no fold)", 2);
  res(resident<1, 0>, {1, 0}, "resident, L1-allocating copies (.ca)", 0);
  str(streamed<4, 1, 0, 1>, {4, 1, 0, 1, 0, 0}, "streamed, ring 4, L1-allocating copies (.ca)", 4, 0, ctas);
  const int c16 = h16.ctas;
  on(h16, in16, streamed<4, 0, 0, 1, 1>, {4, 0, 0, 1, 1, 0}, "alt-k16: streamed, ring 4", 4, 0, c16);
  on(h16, in16, streamed<4, 0, 1, 1, 1>, {4, 0, 1, 1, 1, 0}, "alt-k16: bound: the fold alone (ring 4)", 4, 1, c16);
  on(h16, in16, streamed<4, 0, 4, 1, 1>, {4, 0, 4, 1, 1, 0}, "alt-k16: bound: the fold alone, no producer", 4, 4, c16);
  on(h16, in16, streamed<4, 0, 2, 1, 1>, {4, 0, 2, 1, 1, 0}, "alt-k16: bound: the copies alone (ring 4)", 4, 2, c16);
  const int cc = hc.ctas;
  on(hc, inc, streamed<4, 0, 0, 1, 2>, {4, 0, 0, 1, 2, 0}, "cohort-k16: streamed, ring 4", 4, 0, cc);
  on(hc, inc, streamed<4, 0, 1, 1, 2>, {4, 0, 1, 1, 2, 0}, "cohort-k16: bound: the fold alone (ring 4)", 4, 1, cc);
  on(hc, inc, streamed<4, 0, 4, 1, 2>, {4, 0, 4, 1, 2, 0}, "cohort-k16: bound: the fold alone, no producer", 4, 4, cc);
  on(hc, inc, streamed<4, 0, 2, 1, 2>, {4, 0, 2, 1, 2, 0}, "cohort-k16: bound: the copies alone (ring 4)", 4, 2, cc);
  on(hc, inc, streamed<4, 0, 0, 1, 2, 1>, {4, 0, 0, 1, 2, 1}, "cohort-k16: streamed, ring 4, records on mbarriers", 4,
     0, cc);
  on(h16, in16, streamed<4, 1, 0, 1, 1>, {4, 1, 0, 1, 1, 0}, "alt-k16: streamed, ring 4, L1-allocating copies (.ca)", 4,
     0, c16);
  on(hc, inc, streamed<4, 1, 0, 1, 2>, {4, 1, 0, 1, 2, 0}, "cohort-k16: streamed, ring 4, L1-allocating copies (.ca)",
     4, 0, cc);
  run_staged<0, 0>(b, "staged, ring 4", in, h, st, ctas);
  run_staged<1, 0>(b, "staged: bound: the fold with the producer, no row copies", in, h, st, ctas);
  run_staged<2, 0>(b, "staged: bound: the copies alone", in, h, st, ctas);
  run_staged<0, 1>(b, "alt-k16: staged, ring 4", in16, h16, st16, c16);
  run_staged<1, 1>(b, "alt-k16: staged: bound: the fold with the producer, no row copies", in16, h16, st16, c16);
  run_staged<0, 2>(b, "cohort-k16: staged, ring 4", inc, hc, stc, cc);
  run_staged<1, 2>(b, "cohort-k16: staged: bound: the fold with the producer, no row copies", inc, hc, stc, cc);
  run_staged<0, 0, 1>(b, "staged, ring 4, pads not copied", in, h, st, ctas);
  run_staged<0, 1, 1>(b, "alt-k16: staged, ring 4, pads not copied", in16, h16, st16, c16);
  run_staged<0, 2, 1>(b, "cohort-k16: staged, ring 4, pads not copied", inc, hc, stc, cc);
}
