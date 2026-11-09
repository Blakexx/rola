// measure/lab/experiments/read_stream.cu -- QUESTION: when the readout's rows are streamed a digit at a time into each
// warp's ring (merge width 1) instead of gathered from a staged pool, what does the stream cost beyond the readout,
// at what ring depth, and how few instructions a MMA do the readout and its stream take? METHOD: the readout's
// tokens-on-N form (a tile 8 tokens, 4 bf16 MMAs against the digit's state from the image, each lane's two tokens'
// outputs scaled by their coefficients, packed exact and added into their panel rows; each pass's every digit's first
// tile in one straight-line block, then each digit's rest; the panel drained a pass) over three cells' read lists
// (`read_stream.inputs.py`): 80 CTAs, box `b` of one head, a warp its digits w, w + 8, w + 16, w + 24; a stage a
// tile's 8 read rows (a token's 64 bytes of its slice group, shared between the boxes as the carry's are, so L2
// serves what one box fetched to the others) and its 8 entries lane-ready (fp32 coefficient, panel offset), copied
// ahead across digits, passes and windows from the warp's window records (a pre-pass's plan). The lane's addresses
// held in registers; the drain unrolled on held addresses. Bounds: the readout with the producer and no row copies,
// the copies alone, the readout with no producer, and the same with no drain. A sample is the CTA's span a window;
// each forming variant's drained panel is checked word for word.
// see docs/internals/tools/lab.md#read-stream
#include <cstdlib>

#include "read_image.cuh"

namespace read_stream {

using read_image::kPanelPitch;
using read_image::kPlaneBytes;
using read_image::kPlanes;
using read_image::kSlots;
using read_image::kThreads;
using read_image::kWarps;
using read_image::kWords;

constexpr int kWindow = 512, kGroups = 8, kTiles = 2, kSliceGroups = 8, kRow = 64, kPasses = 2;
//: THE LAYOUT: the image's planes, each warp's ring of `D` stages (8 read rows, then their 8 entries, 8 bytes each:
//: 576 bytes, the rows' 32-byte phase kept), the lanes' pad bases (128 bytes), the 8 dummy panel rows and the panel,
//: each warp's two token records
constexpr int kRing = kPlanes * kPlaneBytes, kStage = 8 * kRow + 64;

template <int D>
struct Layout {
  static constexpr int kPads = kRing + kWarps * D * kStage, kPanel = kPads + 128 + 8 * kPanelPitch * 4;
  static constexpr int kRec = kPanel + kSlots * kPanelPitch * 4;
};

//: THE INPUTS on the device: a box-window's passes `[windows][8]`, a read digit's tiles in a pass `[windows][2][256]`,
//: the token records (`token_bytes` each) and word records (`word_bytes`), per window, read group and warp, a token's
//: read rows `[windows x 512][8 slice groups][64 B]`, and each CTA's drained panel `[ctas][2][256][16]`
struct In {
  const uint16_t* passes;
  const uint16_t* tiles;
  const uint8_t* rec;
  const uint8_t* words;
  const uint8_t* rows;
  uint32_t* panel_out;
  int windows, token_bytes, word_bytes;
};

//: THIS CTA'S BOX (the carry's order, its channel tile fastest): its slice group and read group
struct Box {
  int slice, read;
};

__device__ __forceinline__ Box box_of() {
  const int b = static_cast<int>(blockIdx.x);
  return Box{(b / kTiles) % kSliceGroups, b / (kTiles * kSliceGroups)};
}

//: A LANE'S CONSTANTS for tokens on N (`read_image::lane_of<1>`), its pad base in this layout
__device__ __forceinline__ read_image::Lane lane_of(uint32_t sm, int warp, int lane, int pads) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t plane = sm + (warp % kPlanes) * kPlaneBytes;
  const int srow = 8 * (lane >> 4) + r, sc = q & 1;
  const auto h = [](auto v) { return static_cast<decltype(v)>(lab::held(static_cast<uint32_t>(v))); };
  return {h(r), h(2 * t), h(2 * t + 1), h(16u * q), h(sm + pads + 4u * g + 32u * t),
          h(plane + read_image::in_plane(srow, sc)), h(plane + read_image::in_plane(srow, 2 + sc)), h(8u * r)};
}

//: a 16-byte copy; `CA` allocates it in L1
template <int CA>
__device__ __forceinline__ void copy16(uint32_t dst, const void* src) {
  if constexpr (CA) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  }
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

__device__ __forceinline__ uint32_t lds16(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}

//: A TILE: 8 tokens on N -- their read rows (B) by one `ldmatrix.x4` at `rows`, 4 MMAs against the digit's state (A),
//: each lane's two tokens' outputs scaled by their coefficients (`e`: coefficient and panel offset of tokens 2t and
//: 2t + 1), packed exact and added at their panel rows
__device__ __forceinline__ void tile(uint32_t rows, uint4 e, uint32_t pad, const uint32_t (&f)[16]) {
  uint32_t b[4];
  lab::ldsm(b, rows);
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
  const float c0 = __uint_as_float(e.x), c1 = __uint_as_float(e.z);
  const uint32_t p0 = pad + e.y, p1 = pad + e.w;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    read_image::panel_red(p0 + 32 * m, read_image::s16_exact(acc[m][0], acc[m][2], c0));
    read_image::panel_red(p1 + 32 * m, read_image::s16_exact(acc[m][1], acc[m][3], c1));
  }
}

//: THE READOUT STREAMED: each warp's ring of `D` stages (`D >= 4`: the first tiles' block holds four); the warp's
//: stages its window records' in order, each copied as soon as a slot no tile holds is free, across digits, passes and
//: windows -- the producer reading a stage's tokens from its token record (copied a window ahead with the stage group
//: it is issued in: every warp-window holds at least 4 stages, so at `D <= 5` a window's entry waits only on the groups
//: the ring's next stage needs) and copying its rows and its entries; the consumer its passes and its digits' tiles a
//: window ahead into registers, the panel drained after each pass. `PART` 1 the readout with the producer, no row
//: copies (the ring's stale rows), 2 the copies alone, 3 the readout with no producer (its entries from global
//: memory, a tile ahead), 4 the stream with no waits on its stages (a bound: what waiting on the copies costs), 5 as 3
//: with no drain. `SET` names the dataset's instantiation, so the profiler keeps each dataset's counters apart; `CA`
//: allocates the copies in L1 (`cp.async.ca`), else not (`.cg`)
template <int D, int PART, int SET, int CA>
__global__ void __launch_bounds__(kThreads, 1) stream(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<D>;
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, C = in.token_bytes, CW = in.word_bytes,
            r = lane >> 2, c = lane & 3;
  const Box x = box_of();
  const uint32_t sm = lab::smem(smem);
  constexpr bool kProducer = PART != 3 && PART != 5;
  //: the image's even channels -0.0625 and odd +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < (L::kRec + kWarps * 2 * C) / 16; k += kThreads) {
    const uint32_t v = k < kRing / 16 ? 0x3d80bd80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const read_image::Lane lc = lane_of(sm, warp, lane, L::kPads);
  //: the lane's constants held in registers (opaque, so ptxas keeps them rather than re-forming them each tile)
  const uint32_t ring = lab::held(sm + kRing + warp * D * kStage), ring_last = lab::held(ring + (D - 1) * kStage);
  const uint32_t recs = sm + L::kRec + warp * 2 * C;
  const auto next_slot = [&](uint32_t slot) { return slot == ring_last ? ring : slot + kStage; };
  const auto warp_record = [&](int w) {
    return (static_cast<size_t>(w) * kGroups + x.read) * kWarps + warp;
  };
  //: window `w`'s token record into its buffer `w % 2` (fetched a window ahead), a lane a 16-byte chunk
  const auto fetch = [&](int w) {
#pragma unroll 1
    for (int q = lane; q < (w < W ? C / 16 : 0); q += 32)
      copy16<CA>(recs + (w & 1) * C + 16 * q, in.rec + warp_record(w) * C + 16 * q);
  };
  //: THE PRODUCER'S CURSOR: the window and stage it copies next and the window's stages; its running addresses -- the
  //: lane's token in the window's record, the stage's entries in global memory, its row's base in the window's rows,
  //: the ring slot it fills next; the groups committed and the group carrying the next window's record
  int pw = 0, ps = 0, groups = 0, carrier = 0;
  fetch(0);
  commit();
  wait_groups<0>();
  __syncwarp();
  fetch(1);
  const uint32_t row_c = lab::held(r * kRow + 16 * (c ^ ((r >> 1) & 3)));
  const uint8_t* const rows_c = in.rows + x.slice * kRow + 16 * c;
  uint32_t pe = recs + 16 + 2 * r, pslot = ring;
  const uint8_t* pwd = in.words + warp_record(0) * CW + 16 * (lane & 3);
  const uint8_t* prow = rows_c;
  int S = static_cast<int>(lds16(recs));
  //: the cursor to the next window holding stages, its record landed first (a wait on every group but the newest D - 1,
  //: or every group where fewer have been committed since)
  const auto enter = [&] {
#pragma unroll 1
    do {
      ++pw;
      ps = 0;
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
      fetch(pw + 1);
      carrier = groups;
      const uint32_t rec = recs + (pw & 1) * C;
      pe = rec + 16 + 2 * r;
      pwd = in.words + warp_record(pw) * CW + 16 * (lane & 3);
      prow = rows_c + static_cast<size_t>(pw) * kWindow * kSliceGroups * kRow;
      S = pw < W ? static_cast<int>(lds16(rec)) : 0;
    } while ((pw < W) & (S == 0));
  };
  const auto issue = [&] {
    if (pw < W) {
      const uint32_t tk = lds16(pe);
      if constexpr (PART != 1) copy16<CA>(pslot + row_c, prow + tk * (kSliceGroups * kRow));
      if (lane < 4) copy16<CA>(pslot + 8 * kRow + 16 * lane, pwd);
      pe += 16;
      pwd += 64;
      if (++ps >= S) enter();
    }
    pslot = next_slot(pslot);
    commit();
    ++groups;
  };
  const long long t0 = clock64();
  if constexpr (kProducer) {
#pragma unroll 1
    for (int q = 0; q < D - 1; ++q) issue();
  }
  //: THE CONSUMER: its next ring slot, its window's passes and its digits' tiles a pass (lane `i` digit `w + 8 i`'s, of
  //: pass 0 and 1), each held a window ahead; the lane's row address and entries' offset in a stage held; with no
  //: producer, its window's entries from global memory a tile ahead
  const auto passes_of = [&](int w) {
    return w < W ? static_cast<int>(in.passes[static_cast<size_t>(w) * kGroups + x.read]) : 0;
  };
  const auto tiles_of = [&](int w, int p) {
    return w < W && lane < 4 ? static_cast<int>(in.tiles[(static_cast<size_t>(w) * kPasses + p) * 256 + 32 * x.read +
                                                         warp + 8 * lane])
                             : 0;
  };
  int np = passes_of(0), np_next = passes_of(1), ct0 = tiles_of(0, 0), ct1 = tiles_of(0, 1);
  int ct0_next = tiles_of(1, 0), ct1_next = tiles_of(1, 1);
  uint32_t cslot = ring;
  const uint32_t row_q = lab::held((lane & 7) * kRow + 16 * ((((lane & 7) >> 1) & 3) ^ (lane >> 3)));
  const uint32_t ent = lab::held(8 * kRow + 16 * (lane & 3));
  const uint8_t* cwd = nullptr;
  uint4 e_next = {};
  //: a tile at ring slot `slot`: its rows and entries from the slot (from global memory where no producer runs)
  const auto step = [&](const uint32_t (&f)[16], uint32_t slot) {
    uint4 e;
    if constexpr (kProducer) {
      e = lds128(slot + ent);
    } else {
      e = e_next;
      cwd += 64;
      e_next = __ldg(reinterpret_cast<const uint4*>(cwd + 16 * (lane & 3)));
    }
    tile(slot + row_q, e, lc.pad, f);
  };
  //: the drain of pass `p`: each warp's 32 slots, two a request (a 128-byte line), unrolled on held addresses, each
  //: word cleared
  const uint32_t cell = lab::held(sm + L::kPanel + ((32 * warp + (lane >> 4)) * kPanelPitch + (lane & 15)) * 4);
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
    if constexpr (!kProducer) {
      cwd = in.words + warp_record(w) * CW;
      e_next = __ldg(reinterpret_cast<const uint4*>(cwd + 16 * (lane & 3)));
    }
#pragma unroll 1
    for (int p = 0; p < np; ++p) {
      const int ct = p == 0 ? ct0 : ct1;
      //: every digit's first tile, one block: its four slots held, the ring's other slots ahead
      if constexpr (kProducer) {
        issue();
        if constexpr (PART != 4) wait_groups<D - 4>();
        __syncwarp();
      }
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (PART != 2) {
          read_image::state_in<1>(f[i], lc, read_image::plane_off(i));
          step(f[i], cslot);
        }
        cslot = next_slot(cslot);
      }
      if constexpr (kProducer) {
#pragma unroll 1
        for (int q = 0; q < 3; ++q) issue();
      }
      //: each digit's rest, a tile waiting for its stage
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = __shfl_sync(0xffffffffu, ct, i) - 1;
#pragma unroll 1
        for (int s = 0; s < n; ++s) {
          if constexpr (kProducer) {
            issue();
            if constexpr (PART != 4) wait_groups<D - 1>();
            __syncwarp();
          }
          if constexpr (PART != 2) step(f[i], cslot);
          cslot = next_slot(cslot);
        }
      }
      __syncthreads();
      if constexpr (PART != 2 && PART != 5) drain(p);
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

//: THE STAGED STREAM, the stream's floor form: each warp's ring of 8 stages of 608 bytes -- a tile's 8 read rows, its
//: 8 entries lane-ready, the 8 tokens of the stage 8 on -- its stages one record stream across windows and passes
//: (`staged.bin`), so the producer reads a stage's tokens from the slot it refills, copies its rows (a pad's as zeros,
//: read from nothing) and its record (its entries, the tokens 8 on) in one burst, and checks no window. `PART` 0, 1
//: the readout with the producer and no row copies, 2 the copies alone
constexpr int kStaged = 8, kStagedStage = 8 * kRow + 64 + 32, kStagedRecord = 96;
constexpr int kStagedPads = kRing + kWarps * kStaged * kStagedStage;
constexpr int kStagedPanel = kStagedPads + 128 + 8 * kPanelPitch * 4;
constexpr int kStagedBytes = kStagedPanel + kSlots * kPanelPitch * 4;

template <int PART, int SET, int SKIP = 0>
__global__ void __launch_bounds__(kThreads, 1) staged(In in, const uint8_t* records, int stream_records,
                                                      long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, r = lane >> 2, c = lane & 3;
  const Box x = box_of();
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kStagedBytes / 16; k += kThreads) {
    const uint32_t v = k < kRing / 16 ? 0x3d80bd80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const read_image::Lane lc = lane_of(sm, warp, lane, kStagedPads);
  //: the lane's constants held in registers: the ring, its 16 bytes of each record (lanes 0-5), its row's source base
  //: and offsets in a stage
  const uint32_t ring = lab::held(sm + kRing + warp * kStaged * kStagedStage);
  const uint8_t* const stream =
      records + (static_cast<size_t>(x.read) * kWarps + warp) * stream_records * kStagedRecord;
  const uint8_t* const rec_lane = held64(stream + kStaged * kStagedRecord + 16 * lane);
  const uint8_t* const rows_c = held64(in.rows + x.slice * kRow + 16 * c);
  const uint32_t row_c = lab::held(r * kRow + 16 * (c ^ ((r >> 1) & 3))), rec_dst = lab::held(8 * kRow + 16 * lane);
  const uint32_t rec_live = lab::held(lane < 6 ? 1u : 0u), tok_at = lab::held(8 * kRow + 64 + 4 * r);
  const auto slot_of = [&](uint32_t s) { return ring + (s & (kStaged - 1)) * kStagedStage; };
  //: stages 0-7's tokens into their slots (records 0-7's last 32 bytes, two lanes a slot)
  if (lane < 16)
    copy16<0>(slot_of(lane >> 1) + 8 * kRow + 64 + 16 * (lane & 1),
              stream + (lane >> 1) * kStagedRecord + 64 + 16 * (lane & 1));
  commit();
  wait_groups<0>();
  __syncwarp();
  //: THE PRODUCER: stage `ps` into its slot -- its tokens from the slot, then its rows and its record -- a group each
  //: a pad's token, masked to a valid row where its copy writes zeros; where it is not copied its address is unused
  constexpr uint32_t kMask = SKIP ? 0xffffffffu : 0x7fffffffu;
  uint32_t ps = 0;
  const uint8_t* prec = rec_lane;
  const auto issue = [&] {
    const uint32_t slot = slot_of(ps), tk = lds32(slot + tok_at);
    if constexpr (PART != 1) copy16_token<SKIP>(slot + row_c, wide(rows_c, tk & kMask, 8 * kRow), tk);
    copy16_if(slot + rec_dst, prec, rec_live);
    commit();
    prec += kStagedRecord;
    ++ps;
  };
  const long long t0 = clock64();
#pragma unroll 1
  for (int q = 0; q < kStaged - 1; ++q) issue();
  //: THE CONSUMER, `stream`'s: its passes and its digits' tiles a pass held a window ahead, the stage `cs` it reads
  const auto passes_of = [&](int w) {
    return w < W ? static_cast<int>(in.passes[static_cast<size_t>(w) * kGroups + x.read]) : 0;
  };
  const auto tiles_of = [&](int w, int p) {
    return w < W && lane < 4 ? static_cast<int>(in.tiles[(static_cast<size_t>(w) * kPasses + p) * 256 + 32 * x.read +
                                                         warp + 8 * lane])
                             : 0;
  };
  int np = passes_of(0), np_next = passes_of(1), ct0 = tiles_of(0, 0), ct1 = tiles_of(0, 1);
  int ct0_next = tiles_of(1, 0), ct1_next = tiles_of(1, 1);
  uint32_t cs = 0;
  const uint32_t row_q = lab::held((lane & 7) * kRow + 16 * ((((lane & 7) >> 1) & 3) ^ (lane >> 3)));
  const uint32_t ent = lab::held(8 * kRow + 16 * (lane & 3));
  const auto step = [&](const uint32_t (&f)[16]) {
    const uint32_t slot = slot_of(cs);
    tile(slot + row_q, lds128(slot + ent), lc.pad, f);
  };
  const uint32_t cell = lab::held(sm + kStagedPanel + ((32 * warp + (lane >> 4)) * kPanelPitch + (lane & 15)) * 4);
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
      //: every digit's first tile, one block: its four slots held, the ring's other slots ahead
      issue();
      wait_groups<kStaged - 4>();
      __syncwarp();
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (PART != 2) {
          read_image::state_in<1>(f[i], lc, read_image::plane_off(i));
          step(f[i]);
        }
        ++cs;
      }
#pragma unroll 1
      for (int q = 0; q < 3; ++q) issue();
      //: each digit's rest, a tile waiting for its stage
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = __shfl_sync(0xffffffffu, ct, i) - 1;
#pragma unroll 1
        for (int s = 0; s < n; ++s) {
          issue();
          wait_groups<kStaged - 1>();
          __syncwarp();
          if constexpr (PART != 2) step(f[i]);
          ++cs;
        }
      }
      __syncthreads();
      if constexpr (PART != 2) drain(p);
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

//: the inputs from a dataset's directory
struct Host {
  std::vector<uint16_t> passes, tiles;
  std::vector<uint8_t> rec, words, rows, staged;
  std::vector<uint32_t> panel;
  int ctas = 0, windows = 0, token_bytes = 0, word_bytes = 0, stream_records = 0;
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "read_stream: %s is missing\n", path.c_str());
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

Host load(const std::string& dir) {
  Host h;
  const std::vector<int> head = as<int>(file(dir + "/head.bin"));
  h.ctas = head[1];
  h.windows = head[2];
  h.token_bytes = head[3];
  h.word_bytes = head[4];
  h.passes = as<uint16_t>(file(dir + "/passes.bin"));
  h.tiles = as<uint16_t>(file(dir + "/tiles.bin"));
  h.panel = as<uint32_t>(file(dir + "/panel.bin"));
  h.rec = file(dir + "/records.bin");
  h.words = file(dir + "/words.bin");
  h.rows = file(dir + "/rows.bin");
  h.staged = file(dir + "/staged.bin");
  h.stream_records = *reinterpret_cast<const int*>(h.staged.data());
  h.staged.erase(h.staged.begin(), h.staged.begin() + 4);
  return h;
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

//: one launch from a cleared panel, its drained panel against the mirror's: the drains of the timed launches summed
//: over their repetitions would not match
template <class Launch>
void check_panel(const lab::Bench& b, const std::string& variant, const In& in, const Host& h, Launch launch) {
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(h.panel.size());
  LAB_CHECK(cudaMemcpy(got.data(), in.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.panel[i];
  b.check(variant, bad == 0, std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                                 " drained panel words exact (each CTA's, summed over the windows)");
}

template <int D, int PART, int SET, int CA = 0>
void run(const lab::Bench& b, const std::string& variant, const In& in, const Host& h) {
  const std::string name = lab::kernel("read_stream::stream", {D, PART, SET, CA});
  b.expect(name, "LDGSTS", ">=", 1);
  b.expect(name, "HMMA.16816.F32.BF16", PART == 2 ? "==" : ">=", PART == 2 ? 0 : 4);
  //: the producer's cursor runs inside the readout's tile loop, a stage a tile; with no producer, the entries load from
  //: global memory inside it
  b.expect(name, "BSSY", ">=", 0);
  if (PART == 3 || PART == 5) b.expect(name, "STEP_GLOBAL", ">=", 0);
  b.exempt(name, "a copy-overlap probe: the readout's tile over streamed rows, its cost the question");
  if (b.expect_only) return;
  const int bytes = Layout<D>::kRec + kWarps * 2 * h.token_bytes;
  LAB_CHECK(cudaFuncSetAttribute(stream<D, PART, SET, CA>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  const lab::Samples s = b.variant([&] { stream<D, PART, SET, CA><<<h.ctas, kThreads, bytes>>>(in, b.times); },
                                   [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (PART) {
    b.unchecked(variant, PART == 5   ? "a bound: the readout with no producer and no drain, no panel formed"
                         : PART == 2 ? "a bound: copies and their waits alone, no panel formed"
                         : PART == 4 ? "a bound: the stream with no waits, its rows racing their copies"
                                     : "a bound: the readout over stale rows, no panel formed");
  } else {
    check_panel(b, variant, in, h, [&] { stream<D, PART, SET, CA><<<h.ctas, kThreads, bytes>>>(in, b.times); });
  }
  b.record(variant, "cycles a window", s);
}

template <int PART, int SET, int SKIP = 0>
void run_staged(const lab::Bench& b, const std::string& variant, const In& in, const Host& h, const uint8_t* records) {
  const std::string name = lab::kernel("read_stream::staged", {PART, SET, SKIP});
  b.expect(name, "LDGSTS", ">=", PART == 1 ? 1 : 2);
  b.expect(name, "HMMA.16816.F32.BF16", PART == 2 ? "==" : ">=", PART == 2 ? 0 : 4);
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a copy-overlap probe: the readout's tile over staged rows, its cost the question");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(staged<PART, SET, SKIP>, cudaFuncAttributeMaxDynamicSharedMemorySize, kStagedBytes));
  const auto launch = [&] {
    staged<PART, SET, SKIP><<<h.ctas, kThreads, kStagedBytes>>>(in, records, h.stream_records, b.times);
  };
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (PART) {
    b.unchecked(variant, PART == 2 ? "a bound: copies and their waits alone, no panel formed"
                                   : "a bound: the readout over stale rows, no panel formed");
  } else {
    check_panel(b, variant, in, h, launch);
  }
  b.record(variant, "cycles a window", s);
}

}  // namespace read_stream

int main(int argc, char** argv) {
  using namespace read_stream;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  //: each dataset's inputs (`read_stream.inputs.py`'s `DATASETS`): nl64k-alt-k4, flagship-alt-k16 in `k16/` and
  //: flagship-cohort-k16 in `c16/`
  Host h[3];
  In in[3] = {};
  const uint8_t* st[3] = {};
  const char* sub[3] = {"", "/k16", "/c16"};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_stream: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    for (int k = 0; k < 3; ++k) {
      h[k] = load(dir + sub[k]);
      if (h[k].ctas > b.sms) {
        std::fprintf(stderr, "read_stream: %d boxes need as many SMs, the device has %d\n", h[k].ctas, b.sms);
        return 1;
      }
      uint32_t* out = nullptr;
      LAB_CHECK(cudaMalloc(&out, h[k].panel.size() * 4));
      in[k] = In{up(h[k].passes), up(h[k].tiles), up(h[k].rec), up(h[k].words), up(h[k].rows), out,
                 h[k].windows, h[k].token_bytes, h[k].word_bytes};
      st[k] = up(h[k].staged);
    }
  }
  run<4, 0, 0>(b, "streamed, ring 4 (the first tiles' block, nothing ahead)", in[0], h[0]);
  run<5, 0, 0>(b, "streamed, ring 5", in[0], h[0]);
  run<6, 0, 0>(b, "streamed, ring 6", in[0], h[0]);
  run<5, 1, 0>(b, "bound: the readout with the producer, no row copies (ring 5)", in[0], h[0]);
  run<5, 2, 0>(b, "bound: the copies alone (ring 5)", in[0], h[0]);
  run<5, 3, 0>(b, "bound: the readout alone, no producer", in[0], h[0]);
  run<5, 5, 0>(b, "bound: the readout alone, no producer, no drain", in[0], h[0]);
  run<5, 4, 0>(b, "bound: the stream with no waits on its stages (ring 5)", in[0], h[0]);
  run<5, 0, 1>(b, "alt-k16: streamed, ring 5", in[1], h[1]);
  run<5, 1, 1>(b, "alt-k16: bound: the readout with the producer, no row copies", in[1], h[1]);
  run<5, 3, 1>(b, "alt-k16: bound: the readout alone, no producer", in[1], h[1]);
  run<5, 5, 1>(b, "alt-k16: bound: the readout alone, no producer, no drain", in[1], h[1]);
  run<5, 0, 2>(b, "cohort-k16: streamed, ring 5", in[2], h[2]);
  run<5, 1, 2>(b, "cohort-k16: bound: the readout with the producer, no row copies", in[2], h[2]);
  run<5, 3, 2>(b, "cohort-k16: bound: the readout alone, no producer", in[2], h[2]);
  run<5, 5, 2>(b, "cohort-k16: bound: the readout alone, no producer, no drain", in[2], h[2]);
  run<5, 0, 0, 1>(b, "streamed, ring 5, L1-allocating copies (.ca)", in[0], h[0]);
  run<5, 0, 1, 1>(b, "alt-k16: streamed, ring 5, L1-allocating copies (.ca)", in[1], h[1]);
  run<5, 0, 2, 1>(b, "cohort-k16: streamed, ring 5, L1-allocating copies (.ca)", in[2], h[2]);
  run_staged<0, 0>(b, "staged, ring 8", in[0], h[0], st[0]);
  run_staged<1, 0>(b, "staged: bound: the readout with the producer, no row copies", in[0], h[0], st[0]);
  run_staged<2, 0>(b, "staged: bound: the copies alone", in[0], h[0], st[0]);
  run_staged<0, 1>(b, "alt-k16: staged, ring 8", in[1], h[1], st[1]);
  run_staged<1, 1>(b, "alt-k16: staged: bound: the readout with the producer, no row copies", in[1], h[1], st[1]);
  run_staged<0, 2>(b, "cohort-k16: staged, ring 8", in[2], h[2], st[2]);
  run_staged<1, 2>(b, "cohort-k16: staged: bound: the readout with the producer, no row copies", in[2], h[2], st[2]);
  run_staged<0, 0, 1>(b, "staged, ring 8, pads not copied", in[0], h[0], st[0]);
  run_staged<0, 1, 1>(b, "alt-k16: staged, ring 8, pads not copied", in[1], h[1], st[1]);
  run_staged<0, 2, 1>(b, "cohort-k16: staged, ring 8, pads not copied", in[2], h[2], st[2]);
}
