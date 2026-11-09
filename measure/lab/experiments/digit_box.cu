// measure/lab/experiments/digit_box.cu -- QUESTION: on sm_86, which box keeps the carry's hand-off cheapest -- 32
// lists x 16 digits or 16 lists x 32 digits with the whole image resident (32 KB), or 32 x 32 with its image in two
// parts -- at nl64k-alt-k4 and at a box whose lists average under one write a window (as at ~1M leaves)? METHOD:
// `read_image.cu`'s 32-channel tokens-on-N readout (16 lists: one k-tile, 32-byte read rows) and a fold stand-in with
// the fold's shared traffic over the box's lists (a step 8 tokens: an entry, their write rows over the box's digits by
// `ldmatrix.trans`, their 32 channels' values from a zero row, the coefficients, m16n8k8 MMAs), on the lists of
// `digit_box.inputs.py`'s two cells (`alt/`, and `thin/`: each entry kept with probability 1/16). Both skip a digit
// with no readers and a list with no writes. The image: a plane a digit ([list][32 channels]), 16-byte units swizzled
// `(4 list + (c ^ s(list))) ^ (digit & 7)` so the fold's stores and the readout's `ldmatrix.trans` are conflict-free
// (KERNEL_STANDARDS 30). The resident hand-off: the readout, a named barrier, the fold storing each list it folded as
// its last step lands, a named barrier. The two-part hand-off (32 x 32): the fold, then for each half of the digits a
// barrier, the half's stores, a barrier, its readout. Modes: 0 a static image, free-running; 1 the resident hand-off;
// 2 the readout alone; 3 the fold alone; 4 the two-part hand-off. `digit_box32.cu` and `list_box.cu` build this file
// for 32 x 32 and 16 x 32. see docs/internals/tools/lab.md#digit-box
#ifndef READ_IMAGE_DIGITS
#define READ_IMAGE_DIGITS 16
#endif
#include "read_image.cuh"

namespace digit_box {

using namespace read_image;

//: the box's shape: its list boxes a 32-list group (2 at 16 lists), the readout's digits a warp and k-tiles, the
//: fold's lists a warp and m-tiles
constexpr int kListBoxes = 32 / kLists, kReadDigits = kDigits / kWarps, kKb = kLists / 16;
constexpr int kFoldLists = kLists / kWarps, kMTiles = kDigits / 16;
//: the fold's write rows (64 bytes over 32 digits, the fold inputs' layout from 32 KB): the read pool where its rows
//: are 64 bytes, else their own region; a zero row its values; its heads (a warp's 4 lists, double-buffered)
constexpr int kWritePool = kLists == 32 ? kPool : kBytes;
constexpr int kZeroRow = kLists == 32 ? kBytes : kWritePool + kSlots * 64;
constexpr int kFoldHeadBytes = 4 * kHead * 8, kFoldHeads = kZeroRow + 128;
constexpr int kAll = kFoldHeads + 2 * kWarps * kFoldHeadBytes;
constexpr uint32_t kFoldDelta = kWritePool - 32768;

//: a 16-byte unit of digit `d`'s plane for list `l`, chunk `c`
__host__ __device__ constexpr int unit_of(int l, int c, int d) {
  return ((4 * l + (c ^ ((l >> 1) & 3))) ^ (d & 7)) & (kPlaneBytes / 16 - 1);
}

__device__ __forceinline__ uint32_t plane_of(uint32_t sm, int d) {
  return sm + kImage + static_cast<uint32_t>(d % kPlanes) * kPlaneBytes;
}

__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void ldsm_t2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void mma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

//: A DIGIT'S STATE FRAGMENTS (32 channels x the box's lists, the A operand), `(m * kKb + kb) * 4 + i`: `u` the lane's
//: units' bytes (m-tile, k-tile) at digit 0, XORed with the digit's
__device__ __forceinline__ void state_in(uint32_t (&f)[8 * kKb], uint32_t sm, const uint32_t (&u)[2 * kKb], int d) {
  const uint32_t plane = plane_of(sm, d), x = 16u * (d & 7);
#pragma unroll
  for (int k = 0; k < 2 * kKb; ++k) {
    uint32_t a[4];
    lab::ldsm_t(a, plane + (u[k] ^ x));
#pragma unroll
    for (int i = 0; i < 4; ++i) f[k * 4 + i] = a[i];
  }
}

//: THE READOUT'S TILE, 8 tokens on N against the digit's state: `read_image`'s at 32 lists; at 16, their read rows
//: (32 bytes) by one `ldmatrix.x2` and one MMA an m-tile
template <int KB = kKb>
__device__ __forceinline__ void tile(uint32_t sm, uint2 e, const Lane& c, uint32_t half, const uint32_t (&f)[8 * KB],
                                     float& sink) {
  if constexpr (KB == 2) {
    tile_n<1>(sm, e, c, f, sink);
  } else {
    const uint32_t eb = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
    const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.src_g), e1 = __shfl_sync(0xffffffffu, e.y, c.src_h);
    uint32_t b[2];
    ldsm2(b, (sm + eb) ^ half);
    float acc[2][4] = {};
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const uint32_t a[4] = {f[m * 4], f[m * 4 + 1], f[m * 4 + 2], f[m * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[0], b[1]);
    }
    const uint32_t b0 = c.pad + (e0 >> 16), b1 = c.pad + (e1 >> 16);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      panel_red(b0 + 32 * m, s16_exact(acc[m][0], acc[m][2], coef(e0)));
      panel_red(b1 + 32 * m, s16_exact(acc[m][1], acc[m][3], coef(e1)));
    }
  }
}

//: THE FOLD STAND-IN'S STEP, 8 tokens of a list: their write rows over the box's digits (A, `ldmatrix.trans`) times
//: their coefficients, their 32 channels' values from the zero row (B; the address computed as for a real row), and
//: m16n8k8 MMAs into the list's state
__device__ __forceinline__ void fold_step(uint32_t sm, uint2 e, const Lane& c, uint32_t half, uint32_t zmask,
                                          float (&acc)[kMTiles][4][4]) {
  const uint32_t ea = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.src_g), e1 = __shfl_sync(0xffffffffu, e.y, c.src_h);
  const uint32_t row = sm + kFoldDelta + ea;
  uint32_t a[2 * kMTiles];
  if constexpr (kMTiles == 1) {
    ldsm_t2(*reinterpret_cast<uint32_t(*)[2]>(a), row ^ half);
  } else {
    lab::ldsm_t(*reinterpret_cast<uint32_t(*)[4]>(a), row ^ c.chunk);
  }
  const uint32_t cp = prmt(e0, e1, 0x5410u);
#pragma unroll
  for (int k = 0; k < 2 * kMTiles; ++k) a[k] = lab::hmul2(a[k], cp);
  const uint32_t s = (ea - 32768u) >> 6;
  uint32_t b[4];
  lab::ldsm_t(b, sm + kZeroRow + ((64u * s + (c.chunk ^ (16u * ((s >> 1) & 3u)))) & zmask));
#pragma unroll
  for (int m = 0; m < kMTiles; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) mma_k8(acc[m][n], a[2 * m], a[2 * m + 1], b[n]);
}

template <int MODE>
__global__ __launch_bounds__(kThreads, 1) void box(Lists rl, Lists fl, const uint32_t* zeros, long long* times,
                                                    int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kRead = MODE != 3, kFold = MODE != 2, kResident = MODE == 1, kParts = MODE == 4;
  static_assert(!kResident || kPlanes * kPlaneBytes >= kDigits * kPlaneBytes, "the resident image holds every digit");
  static_assert(!kParts || (kDigits == 32 && kLists == 32), "the two-part hand-off is the 32 x 32 box's");
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = rl.width / kDigits * kListBoxes, bx = static_cast<int>(blockIdx.x) % boxes;
  const int rb = bx / kListBoxes, lh = bx % kListBoxes, fg = rb * kDigits / 32;
  const uint32_t sm = lab::smem(smem);
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  //: the image's even channels -0.0625 and odd +0.0625, the read and write pools' rows +0.0625, the rest zero
  for (int k = static_cast<int>(threadIdx.x); k < kAll / 16; k += kThreads) {
    const bool pool = k < kPads / 16 || (kWritePool != kPool && k >= kWritePool / 16 && k < kZeroRow / 16);
    const uint32_t v = k < kPool / 16 ? 0x3d80bd80u : pool ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const Lane c = lane_of<1>(sm, warp, lane);
  const uint32_t half = lab::held(16u * ((lane >> 3) & 1)), zmask = lab::held(zeros[1]);
  uint32_t u[2 * kKb];
#pragma unroll
  for (int k = 0; k < 2 * kKb; ++k)
    u[k] = lab::held(16u * unit_of(8 * (lane >> 4) + (lane & 7) + 16 * (k % kKb), 2 * (k / kKb) + ((lane >> 3) & 1),
                                   0));
  //: THE STATE, a warp's lists w + 8 J by their digits' m-tiles and the channels' n-tiles: -0.0625 even, +0.0625 odd
  const uint32_t zero = zeros[0];
  float acc[kFoldLists][kMTiles][4][4];
#pragma unroll
  for (int k = 0; k < 64 * kFoldLists * kMTiles / 4; ++k)
    (&acc[0][0][0][0])[k] = __uint_as_float(zero | (k & 1 ? 0x3d800000u : 0xbd800000u));
  //: a lane's store address for list J (n-tile 0, row g of digit block 0): its g's plane and unit, its t's word
  const int g = lane >> 2, t = lane & 3;
  uint32_t put[kFoldLists];
#pragma unroll
  for (int j = 0; j < kFoldLists; ++j) put[j] = lab::held(plane_of(sm, g) + 16u * unit_of(warp + 8 * j, 0, g) + 4u * t);
  float sink = 0.0f;
  //: both heads a window ahead: the readout's (8 or 16 bytes a lane), the fold's (16)
  const uint8_t* rsrc = rl.heads + (static_cast<int64_t>(rb) * kWarps + warp) * kHeadBytes + kHeadBytes / 32 * lane;
  const uint8_t* fsrc = fl.heads + (static_cast<int64_t>(fg) * kWarps + warp) * kFoldHeadBytes + 16 * lane;
  const int64_t rstep = static_cast<int64_t>(rl.width / kDigits) * kWarps * kHeadBytes;
  const int64_t fstep = static_cast<int64_t>(fl.width / 32) * kWarps * kFoldHeadBytes;
  const uint32_t rto = lab::held(sm + kHeads + warp * kHeadBytes + (kHeadBytes / 32) * lane);
  const uint32_t fto = lab::held(sm + kFoldHeads + warp * kFoldHeadBytes + 16 * lane);
  const uint32_t rhe = lab::held(sm + kHeads + warp * kHeadBytes + c.entry);
  const uint32_t fhe = lab::held(sm + kFoldHeads + warp * kFoldHeadBytes + c.entry + kHead * 8 * kFoldLists * lh);
  const auto heads_in = [&](uint32_t rflip, uint32_t fflip) {
    if constexpr (kHeadBytes == 256) {
      cp_async8(rto + rflip, rsrc);
    } else {
      lab::cp_async16(rto + rflip, rsrc);
    }
    lab::cp_async16(fto + fflip, fsrc);
  };
  heads_in(0, 0);
  lab::cp_commit();
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < rl.windows; ++w) {
    const uint32_t rflip = (w & 1) * kWarps * kHeadBytes, fflip = (w & 1) * kWarps * kFoldHeadBytes;
    __syncwarp();
    rsrc += rstep;
    fsrc += fstep;
    heads_in(kWarps * kHeadBytes - rflip, kWarps * kFoldHeadBytes - fflip);
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    //: an entry past the staged ones, from its list in global memory (zero past its count): 0.3% of lists at alt-k4
    const auto listed = [&](const Lists& ls, int b, int digits, int d, int e) {
      return ls.entries[(static_cast<int64_t>(w) * ls.width + b * digits + d) * kListCap + e];
    };
    //: THE READOUT of the warp's digits w + 8 i (i in [i0, i1)), a digit with no readers skipped
    const auto readout = [&](int i0, int i1) {
#pragma unroll
      for (int i = i0; i < i1; ++i) {
        const uint2 e = lds64(rhe + rflip + kHead * 8 * i);
        const int steps = static_cast<int>((e.x >> 16) & 0xffu), d = warp + 8 * i;
        for (int k = __any_sync(0xffffffffu, steps > 0) ? 1 : 0; k > 0; --k) {
          uint32_t f[8 * kKb];
          state_in(f, sm, u, d);
          tile(sm, e, c, half, f, sink);
          for (int k2 = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k2 > 0; --k2)
            tile(sm, lds64(rhe + rflip + kHead * 8 * i + 64), c, half, f, sink);
          for (int k2 = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k2 > 0; --k2)
            do tile(sm, listed(rl, rb, kDigits, d, 8 * s + (lane & 7)), c, half, f, sink);
            while (__any_sync(0xffffffffu, ++s < steps));
        }
      }
    };
    //: list J's rows of digit half `m` into the image: 8 bf16 pairs a lane
    const auto store = [&](int j, int m) {
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
          asm volatile("st.shared.u32 [%0], %1;"
                       ::"r"((put[j] ^ (16u * n)) + kPlaneBytes * ((16 * m + 8 * hi) % kPlanes)),
                       "r"(pack_bf16(acc[j][m % kMTiles][n][2 * hi], acc[j][m % kMTiles][n][2 * hi + 1]))
                       : "memory");
    };
    //: THE FOLD of the warp's lists w + 8 J, a list with no writes skipped; the resident hand-off stores each list
    //: folded as its last step lands
    const auto fold = [&] {
#pragma unroll
      for (int j = 0; j < kFoldLists; ++j) {
        const uint2 e = lds64(fhe + fflip + kHead * 8 * j);
        const int steps = static_cast<int>((e.x >> 16) & 0xffu);
        for (int k = __any_sync(0xffffffffu, steps > 0) ? 1 : 0; k > 0; --k) {
          fold_step(sm, e, c, half, zmask, acc[j]);
          for (int k2 = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k2 > 0; --k2)
            fold_step(sm, lds64(fhe + fflip + kHead * 8 * j + 64), c, half, zmask, acc[j]);
          for (int k2 = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k2 > 0; --k2)
            do fold_step(sm, listed(fl, fg, 32, warp + 8 * (kFoldLists * lh + j), 8 * s + (lane & 7)), c, half,
                         zmask, acc[j]);
            while (__any_sync(0xffffffffu, ++s < steps));
          if constexpr (kResident) {
#pragma unroll
            for (int m = 0; m < kMTiles; ++m) store(j, m);
          }
        }
      }
    };
    if constexpr (kParts) {
      fold();
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        lab::bar_sync(1, kThreads);
#pragma unroll
        for (int j = 0; j < kFoldLists; ++j) store(j, p);
        lab::bar_sync(1, kThreads);
        readout(2 * p, 2 * p + 2);
      }
    } else {
      if constexpr (kRead) readout(0, kReadDigits);
      if constexpr (kResident) lab::bar_sync(1, kThreads);
      if constexpr (kFold) fold();
      if constexpr (kResident) lab::bar_sync(1, kThreads);
    }
  }
  clock.stop(times);
  lab::cp_wait_all();
#pragma unroll
  for (int k = 0; k < 64 * kFoldLists * kMTiles / 4; ++k) sink += (&acc[0][0][0][0])[k];
  lab::sink(sinks, lab::bits(sink));
  __syncthreads();
  constexpr int kRowChunks = kWords / 4;
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * kRowChunks; k += kThreads)
    reinterpret_cast<uint4*>(rl.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kRowChunks + k] =
        chunks[(kPanel + (k / kRowChunks) * kPanelPitch * 4) / 16 + k % kRowChunks];
}

template <int MODE>
void variant(const lab::Bench& b, const Lists& rl, const Lists& fl, const uint32_t* zeros, const HostLists& host,
             const char* cell) {
  const std::string name = lab::kernel("digit_box::box", {MODE});
  b.expect(name, "HMMA", ">=", 4);
  b.expect(name, "STEP_GLOBAL", ">=", 0);
  //: one kernel runs both cells, whose amortizations differ, so a budget per MMA cannot hold for both: the forms'
  //: counts are recorded, the chosen form's budget stated at its design line
  b.exempt(name, "a form comparison across box shapes and two cells (counts recorded, not gated)");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(box<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAll));
  const auto launch = [&] { box<MODE><<<b.sms, kThreads, kAll>>>(rl, fl, zeros, b.times, b.sinks); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps) / rl.windows; });
  const char* labels[5] = {"a static image, free-running", "the resident hand-off", "the readout alone",
                           "the fold alone", "the two-part hand-off"};
  const std::string label = std::string(labels[MODE]) + " / " + std::to_string(kLists) + " lists x " +
                            std::to_string(kDigits) + " digits / " + cell;
  b.record(label, "cycles a CTA-window", s);
  if constexpr (MODE == 3) {
    b.unchecked(label, "an isolation: the fold's state is not read out");
  } else {
    LAB_CHECK(cudaMemset(rl.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    const Reference ref{&host, b.sms};
    const Verdict v = verify<1, 1, 0>(ref, rl);
    b.check(label, v.bad == 0 && v.words > 0, v.detail);
  }
}

//: a cell's inputs: the readout's lists from `dir`, the fold's (32 digits a box) from `dir/fold`
struct Cell {
  HostLists host;
  Lists rl, fl;
};

inline Cell load_cell(const std::string& dir, int sms) {
  Cell c;
  Lists unused{};
  load(dir, sms, c.host, unused, c.rl);
  const HostLists fh = read_lists(dir + "/fold/read.bin");
  const size_t entries = static_cast<size_t>(fh.windows) * fh.width * kListCap * 8;
  const size_t heads = static_cast<size_t>(fh.windows) * (fh.width / 32) * kWarps * kFoldHeadBytes;
  c.fl = c.rl;
  c.fl.width = fh.width;
  c.fl.entries = upload<uint2>(dir + "/fold/entries_n.bin", entries);
  c.fl.heads = upload<uint8_t>(dir + "/fold/heads_n.bin", heads, heads / fh.windows);
  return c;
}

}  // namespace digit_box

int main(int argc, char** argv) {
  using namespace digit_box;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Cell cells[2];
  uint32_t* zeros = nullptr;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "digit_box: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    cells[0] = load_cell(dir + "/alt", b.sms);
    cells[1] = load_cell(dir + "/thin", b.sms);
    LAB_CHECK(cudaMalloc(&zeros, 64));
    LAB_CHECK(cudaMemset(zeros, 0, 64));
    std::printf("{\"lab\": \"note\", \"bytes\": %d}\n", kAll);
  }
  const char* names[2] = {"alt-k4", "thinned 1/16"};
  for (int k = 0; k < 2; ++k) {
    const Cell& c = cells[k];
    variant<0>(b, c.rl, c.fl, zeros, c.host, names[k]);
#if READ_IMAGE_DIGITS == 32 && READ_IMAGE_LISTS == 32
    variant<4>(b, c.rl, c.fl, zeros, c.host, names[k]);
#else
    variant<1>(b, c.rl, c.fl, zeros, c.host, names[k]);
#endif
    variant<2>(b, c.rl, c.fl, zeros, c.host, names[k]);
    variant<3>(b, c.rl, c.fl, zeros, c.host, names[k]);
  }
}
