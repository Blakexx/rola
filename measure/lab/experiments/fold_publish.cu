// measure/lab/experiments/fold_publish.cu -- QUESTION: the carry's hand-off with one image on sm_86 at a 16-channel
// box: each window the readout reads the image, a named barrier, the fold folds and stores each list's rows into the
// image as its last step lands (its tail), a named barrier -- what does it cost a window against the same readout and
// fold free-running on a static image, does balancing the readout's digits over the warps cut the barriers' skew, and
// what does the 16-channel box cost a token against 32? METHOD: `read_image.cu`'s tokens-on-N readout built for 16
// channels (one m-tile, panel rows of 12 words) on its colored, placed entries; a 32 KB image of 32 digit planes
// ([list][16 channels], 16-byte units swizzled so the fold's stores -- one list, 8 digits -- and the readout's
// `ldmatrix.trans` -- one digit, 8 lists -- are conflict-free, KERNEL_STANDARDS 30); a fold stand-in with the fold's
// shared traffic on the same lists (a warp's 4 lists, each step 8 tokens: an entry, their write rows by
// `ldmatrix.trans`, their values, the coefficients, 4 m16n8k8 MMAs into 64 fp32 state registers), its values zero so
// the state and the check stay exact. Modes: 0 a static image, free-running; 1 the hand-off; 2 the hand-off, the
// readout's digits balanced over the warps; 3 the hand-off's barriers without the stores; 4 a static image, digits
// balanced; 5 the readout alone. see docs/internals/tools/lab.md#fold-publish
#define READ_IMAGE_CHANNELS 16
#include "read_image.cuh"

namespace fold_publish {

using namespace read_image;

//: the fold's values (a slot's 16 channels, 32 bytes, swizzled as the pool's rows; zero) and its heads (a warp's 4
//: lists' first entries, double-buffered), past the readout's layout
constexpr int kValues = kBytes, kFoldHeads = kValues + kSlots * 32, kAll = kFoldHeads + 2 * kWarps * kHeadBytes;

//: a 16-byte unit of digit `d`'s plane for list `l`, chunk `c`: `(2 l + (c ^ s(l))) ^ (d & 7)`
__host__ __device__ constexpr int unit_of(int l, int c, int d) {
  return ((2 * l + (c ^ ((l >> 2) & 1))) ^ (d & 7)) & 63;
}

__device__ __forceinline__ void ldsm_t2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
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

//: A LANE'S CONSTANTS, once a kernel: the shuffle sources (a tile's token r, the outputs' and coefficients' tokens 2t
//: and 2t + 1), the read rows' chunk XOR, the pad base, the state loads' plane units (kb = 0, 1), a values chunk
struct Lane {
  int r, t2;
  uint32_t chunk, pad, unit0, unit1, half, entry;
};

__device__ __forceinline__ Lane lane_of(uint32_t sm, int lane) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3, c = q & 1, l = 8 * (lane >> 4) + r;
  return {static_cast<int>(lab::held(r)), static_cast<int>(lab::held(2 * t)), lab::held(16u * q),
          lab::held(sm + kPads + 4u * g + 32u * t), lab::held(16u * unit_of(l, c, 0)),
          lab::held(16u * unit_of(l + 16, c, 0)), lab::held(c), lab::held(8u * r)};
}

//: A DIGIT'S STATE FRAGMENTS (16 channels x 32 lists, the A operand, 8 registers) from its plane
__device__ __forceinline__ void state_in(uint32_t (&f)[8], uint32_t sm, const Lane& c, int d) {
  const uint32_t plane = sm + kImage + 1024u * d, x = 16u * (d & 7);
  uint32_t a[4], b[4];
  lab::ldsm_t(a, plane + (c.unit0 ^ x));
  lab::ldsm_t(b, plane + (c.unit1 ^ x));
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    f[k] = a[k];
    f[4 + k] = b[k];
  }
}

//: THE READOUT'S TILE, 8 tokens on N: their read rows by one `ldmatrix.x4`, 2 MMAs against the digit's 16 channels,
//: each lane's two tokens' exact pairs added at their panel bases
__device__ __forceinline__ void tile16(uint32_t sm, uint2 e, const Lane& c, const uint32_t (&f)[8]) {
  const uint32_t eb = __shfl_sync(0xffffffffu, e.x, c.r) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.t2), e1 = __shfl_sync(0xffffffffu, e.y, c.t2 + 1);
  uint32_t b[4];
  lab::ldsm(b, (sm + eb) ^ c.chunk);
  float acc[4] = {};
  const uint32_t a0[4] = {f[0], f[1], f[2], f[3]}, a1[4] = {f[4], f[5], f[6], f[7]};
  lab::mma_bf16_f32(acc, a0, b[0], b[1]);
  lab::mma_bf16_f32(acc, a1, b[2], b[3]);
  panel_red(c.pad + (e0 >> 16), s16_exact(acc[0], acc[2], coef(e0)));
  panel_red(c.pad + (e1 >> 16), s16_exact(acc[1], acc[3], coef(e1)));
}

//: THE FOLD STAND-IN'S STEP, 8 tokens of a list: their write rows (32 digits, the A operand by `ldmatrix.trans`)
//: times their coefficients, their values (16 channels, B), 4 m16n8k8 MMAs into the list's state
__device__ __forceinline__ void fold_step(uint32_t sm, uint2 e, const Lane& c, float (&acc)[2][2][4]) {
  const uint32_t ea = __shfl_sync(0xffffffffu, e.x, c.r) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.t2), e1 = __shfl_sync(0xffffffffu, e.y, c.t2 + 1);
  uint32_t a[4];
  lab::ldsm_t(a, (sm + ea) ^ c.chunk);
  const uint32_t cp = prmt(e0, e1, 0x5410u);
#pragma unroll
  for (int k = 0; k < 4; ++k) a[k] = lab::hmul2(a[k], cp);
  const uint32_t s = (ea - kPool) >> 6;
  uint32_t b[2];
  ldsm_t2(b, sm + kValues + 32u * s + 16u * (c.half ^ ((s >> 2) & 1u)));
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < 2; ++n) mma_k8(acc[m][n], a[2 * m], a[2 * m + 1], b[n]);
}

template <int MODE>
__global__ __launch_bounds__(kThreads, 1) void hand(Lists rl, Lists fl, const uint32_t* zeros, long long* times,
                                                     int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kStore = MODE == 1 || MODE == 2, kBarriers = MODE >= 1 && MODE <= 3, kFold = MODE != 5;
  constexpr bool kBalanced = MODE == 2 || MODE == 4;
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = rl.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kAll / 16; k += kThreads) {
    const uint32_t v = k < kPool / 16 ? 0x3d80bd80u : k < kPads / 16 ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const Lane c = lane_of(sm, lane);
  //: THE STATE, a warp's lists w + 8 J by their digits' m-tiles and channels' n-tiles: -0.0625 even, +0.0625 odd
  const uint32_t zero = zeros[0];
  float acc[4][2][2][4];
#pragma unroll
  for (int k = 0; k < 64; ++k) (&acc[0][0][0][0])[k] = __uint_as_float(zero | (k & 1 ? 0x3d800000u : 0xbd800000u));
  //: a lane's store address for list J (n-tile 0, digit row 0): its g's plane and unit, its t's word
  const int g = lane >> 2, t = lane & 3;
  uint32_t put[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) put[j] = lab::held(sm + kImage + 1024u * g + 16u * unit_of(warp + 8 * j, 0, g) + 4u * t);
  float sink = 0.0f;
  //: both heads a window ahead, a 16-byte chunk a lane each
  const uint8_t* rsrc = rl.heads + (static_cast<int64_t>(box) * kWarps + warp) * kHeadBytes + 16 * lane;
  const uint8_t* fsrc = fl.heads + (static_cast<int64_t>(box) * kWarps + warp) * kHeadBytes + 16 * lane;
  const int64_t src_step = static_cast<int64_t>(boxes) * kWarps * kHeadBytes;
  const uint32_t hb_step = kWarps * kHeadBytes;
  const uint32_t rto = lab::held(sm + kHeads + warp * kHeadBytes + 16 * lane);
  const uint32_t fto = lab::held(sm + kFoldHeads + warp * kHeadBytes + 16 * lane);
  const uint32_t rhe = lab::held(sm + kHeads + warp * kHeadBytes + c.entry);
  const uint32_t fhe = lab::held(sm + kFoldHeads + warp * kHeadBytes + c.entry);
  lab::cp_async16(rto, rsrc);
  lab::cp_async16(fto, fsrc);
  lab::cp_commit();
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < rl.windows; ++w) {
    const uint32_t flip = (w & 1) * hb_step;
    __syncwarp();
    rsrc += src_step;
    fsrc += src_step;
    lab::cp_async16(rto + (hb_step - flip), rsrc);
    lab::cp_async16(fto + (hb_step - flip), fsrc);
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    //: an entry past the staged ones, from its list in global memory (zero past its count): 0.3% of lists at alt-k4
    const auto listed = [&](const Lists& ls, int d, int e) {
      return ls.entries[(static_cast<int64_t>(w) * ls.width + box * kDigits + d) * kListCap + e];
    };
    //: THE READOUT, the warp's 4 digits (fixed w + 8 i, or the balanced deal's, read from the tag)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint2 e = lds64(rhe + flip + kHead * 8 * i);
      const int steps = static_cast<int>((e.x >> 16) & 0xffu);
      const int d = kBalanced ? __shfl_sync(0xffffffffu, static_cast<int>((e.x >> 25) & 31u), 0) : warp + 8 * i;
      uint32_t f[8];
      state_in(f, sm, c, d);
      tile16(sm, e, c, f);
      for (int k = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k > 0; --k)
        tile16(sm, lds64(rhe + flip + kHead * 8 * i + 64), c, f);
      for (int k = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k > 0; --k)
        do tile16(sm, listed(rl, d, 8 * s + (lane & 7)), c, f);
        while (__any_sync(0xffffffffu, ++s < steps));
    }
    if constexpr (kBarriers) lab::bar_sync(1, kThreads);
    //: THE FOLD, the warp's 4 lists, each list's rows stored into the image as its last step lands
    if constexpr (kFold) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint2 e = lds64(fhe + flip + kHead * 8 * j);
        const int steps = static_cast<int>((e.x >> 16) & 0xffu);
        fold_step(sm, e, c, acc[j]);
        for (int k = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k > 0; --k)
          fold_step(sm, lds64(fhe + flip + kHead * 8 * j + 64), c, acc[j]);
        for (int k = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k > 0; --k)
          do fold_step(sm, listed(fl, warp + 8 * j, 8 * s + (lane & 7)), c, acc[j]);
          while (__any_sync(0xffffffffu, ++s < steps));
        if constexpr (kStore) {
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
              for (int hi = 0; hi < 2; ++hi)
                asm volatile("st.shared.u32 [%0], %1;" ::"r"((put[j] ^ (16u * n)) + 1024u * (16 * m + 8 * hi)),
                             "r"(pack_bf16(acc[j][m][n][2 * hi], acc[j][m][n][2 * hi + 1]))
                             : "memory");
        }
      }
    }
    if constexpr (kBarriers) lab::bar_sync(1, kThreads);
  }
  clock.stop(times);
  lab::cp_wait_all();
#pragma unroll
  for (int k = 0; k < 64; ++k) sink += (&acc[0][0][0][0])[k];
  lab::sink(sinks, lab::bits(sink));
  __syncthreads();
  constexpr int kRowChunks = kWords / 4;
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * kRowChunks; k += kThreads)
    reinterpret_cast<uint4*>(rl.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kRowChunks + k] =
        chunks[(kPanel + (k / kRowChunks) * kPanelPitch * 4) / 16 + k % kRowChunks];
}

//: THE BUDGET (KERNEL_STANDARDS sections 29-30) of the hand-off (modes 1-2), per MMA at alt-k4: a warp-window's ~5.6
//: readout tiles (2 MMAs each) and ~5.6 fold steps (4 m16n8k8 each), 33.8 MMAs. A tile: an entry and a read-row load,
//: 3 shuffles, 4 FFMAs, ~8 integer (a mask-XOR, 2 coefficients' shifts, 2 bases, 2 packs, a filler), 2 adds. A step:
//: an entry, the write rows' and values' loads, 3 shuffles, a PRMT, 4 HMUL2s, ~7 integer (the rows' XOR, the values'
//: address). A digit: 2 state loads, its plane (~3), its tag and 2 votes; a list: its tag and votes, its 8 packs and 8
//: stores. A window: 2 heads' copies and their 6 never-executed `LDS`, ~20 integer, 2 barriers, ~10 control.
//: Wavefronts: a tile 8 (entry 2, rows 4, adds 2), a step 8 (entry 2, rows 4, values 2), a digit's state 8, a list's
//: stores 8, the heads' copies 16, ~0.6 a shuffle: 190 a warp-window.
struct Budget {
  double shared_load, shuffle, float_, pack, integer, select, control, shared_store, atomic, global, wavefronts;
};

constexpr double kMmas = 33.8;
constexpr Budget kBudget = {42 / kMmas, 34 / kMmas, 45 / kMmas, 38 / kMmas, 140 / kMmas, 20 / kMmas,
                            28 / kMmas, 32 / kMmas, 11.5 / kMmas, 2.5 / kMmas, 195 / kMmas};

template <int MODE>
void variant(const lab::Bench& b, const Lists& rl, const Lists& fl, const uint32_t* zeros, const Reference& ref) {
  const std::string name = lab::kernel("fold_publish::hand", {MODE});
  b.expect(name, "HMMA.16816.F32.BF16", ">=", 2);
  b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (MODE == 1 || MODE == 2) {
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", kBudget.shared_load);
    b.budget(name, "shared store", kBudget.shared_store);
    b.budget(name, "shared atomic", kBudget.atomic);
    b.budget(name, "shuffle", kBudget.shuffle);
    b.budget(name, "pack", kBudget.pack);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "float", kBudget.float_);
    b.budget(name, "select/compare", kBudget.select);
    b.budget(name, "integer/address", kBudget.integer);
    b.budget(name, "control", kBudget.control);
    b.budget(name, "global", kBudget.global);
    b.budget(name, "other", 0.0);
    b.budget(name, "wavefronts", kBudget.wavefronts);
  } else {
    b.exempt(name, MODE == 3   ? "an isolation: the hand-off's barriers without its stores"
                   : MODE == 5 ? "the readout alone, the 16-channel box's per-token reference"
                               : "a baseline: the same readout and fold on a static image");
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(hand<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAll));
  const auto launch = [&] { hand<MODE><<<b.sms, kThreads, kAll>>>(rl, fl, zeros, b.times, b.sinks); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps) / rl.windows; });
  const char* labels[6] = {"a static image, free-running", "the hand-off: readout, barrier, fold storing, barrier",
                           "the hand-off, the readout's digits balanced", "the hand-off's barriers alone (no stores)",
                           "a static image, digits balanced, free-running", "the readout alone, free-running"};
  b.record(labels[MODE], "cycles a CTA-window (16 channels)", s);
  LAB_CHECK(cudaMemset(rl.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  const Verdict v = verify<1, 1, 0>(ref, rl);
  b.check(labels[MODE], v.bad == 0 && v.words > 0, v.detail);
}

}  // namespace fold_publish

int main(int argc, char** argv) {
  using namespace fold_publish;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists rl{}, bl{}, unused{};
  HostLists host;
  uint32_t* zeros = nullptr;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "fold_publish: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    load(dir, b.sms, host, unused, rl);
    bl = rl;
    const size_t heads = static_cast<size_t>(host.windows) * (host.width / kDigits) * kWarps * kHeadBytes;
    bl.heads = upload<uint8_t>(dir + "/heads_bal_n.bin", heads, heads / host.windows);
    LAB_CHECK(cudaMalloc(&zeros, 64));
    LAB_CHECK(cudaMemset(zeros, 0, 64));
    std::printf("{\"lab\": \"note\", \"free_floor_n\": %.1f, \"bytes\": %d}\n", floor_of(host, 1, false, true), kAll);
  }
  const Reference ref{&host, b.sms};
  variant<0>(b, rl, rl, zeros, ref);
  variant<1>(b, rl, rl, zeros, ref);
  variant<2>(b, bl, rl, zeros, ref);
  variant<3>(b, rl, rl, zeros, ref);
  variant<4>(b, bl, rl, zeros, ref);
  variant<5>(b, rl, rl, zeros, ref);
}
