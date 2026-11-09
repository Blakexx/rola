// measure/lab/experiments/read_alone.cu -- QUESTION: what does the value-carve readout cost a warp-window ALONE --
// production's `read_chunk` over a read chunk drawn as a cell's is, nothing else in the window -- against its tensor and
// shared-memory floors and against the composed kernel's readout phase? METHOD: one CTA an SM, a box's plan tenants; at
// the 32 x 32 x 32 box each window a 216-slot union whose 32 digits hold ~8 entries each (slot `s` in digit `d`'s list
// with probability 10 / 256; at alt-k16's density 18 / 256 and a DENSE 40 / 256 past it); at the 32 x 16 x 64 box a
// read chunk as nl64k-alt-k4's (120 slots, each a digit's and any other's at 1 / 256) or flagship-alt-k16's (192, 9 /
// 256); the image in halves as sm_86 runs it, both halves read with a rendezvous between.
// see docs/lab/read_alone.md
#include <cstdio>
#include <vector>

#include "carry/ampere/carry_kernel.cuh"
#include "common/arm.cuh"
#include "lab.cuh"

namespace read_alone {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
constexpr int kWindows = 64, kLiveOf256 = 10, kK16Of256 = 18, kDenseOf256 = 40;

//: THE BOXES: 0 the 32 x 32 x 32 box (two token groups by four channel groups), 1 the 32 x 16 x 64 box (one by eight)
template <int BOX>
struct Box;

template <>
struct Box<0> {
  using A = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
  using P = rc::Plan<A, lab::kLabBuild>;
};

template <>
struct Box<1> {
  using A = rola::Arm<0, 16, 1, 32, 1, 64, 0>;
  using P = rc::Plan<A, lab::kLabBuild>;
};

//: a variant's draw: its union's slots, and at the 32 x 16 x 64 box each slot's other digits a 256th
template <int BOX, int MODE>
constexpr int slots_of() {
  return BOX == 0 ? 216 : MODE == 13 ? 192 : 120;
}

template <int BOX, int MODE>
constexpr int of256() {
  if constexpr (BOX == 1) return MODE == 13 ? 9 : 1;
  return MODE == 12 ? kDenseOf256 : MODE == 13 ? kK16Of256 : kLiveOf256;
}

__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

//: whether slot `s` is in digit `d`'s list in window `w` of block `b` (both sides draw it: the kernel its runs, the
//: host the floors): RANDOM slots, or (`colored`) the coloring's best case -- `n` = popcount of 16 drawn bits entries,
//: entry `k` on slot `8 (27 k / n) + k mod 8`, so any 8 consecutive entries meet 8 residues
__host__ __device__ __forceinline__ uint32_t key(int b, int w, int d, int s) {
  return mix(static_cast<uint32_t>(b) * 7919u + static_cast<uint32_t>(w) * 104729u
             + static_cast<uint32_t>(d) * 31u + static_cast<uint32_t>(s) * 977u);
}

__host__ __device__ __forceinline__ int popc16(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __popc(x & 0xffffu);
#else
  return __builtin_popcount(x & 0xffffu);
#endif
}

//: at the 32 x 32 x 32 box each (digit, slot) at `of256`; at the 32 x 16 x 64 box a union's slot as a cell's: one digit
//: its own (every slot read) and each other at `of256`
template <int BOX>
__host__ __device__ __forceinline__ bool live(int b, int w, int d, int s, bool colored, int of256) {
  if constexpr (BOX == 1) {
    constexpr int kDigits = Box<1>::P::kDigits;
    if (d >= kDigits) return false;
    return static_cast<int>((key(b, w, -1, s) >> 8) % kDigits) == d
           || (key(b, w, d, s) & 0xffu) < static_cast<uint32_t>(of256);
  }
  if (!colored) return (key(b, w, d, s) & 0xffu) < static_cast<uint32_t>(of256);
  const int n = popc16(key(b, w, d, -1)), u = s / 8, k0 = s % 8;
  for (int k = k0; k < n; k += 8)
    if (27 * k / n == u) return true;
  return false;
}

//: MODE 0: an f16 panel, fp16 accumulate (sm_86's production form); 1 an f16 panel, fp32 accumulate (the A100's);
//: 2 an fp32 panel, fp32 accumulate; 3-5 the same on colored slots
template <int BOX, int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void read(long long* t, int* out) {
  //: MODE 10 and 11 the sections, in sm_86's form on random slots; 12 and 13 sm_86's form on dense lists and at
  //: alt-k16's density
  using P = typename Box<BOX>::P;
  constexpr bool kF32 = MODE < 10 && MODE % 3 == 2, kF16 = MODE >= 10 || MODE % 3 == 0,
                 kColored = MODE < 10 && MODE % 6 >= 3, kEmpty = MODE >= 6 && MODE < 10,
                 kFirst = MODE != 10, kSecond = MODE != 10 && MODE != 11;
  constexpr int kSlots = slots_of<BOX, MODE>(), kOf256 = of256<BOX, MODE>(), kHalf = P::kHalfPlanes;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  rc::CarryParams p{};
  p.tables = rc::tables_of<P::kWindow>(1, P::kSmemBytes);
  p.reduce = kF32 ? rc::kReduceF32 : rc::kReduceF16;
  const int split = rc::group_split(kSlots);
  const ops::SmemAddr image = ops::smem_addr(smem + P::kAreaAt);
  const ops::SmemAddr rows = image + static_cast<uint32_t>(P::kImageRing);
  const ops::SmemAddr panel = rows + static_cast<uint32_t>(rc::read_panel<P>(kSlots)),
                      dens = rows + static_cast<uint32_t>(rc::read_dens<P>(kSlots, p.reduce)),
                      ents = rows + static_cast<uint32_t>(rc::read_ents<P>(kSlots, p.reduce));
  //: the image half, the union's rows and the den image small f16 values; each digit's exponent and den factor; every
  //: token's scale byte
  for (int i = tid; i < (P::kImageRing + kSlots * P::kReadRow) / 4; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kAreaAt)[i] =
        0x24002400u ^ (mix(static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kMaxExtent * rc::kMaxExtent / 2; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kDenImage)[i] =
        0x24002400u ^ (mix(~static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kWarps * rc::kMaxExtent; i += rc::kThreads)
    reinterpret_cast<int*>(smem + P::kExps)[i] = 4;
  if (tid < rc::kMaxExtent) reinterpret_cast<float*>(smem + P::kDenFactor)[tid] = 1.0f;
  for (int i = tid; i < 2 * P::kWindow; i += rc::kThreads) (smem + p.tables.scale_at)[i] = 2;
  __syncthreads();

  long long busy = 0;
  float fold = 0.0f;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: lane `d` digit `d`'s run: its entries in slot order, its count below the token groups' split, its place (the
    //: counts before it), as `chunk_runs` gives them; warp 0 writes every run's entries
    int n = 0, n0 = 0;
    for (int s = 0; s < kSlots; ++s)
      if (live<BOX>(blockIdx.x, w, lane, s, kColored, kOf256)) {
        ++n;
        n0 += s < split;
      }
    int to = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, to, o);
      if (lane >= o) to += v;
    }
    //: the draw through shared memory, stored before the barrier and loaded inside the span: register arithmetic
    //: alone would sink past the barrier and the clock into the timed span
    int* const drawn =
        reinterpret_cast<int*>(smem + P::kUnionW);  // a tenant the readout never reads
    if (warp == 0) {
      drawn[lane] = n;
      drawn[32 + lane] = n0;
      drawn[64 + lane] = to - n;
    }
    if (warp == 0) {
      int k = to - n;
      for (int s = 0; s < kSlots; ++s)
        if (live<BOX>(blockIdx.x, w, lane, s, kColored, kOf256)) {
          const uint32_t sl = static_cast<uint32_t>(s);
          reinterpret_cast<uint2*>(smem + (ents - ops::smem_addr(smem)))[k++] =
              make_uint2((sl * (P::kReadRow / 16) + rc::pool_chunk(P::kReadRow, sl, 0u)) | sl << 16,
                         0x3800u ^ (mix(sl ^ static_cast<uint32_t>(lane)) & 0x3ffu));
        }
    }
    //: the panel and the den words cleared, as the chunk's fill clears them
    for (int i = tid; i < (rc::read_ents<P>(kSlots, p.reduce) - rc::read_panel<P>(kSlots)) / 4;
         i += rc::kThreads)
      reinterpret_cast<uint32_t*>(smem + (panel - ops::smem_addr(smem)))[i] = 0u;
    //: two barriers before the clock: after one, the empty span read 19.4K cycles a window (the warps not lined up
    //: by it, measured); after two, 38
    __syncthreads();
    __syncthreads();
    const long long t0 = clock64();
    const rc::Run run{0, drawn[lane], drawn[32 + lane], drawn[64 + lane]};
    if constexpr (!kEmpty) rc::read_factors<P>(p, smem, run, w, 0, ents, warp, lane);
    ops::rendezvous();
    if constexpr (!kEmpty && kFirst)
      rc::read_chunk<P, kF32, kF16>(smem, rc::read_run<P>(run, warp, lane), 0, kHalf, image,
                                    kHalf - 1u, rows, panel, dens, ents, warp, lane);
    ops::rendezvous();
    if constexpr (!kEmpty && kSecond)
      rc::read_chunk<P, kF32, kF16>(smem, rc::read_run<P>(run, warp, lane), kHalf, 2 * kHalf, image,
                                    kHalf - 1u, rows, panel, dens, ents, warp, lane);
    const long long t1 = clock64();
    busy += t1 - t0;
    //: the sink read between barriers: every warp's den words written before it, none cleared until after it
    __syncthreads();
    fold += reinterpret_cast<const float*>(smem + (dens - ops::smem_addr(smem)))[tid];
    __syncthreads();
  }
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = busy;
  lab::sink(out, lab::bits(fold));
}

//: THE FLOORS a window, from the same draw: TENSOR, a scheduler's warps' MMAs (4 a tile a channel group: the num's 2
//: and the den's 2, `mma` cycles each); SHARED, the CTA's wavefront-cycles (a 128-byte load one, a store two): a digit
//: a warp its two slices (8), its exponent and den factor (2); a tile a warp its entries (2), its rows (8), and a token
//: pair's panel and den words (a load and a store each, the f16 panel's scale byte a load: 14 f16, 18 fp32)
struct Floor {
  double tensor = 0, shared = 0;
};

template <int BOX, int MODE>
Floor floor_of(int blocks, double mma, double tile_shared, bool colored) {
  using P = typename Box<BOX>::P;
  constexpr int kSlots = slots_of<BOX, MODE>(), kOf256 = of256<BOX, MODE>(),
                kCg = P::kChannelGroups;
  Floor f;
  const int split = rc::group_split(kSlots);
  for (int b = 0; b < blocks; ++b)
    for (int w = 0; w < kWindows; ++w) {
      double tiles = 0, digits = 0;
      for (int d = 0; d < P::kDigits; ++d) {
        int n[2] = {0, 0};
        for (int s = 0; s < kSlots; ++s)
          if (live<BOX>(b, w, d, s, colored, kOf256)) n[P::kTokenGroups == 2 && s >= split]++;
        for (int g = 0; g < P::kTokenGroups; ++g) tiles += (n[g] + 15) / 16;
        digits += P::kTokenGroups;
      }
      f.tensor += tiles * kCg * mma;
      f.shared += kCg * (digits * 10 + tiles * tile_shared);
    }
  f.tensor /= double(blocks) * kWindows;
  f.shared /= double(blocks) * kWindows;
  return f;
}

template <int BOX, int MODE>
void variant(const lab::Bench& b, const char* label) {
  using P = typename Box<BOX>::P;
  const std::string name = lab::kernel("read", {BOX, MODE});
  if constexpr (MODE < 6 || MODE >= 11)
    b.expect(name, MODE % 3 == 0 || MODE >= 11 ? "HMMA.16816.F16" : "HMMA.16816.F32", ">=", 4);
  b.exempt(name,
           "an isolation of the production readout; its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  using A = typename Box<BOX>::A;
  //: the launch's shared bytes, the lab build's plan
  const int smem = rc::Plan<A, lab::kLabBuild>::kSmemBytes;
  LAB_CHECK(
      cudaFuncSetAttribute(read<BOX, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const lab::Samples s = b.variant(
      [&] { read<BOX, MODE><<<b.sms, rc::kThreads, smem>>>(b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(label,
              "a timing isolation; the readout's arithmetic is the oracle's (test_carry_readout)");
  b.record(label, "cycles a warp-window", s);
  //: the floors are the whole readout's: the controls and the sections have none
  if constexpr (MODE >= 6 && MODE < 12) return;
  //: sm_86's m16n8k16 at fp32 accumulate is half rate (32 cycles a scheduler), at fp16 16; sm_80's 8 either way
  cudaDeviceProp prop{};
  LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
  const bool geforce = prop.major == 8 && prop.minor == 6;
  //: the dtypes as `read`'s: MODE 12 and 13 sm_86's form
  constexpr bool kF16 = MODE >= 10 || MODE % 3 == 0, kF32 = MODE < 10 && MODE % 3 == 2;
  const Floor f = floor_of<BOX, MODE>(b.sms, geforce ? (kF16 ? 16.0 : 32.0) : 8.0,
                                      kF32 ? 26.0 : 22.0, MODE >= 3 && MODE < 6);
  for (const auto& [what, v] :
       {std::pair<const char*, double>{"tensor floor", f.tensor}, {"shared floor", f.shared}}) {
    const std::string nm = std::string(label) + ": " + what;
    b.unchecked(nm, "computed on the host from the same draw");
    std::printf(
        "{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"cycles a warp-window\", \"samples\": [%.1f]}\n",
        nm.c_str(), v);
  }
}

}  // namespace read_alone

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  read_alone::variant<0, 0>(b, "f16 panel, fp16 accumulate (sm_86's form)");
  read_alone::variant<0, 1>(b, "f16 panel, fp32 accumulate (the A100's form)");
  read_alone::variant<0, 2>(b, "fp32 panel, fp32 accumulate");
  read_alone::variant<0, 3>(b, "colored slots: f16 panel, fp16 accumulate");
  read_alone::variant<0, 4>(b, "colored slots: f16 panel, fp32 accumulate");
  read_alone::variant<0, 5>(b, "colored slots: fp32 panel, fp32 accumulate");
  read_alone::variant<0, 6>(b, "control: the timed span without the readout");
  read_alone::variant<0, 9>(b, "control: the timed span without the readout, colored slots");
  read_alone::variant<0, 10>(b, "section: the factors and their rendezvous alone (sm_86's form)");
  read_alone::variant<0, 11>(b, "section: the factors and the first half (sm_86's form)");
  read_alone::variant<0, 13>(b, "alt-k16's density (18 / 256): f16 panel, fp16 accumulate");
  read_alone::variant<0, 12>(b, "dense lists (40 / 256): f16 panel, fp16 accumulate");
  read_alone::variant<1, 0>(
      b, "32 x 16 x 64 box, nl64k-alt-k4's chunk: f16 panel, fp16 accumulate (sm_86's form)");
  read_alone::variant<1, 1>(
      b, "32 x 16 x 64 box, nl64k-alt-k4's chunk: f16 panel, fp32 accumulate (the A100's form)");
  read_alone::variant<1, 2>(b,
                            "32 x 16 x 64 box, nl64k-alt-k4's chunk: fp32 panel, fp32 accumulate");
  read_alone::variant<1, 6>(b, "32 x 16 x 64 box: control, the timed span without the readout");
  read_alone::variant<1, 10>(b,
                             "32 x 16 x 64 box, nl64k-alt-k4's chunk: section, the factors alone");
  read_alone::variant<1, 11>(
      b, "32 x 16 x 64 box, nl64k-alt-k4's chunk: section, the factors and the first half");
  read_alone::variant<1, 13>(
      b, "32 x 16 x 64 box, flagship-alt-k16's chunk: f16 panel, fp16 accumulate");
  return 0;
}
