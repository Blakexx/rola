// measure/lab/experiments/row_ring.cu -- QUESTION: producer warps expand u8 rows, landed in shared memory by a bulk
// copy, into a ring of wgmma B operands (bf16, each row times its fp32 scale); how many cycles a tile does a
// producer take against the warpgroup consuming it, and how many producers keep one warpgroup fed? METHOD: a CTA an
// SM: warps 0-3 a consuming warpgroup (one m64n64k16 wgmma a tile, A in registers), then 1, 2 or 4 producer warps
// over a ring of 4 operand slots and 8 slots of bytes. A producer's tile: its 1 KB of bytes arrives under an
// mbarrier, issued before the tile ahead of it is expanded (two loads in flight); a byte becomes 2^23 + byte by one
// PRMT, the product by one FFMA (times the scale, less 2^23 scales), a pair a bf16 word by one conversion; full and
// empty barriers a slot hand tiles over. Each side also runs alone, the other doing nothing but the barriers'
// arrivals. The readout's accumulators are checked against the host's.
// see docs/internals/tools/lab.md#row-ring
#include "lab.cuh"

namespace row_ring {

constexpr int kM = 64, kK = 16, kN = 64, kSlots = 4, kTileBytes = 2048, kRawBytes = kN * kK, kSrcTiles = 32;
//: the bytes' ring: twice the operands', so a producer's next tile lands while it expands this one at any count
constexpr int kRawSlots = 2 * kSlots;
enum Mode : int { kBoth = 0, kProducerAlone, kConsumerAlone };
const char* kModeName[3] = {"producers and consumer", "producers alone", "the consumer alone"};

constexpr int read_a(int m, int k) { return (m + k) % 7 - 3; }
//: a tile's byte at token `n`, K `k`, and the token's scale, a power of two: every product exact in bf16
constexpr int byte_of(int tile, int n, int k) { return (tile * 31 + n * 7 + k * 3 + 1) % 256; }
constexpr float scale_of(int n) { return 1.0f / static_cast<float>(1 << (n % 5)); }

//: a K-major tile's byte offset: token `n`'s K half `q`, 16 bytes (`lab::wg_desc(a, 8, 16)`)
constexpr int line_at(int n, int q) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * q); }

constexpr int kTiles = 0, kRaw = kTiles + kSlots * kTileBytes, kBars = kRaw + kRawSlots * kRawBytes,
              kBarCount = 2 * kSlots + kRawSlots, kSmem = kBars + kBarCount * 8;

//: a tile's barriers: its operand slot's full and empty, its bytes' slot's landed
struct Bars {
  uint32_t full, empty, raw;
};

__device__ __forceinline__ Bars bars_of(uint32_t base, int tile) {
  return {base + kBars + 8 * (tile % kSlots), base + kBars + 8 * (kSlots + tile % kSlots),
          base + kBars + 8 * (2 * kSlots + tile % kRawSlots)};
}

//: A TILE'S BYTES, issued: the one arrival that states them, then the copy into the tile's slot of the bytes' ring.
//: ISSUE EDGE: the slot's last reader is this warp's tile `kRawSlots` back, expanded and synchronized iterations ago
__device__ __forceinline__ void fetch(uint32_t base, const uint8_t* src, int tile) {
  lab::mbar_expect(bars_of(base, tile).raw, kRawBytes);
  lab::bulk_load(base + kRaw + (tile % kRawSlots) * kRawBytes, src + (tile % kSrcTiles) * kRawBytes, kRawBytes,
                 bars_of(base, tile).raw);
}

//: A BYTE AS 2^23 + byte, then the product `byte * s` in one FFMA (`c` = -2^23 s)
__device__ __forceinline__ float expand(uint32_t word, int i, float s, float c) {
  uint32_t x;
  if (i == 0) asm volatile("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 1) asm volatile("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 2) asm volatile("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 3) asm volatile("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  return fmaf(__uint_as_float(x), s, c);
}

//: A PRODUCER WARP'S TILES: every `kProducers`-th, the next one's bytes issued before this one is expanded
template <int MODE, int kProducers>
__device__ __forceinline__ void produce(uint8_t* smem, const uint8_t* src, int tiles, int mine) {
  static_assert(kProducers < kRawSlots, "a producer's next tile has its own slot of bytes");
  constexpr bool kExpands = MODE != kConsumerAlone;
  const int lane = threadIdx.x & 31;
  const uint32_t base = lab::smem(smem);
  //: this lane's two tokens' scales 2^-e and -2^23 times them, built as bits, once
  float s[2], c[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint32_t e = static_cast<uint32_t>((2 * lane + j) % 5);
    s[j] = __uint_as_float((127u - e) << 23);
    c[j] = __uint_as_float(0x80000000u | (150u - e) << 23);
  }
  if (kExpands && lane == 0 && mine < tiles) fetch(base, src, mine);
#pragma unroll 1
  for (int t = mine; t < tiles; t += kProducers) {
    const int slot = t % kSlots, use = t / kSlots;
    const Bars b = bars_of(base, t);
    if constexpr (kExpands) {
      //: THE NEXT TILE'S BYTES, in flight under this tile's work
      if (lane == 0 && t + kProducers < tiles) fetch(base, src, t + kProducers);
    }
    //: THE WAITS: the slot's last tile consumed, this tile's bytes landed
    if (use > 0) lab::mbar_wait(b.empty, (use - 1) & 1);
    if constexpr (kExpands) {
      lab::mbar_wait(b.raw, (t / kRawSlots) & 1);
      //: THE EXPANSION: this lane's two tokens, 16 bytes each
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int n = 2 * lane + j;
        const uint4 in = *reinterpret_cast<const uint4*>(smem + kRaw + (t % kRawSlots) * kRawBytes + 16 * n);
        const uint32_t w[4] = {in.x, in.y, in.z, in.w};
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          uint4 o;
          o.x = lab::bf16x2_round(expand(w[2 * q], 0, s[j], c[j]), expand(w[2 * q], 1, s[j], c[j]));
          o.y = lab::bf16x2_round(expand(w[2 * q], 2, s[j], c[j]), expand(w[2 * q], 3, s[j], c[j]));
          o.z = lab::bf16x2_round(expand(w[2 * q + 1], 0, s[j], c[j]), expand(w[2 * q + 1], 1, s[j], c[j]));
          o.w = lab::bf16x2_round(expand(w[2 * q + 1], 2, s[j], c[j]), expand(w[2 * q + 1], 3, s[j], c[j]));
          *reinterpret_cast<uint4*>(smem + kTiles + slot * kTileBytes + line_at(n, q)) = o;
        }
      }
      //: THE WRITERS' EDGE to the wgmma that reads the tile
      lab::bulk_fence();
    }
    //: THE HAND-OVER
    __syncwarp();
    if (lane == 0) lab::mbar_arrive(b.full);
  }
}

//: THE CONSUMER: a tile a wgmma, the slot handed back
template <int MODE>
__device__ __forceinline__ void consume(uint8_t* smem, float* out, int tiles) {
  const int tid = threadIdx.x, lane = tid & 31;
  const uint32_t base = lab::smem(smem);
  uint32_t a[4];
  const int g = 16 * (tid >> 5) + lane / 4, kt = lane & 3;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int m = g + 8 * (w & 1), k = 8 * (w >> 1) + 2 * kt;
    a[w] = (__float_as_uint(static_cast<float>(read_a(m, k))) >> 16)
           | (__float_as_uint(static_cast<float>(read_a(m, k + 1))) & 0xffff0000u);
  }
  float d[32] = {};
#pragma unroll 1
  for (int t = 0; t < tiles; ++t) {
    const int slot = t % kSlots, use = t / kSlots;
    const Bars b = bars_of(base, t);
    lab::mbar_wait(b.full, use & 1);
    if constexpr (MODE != kProducerAlone) {
      lab::wg_fence();
      lab::wg_bf16_n64_rs(d, a, lab::wg_desc(base + kTiles + slot * kTileBytes, 8, 16));
      lab::wg_commit();
      lab::wg_wait<0>();
    }
    if (tid == 0) lab::mbar_arrive(b.empty);
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) lab::wg_hold(d[i]);
  float* mine = out + (static_cast<long long>(blockIdx.x) * 128 + tid) * 32;
#pragma unroll
  for (int i = 0; i < 32; ++i) mine[i] = d[i];
}

template <int MODE, int kProducers>
__global__ void ring(long long* times, float* out, const uint8_t* src, int tiles) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, warp = tid >> 5;
  const uint32_t base = lab::smem(smem);
  //: the barriers, one arrival a phase; the tiles zero (the consumer alone reads them as they are)
  if (tid < kBarCount) lab::mbar_init(base + kBars + 8 * tid, 1);
#pragma unroll 1
  for (int i = tid; i < kSlots * kTileBytes / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
  if (warp < 4) consume<MODE>(smem, out, tiles);
  else produce<MODE, kProducers>(smem, src, tiles, warp - 4);
  clock.stop(times);
}

}  // namespace row_ring

template <int MODE, int kProducers>
void run(lab::Bench& b, float* out, const uint8_t* src) {
  using namespace row_ring;
  constexpr int kCheckTiles = 32, kRunTiles = 4096;
  const int threads = 128 + 32 * kProducers, warps = threads / 32;
  const std::string name = std::string(kModeName[MODE]) + ", " + std::to_string(kProducers)
                           + (kProducers == 1 ? " producer warp" : " producer warps");
  if (MODE == kBoth) {
    ring<MODE, kProducers><<<b.sms, threads>>>(b.times, out, src, kCheckTiles);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(static_cast<size_t>(b.sms) * 128 * 32);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<double> d(kM * kN, 0.0);
    for (int t = 0; t < kCheckTiles; ++t)
      for (int m = 0; m < kM; ++m)
        for (int n = 0; n < kN; ++n)
          for (int k = 0; k < kK; ++k)
            d[m * kN + n] += read_a(m, k) * (byte_of(t % kSrcTiles, n, k) * static_cast<double>(scale_of(n)));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < 128; ++tid) {
        const float* mine = got.data() + (static_cast<size_t>(blk) * 128 + tid) * 32;
        const int g = 16 * (tid >> 5) + (tid & 31) / 4, t = tid & 3;
        for (int i = 0; i < 32; ++i) {
          const int m = g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1);
          wrong += mine[i] != static_cast<float>(d[m * kN + n]);
        }
      }
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(got.size()) + " accumulators differ from the host's at "
                + std::to_string(kCheckTiles) + " tiles");
  } else {
    b.unchecked(name, "an isolation: one side only arrives at the barriers, so the accumulators are not the product");
  }
  const lab::Samples cycles = b.variant([&] { ring<MODE, kProducers><<<b.sms, threads>>>(b.times, out, src, kRunTiles); },
                                        [&] { return b.span(warps) / kRunTiles; });
  b.record(name, "cycles a tile", cycles);
}

int main(int argc, char** argv) {
  using namespace row_ring;
  lab::Bench b(argc, argv);
  for (int mode = 0; mode < 3; ++mode)
    for (int p : {1, 2, 4}) {
      if (mode != kBoth && p != 1) continue;
      const std::string k = lab::kernel("ring", {mode, p});
      b.expect(k, "HGMMA", "==", mode == kProducerAlone ? 0 : 1);
      b.expect(k, "UBLKCP", "==", mode == kConsumerAlone ? 0 : 2);
      b.expect(k, "FFMA", "==", mode == kConsumerAlone ? 0 : 32);
      b.expect(k, "F2FP", "==", mode == kConsumerAlone ? 0 : 16);
      if (mode != kProducerAlone) b.exempt(k, "a rate probe: a producer's tile against a warpgroup's, no kernel's step");
    }
  if (b.expect_only) return 0;
  float* out = nullptr;
  uint8_t* src = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * 128 * 32));
  LAB_CHECK(cudaMalloc(&src, kSrcTiles * kRawBytes));
  std::vector<uint8_t> bytes(kSrcTiles * kRawBytes);
  for (int t = 0; t < kSrcTiles; ++t)
    for (int n = 0; n < kN; ++n)
      for (int k = 0; k < kK; ++k) bytes[(t * kN + n) * kK + k] = static_cast<uint8_t>(byte_of(t, n, k));
  LAB_CHECK(cudaMemcpy(src, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
  run<kBoth, 1>(b, out, src);
  run<kBoth, 2>(b, out, src);
  run<kBoth, 4>(b, out, src);
  run<kProducerAlone, 1>(b, out, src);
  run<kConsumerAlone, 1>(b, out, src);
  return 0;
}
