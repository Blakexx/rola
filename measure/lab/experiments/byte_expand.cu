// measure/lab/experiments/byte_expand.cu -- QUESTION: what does a warp take to expand a tile of u8 bytes into bf16
// times each token's fp32 scale, with nothing else in its loop -- no ring, no barrier -- and which of its
// instruction classes holds an SM's rate? METHOD: 1, 4 or 8 warps a CTA, a CTA an SM; each warp expands its own 1 KB
// of bytes (64 tokens x 16 K, in shared memory) into its own 2 KB operand tile, over and over: a lane's two tokens,
// a byte to 2^23 + byte by one PRMT, the product by one FFMA, a pair to a bf16 word by one conversion (`row_ring`'s
// expansion). Four more forms at 4 and 8 warps each drop one class: the pair packed by truncation (a PRMT), no
// product, no reload of the bytes, no store of the tile. The checked forms' last tile is the host's.
// see docs/internals/tools/lab.md#byte-expand
#include "lab.cuh"

namespace byte_expand {

constexpr int kN = 64, kK = 16, kRawBytes = kN * kK, kTileBytes = 2 * kRawBytes, kMaxWarps = 8;
enum Form : int { kWhole = 0, kTruncated, kNoProduct, kNoReload, kNoStore, kForms };
const char* kFormName[kForms] = {"the expansion", "the pair packed by truncation", "no product", "no reload of the bytes",
                                 "no store of the tile"};

constexpr int byte_of(int warp, int n, int k) { return (warp * 31 + n * 7 + k * 3 + 1) % 256; }
constexpr int line_at(int n, int q) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * q); }

template <int FORM>
__device__ __forceinline__ float expand(uint32_t word, int i, float s, float c) {
  uint32_t x;
  if (i == 0) asm volatile("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 1) asm volatile("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 2) asm volatile("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if (i == 3) asm volatile("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(x) : "r"(word), "r"(0x4B000000u));
  if constexpr (FORM == kNoProduct) return __uint_as_float(x);
  return fmaf(__uint_as_float(x), s, c);
}

//: a pair as a bf16 word: rounded by one conversion, or (`kTruncated`) each value's high half by one PRMT
template <int FORM>
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  if constexpr (FORM == kTruncated) return lab::bf16x2_trunc(lo, hi);
  return lab::bf16x2_round(lo, hi);
}

template <int kWarps, int FORM>
__global__ void tiles(long long* times, int* sinks, uint16_t* out, int rounds) {
  __shared__ alignas(128) uint8_t smem[kWarps * (kRawBytes + kTileBytes)];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint8_t* const raw = smem + warp * (kRawBytes + kTileBytes);
  uint8_t* const tile = raw + kRawBytes;
  //: this lane's two tokens' bytes, as it will rewrite them every round, and their scales 2^-e as bits
  uint4 mine[2];
  float s[2], c[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int n = 2 * lane + j;
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      w[q] = byte_of(warp, n, 4 * q) | byte_of(warp, n, 4 * q + 1) << 8 | byte_of(warp, n, 4 * q + 2) << 16
             | static_cast<uint32_t>(byte_of(warp, n, 4 * q + 3)) << 24;
    mine[j] = make_uint4(lab::held(w[0]), lab::held(w[1]), lab::held(w[2]), lab::held(w[3]));
    const uint32_t e = static_cast<uint32_t>(n % 5);
    s[j] = __uint_as_float((127u - e) << 23);
    c[j] = __uint_as_float(0x80000000u | (150u - e) << 23);
  }

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int n = 2 * lane + j;
      //: THE BYTES, stored then loaded: the round's own, as a landed copy's would be
      uint4 in = mine[j];
      if constexpr (FORM != kNoReload) {
        *reinterpret_cast<uint4*>(raw + 16 * n) = mine[j];
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(in.x), "=r"(in.y), "=r"(in.z), "=r"(in.w)
                     : "r"(lab::smem(raw + 16 * n))
                     : "memory");
      }
      const uint32_t w[4] = {in.x, in.y, in.z, in.w};
      //: THE EXPANSION
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint4 o;
        o.x = pack<FORM>(expand<FORM>(w[2 * q], 0, s[j], c[j]), expand<FORM>(w[2 * q], 1, s[j], c[j]));
        o.y = pack<FORM>(expand<FORM>(w[2 * q], 2, s[j], c[j]), expand<FORM>(w[2 * q], 3, s[j], c[j]));
        o.z = pack<FORM>(expand<FORM>(w[2 * q + 1], 0, s[j], c[j]), expand<FORM>(w[2 * q + 1], 1, s[j], c[j]));
        o.w = pack<FORM>(expand<FORM>(w[2 * q + 1], 2, s[j], c[j]), expand<FORM>(w[2 * q + 1], 3, s[j], c[j]));
        if constexpr (FORM == kNoStore) seen ^= (o.x + o.y) ^ (o.z + o.w) ^ static_cast<uint32_t>(r);
        else *reinterpret_cast<uint4*>(tile + line_at(n, q)) = o;
      }
    }
  }
  clock.stop(times);
  lab::sink(sinks, seen);
  __syncthreads();
  //: the last tile out: this lane's 32 words
  uint16_t* dst = out + (static_cast<long long>(blockIdx.x) * kMaxWarps + warp) * kRawBytes;
#pragma unroll 1
  for (int i = lane; i < kRawBytes; i += 32) dst[i] = reinterpret_cast<const uint16_t*>(tile)[i];
}

}  // namespace byte_expand

template <int kWarps, int FORM>
void run(lab::Bench& b, uint16_t* out) {
  using namespace byte_expand;
  constexpr int kRounds = 4096;
  constexpr bool kChecked = FORM == kWhole || FORM == kTruncated || FORM == kNoReload;
  const std::string name = std::string(kFormName[FORM]) + ", " + std::to_string(kWarps)
                           + (kWarps == 1 ? " warp" : " warps") + " a CTA";
  bool checked = false;
  long long wrong = 0, total = 0;
  const lab::Samples cycles = b.variant([&] { tiles<kWarps, FORM><<<b.sms, 32 * kWarps>>>(b.times, b.sinks, out, kRounds); }, [&] {
    if (kChecked && !checked) {
      checked = true;
      std::vector<uint16_t> got(static_cast<size_t>(b.sms) * kMaxWarps * kRawBytes);
      LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      for (int blk = 0; blk < b.sms; ++blk)
        for (int w = 0; w < kWarps; ++w)
          for (int n = 0; n < kN; ++n)
            for (int k = 0; k < kK; ++k, ++total) {
              //: a byte times a power of two is exact in bf16, so rounded and truncated agree
              const float v = byte_of(w, n, k) / static_cast<float>(1 << (n % 5));
              uint32_t bits;
              std::memcpy(&bits, &v, 4);
              const size_t at = (static_cast<size_t>(blk) * kMaxWarps + w) * kRawBytes + (line_at(n, k / 8) / 2) + k % 8;
              wrong += got[at] != static_cast<uint16_t>(bits >> 16);
            }
    }
    return b.span(kWarps) / kRounds;
  });
  if (kChecked)
    b.check(name, wrong == 0 && total > 0,
            std::to_string(wrong) + " of " + std::to_string(total) + " operand words differ from the host's");
  else b.unchecked(name, "an isolation: one instruction class is dropped, so the tile is not the product");
  b.record(name, "cycles a tile a warp", cycles);
}

int main(int argc, char** argv) {
  using namespace byte_expand;
  lab::Bench b(argc, argv);
  for (int w : {1, 4, 8})
    for (int f = 0; f < kForms; ++f) {
      if (w == 1 && f != kWhole) continue;
      const std::string k = lab::kernel("tiles", {w, f});
      b.expect(k, "FFMA", "==", f == kNoProduct ? 0 : 32);
      b.expect(k, "F2FP", "==", f == kTruncated ? 0 : 16);
      b.expect(k, "STS.128", "==", f == kNoStore ? 2 : f == kNoReload ? 4 : 6);
    }
  if (b.expect_only) return 0;
  uint16_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(uint16_t) * b.sms * kMaxWarps * kRawBytes));
  run<1, kWhole>(b, out);
  run<4, kWhole>(b, out);
  run<8, kWhole>(b, out);
  run<4, kTruncated>(b, out);
  run<8, kTruncated>(b, out);
  run<4, kNoProduct>(b, out);
  run<8, kNoProduct>(b, out);
  run<4, kNoReload>(b, out);
  run<8, kNoReload>(b, out);
  run<4, kNoStore>(b, out);
  run<8, kNoStore>(b, out);
  return 0;
}
