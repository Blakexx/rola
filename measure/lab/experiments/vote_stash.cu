// measure/lab/experiments/vote_stash.cu -- QUESTION: what does the facts pass's first stage pay to keep each digit's
// live bytes -- a STASH a (window, digit), its live bytes in token order, one byte an entry -- beside the vote that
// turns its rows into bitlists, and does the row stream keep its rate? METHOD: rows [level-side, T, H, 256] u8, each
// token's head live in k random digits (bytes 1..255, the rest zero); one block of eight warps an SM, each warp taking
// items in order from a counter -- (window of 512 tokens, level-side, head, 128-digit span), the span innermost --
// and streaming an item as 16 blocks of 32 tokens x 128 bytes (4 KB; a ring of three, two in flight ahead of the one
// read and running on into the next item; each lane one token's 128-byte run). A block's four 32-digit sub-words are
// voted as the facts pass votes a word (lane `(tk, dq)` four tokens' four digits a load, a 4 x 4 byte transpose, then
// digit `4 dq + tk`'s 32 tokens as one word) and stored word-major; EXTRACT 1 also walks each lane's digit's live
// tokens, reads each one's byte from the stage and stores it to the digit's stash at the digit's running count;
// EXTRACT 2 appends them to the item's one packed stash instead (a sub-word's live bytes lane by lane after a scan of
// the lanes' counts, each lane's in token order). Against vote alone, at k 4, 16 and 64: the stream's bytes a cycle,
// and from the profiler each variant's instructions. TILED: each warp keeps its item's bits in a shared tile (its
// span's digits x the window's 16 words, 8 KB) and stores the tile's rows at the item's end, 64 contiguous bytes a
// row (the footprint the lists' second stage needs on chip), at fewer warps an SM.
// see docs/internals/tools/lab.md#vote-stash
#include "lab.cuh"

namespace vote_stash {

constexpr int kSpan = 128, kSubWords = kSpan / 32, kTileStride = 17, kTileBytes = kSpan * kTileStride * 4;
constexpr int kBlockBytes = 32 * kSpan, kWindow = 512, kBlocks = kWindow / 32;
constexpr int kTokens = 65536, kHeads = 4, kWidth = 256, kLevelSides = 4;
constexpr long long kLevelBytes = static_cast<long long>(kTokens) * kHeads * kWidth;
constexpr long long kBytes = kLevelBytes * kLevelSides;
constexpr int kSpans = kWidth / kSpan, kWindows = kTokens / kWindow;
constexpr int kItems = kWindows * kLevelSides * kHeads * kSpans;

__device__ __forceinline__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

//: a token's head live in `k` random digits of 256, bytes 1..255 (the rows zeroed first)
__global__ void fill(uint8_t* rows, int k) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<long long>(kLevelSides) * kTokens * kHeads) return;
  for (int j = 0; j < k; ++j) {
    const uint32_t h = mix(static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0x5eedu);
    rows[i * kWidth + (h & 255u)] = static_cast<uint8_t>(1u + (h >> 8) % 255u);
  }
}

__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

//: a block's copies: quantum `32 q + lane` of the block's 256 is token `4 q + lane / 8`'s eighth `lane % 8` of its
//: 128-byte run, so eight consecutive lanes read one run; each 32-byte quarter lands in its sub-word's facts layout
__device__ __forceinline__ void issue(const uint8_t* block, uint32_t stage, int lane) {
#pragma unroll
  for (int q = 0; q < kSpan / 16; ++q) {
    const int token = 4 * q + (lane >> 3), part = lane & 7;
    const uint32_t dst = stage + (part >> 1) * 1024u + ring_row(token) * 32u + 16u * (part & 1);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                 "l"(block + static_cast<long long>(token) * kHeads * kWidth + 16 * part));
  }
}

template <int EXTRACT, int WARPS, int STAGES, int TILED>
__global__ void __launch_bounds__(WARPS * 32, 1) stage1(const uint8_t* rows, uint32_t* bits_out, uint8_t* stash,
                                                         int* counter, long long* t) {
  constexpr int kStages = STAGES;
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = lab::lane_id(), warp = lab::warp_id(), tk = lane >> 3, dq = lane & 7;
  uint8_t* const ring = smem + warp * (kStages * kBlockBytes + TILED * kTileBytes);
  uint32_t* const tile = reinterpret_cast<uint32_t*>(ring + kStages * kBlockBytes);
  const uint32_t ring_at = lab::smem(ring);
  lab::WarpSpan clock;
  clock.start();
  const auto take = [&] {
    int item = 0;
    if (lane == 0) item = atomicAdd(counter, 1);
    return __shfl_sync(0xffffffffu, item, 0);
  };
  //: block `b` of `item`: its first token's row's span bytes
  const auto source = [&](int item, int b) {
    const int span = item % kSpans, head = (item / kSpans) % kHeads, ls = (item / kSpans / kHeads) % kLevelSides;
    const int window = item / kSpans / kHeads / kLevelSides;
    return rows + ls * kLevelBytes
           + (static_cast<long long>(window * kWindow + 32 * b) * kHeads + head) * kWidth + span * kSpan;
  };
  int item = take(), next = take(), seq = 0;
  if (item < kItems) {
#pragma unroll
    for (int k = 0; k < kStages - 1; ++k) {
      issue(source(item, k), ring_at + k * kBlockBytes, lane);
      commit();
    }
  }
#pragma unroll 1
  for (; item < kItems; item = next, next = take()) {
    int count[kSubWords] = {}, written = 0;
#pragma unroll 1
    for (int b = 0; b < kBlocks; ++b, ++seq) {
      const int ahead = b + kStages - 1;
      const uint32_t slot_at = ring_at + ((seq + kStages - 1) % kStages) * kBlockBytes;
      if (ahead < kBlocks)
        issue(source(item, ahead), slot_at, lane);
      else if (next < kItems)
        issue(source(next, ahead - kBlocks), slot_at, lane);
      commit();
      wait<kStages - 1>();
      __syncwarp();
      const uint8_t* const stage = ring + (seq % kStages) * kBlockBytes;
#pragma unroll
      for (int s = 0; s < kSubWords; ++s) {
        const uint8_t* const slot = stage + s * 1024;
        uint32_t flags = 0u;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint32_t x = *reinterpret_cast<const uint32_t*>(slot + (4 * i + tk) * 32 + 4 * dq);
          flags |= ((((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u) >> (7 - i);
        }
        uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
        flags = (tk & 2) ? prmt<0x3276u>(flags, y) : prmt<0x5410u>(flags, y);
        y = __shfl_xor_sync(0xffffffffu, flags, 8);
        flags = (tk & 1) ? prmt<0x3715u>(flags, y) : prmt<0x6240u>(flags, y);
        const int digit = 4 * dq + tk;
        if constexpr (TILED)
          tile[(32 * s + digit) * kTileStride + b] = flags;
        else
          bits_out[(static_cast<long long>(item) * kBlocks + b) * kSpan + 32 * s + digit] = flags;
        if constexpr (EXTRACT == 2) {
          //: PACKED: the sub-word's live bytes appended to the item's stash, lane by lane (a scan of their counts),
          //: each lane its digit's in token order
          const int n = __popc(flags);
          int first = n;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, first, off);
            first += lane >= off ? x : 0;
          }
          const int total = __shfl_sync(0xffffffffu, first, 31);
          uint8_t* out = stash + static_cast<long long>(item) * kSpan * kWindow + written + first - n;
          uint32_t live = flags;
#pragma unroll 1
          while (live != 0u) {
            const int tok = __ffs(live) - 1;
            live &= live - 1u;
            *out++ = slot[ring_row(tok) * 32 + digit];
          }
          written += total;
        }
        if constexpr (EXTRACT == 1) {
          uint8_t* const out = stash + (static_cast<long long>(item) * kSpan + 32 * s + digit) * kWindow;
          uint32_t live = flags;
#pragma unroll 1
          while (live != 0u) {
            const int tok = __ffs(live) - 1;
            live &= live - 1u;
            out[count[s]++] = slot[ring_row(tok) * 32 + digit];
          }
        }
      }
      __syncwarp();
    }
    if constexpr (TILED) {
      //: the tile's rows out: row r of the span the item's (level-side, head) row, its window's 16 words one run
      const int span = item % kSpans, head = (item / kSpans) % kHeads, ls = (item / kSpans / kHeads) % kLevelSides;
      const int window = item / kSpans / kHeads / kLevelSides;
#pragma unroll 4
      for (int i = lane; i < kSpan * 4; i += 32) {
        const int r = i >> 2, q = i & 3;
        const uint32_t* const at = tile + r * kTileStride + 4 * q;
        const long long row = (static_cast<long long>(ls) * kHeads + head) * kWidth + span * kSpan + r;
        *reinterpret_cast<uint4*>(bits_out + row * (kTokens / 32) + window * kBlocks + 4 * q) =
            make_uint4(at[0], at[1], at[2], at[3]);
      }
      __syncwarp();
    }
  }
  clock.stop(t);
}

template <int EXTRACT, int K, int WARPS = 8, int STAGES = 3, int TILED = 0>
void variant(const lab::Bench& b, uint8_t* rows, uint32_t* bits, uint8_t* stash, int* counter) {
  const std::string name = lab::kernel("stage1", {EXTRACT, WARPS, STAGES, TILED});
  b.expect(name, "LDGSTS", ">=", 8);
  if (b.expect_only) return;
  LAB_CHECK(cudaMemset(rows, 0, kBytes));
  const long long heads = static_cast<long long>(kLevelSides) * kTokens * kHeads;
  fill<<<static_cast<int>((heads + 255) / 256), 256>>>(rows, K);
  LAB_CHECK(cudaDeviceSynchronize());
  const int bytes = WARPS * (STAGES * kBlockBytes + TILED * kTileBytes);
  LAB_CHECK(cudaFuncSetAttribute(stage1<EXTRACT, WARPS, STAGES, TILED>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes));
  const lab::Samples s = b.variant(
      [&] {
        LAB_CHECK(cudaMemset(counter, 0, sizeof(int)));
        stage1<EXTRACT, WARPS, STAGES, TILED><<<b.sms, WARPS * 32, bytes>>>(rows, bits, stash, counter, b.times);
      },
      [&] { return static_cast<double>(kBytes) / b.span(WARPS); });
  const char* kind[3] = {"vote", "vote + stash a digit", "vote + stash packed"};
  std::string label = std::string(kind[EXTRACT]) + " / k " + std::to_string(K);
  if (TILED || WARPS != 8 || STAGES != 3)
    label += " / " + std::to_string(WARPS) + " warps, " + std::to_string(STAGES) + " stages" + (TILED ? ", tiled" : "");
  b.record(label, "bytes a cycle", s);
}

}  // namespace vote_stash

int main(int argc, char** argv) {
  using namespace vote_stash;
  lab::Bench b(argc, argv);
  uint8_t *rows = nullptr, *stash = nullptr;
  uint32_t* bits = nullptr;
  int* counter = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&rows, kBytes));
    LAB_CHECK(cudaMalloc(&bits, kBytes / 8));
    LAB_CHECK(cudaMalloc(&stash, static_cast<size_t>(kItems) * kSpan * kWindow));
    LAB_CHECK(cudaMalloc(&counter, sizeof(int)));
  }
  variant<0, 4>(b, rows, bits, stash, counter);
  variant<1, 4>(b, rows, bits, stash, counter);
  variant<2, 4>(b, rows, bits, stash, counter);
  variant<0, 16>(b, rows, bits, stash, counter);
  variant<1, 16>(b, rows, bits, stash, counter);
  variant<2, 16>(b, rows, bits, stash, counter);
  variant<0, 64>(b, rows, bits, stash, counter);
  variant<1, 64>(b, rows, bits, stash, counter);
  variant<2, 64>(b, rows, bits, stash, counter);
  //: the bits kept on chip for the second stage: 6 warps of two stages, 4 of three, against 8 of two untiled
  variant<2, 4, 8, 2, 0>(b, rows, bits, stash, counter);
  variant<2, 4, 6, 2, 1>(b, rows, bits, stash, counter);
  variant<2, 4, 4, 3, 1>(b, rows, bits, stash, counter);
  variant<2, 16, 6, 2, 1>(b, rows, bits, stash, counter);
  variant<2, 16, 4, 3, 1>(b, rows, bits, stash, counter);
}
