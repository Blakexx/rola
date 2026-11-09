// measure/lab/experiments/handoff.cu -- QUESTION: what does the carry's hand-off cost a window on sm_86 -- each
// digit's exponent from the fold's state (its largest value over the box's 32 lists, each list at its own unit) and the
// state stored into the image, f16 at each digit's power of two -- in the carry's form and in a lean form, the image
// whole (sm_80's form) or in two halves with a barrier between (sm_86's), and how near its floor does it run?
// METHOD: 8 warps hold a box's state in their fold accumulators (lists w + 8 j; `handoff.inputs.py`), made opaque each
// window so nothing is hoisted; a window is the maxima, a CTA barrier, the exponents, the stores and a barrier; a
// sample is the CTA's span over 64 windows, 80 CTAs. Each form's image and exponents are checked byte for byte
// against the mirror's. see docs/lab/handoff.md
#include <cuda_fp16.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/static_for.cuh"
#include "lab.cuh"

namespace handoff {

constexpr int kWarps = 8, kThreads = 32 * kWarps, kLists = 32, kDigits = 32, kChannels = 32, kWindows = 64;
//: the CTAs launched: 80 boxes, one an SM on sm_86 (fixed, so a CTA-window is the same unit on every card)
constexpr int kCtas = 80;
constexpr int kPlaneBytes = kLists * kChannels * 2, kImageBytes = kDigits * kPlaneBytes, kImageTop = 10;
constexpr int kUnitNone = -1000;
//: THE LAYOUT: the image, the lists' units, the carry's posts (a window's parity, a warp's 32 digits, int16), the
//: digits' exponents, the lean form's keys (a window's parity, a lane's 4 digits at 4 g + 2 M + H)
constexpr int kUnitsAt = kImageBytes, kPostsAt = kUnitsAt + 4 * kLists, kExpsAt = kPostsAt + 2 * kWarps * kDigits * 2;
//: and the 16-byte form's scale constants (2^23, 127 x 2^23), loaded so ptxas cannot fold them into a shift
constexpr int kKeysAt = kExpsAt + 4 * kDigits, kConstAt = kKeysAt + 2 * 4 * kDigits, kBytes = kConstAt + 8;

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(static_cast<uint32_t>(e + 127) << 23); }

//: `parts.cuh`'s: the smallest `e` with `x <= 2^e`, `kUnitNone` for zero
__device__ __forceinline__ int ceil_exp(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits == 0u ? kUnitNone : static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

__host__ __device__ __forceinline__ constexpr uint32_t unit_of(int l, int k, int d) {
  return static_cast<uint32_t>(((4 * l + (k ^ ((l >> 1) & 3))) ^ (d & 7)) & (kPlaneBytes / 16 - 1));
}

//: THE 16-BYTE IMAGE: a unit holds a writer lane's 8 channels (8 n + 2 t + b, position 2 n + b) for chunk `t`; the
//: digit's swizzle moves its low bit to bit 2, so a quarter-warp's stores (two digits, four chunks) take eight bank
//: groups and the readout's `ldmatrix.trans` (eight lists of one chunk) stays conflict-free
__host__ __device__ __forceinline__ constexpr uint32_t swizzle16(int d) {
  return static_cast<uint32_t>(((d & 1) << 2) | ((d >> 1) & 3));
}

__host__ __device__ __forceinline__ constexpr uint32_t unit16_of(int l, int k, int d) {
  return ((4u * static_cast<uint32_t>(l) + static_cast<uint32_t>(k ^ ((l >> 1) & 3))) ^ swizzle16(d)) &
         (kPlaneBytes / 16 - 1);
}

__device__ __forceinline__ void sts128(uint32_t at, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(at), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

__device__ __forceinline__ void sts32(uint32_t at, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

using State = float[4][2][4][4];

//: a lane's largest |value| of digit row (M, H) in list J: its 8 values (4 channel tiles x 2)
template <int J, int M, int H>
__device__ __forceinline__ float row_max(const State& acc) {
  float x = 0.0f;
  rola::static_for<4>([&](auto N) {
    x = fmaxf(x, fmaxf(fabsf(acc[J][M][N][2 * H]), fabsf(acc[J][M][N][2 * H + 1])));
  });
  return x;
}

//: the same from its first pair: 7 maxima, not 8
template <int J, int M, int H>
__device__ __forceinline__ float row_max7(const State& acc) {
  float x = fmaxf(fabsf(acc[J][M][0][2 * H]), fabsf(acc[J][M][0][2 * H + 1]));
  rola::static_for<3>([&](auto N) {
    x = fmaxf(x, fmaxf(fabsf(acc[J][M][N + 1][2 * H]), fabsf(acc[J][M][N + 1][2 * H + 1])));
  });
  return x;
}

//: THE CARRY'S FORM (`parts.cuh`): a warp's maxima as exponents (a list's `ceil_exp` plus its unit), two shuffles a
//: digit row, posted int16 at the window's parity; after the barrier lane `d` the largest of the 8 warps' posts
template <int J>
__device__ __forceinline__ int list_unit(const uint8_t* sm, int warp) {
  return reinterpret_cast<const volatile int*>(sm + kUnitsAt)[warp + kWarps * J];
}

__device__ __forceinline__ void carry_maxima(const State& acc, uint8_t* sm, int w, int warp, int lane) {
  int16_t* const posts = reinterpret_cast<int16_t*>(sm + kPostsAt) + ((w & 1) * kWarps + warp) * kDigits;
  int u[4];
  rola::static_for<4>([&](auto J) { u[J] = list_unit<J>(sm, warp); });
  rola::static_for<2>([&](auto M) {
    rola::static_for<2>([&](auto H) {
      int e = 2 * kUnitNone;
      rola::static_for<4>([&](auto J) { e = max(e, ceil_exp(row_max<J, M, H>(acc)) + u[J]); });
      e = max(e, __shfl_xor_sync(0xffffffffu, e, 1));
      e = max(e, __shfl_xor_sync(0xffffffffu, e, 2));
      if ((lane & 3) == 0) posts[16 * M + 8 * H + (lane >> 2)] = static_cast<int16_t>(e);
    });
  });
}

__device__ __forceinline__ int carry_digit_exp(const uint8_t* sm, int w, int d) {
  const int16_t* posts = reinterpret_cast<const int16_t*>(sm + kPostsAt) + (w & 1) * kWarps * kDigits + d;
  int e = kImageTop - 127;
  rola::static_for<kWarps>([&](auto W) { e = max(e, static_cast<int>(posts[W * kDigits])); });
  return e;
}

//: the carry's `store_batches` for m-tile rows `M_LO .. M_HI - 1` (16 digits each), the lane's exponent `dexp`
template <int M_LO, int M_HI>
__device__ __forceinline__ void carry_stores(const State& acc, uint8_t* sm, int dexp, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint32_t base;
  asm volatile("mov.u32 %0, %1;" : "=r"(base) : "r"(unit_of(warp, 0, g)));
  const uint32_t lane0 = lab::smem(sm) + 4u * t;
  int u[4];
  rola::static_for<4>([&](auto J) { u[J] = list_unit<J>(sm, warp); });
  rola::static_for<2>([&](auto M) {
    rola::static_for<2>([&](auto H) {
      const int d = 16 * M + 8 * H + g;
      const int e = __shfl_sync(0xffffffffu, dexp, d);
      if (M >= M_LO && M < M_HI) {
        const uint32_t plane = static_cast<uint32_t>(d & (kDigits - 1)) * kPlaneBytes;
        rola::static_for<4>([&](auto J) {
          const float scale = pow2(min(max(u[J] + kImageTop - e, -126), 127));
          rola::static_for<4>([&](auto N) {
            sts32(lane0 + 16u * (base ^ static_cast<uint32_t>(N)) + plane + 512u * static_cast<uint32_t>(J),
                  pack(acc[J][M][N][2 * H] * scale, acc[J][M][N][2 * H + 1] * scale));
          });
        });
      }
    });
  });
}

//: THE LEAN FORM. A list's value at its unit compared across lists as an integer KEY -- its fp32 bits plus (its unit
//: less the box's top unit) x 2^23, the exponent field shifted, the mantissa kept; a zero INT_MIN -- so a digit's
//: largest is one integer max a list and one shared atomic max a lane (its digit row's 4 lanes joined first), and its
//: exponent `(key >> 23) - 127 + (mantissa != 0)` plus the top unit, exact while the live units span under 255 (the
//: carry's span from `kUnitMin`; a never-live list's state is zero, its key INT_MIN). The keys sit at the
//: window's parity, a lane's 4 digits at 4 g + 2 M + H, so a lane reads its own after the barrier in one load.
struct Lean {
  int u[4], top;
};

__device__ __forceinline__ Lean lean_units(const uint8_t* sm, int warp, int lane) {
  Lean l;
  rola::static_for<4>([&](auto J) { l.u[J] = list_unit<J>(sm, warp); });
  l.top = __reduce_max_sync(0xffffffffu, reinterpret_cast<const volatile int*>(sm + kUnitsAt)[lane]);
  return l;
}

template <bool SEVEN>
__device__ __forceinline__ void lean_maxima(const State& acc, uint8_t* sm, const Lean& l, int w, int lane) {
  int shift[4];
  rola::static_for<4>([&](auto J) { shift[J] = max(l.u[J] - l.top, -255) * (1 << 23); });
  int key[2][2];
  rola::static_for<2>([&](auto M) {
    rola::static_for<2>([&](auto H) {
      int k = INT_MIN;
      rola::static_for<4>([&](auto J) {
        const int bits = __float_as_int(SEVEN ? row_max7<J, M, H>(acc) : row_max<J, M, H>(acc));
        k = max(k, bits == 0 ? INT_MIN : bits + shift[J]);
      });
      k = max(k, __shfl_xor_sync(0xffffffffu, k, 1));
      key[M][H] = max(k, __shfl_xor_sync(0xffffffffu, k, 2));
    });
  });
  const int t = lane & 3;
  const int mine = t & 2 ? (t & 1 ? key[1][1] : key[1][0]) : (t & 1 ? key[0][1] : key[0][0]);
  atomicMax(reinterpret_cast<int*>(sm + kKeysAt) + (w & 1) * kDigits + lane, mine);
}

//: a lane's 4 digits' exponents (digit 16 M + 8 H + g at 2 M + H), its keys reset for the window two on (every warp
//: the same, so no warp branches)
__device__ __forceinline__ int4 lean_exps(uint8_t* sm, const Lean& l, int w, int lane) {
  int* const keys = reinterpret_cast<int*>(sm + kKeysAt);
  int4 k;
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(k.x), "=r"(k.y), "=r"(k.z), "=r"(k.w)
               : "r"(lab::smem(keys + (w & 1) * kDigits + 4 * (lane >> 2)))
               : "memory");
  keys[((w + 1) & 1) * kDigits + lane] = INT_MIN;
  const auto e = [&](int key) {
    return max((key >> 23) - 127 + ((key & 0x7fffff) != 0) + l.top, kImageTop - 127);
  };
  return make_int4(e(k.x), e(k.y), e(k.z), e(k.w));
}

//: the 16-byte form's exponents: `(key + 0x7fffff) >> 23` is the key's exponent field rounded up past a nonzero
//: mantissa, so a digit's exponent is that plus the top unit less 127, at least `kImageTop - 127`
__device__ __forceinline__ int4 lean16_exps(uint8_t* sm, const Lean& l, int w, int lane) {
  int* const keys = reinterpret_cast<int*>(sm + kKeysAt);
  int4 k;
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(k.x), "=r"(k.y), "=r"(k.z), "=r"(k.w)
               : "r"(lab::smem(keys + (w & 1) * kDigits + 4 * (lane >> 2)))
               : "memory");
  keys[((w + 1) & 1) * kDigits + lane] = INT_MIN;
  const int top = l.top - 127;
  const auto e = [&](int key) { return max(((key + 0x7fffff) >> 23) + top, kImageTop - 127); };
  return make_int4(e(k.x), e(k.y), e(k.z), e(k.w));
}

//: a list's scale for a digit, `2^(a - e)` held to a normal float's exponents: the difference and the float's bits each
//: one integer multiply-add (the FMA pipe; the bits' multiplier `two23` and addend `bias` loaded, so ptxas cannot fold
//: them into a shift on the ALU pipe), only the two clamps on the ALU pipe
__device__ __forceinline__ float scale16(int a, int e, uint32_t two23, uint32_t bias) {
  int x;
  asm("mad.lo.s32 %0, %1, -1, %2;" : "=r"(x) : "r"(e), "r"(a));
  x = min(max(x, -126), 127);
  uint32_t bits;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(two23), "r"(bias));
  return __uint_as_float(bits);
}

//: the 16-byte stores of m-tile rows `M_LO .. M_HI - 1`: a scale a (digit row, list), the lane's 8 values of the row
//: scaled, packed and stored as one unit at the lane's held base plus a compile-time offset
template <int M_LO, int M_HI>
__device__ __forceinline__ void lean16_stores(const State& acc, const Lean& l, int4 ex, uint32_t at, uint32_t two23,
                                              uint32_t bias) {
  const int es[2][2] = {{ex.x, ex.y}, {ex.z, ex.w}};
  int a[4];
  rola::static_for<4>([&](auto J) { a[J] = l.u[J] + kImageTop; });
  rola::static_for<2>([&](auto M) {
    if constexpr (M >= M_LO && M < M_HI) {
      rola::static_for<2>([&](auto H) {
        rola::static_for<4>([&](auto J) {
          const float scale = scale16(a[J], es[M][H], two23, bias);
          uint32_t v[4];
          rola::static_for<4>([&](auto N) {
            v[N] = pack(acc[J][M][N][2 * H] * scale, acc[J][M][N][2 * H + 1] * scale);
          });
          sts128(at + static_cast<uint32_t>((16 * M + 8 * H) * kPlaneBytes + 512 * J), v[0], v[1], v[2], v[3]);
        });
      });
    }
  });
}

//: the lean stores of m-tile rows `M_LO .. M_HI - 1`: a scale a (digit row, list), each pair scaled, packed and stored
//: at one of the lane's 4 held bases (a channel tile's) plus a compile-time offset
template <int M_LO, int M_HI>
__device__ __forceinline__ void lean_stores(const State& acc, const Lean& l, int4 ex, const uint32_t (&at)[4]) {
  const int es[2][2] = {{ex.x, ex.y}, {ex.z, ex.w}};
  rola::static_for<2>([&](auto M) {
    if constexpr (M >= M_LO && M < M_HI) {
      rola::static_for<2>([&](auto H) {
        rola::static_for<4>([&](auto J) {
          const float scale = pow2(min(max(l.u[J] + kImageTop - es[M][H], -126), 127));
          rola::static_for<4>([&](auto N) {
            sts32(at[N] + static_cast<uint32_t>((16 * M + 8 * H) * kPlaneBytes + 512 * J),
                  pack(acc[J][M][N][2 * H] * scale, acc[J][M][N][2 * H + 1] * scale));
          });
        });
      });
    }
  });
}

//: FORM 0 the carry's, 1 lean, 2 lean's maxima alone, 3 lean's stores alone (its exponents formed once), 4 lean into
//: the 16-byte image (`unit16_of`); `HALVES` the stores in two halves of 16 digits with a CTA barrier between
template <int FORM, int HALVES>
__global__ void __launch_bounds__(kThreads, 1) handoff(const float* state, const int* units, uint8_t* image_out,
                                                       int* exps_out, long long* t) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), g = lane >> 2, t4 = lane & 3;
  if (threadIdx.x < kLists) reinterpret_cast<int*>(sm + kUnitsAt)[threadIdx.x] = units[threadIdx.x];
  if (threadIdx.x < 2 * kDigits) reinterpret_cast<int*>(sm + kKeysAt)[threadIdx.x] = INT_MIN;
  if (threadIdx.x < 2) reinterpret_cast<uint32_t*>(sm + kConstAt)[threadIdx.x] = threadIdx.x ? 127u << 23 : 1u << 23;
  //: the warp's state: list warp + 8 J, digit 16 M + g + 8 (I >> 1), channel 8 N + 2 t + (I & 1)
  State acc;
  rola::static_for<4>([&](auto J) {
    rola::static_for<2>([&](auto M) {
      rola::static_for<4>([&](auto N) {
        rola::static_for<4>([&](auto I) {
          acc[J][M][N][I] = state[((warp + kWarps * J) * kDigits + 16 * M + g + 8 * (I >> 1)) * kChannels + 8 * N +
                                  2 * t4 + (I & 1)];
        });
      });
    });
  });
  //: the lean stores' bases, a channel tile's, held
  uint32_t at[4];
  rola::static_for<4>([&](auto N) {
    at[N] = lab::held(lab::smem(sm) + 4u * t4 + 16u * unit_of(warp, N, g) + static_cast<uint32_t>(g) * kPlaneBytes);
  });
  //: the 16-byte stores' base: the lane's unit for chunk t at its residue's digit, held
  const uint32_t at16 =
      lab::held(lab::smem(sm) + 16u * unit16_of(warp, t4, g) + static_cast<uint32_t>(g) * kPlaneBytes);
  __syncthreads();
  const uint32_t two23 = reinterpret_cast<const volatile uint32_t*>(sm + kConstAt)[0],
                 bias = reinterpret_cast<const volatile uint32_t*>(sm + kConstAt)[1];
  int4 ex0 = {};
  if constexpr (FORM == 3) {
    const Lean l = lean_units(sm, warp, lane);
    lean_maxima<false>(acc, sm, l, 0, lane);
    __syncthreads();
    ex0 = lean_exps(sm, l, 0, lane);
    __syncthreads();
  }
  int dexp = 0;
  int4 last = {};
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    rola::static_for<4>([&](auto J) {
      rola::static_for<2>([&](auto M) {
        rola::static_for<4>([&](auto N) {
          rola::static_for<4>([&](auto I) { asm volatile("" : "+f"(acc[J][M][N][I])); });
        });
      });
    });
    if constexpr (FORM == 0) {
      carry_maxima(acc, sm, w, warp, lane);
      __syncthreads();
      reinterpret_cast<volatile int*>(sm + kExpsAt)[lane] = carry_digit_exp(sm, w, lane);
      dexp = reinterpret_cast<const volatile int*>(sm + kExpsAt)[lane];
      if constexpr (HALVES) {
        carry_stores<0, 1>(acc, sm, dexp, warp, lane);
        __syncthreads();
        carry_stores<1, 2>(acc, sm, dexp, warp, lane);
      } else {
        carry_stores<0, 2>(acc, sm, dexp, warp, lane);
      }
    } else if constexpr (FORM == 4) {
      const Lean l = lean_units(sm, warp, lane);
      lean_maxima<true>(acc, sm, l, w, lane);
      __syncthreads();
      const int4 ex = lean16_exps(sm, l, w, lane);
      if constexpr (HALVES) {
        lean16_stores<0, 1>(acc, l, ex, at16, two23, bias);
        __syncthreads();
        lean16_stores<1, 2>(acc, l, ex, at16, two23, bias);
      } else {
        lean16_stores<0, 2>(acc, l, ex, at16, two23, bias);
      }
      last = ex;
    } else {
      const Lean l = lean_units(sm, warp, lane);
      int4 ex = ex0;
      if constexpr (FORM != 3) {
        lean_maxima<false>(acc, sm, l, w, lane);
        __syncthreads();
        ex = lean_exps(sm, l, w, lane);
      }
      if constexpr (FORM != 2) {
        if constexpr (HALVES) {
          lean_stores<0, 1>(acc, l, ex, at);
          __syncthreads();
          lean_stores<1, 2>(acc, l, ex, at);
        } else {
          lean_stores<0, 2>(acc, l, ex, at);
        }
      }
      last = ex;
    }
    __syncthreads();
  }
  const long long t1 = clock64();
  if constexpr (FORM != 0) dexp = t4 == 0 ? last.x : t4 == 1 ? last.y : t4 == 2 ? last.z : last.w;
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
  //: CTA 0's image and exponents out: the carry's lane d digit d's, the lean's lane 4 g + t digit
  //: 16 (t >> 1) + 8 (t & 1) + g's
  if (blockIdx.x == 0) {
    for (int k = static_cast<int>(threadIdx.x); k < kImageBytes / 16; k += kThreads)
      reinterpret_cast<uint4*>(image_out)[k] = reinterpret_cast<const uint4*>(sm)[k];
    if (warp == 0) exps_out[FORM == 0 ? lane : 16 * (t4 >> 1) + 8 * (t4 & 1) + g] = dexp;
  }
}

//: the inputs from `--inputs`
struct Host {
  std::vector<float> state;
  std::vector<int> units, exps;
  std::vector<uint8_t> image, image16;
};

template <class T>
std::vector<T> file(const std::string& path, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "handoff: %s is missing\n", path.c_str());
    std::exit(1);
  }
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  return v;
}

template <int FORM, int HALVES>
void run(const lab::Bench& b, const char* variant, const float* state, const int* units, uint8_t* image, int* exps,
         const Host& h) {
  const std::string name = lab::kernel("handoff", {FORM, HALVES});
  if (FORM != 2 && FORM != 4) b.expect(name, "STS", ">=", 64);
  if (FORM == 4) b.expect(name, "STS.128", ">=", 16);
  b.expect(name, "BAR", ">=", 1);
  if (FORM >= 1) b.expect(name, "ATOMS", FORM == 3 ? ">=" : ">=", 1);
  b.units(name, "a CTA-window", static_cast<long long>(kCtas) * kWindows);
  if (FORM == 1 && !HALVES) {
    //: THE LEAN BUDGET a CTA-window (8 warps), from the design's count a warp: 128 FMNMX and 128 FMUL, 64 packs, 64
    //: stores and a key reset, 1 atomic, 4 unit loads and a key load, 8 shuffles and a REDUX, 16 key adds and 16 scale
    //: exponents and 4 exponents' (~150 integer), their selects (~60), the window's loop and barriers (~20)
    b.budget(name, "float", 8 * 256 + 64);
    b.budget(name, "pack", 8 * 64 + 16);
    b.budget(name, "shared store", 8 * 64 + 8);
    b.budget(name, "shared atomic", 8 * 1 + 1);
    b.budget(name, "shared load", 8 * 6 + 8);
    b.budget(name, "shuffle", 8 * 10 + 8);
    b.budget(name, "integer/address", 8 * 150);
    b.budget(name, "select/compare", 8 * 60);
    b.budget(name, "control", 8 * 20);
  }
  if (FORM == 4 && !HALVES) {
    //: THE 16-BYTE BUDGET a CTA-window, a warp's: 112 FMNMX (7 a list's digit row) and 128 FMUL, 64 packs, 16
    //: 16-byte stores and a key reset, the lean form's atomic, loads and shuffles; integer work 136 (60 IMNMX: 16
    //: scales' two clamps, 12 + 8 key maxima, 4 shift and 4 exponent floors; 32 IMAD: 16 key adds, 16 scale bits; 20
    //: scale differences; ~24 exponents and addresses -- the first budget's 118 priced the 16 scale-bit builds as
    //: address forms the held base removes, wrongly), its selects (~36). Binding: the ALU pipe, 292 a warp at 2
    //: cycles (112 FMNMX, 64 packs, 60 IMNMX, 35 key selects, ~21 adds and logic), ~1,170 cycles a window; issue
    //: ~1,030; the shared pipe ~830 an SM.
    b.budget(name, "float", 8 * 240 + 64);
    b.budget(name, "pack", 8 * 64 + 16);
    b.budget(name, "shared store", 8 * 17 + 8);
    b.budget(name, "shared atomic", 8 * 1 + 1);
    b.budget(name, "shared load", 8 * 6 + 8);
    b.budget(name, "shuffle", 8 * 10 + 8);
    b.budget(name, "integer/address", 8 * 136);
    b.budget(name, "select/compare", 8 * 36);
    b.budget(name, "control", 8 * 5);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(handoff<FORM, HALVES>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const lab::Samples s = b.variant(
      [&] { handoff<FORM, HALVES><<<kCtas, kThreads, kBytes>>>(state, units, image, exps, b.times); },
      [&] { return b.span(kWarps, kCtas) / kWindows; });
  if (FORM == 2) {
    b.unchecked(variant, "a bound: the maxima alone, no image formed");
  } else {
    std::vector<uint8_t> got(kImageBytes);
    std::vector<int> ge(kDigits);
    LAB_CHECK(cudaMemcpy(got.data(), image, got.size(), cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(ge.data(), exps, ge.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0, bad_e = 0;
    const std::vector<uint8_t>& want = FORM == 4 ? h.image16 : h.image;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
    for (int d = 0; d < kDigits; ++d) bad_e += ge[d] != h.exps[d];
    b.check(variant, bad == 0 && bad_e == 0,
            std::to_string(bad) + " of " + std::to_string(got.size()) + " image bytes and " + std::to_string(bad_e) +
                " of 32 exponents differ");
  }
  b.record(variant, "cycles a window", s);
}

}  // namespace handoff

int main(int argc, char** argv) {
  using namespace handoff;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Host h;
  float* state = nullptr;
  int *units = nullptr, *exps = nullptr;
  uint8_t* image = nullptr;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "handoff: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    h.state = file<float>(dir + "/state.bin", kLists * kDigits * kChannels);
    h.units = file<int>(dir + "/units.bin", kLists);
    h.exps = file<int>(dir + "/exps.bin", kDigits);
    h.image = file<uint8_t>(dir + "/image.bin", kImageBytes);
    h.image16 = file<uint8_t>(dir + "/image16.bin", kImageBytes);
    LAB_CHECK(cudaMalloc(&state, h.state.size() * 4));
    LAB_CHECK(cudaMemcpy(state, h.state.data(), h.state.size() * 4, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMalloc(&units, kLists * 4));
    LAB_CHECK(cudaMemcpy(units, h.units.data(), kLists * 4, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMalloc(&exps, kDigits * 4));
    LAB_CHECK(cudaMalloc(&image, kImageBytes));
  }
  run<0, 0>(b, "the carry's form, the image whole", state, units, image, exps, h);
  run<0, 1>(b, "the carry's form, two halves", state, units, image, exps, h);
  run<1, 0>(b, "lean, the image whole", state, units, image, exps, h);
  run<1, 1>(b, "lean, two halves", state, units, image, exps, h);
  run<2, 0>(b, "bound: lean, the maxima alone", state, units, image, exps, h);
  run<3, 0>(b, "bound: lean, the stores alone", state, units, image, exps, h);
  run<4, 0>(b, "lean, 16-byte units, the image whole", state, units, image, exps, h);
  run<4, 1>(b, "lean, 16-byte units, two halves", state, units, image, exps, h);
}
