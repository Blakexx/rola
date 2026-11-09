// measure/lab/experiments/drain.cu -- QUESTION: what does the readout's drain -- a pass's panel words added into the
// head's `num` by `red.global.add` at each slot's token, the panel cleared for the next pass's adds -- cost a
// CTA-window, in today's form and in forms that pair conflict-free slots, clear by exchange or by a 16-byte sweep,
// and beside a stream's copies? METHOD: 80 CTAs, box b of one head (`drain.inputs.py`: `read_stream`'s three cells,
// each box-window's union in rank order); each pass every thread fills its slot's 16 panel words (four 16-byte
// stores) and its token, a CTA barrier, the drain, a barrier. A warp drains its 32 slots, two a `red.global` (a
// lane a word: lanes 0-15 one slot's 16 words, 16-31 another's), each request a token's half line of `num` (d_v 64:
// two channel tiles a 128-byte row). The forms differ in which two slots a request pairs and how a word is cleared:
// FORM 0 slots (s, s + 1) at the panel's 20-word pitch (4 banks shared: 2 wavefronts an access), a load and a 4-byte
// store; 1 slots (s, s + 4) (16 banks apart: 1 wavefront); 2 as 1 with one `atom.shared.exch` reading and clearing; 3
// as 1 with no clear in the drain, a 16-byte sweep after the barrier; 4 as 1 with a warp's `cp.async` of 512 bytes
// every 4 adds (a stream beside it); 7 the same every add; 6 as 1 with no clear (a bound); 5 the fill alone (a
// bound); 8 as 3 with the warp's groups of 4 requests past its slots in the union not issued (a warp-uniform count);
// 9 as 8 into an fp32 `num`: a slot's 32 channels a request (its channel tile's full line), each lane one channel from
// its pair word's half, converted exactly by the magic offset and scaled by the slot's unit 2^e; 10 as 9 converted by
// an offset-free half (a lane's add and shift) OR'd under the magic and one FFMA by the slot's (2^e, bias) pair; 11
// as 10 into an s32 `num` by integer adds: the half sign-extended and shifted left by the slot's exponent; 12 as 8 (two
// slots a request, a lane a word) into an f16-pair `num`: each word's halves converted exactly, scaled by the slot's
// (scale, bias) pair, packed to f16 and added by `red.global.add.noftz.f16x2` (its panel filled with 5-bit halves, so
// every f16 sum is an exact integer and the check stays word for word); 13 as 12 with no conversion (the raw word
// added as an f16 pair: L2's float adds alone); 14 as 12's conversion added as an integer (the conversion alone).
// Each draining form's `num` is checked word for word against the mirror's; a row prices a line request from form 8.
// A sample is the CTA's span a window. see docs/lab/drain.md
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "lab.cuh"

namespace drain_lab {

constexpr int kWarps = 8, kThreads = 32 * kWarps, kSlots = 256, kPitch = 20, kWords = 16,
              kGroups = 8;
//: THE LAYOUT: the panel (256 slots of 80 bytes), each slot's token, unit exponent and (scale, bias) pair, each warp's
//: 4 KB copy ring
constexpr int kPanel = 0, kTok = kSlots * kPitch * 4, kExp = kTok + 4 * kSlots,
              kScale = kExp + 4 * kSlots, kRing = kScale + 8 * kSlots,
              kBytes = kRing + kWarps * 4096;
//: the copies' table: 1 MB, L2-resident beside the head's `num`
constexpr uint32_t kTable = 1u << 20;

struct In {
  const uint16_t* counts;
  const uint16_t* tokens;
  uint32_t* num;
  float* numf;
  uint32_t* num16;
  const uint8_t* table;
  int windows;
};

__device__ __forceinline__ uint32_t pattern(int w, int b, int s, int j) {
  return static_cast<uint32_t>(w) * 2654435761u + static_cast<uint32_t>(b) * 40503u
         + static_cast<uint32_t>(s) * 16u + static_cast<uint32_t>(j);
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}

__device__ __forceinline__ uint32_t exch0(uint32_t a) {
  uint32_t v;
  asm volatile("atom.shared.exch.b32 %0, [%1], 0;" : "=r"(v) : "r"(a) : "memory");
  return v;
}

//: an add into `num`: unpredicated -- ptxas puts a predicated global add in a branch and a convergence region (BSSY,
//: BRA, BSYNC around it), so a slot past the pass's union adds zero at its CTA's own dummy row past the head's tokens
//: (a shared dummy row would be one line every CTA's dead slots serialize on)
__device__ __forceinline__ void red(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void redf(float* at, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(at), "f"(v) : "memory");
}

__device__ __forceinline__ void redh2(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void copy512(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

//: THE DRAIN of a pass's `m` slots by FORM (header): a warp's 16 requests, each slot's word and token loaded four
//: requests ahead of their adds
template <int FORM>
__device__ __forceinline__ void drain_pass(const In& in, uint32_t sm, int m, int ct, uint32_t dummy,
                                           int warp, int lane, int& copies,
                                           const uint8_t* table_lane, uint32_t ring_lane) {
  const int j = lane & 15, h = lane >> 4;
  uint32_t* const row = in.num + ct * kWords + j;
  //: FORM 8 issues the groups holding a live slot (a group q its slots 32 w + 8 q .. + 7), the rest all of them
  const int groups = FORM == 8 ? min(4, max(0, (m - 32 * warp + 7) / 8)) : 4;
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    uint32_t v[4], tok[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 4 * q + i;
      const int s = FORM == 0 ? 32 * warp + h + 2 * k : 32 * warp + 8 * (k >> 2) + (k & 3) + 4 * h;
      const uint32_t at = sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4);
      if constexpr (FORM == 2) {
        v[i] = exch0(at);
      } else {
        v[i] = lds32(at);
        if constexpr (FORM == 0 || FORM == 1 || FORM == 4 || FORM == 7) sts32(at, 0u);
      }
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? v[i] : 0u;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      red(row + static_cast<size_t>(tok[i]) * (2 * kWords), v[i]);
      if (FORM == 7 || (FORM == 4 && i == 0)) {
        copy512(ring_lane + (copies & 7) * 512,
                table_lane + static_cast<size_t>(copies & 255) * 512);
        ++copies;
      }
    }
  }
  if constexpr (FORM == 4 || FORM == 7)
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

//: FORM 9's drain of a pass's `m` slots: a warp's groups of 8 slots holding a live one, a slot a request (lane `l` its
//: channel `l`: word `l & 15`'s low half below 16, its high half above), each slot's word, token and unit exponent
//: loaded for the group before its adds
__device__ __forceinline__ void drain_pass_f32(const In& in, uint32_t sm, int m, int ct,
                                               uint32_t dummy, int warp, int lane) {
  const int j = lane & 15, high = lane >> 4;
  float* const row = in.numf + ct * 32 + lane;
  const int groups = min(4, max(0, (m - 32 * warp + 7) / 8));
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    float v[8];
    uint32_t tok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int s = 32 * warp + 8 * q + i;
      const uint32_t w = lds32(sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4));
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      const int e = static_cast<int>(lds32(sm + kExp + 4 * s));
      const int lo = static_cast<int16_t>(w & 0xffffu);
      const int x = high ? static_cast<int16_t>((w - static_cast<uint32_t>(lo)) >> 16) : lo;
      const float f =
          (__int_as_float(0x4B400000 + x) - 12582912.0f) * __int_as_float((127 + e) << 23);
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? f : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) redf(row + static_cast<size_t>(tok[i]) * 64, v[i]);
  }
}

//: FORM 10's drain: as `drain_pass_f32`, a lane's half made offset-free by a lane-constant add and shift (the high half
//: `(w + 0x8000) >> 16`, which takes the low half's borrow back), biased by 0x8000 under the magic 0x4B400000, and one
//: FFMA by the slot's (2^e, -(12582912 + 32768) 2^e) pair
__device__ __forceinline__ void drain_pass_f32b(const In& in, uint32_t sm, int m, int ct,
                                                uint32_t dummy, int warp, int lane) {
  const int j = lane & 15;
  const uint32_t add = lane >> 4 ? 0x8000u : 0u, shift = lane >> 4 ? 16u : 0u;
  float* const row = in.numf + ct * 32 + lane;
  const int groups = min(4, max(0, (m - 32 * warp + 7) / 8));
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    float v[8];
    uint32_t tok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int s = 32 * warp + 8 * q + i;
      const uint32_t w = lds32(sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4));
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      float2 sb;
      asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                   : "=f"(sb.x), "=f"(sb.y)
                   : "r"(sm + kScale + 8 * s));
      const uint32_t t = (w + add) >> shift;
      const float f = fmaf(__uint_as_float(((t ^ 0x8000u) & 0xffffu) | 0x4B400000u), sb.x, sb.y);
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? f : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) redf(row + static_cast<size_t>(tok[i]) * 64, v[i]);
  }
}

//: FORM 11's drain: as `drain_pass_f32b` with the lane's half sign-extended, shifted by the slot's exponent and added
//: as a 32-bit integer
__device__ __forceinline__ void drain_pass_s32(const In& in, uint32_t sm, int m, int ct,
                                               uint32_t dummy, int warp, int lane) {
  const int j = lane & 15;
  const uint32_t add = lane >> 4 ? 0x8000u : 0u, shift = lane >> 4 ? 16u : 0u;
  uint32_t* const row = reinterpret_cast<uint32_t*>(in.numf) + ct * 32 + lane;
  const int groups = min(4, max(0, (m - 32 * warp + 7) / 8));
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    uint32_t v[8], tok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int s = 32 * warp + 8 * q + i;
      const uint32_t w = lds32(sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4));
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      const uint32_t e = lds32(sm + kExp + 4 * s);
      const uint32_t t = (w + add) >> shift;
      const uint32_t x = static_cast<uint32_t>(static_cast<int>(static_cast<int16_t>(t & 0xffffu)))
                         << e;
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? x : 0u;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) red(row + static_cast<size_t>(tok[i]) * 64, v[i]);
  }
}

//: FORM 12's drain: FORM 8's requests (two slots a request, lane `l` word `l & 15` of slot `l >> 4`'s), each word's
//: halves made offset-free under the magic, one FFMA each by the slot's (scale, bias) pair, packed to an f16 pair
__device__ __forceinline__ void drain_pass_h2(const In& in, uint32_t sm, int m, int ct,
                                              uint32_t dummy, int warp, int lane) {
  const int j = lane & 15, h = lane >> 4;
  uint32_t* const row = in.num16 + ct * kWords + j;
  const int groups = min(4, max(0, (m - 32 * warp + 7) / 8));
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    uint32_t v[4], tok[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int s = 32 * warp + 8 * q + i + 4 * h;
      const uint32_t w = lds32(sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4));
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      float2 sb;
      asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                   : "=f"(sb.x), "=f"(sb.y)
                   : "r"(sm + kScale + 8 * s));
      const float lo = fmaf(__uint_as_float(((w ^ 0x8000u) & 0xffffu) | 0x4B400000u), sb.x, sb.y);
      const float hi = fmaf(
          __uint_as_float(((((w + 0x8000u) >> 16) ^ 0x8000u) & 0xffffu) | 0x4B400000u), sb.x, sb.y);
      const __half2 pair = __floats2half2_rn(lo, hi);
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? *reinterpret_cast<const uint32_t*>(&pair) : 0u;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) redh2(row + static_cast<size_t>(tok[i]) * (2 * kWords), v[i]);
  }
}

//: FORMS 13 and 14 split FORM 12's cost: 13 adds each raw word as an f16 pair (L2's float adds, no conversion), 14
//: converts as 12 and adds the packed bits as an integer (the conversion, integer adds)
template <int FORM>
__device__ __forceinline__ void drain_pass_split(const In& in, uint32_t sm, int m, int ct,
                                                 uint32_t dummy, int warp, int lane) {
  const int j = lane & 15, h = lane >> 4;
  uint32_t* const row = in.num16 + ct * kWords + j;
  const int groups = min(4, max(0, (m - 32 * warp + 7) / 8));
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    uint32_t v[4], tok[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int s = 32 * warp + 8 * q + i + 4 * h;
      const uint32_t w = lds32(sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4));
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      uint32_t x = w;
      if constexpr (FORM == 14) {
        float2 sb;
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                     : "=f"(sb.x), "=f"(sb.y)
                     : "r"(sm + kScale + 8 * s));
        const float lo = fmaf(__uint_as_float(((w ^ 0x8000u) & 0xffffu) | 0x4B400000u), sb.x, sb.y);
        const float hi =
            fmaf(__uint_as_float(((((w + 0x8000u) >> 16) ^ 0x8000u) & 0xffffu) | 0x4B400000u), sb.x,
                 sb.y);
        const __half2 pair = __floats2half2_rn(lo, hi);
        x = *reinterpret_cast<const uint32_t*>(&pair);
      }
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? x : 0u;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (FORM == 13) {
        redh2(row + static_cast<size_t>(tok[i]) * (2 * kWords), v[i]);
      } else {
        red(row + static_cast<size_t>(tok[i]) * (2 * kWords), v[i]);
      }
    }
  }
}

template <int FORM, int SET>
__global__ void __launch_bounds__(kThreads, 1) drain(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  const int b = static_cast<int>(blockIdx.x), ct = b % 2, g = b / 16;
  const uint32_t sm = lab::smem(smem);
  const uint8_t* const table_lane =
      in.table + 16 * lane + static_cast<size_t>(warp) * (kTable / kWarps);
  const uint32_t ring_lane = sm + kRing + warp * 4096 + 16 * lane;
  int copies = 0;
  //: this CTA's dummy row, past the head's 512 x windows tokens
  const uint32_t dummy = 512u * static_cast<uint32_t>(in.windows) + static_cast<uint32_t>(b);
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < in.windows; ++w) {
    const int n = in.counts[w * kGroups + g];
    const uint16_t* const toks = in.tokens + (static_cast<size_t>(w) * kGroups + g) * 512;
#pragma unroll 1
    for (int p = 0; p * kSlots < n; ++p) {
      const int m = min(kSlots, n - p * kSlots);
      //: the pass's fill: a thread its slot's 16 words and its token
      if (tid < m) {
        const int s = p * kSlots + tid;
        const uint32_t at = sm + kPanel + static_cast<uint32_t>(tid * kPitch * 4);
        constexpr uint32_t keep = FORM >= 12 ? 0x001f001fu : 0xffffffffu;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          sts128(at + 16 * q, pattern(w, b, s, 4 * q) & keep, pattern(w, b, s, 4 * q + 1) & keep,
                 pattern(w, b, s, 4 * q + 2) & keep, pattern(w, b, s, 4 * q + 3) & keep);
        sts32(sm + kTok + 4 * tid, 512u * static_cast<uint32_t>(w) + toks[s]);
        if constexpr (FORM == 9 || FORM == 11) sts32(sm + kExp + 4 * tid, 0u);
        if constexpr (FORM == 10 || FORM == 12 || FORM == 14) {
          asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(sm + kScale + 8 * tid), "f"(1.0f),
                       "f"(-12615680.0f)
                       : "memory");
        }
      }
      __syncthreads();
      if constexpr (FORM == 9) {
        drain_pass_f32(in, sm, m, ct, dummy, warp, lane);
      } else if constexpr (FORM == 10) {
        drain_pass_f32b(in, sm, m, ct, dummy, warp, lane);
      } else if constexpr (FORM == 11) {
        drain_pass_s32(in, sm, m, ct, dummy, warp, lane);
      } else if constexpr (FORM == 12) {
        drain_pass_h2(in, sm, m, ct, dummy, warp, lane);
      } else if constexpr (FORM == 13 || FORM == 14) {
        drain_pass_split<FORM>(in, sm, m, ct, dummy, warp, lane);
      } else if constexpr (FORM != 5) {
        drain_pass<FORM>(in, sm, m, ct, dummy, warp, lane, copies, table_lane, ring_lane);
      }
      __syncthreads();
      //: FORM 3's clear: a thread its slot's 16 words by four 16-byte stores
      if constexpr (FORM == 3 || FORM >= 8) {
        const uint32_t at = sm + kPanel + static_cast<uint32_t>(tid * kPitch * 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) sts128(at + 16 * q, 0u, 0u, 0u, 0u);
      }
    }
  }
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
}

struct Host {
  std::vector<uint16_t> counts, tokens;
  std::vector<uint32_t> num;
  std::vector<float> numf, num16;
  int ctas = 0, windows = 0;
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "drain: %s is missing\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
  return v;
}

template <class T>
std::vector<T> as(const std::vector<uint8_t>& b) {
  return std::vector<T>(reinterpret_cast<const T*>(b.data()),
                        reinterpret_cast<const T*>(b.data() + b.size()));
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

//: an f16's value (normal or zero; the check's sums are integers under 2048)
float half_value(uint16_t x) {
  const int e = (x >> 10) & 31, m = x & 1023;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                         : std::ldexp(static_cast<float>(1024 + m), e - 25);
  return x & 0x8000 ? -v : v;
}

template <int FORM, int SET>
double run(const lab::Bench& b, const std::string& variant, const In& in, const Host& h) {
  const std::string name = lab::kernel("drain_lab::drain", {FORM, SET});
  if (FORM != 5) b.expect(name, "RED.E.ADD", ">=", 4);
  if (FORM == 2) b.expect(name, "ATOMS.EXCH", ">=", 4);
  if (FORM == 4 || FORM == 7) b.expect(name, "LDGSTS", ">=", 1);
  //: fixed counts (the expect pass runs before any input is read): 80 boxes, the cell's 64 or 32 windows
  b.units(name, "a CTA-window", 80LL * (SET == 0 ? 64 : 32));
  b.exempt(name, "a rate probe of the drain's forms; no MMAs");
  if (b.expect_only) return 0.0;
  LAB_CHECK(
      cudaFuncSetAttribute(drain<FORM, SET>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] { drain<FORM, SET><<<h.ctas, kThreads, kBytes>>>(in, b.times); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (FORM == 5) {
    b.unchecked(variant, "a bound: the fill alone, nothing drained");
  } else if (FORM == 13 || FORM == 14) {
    b.unchecked(variant,
                "a split of form 12's cost: its adds or its conversion, the sums meaningless");
  } else if (FORM == 12) {
    LAB_CHECK(cudaMemset(in.num16, 0, h.num16.size() * 2));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(h.num16.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.num16, got.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += half_value(got[i]) != h.num16[i];
    b.check(variant, bad == 0,
            std::to_string(got.size() - bad) + " of " + std::to_string(got.size())
                + " f16 num halves exact");
  } else if (FORM == 9 || FORM == 10 || FORM == 11) {
    LAB_CHECK(cudaMemset(in.numf, 0, h.numf.size() * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(h.numf.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.numf, got.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      int32_t as_int;
      std::memcpy(&as_int, &got[i], 4);
      bad += (FORM == 11 ? static_cast<float>(as_int) : got[i]) != h.numf[i];
    }
    b.check(variant, bad == 0,
            std::to_string(got.size() - bad) + " of " + std::to_string(got.size())
                + " fp32 num slots exact");
  } else {
    LAB_CHECK(cudaMemset(in.num, 0, h.num.size() * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(h.num.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.num, got.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.num[i];
    b.check(variant, bad == 0,
            std::to_string(got.size() - bad) + " of " + std::to_string(got.size())
                + " num words exact");
  }
  b.record(variant, "cycles a window", s);
  return s.median();
}

//: the half-line requests a CTA-window FORM 8 issues (a warp's groups holding a live slot, 4 adds of two lines each),
//: the mean over the cell's CTA-windows
double requests_skipping(const Host& h) {
  double n = 0;
  for (int w = 0; w < h.windows; ++w)
    for (int b = 0; b < h.ctas; ++b) {
      const int u = h.counts[w * kGroups + b / 16];
      for (int p = 0; p * kSlots < u; ++p) {
        const int m = std::min(kSlots, u - p * kSlots);
        for (int warp = 0; warp < kWarps; ++warp)
          n += 8.0 * std::min(4, std::max(0, (m - 32 * warp + 7) / 8));
      }
    }
  return n / (static_cast<double>(h.windows) * h.ctas);
}

//: the global path's price of a line request with every SM draining (`common/hardware.json`'s `global.red_request`):
//: the skipping form's cycles less the fill's, over its requests a CTA-window
void red_request(const lab::Bench& b, const std::string& label, double skipping, double fill,
                 const Host& h) {
  if (b.expect_only) return;
  const std::string row = label + "cycles a line request, every SM draining";
  b.unchecked(row, "the difference of two checked forms' medians over the requests the inputs fix");
  b.record(row, "cycles a request", lab::Samples{{(skipping - fill) / requests_skipping(h)}});
}

template <int SET>
void cell(const lab::Bench& b, const char* label, const In& in, const Host& h) {
  const std::string l = label;
  const double fill = run<5, SET>(b, l + "bound: the fill alone", in, h);
  run<0, SET>(b, l + "slots (s, s + 1), load and 4-byte clear", in, h);
  run<1, SET>(b, l + "slots (s, s + 4), load and 4-byte clear", in, h);
  run<2, SET>(b, l + "slots (s, s + 4), exchange", in, h);
  run<3, SET>(b, l + "slots (s, s + 4), load; 16-byte clear sweep", in, h);
  run<6, SET>(b, l + "bound: slots (s, s + 4), no clear", in, h);
  run<4, SET>(b, l + "slots (s, s + 4) beside copies, 512 bytes every 4 adds", in, h);
  run<7, SET>(b, l + "slots (s, s + 4) beside copies, 512 bytes every add", in, h);
  const double skipping = run<8, SET>(
      b, l + "slots (s, s + 4), load; 16-byte clear sweep; groups past the union skipped", in, h);
  red_request(b, l, skipping, fill, h);
  run<9, SET>(b, l + "fp32: a slot's 32 channels a request, sweep, groups past the union skipped",
              in, h);
  run<10, SET>(b, l + "fp32, the offset conversion and one FFMA a channel", in, h);
  run<11, SET>(b, l + "s32: a slot's 32 channels a request by integer adds", in, h);
  run<12, SET>(b, l + "f16 pairs: two slots a request, each word converted and packed", in, h);
  run<13, SET>(b, l + "split: raw words added as f16 pairs (no conversion)", in, h);
  run<14, SET>(b, l + "split: the conversion, added as integers", in, h);
}

}  // namespace drain_lab

int main(int argc, char** argv) {
  using namespace drain_lab;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Host h[3];
  In in[3] = {};
  const char* sub[3] = {"", "/k16", "/c16"};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "drain: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    uint8_t* table = nullptr;
    LAB_CHECK(cudaMalloc(&table, kTable));
    LAB_CHECK(cudaMemset(table, 0x5a, kTable));
    for (int k = 0; k < 3; ++k) {
      const std::vector<int> head = as<int>(file(dir + sub[k] + "/head.bin"));
      h[k].ctas = head[1];
      h[k].windows = head[2];
      h[k].counts = as<uint16_t>(file(dir + sub[k] + "/counts.bin"));
      h[k].tokens = as<uint16_t>(file(dir + sub[k] + "/tokens.bin"));
      h[k].num = as<uint32_t>(file(dir + sub[k] + "/num.bin"));
      h[k].numf = as<float>(file(dir + sub[k] + "/numf.bin"));
      h[k].num16 = as<float>(file(dir + sub[k] + "/num16.bin"));
      uint32_t* num = nullptr;
      const size_t words = h[k].num.size() + static_cast<size_t>(h[k].ctas) * 2 * kWords;
      LAB_CHECK(cudaMalloc(&num, words * 4));
      LAB_CHECK(cudaMemset(num, 0, words * 4));
      float* numf = nullptr;
      const size_t slots = h[k].numf.size() + static_cast<size_t>(h[k].ctas) * 64;
      LAB_CHECK(cudaMalloc(&numf, slots * 4));
      LAB_CHECK(cudaMemset(numf, 0, slots * 4));
      uint32_t* num16 = nullptr;
      const size_t words16 = h[k].num16.size() / 2 + static_cast<size_t>(h[k].ctas) * 2 * kWords;
      LAB_CHECK(cudaMalloc(&num16, words16 * 4));
      LAB_CHECK(cudaMemset(num16, 0, words16 * 4));
      in[k] = In{up(h[k].counts), up(h[k].tokens), num, numf, num16, table, h[k].windows};
    }
  }
  cell<0>(b, "alt-k4: ", in[0], h[0]);
  cell<1>(b, "alt-k16: ", in[1], h[1]);
  cell<2>(b, "cohort-k16: ", in[2], h[2]);
}
