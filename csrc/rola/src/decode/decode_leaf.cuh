#pragma once

// THE LEAF STEP: one leaf's channels, eight a lane -- its read, its full re-read or its deposit -- the one
// statement of the arithmetic both walk forms (a leaf a lane group, an atom a warp) run.
// see docs/internals/decode/decode_leaf.md

#include <cstdint>

#include "common/state_page.cuh"
#include "decode.cuh"

namespace rola {
namespace decode {

//: A leaf's lane slice in its page: eight channels' hi and lo words, its group's exponent byte, the mass word. -- see docs/internals/decode/decode_leaf.md#leaf-addr
struct LeafAddr {
  uint8_t* hi;
  uint8_t* lo;
  uint8_t* ex;
  uint32_t* mass;
};

template <int DV>
__device__ __forceinline__ LeafAddr leaf_addr(uint8_t* page, int row, int ch0) {
  using Page = state_page::Blocks<DV>;
  return LeafAddr{page + Page::kHi + (row * DV + ch0) * 2, page + Page::kLo + row * DV + ch0,
                  page + Page::kExp + row * Page::kGroups + (ch0 >> 5),
                  reinterpret_cast<uint32_t*>(page + Page::kMass) + row};
}

//: The slice as loaded: an absent atom's leaf reads as zeros; `full` loads the lo plane too; the mass is the row's first lane's.
struct LeafSlice {
  uint4 hi;
  uint2 lo;
  int e;
  float mass;
};

__device__ __forceinline__ LeafSlice load_leaf(const LeafAddr& a, bool present, bool full,
                                               bool row_lead) {
  LeafSlice s{make_uint4(0u, 0u, 0u, 0u), make_uint2(0u, 0u), 0, 0.0f};
  if (present) {
    s.hi = *reinterpret_cast<const uint4*>(a.hi);
    if (full) s.lo = *reinterpret_cast<const uint2*>(a.lo);
    s.e = *reinterpret_cast<const int8_t*>(a.ex);
    if (row_lead) s.mass = __uint_as_float(*a.mass);
  }
  return s;
}

//: A leaf this step only reads: the hi plane at `2^(e + 8)`, into the lane's eight partials.
__device__ __forceinline__ void read_hi(const LeafSlice& s, float ar, bool row_lead,
                                        float (&acc)[8], float& acc_m) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    acc[c] =
        fmaf(static_cast<float>(state_page::hi8(s.hi, c)), ar * state_page::pow2(s.e + 8), acc[c]);
  }
  if (row_lead) acc_m += ar * s.mass;
}

//: A written leaf re-read without depositing (a gated replay's third outcome): its full stored value.
__device__ __forceinline__ void read_full(const LeafSlice& s, float ar, bool row_lead,
                                          float (&acc)[8], float& acc_m) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    acc[c] = fmaf(static_cast<float>(state_page::mantissa8(s.hi, s.lo, c)),
                  ar * state_page::pow2(s.e), acc[c]);
  }
  if (row_lead) acc_m += ar * s.mass;
}

//: THE DEPOSIT, UPDATE FIRST THEN READ THE STORED VALUE: fp32 update, the group's exponent over its four lanes -- see docs/internals/decode/decode_leaf.md#deposit
template <bool DECAY>
__device__ __forceinline__ void deposit_leaf(const LeafSlice& s, const LeafAddr& a, float ar,
                                             float aw, float rate, float share, const float* v_tok,
                                             int ch0, bool present, bool row_lead, bool exp_lead,
                                             float (&acc)[8], float& acc_m) {
  float keep = 1.0f;
  if (DECAY) keep = expf(fminf(aw * share * log1pf(-rate), kDecodeMaxExponent));
  const float kept = keep * state_page::pow2(s.e);
  const float4 v0 = *reinterpret_cast<const float4*>(v_tok + ch0);
  const float4 v1 = *reinterpret_cast<const float4*>(v_tok + ch0 + 4);
  const float vt[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  float sv[8];
  float amax = 0.0f;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    sv[c] = fmaf(static_cast<float>(state_page::mantissa8(s.hi, s.lo, c)), kept, aw * vt[c]);
    amax = fmaxf(amax, fabsf(sv[c]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const int en = state_page::group_exponent(amax);
  int m[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) m[c] = state_page::mantissa(sv[c], en);
  if (present) {
    *reinterpret_cast<uint4*>(a.hi) = state_page::pack_hi8(m);
    *reinterpret_cast<uint2*>(a.lo) = state_page::pack_lo8(m);
    if (exp_lead) *reinterpret_cast<int8_t*>(a.ex) = static_cast<int8_t>(en);
  }
  const float arp = ar * state_page::pow2(en);
#pragma unroll
  for (int c = 0; c < 8; ++c) acc[c] = fmaf(static_cast<float>(m[c]), arp, acc[c]);
  if (row_lead) {
    const float mass = keep * s.mass + aw;
    if (present) *a.mass = __float_as_uint(mass);
    acc_m += ar * mass;
  }
}

}  // namespace decode
}  // namespace rola
