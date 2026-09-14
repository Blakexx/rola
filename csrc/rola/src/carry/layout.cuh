// csrc/rola/src/carry/layout.cuh -- A SIDE'S LAYOUT: how its carve order's leaves fall
// against the sixteen-leaf box, derived once on the host and handed to the kernel in the
// parameter block, so every field is a constant-bank operand and none a register.
// See docs/internals/carry/carry_kernel.md#layout
#pragma once
#include "common/geom.cuh"

namespace rola::carry {

//: THE BOX'S ROUTING FOOTPRINT, in amplitude columns, and one level's base in it.
//: -- see docs/internals/carry/carry_kernel.md#factor-rows
constexpr int span_total(int D, int bc) {
  int total = 0;
  for (int l = 0; l < D; ++l) total += owner_span(D, bc, l);
  return total;
}

constexpr int span_base(int D, int bc, int l) {
  int base = 0;
  for (int i = 0; i < l; ++i) base += owner_span(D, bc, i);
  return base;
}

//: A SIDE'S LAYOUT: its leaves in ITS carve order, rank 0 innermost; a box is sixteen
//: consecutive leaves, the low four bits of a leaf its POSITION. A level's digit lies within
//: the position bits (INNER), above them (OUTER: a scalar a box), or across bit four
//: (STRADDLING: its high part the box's CLASS). Indexed with compile-time levels only.
//: -- see docs/internals/carry/layout.md
struct SideLayout {
  int shift[kGeomMaxLevels];   //: level -> the side-order leaf's bit offset of its digit
  int bits[kGeomMaxLevels];    //: level -> its digit's width
  int dmask[kGeomMaxLevels];   //: level -> (1 << bits) - 1
  int bshift[kGeomMaxLevels];  //: level -> max(shift - 4, 0): the digit's offset in the box index
  int rshift[kGeomMaxLevels];  //: level -> the digit's bit offset in the read order
  int lshift
      [kGeomMaxLevels];  //: level -> the digit's bit offset in the owner-local canonical index
  int inner, outer;      //: the inner and outer levels, bit sets
  int straddle;          //: the straddling level, or -1
  int cbits;             //: straddle: the class bits (the digit's high part)
  int sshift;            //: straddle: the level's bit offset (its position part is 4 - sshift wide)
  int srow;              //: straddle: the level's first digit row in the box words
  int single_inner, single_outer;  //: the one inner / outer level, or -1 when composed or none
  int inner_row, outer_row;        //: their first digit rows in the box words
  //: THE INNER AND OUTER LEVELS LISTED, for the head's uniform loops: an inner level's first
  //: digit row and its digit count; an outer level's first digit row, its digit's offset in
  //: the box index and its digit mask.
  int n_inner, n_outer;
  int in_row[kGeomMaxLevels], in_span[kGeomMaxLevels];
  int out_row[kGeomMaxLevels], out_bshift[kGeomMaxLevels], out_mask[kGeomMaxLevels];

  __host__ __device__ __forceinline__ int digit_of_pos(int l, int pos) const {
    return (pos >> shift[l]) & dmask[l];
  }

  __host__ __device__ __forceinline__ int digit_of_box(int l, int box) const {
    return (box >> bshift[l]) & dmask[l];
  }

  //: the straddling level's digit for position `pos` of box `box`.
  __host__ __device__ __forceinline__ int straddle_digit(int box, int pos) const {
    return ((box & ((1 << cbits) - 1)) << (kAtomLeavesBits - sshift)) | (pos >> sshift);
  }

  //: the read order's index of this side's leaf `v`.
  template <int D>
  __host__ __device__ __forceinline__ int plane_row(int v) const {
    int row = 0;
#pragma unroll
    for (int l = 0; l < D; ++l) row += ((v >> shift[l]) & dmask[l]) << rshift[l];
    return row;
  }

  //: the owner-local canonical index of this side's leaf `v`: its page and row.
  template <int D>
  __host__ __device__ __forceinline__ int local(int v) const {
    int row = 0;
#pragma unroll
    for (int l = 0; l < D; ++l) row += ((v >> shift[l]) & dmask[l]) << lshift[l];
    return row;
  }
};

//: THE DERIVATION, on the host: side `s` laid against the read order `r`.
inline SideLayout make_side_layout(int D, int BC, const int* srank, const int* rrank) {
  SideLayout L{};
  L.inner = 0;
  L.outer = 0;
  L.straddle = -1;

  for (int l = 0; l < D; ++l) {
    int sh = 0, rsh = 0;
    for (int m = 0; m < D; ++m) {
      if (srank[m] < srank[l]) sh += owner_span_bits(D, BC, m);
      if (rrank[m] < rrank[l]) rsh += owner_span_bits(D, BC, m);
    }
    L.shift[l] = sh;
    L.bits[l] = owner_span_bits(D, BC, l);
    L.dmask[l] = (1 << L.bits[l]) - 1;
    L.bshift[l] = sh > kAtomLeavesBits ? sh - kAtomLeavesBits : 0;
    L.rshift[l] = rsh;
    L.lshift[l] = owner_run_shift(D, BC, l);
    if (sh + L.bits[l] <= kAtomLeavesBits)
      L.inner |= 1 << l;
    else if (sh >= kAtomLeavesBits)
      L.outer |= 1 << l;
    else {
      L.straddle = l;
      L.cbits = sh + L.bits[l] - kAtomLeavesBits;
      L.sshift = sh;
      L.srow = span_base(D, BC, l);
    }
  }

  for (int l = D; l < kGeomMaxLevels; ++l) {
    L.shift[l] = L.bits[l] = L.dmask[l] = L.bshift[l] = L.rshift[l] = L.lshift[l] = 0;
  }

  const auto only = [&](int set) {
    int n = 0, which = -1;
    for (int l = 0; l < D; ++l)
      if ((set >> l) & 1) {
        ++n;
        which = l;
      }
    return n == 1 ? which : -1;
  };
  L.n_inner = 0;
  L.n_outer = 0;

  for (int i = 0; i < kGeomMaxLevels; ++i) {
    L.in_row[i] = L.in_span[i] = L.out_row[i] = L.out_bshift[i] = L.out_mask[i] = 0;
  }

  for (int l = 0; l < D; ++l) {
    if ((L.inner >> l) & 1) {
      L.in_row[L.n_inner] = span_base(D, BC, l);
      L.in_span[L.n_inner] = 1 << L.bits[l];
      ++L.n_inner;
    } else if ((L.outer >> l) & 1) {
      L.out_row[L.n_outer] = span_base(D, BC, l);
      L.out_bshift[L.n_outer] = L.bshift[l];
      L.out_mask[L.n_outer] = L.dmask[l];
      ++L.n_outer;
    }
  }
  L.single_inner = only(L.inner);
  L.single_outer = L.straddle < 0 ? only(L.outer) : -1;
  L.inner_row = L.single_inner >= 0 ? span_base(D, BC, L.single_inner) : 0;
  L.outer_row = L.single_outer >= 0 ? span_base(D, BC, L.single_outer) : 0;

  return L;
}

}  // namespace rola::carry
