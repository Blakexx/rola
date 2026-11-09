// measure/lab/experiments/read_image.cuh -- THE READOUT LAB'S SHARED PARTS: the layout the facts pass's entries are
// emitted against, a lane's constants, the state's fragments, the steps of both forms and the dense cell's, and on the
// host the lists, the inputs' upload, the tensor floors and the exact check -- for `read_image.cu`, `read_w1024.cu`
// and `image_ring.cu`. see docs/internals/tools/lab.md#read-image
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lab.cuh"

namespace read_image {

constexpr int kWarps = 8, kThreads = 32 * kWarps;
//: the box's channels: 32, or 16 where `fold_publish.cu` builds this file (the parts that assume 32 drop out)
#ifndef READ_IMAGE_CHANNELS
#define READ_IMAGE_CHANNELS 32
#endif
//: the box's read digits: 32, or 16 where `digit_box.cu` builds this file (2 a warp)
#ifndef READ_IMAGE_DIGITS
#define READ_IMAGE_DIGITS 32
#endif
//: the box's lists: 32, or 16 where `list_box.cu` builds this file (the read rows 32 bytes)
#ifndef READ_IMAGE_LISTS
#define READ_IMAGE_LISTS 32
#endif
constexpr int kLists = READ_IMAGE_LISTS, kDigits = READ_IMAGE_DIGITS, kChannels = READ_IMAGE_CHANNELS;
//: a readout pass's slots: 256 (sm_86's), or 512 (sm_80's) where `read_w1024.cu` builds this file for W = 1024
#ifndef READ_IMAGE_SLOTS
#define READ_IMAGE_SLOTS 256
#endif
constexpr int kListCap = 512, kListAt = 8, kSlots = READ_IMAGE_SLOTS, kWords = kChannels / 2;
//: a panel row's pitch in words, its runs on distinct banks under the facts pass's residue placement: 20 for 32
//: channels, 12 for 16
constexpr int kPanelPitch = kChannels == 32 ? 20 : 12;
//: the image's planes (a plane's access pattern is all a digit's reads see) and a digit's entries staged: 16 and 16, or
//: 4 and 32 beside a 512-slot pass (so the counts fit sm_86)
constexpr int kPlaneBytes = kLists * kChannels * 2, kRowBytes = kLists * 2;
constexpr int kPlanes = (kSlots == 512 ? 8192 : 32768) / kPlaneBytes, kHead = kSlots == 512 ? 32 : 16;
//: THE LAYOUT the entries are emitted against (`read_image.inputs.py` mirrors it): image, pool, the lanes' pad bases
//: (128 bytes, never themselves touched), 8 dummy panel rows (a pad's add lands on its residue's, the banks a real row
//: of that residue would use), panel, heads
constexpr int kImage = 0, kPool = kImage + kPlanes * kPlaneBytes, kPads = kPool + kSlots * kRowBytes;
constexpr int kPanel = kPads + 128 + 8 * kPanelPitch * 4, kHeads = kPanel + kSlots * kPanelPitch * 4;
//: a warp's heads for a window, double-buffered: its digits' first entries (4, or 2 in a 16-digit box)
constexpr int kHeadBytes = kDigits / kWarps * kHead * 8, kBytes = kHeads + 2 * kWarps * kHeadBytes;
//: the dense cell's coefficient times the unit
constexpr float kUnit = 256.0f;

struct Lists {
  const uint2* entries;
  const uint8_t* heads;
  const uint16_t* count;
  int windows, width, groups;
  uint32_t* out;
  uint32_t* panel_out;
};

#if READ_IMAGE_CHANNELS == 32
//: A LANE'S CONSTANTS, once a kernel: its shuffle sources (the read row's token, the outputs' two tokens), the read
//: rows' chunk XOR, its pad word, its state loads' two addresses in its plane (each + 1024 bytes for lists 16-31), its
//: entry's offset in a head block
struct Lane {
  int src_a, src_g, src_h;
  uint32_t chunk, pad, st0, st1, entry;
};

__device__ __forceinline__ uint32_t in_plane(int row, int c) {
  return static_cast<uint32_t>(row * kRowBytes + 16 * (c ^ ((row >> 1) & 3)));
}

template <int FORM>
__device__ __forceinline__ Lane lane_of(uint32_t sm, int warp, int lane) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t plane = sm + kImage + (warp % kPlanes) * kPlaneBytes;
  const int srow = FORM == 0 ? 8 * (q & 1) + r : 8 * (lane >> 4) + r, sc = FORM == 0 ? q >> 1 : q & 1;
  const auto h = [](auto v) { return static_cast<decltype(v)>(lab::held(static_cast<uint32_t>(v))); };
  return {h(FORM == 0 ? 8 * (q & 1) + r : r), h(FORM == 0 ? g : 2 * t), h(FORM == 0 ? g + 8 : 2 * t + 1),
          h(FORM == 0 ? 16u * (q >> 1) : 16u * q), h(sm + kPads + (FORM == 0 ? 4u * lane : 4u * g + 32u * t)),
          h(plane + in_plane(srow, sc)), h(plane + in_plane(srow, 2 + sc)),
          h(8u * (FORM == 0 ? lane & 15 : r))};
}

//: digit `i`'s plane, bytes past the warp's first: a compile-time offset for an unrolled `i`
__device__ __forceinline__ constexpr uint32_t plane_off(int i) { return (8 * i % kPlanes) * kPlaneBytes; }

//: A DIGIT'S STATE FRAGMENTS, 16 registers: form 0's B (K = lists, N = channels) at `(kb * 4 + nt) * 2 + h`, form 1's
//: A (M = channels, K = lists) at `(m * 2 + kb) * 4 + i`
template <int FORM>
__device__ __forceinline__ void state_in(uint32_t (&f)[16], const Lane& c, uint32_t off) {
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    uint32_t x[4], y[4];
    lab::ldsm_t(x, (v ? c.st1 : c.st0) + off);
    lab::ldsm_t(y, (v ? c.st1 : c.st0) + off + 1024u);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      f[(FORM == 0 ? 4 : 8) * v + k] = x[k];
      f[(FORM == 0 ? 8 + 4 * v : 8 * v + 4) + k] = y[k];
    }
  }
}

#endif

__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: THE EXACT PAIR: two outputs times (coefficient x unit) to the integer hi x 2^16 + lo, so u32 adds sum both channels
//: exactly while each final fits s16 -- lo's FFMA adds 1.5 x 2^23 (its bits 0x4B400000 + lo), hi's adds 0x4B40 less
//: (its low half hi - 0x4B40), and one LEA (hi bits << 16, plus lo bits) cancels lo's exponent bits
__device__ __forceinline__ uint32_t s16_exact(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

//: an entry word's coefficient times the unit (bf16, its low half) as fp32
__device__ __forceinline__ float coef(uint32_t word) { return __uint_as_float(word << 16); }

#if READ_IMAGE_CHANNELS == 32
//: FORM 0'S STEP, 16 tokens on M: their read rows by two `ldmatrix.x4` (an address one add and an XOR), 8 MMAs against
//: the digit's state, each output pair exact and added at its token's panel base (one add a token)
template <int OUT>
__device__ __forceinline__ void step_m(uint32_t sm, uint2 e, const Lane& c, const uint32_t (&f)[16], float& sink) {
  const uint32_t ea = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t eg = __shfl_sync(0xffffffffu, e.y, c.src_g), eh = __shfl_sync(0xffffffffu, e.y, c.src_h);
  const uint32_t at = (sm + ea) ^ c.chunk;
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    uint32_t a[4];
    lab::ldsm(a, at ^ (32u * kb));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
  const uint32_t bg = c.pad + (eg >> 16), bh = c.pad + (eh >> 16);
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
    } else {
      panel_red(bg + 16 * nt, s16_exact(acc[nt][0], acc[nt][1], coef(eg)));
      panel_red(bh + 16 * nt, s16_exact(acc[nt][2], acc[nt][3], coef(eh)));
    }
  }
}

//: FORM 1'S TILE, 8 tokens on N: their read rows (B: K = lists, N = tokens) by one `ldmatrix.x4`, 4 MMAs against the
//: digit's state (A), each lane's two tokens' outputs exact and added at their panel bases
template <int OUT>
__device__ __forceinline__ void tile_n(uint32_t sm, uint2 e, const Lane& c, const uint32_t (&f)[16], float& sink) {
  const uint32_t eb = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.src_g), e1 = __shfl_sync(0xffffffffu, e.y, c.src_h);
  uint32_t b[4];
  lab::ldsm(b, (sm + eb) ^ c.chunk);
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
  const uint32_t b0 = c.pad + (e0 >> 16), b1 = c.pad + (e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    if constexpr (OUT == 0) {
      sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
    } else {
      panel_red(b0 + 32 * m, s16_exact(acc[m][0], acc[m][2], coef(e0)));
      panel_red(b1 + 32 * m, s16_exact(acc[m][1], acc[m][3], coef(e1)));
    }
  }
}

//: THE DENSE CELL'S STEP `s` of a pass (every slot a token): a lane's read row and panel row its bases plus
//: compile-time offsets; the coefficient the unit
template <int OUT>
__device__ __forceinline__ void step_dense(uint32_t a0, uint32_t p0, const uint32_t (&f)[16], float& sink, int s) {
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    uint32_t a[4];
    lab::ldsm(a, (a0 ^ (32u * kb)) + 1024u * s);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
    } else {
      panel_red(p0 + 1280u * s + 16 * nt, s16_exact(acc[nt][0], acc[nt][1], kUnit));
      panel_red(p0 + 1280u * s + 640 + 16 * nt, s16_exact(acc[nt][2], acc[nt][3], kUnit));
    }
  }
}

//: THE DENSE CELL'S TILE `s` (tokens on N): 8 consecutive slots' read rows by one `ldmatrix.x4`, a lane's read row
//: and panel rows its bases plus compile-time offsets; the coefficient the unit
template <int OUT>
__device__ __forceinline__ void tile_dense(uint32_t b0, uint32_t p0, const uint32_t (&f)[16], float& sink, int s) {
  uint32_t b[4];
  lab::ldsm(b, b0 + 512u * s);
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    if constexpr (OUT == 0) {
      sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
    } else {
      panel_red(p0 + 640u * s + 32 * m, s16_exact(acc[m][0], acc[m][2], kUnit));
      panel_red(p0 + 640u * s + 80 + 32 * m, s16_exact(acc[m][1], acc[m][3], kUnit));
    }
  }
}
#endif

struct HostLists {
  int windows = 0, width = 0, groups = 0;
  std::vector<uint16_t> slots, at, count;
};

inline HostLists read_lists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  int head[4];
  f.read(reinterpret_cast<char*>(head), sizeof(head));
  if (!f || head[0] != 0x44414552) {
    std::fprintf(stderr, "read_image: %s is not a read lists file\n", path.c_str());
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.slots.resize(static_cast<size_t>(h.windows) * h.width * kListCap);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.slots.data()), static_cast<std::streamsize>(h.slots.size() * 2));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: a file's first `bytes` on the device, then `pad` zero bytes (a window past the last, for the prefetch)
template <typename T>
T* upload(const std::string& path, size_t bytes, size_t pad = 0) {
  std::vector<uint8_t> host(bytes + pad, 0);
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(bytes));
  if (!f) {
    std::fprintf(stderr, "read_image: %s is short\n", path.c_str());
    std::exit(1);
  }
  T* dev = nullptr;
  LAB_CHECK(cudaMalloc(&dev, bytes + pad));
  LAB_CHECK(cudaMemcpy(dev, host.data(), bytes + pad, cudaMemcpyHostToDevice));
  return dev;
}

//: a digit's entries in its window's one pass (the inputs assert a box's union fits it)
inline int entries_of(const HostLists& h, int w, int d) {
  const uint16_t* a = &h.at[(static_cast<size_t>(w) * h.width + d) * kListAt];
  return a[kListAt - 1] - a[0];
}

//: THE TENSOR FLOOR a CTA-window: the busiest scheduler's two warps' MMAs (form 0: 8 a 16-token step; form 1: 4 an
//: 8-token tile) at 32 cycles (bf16, fp32 accumulate, sm_86), averaged over the boxes and windows -- each window's
//: busiest when a barrier ends every window, the busiest over all windows when warps free-run (a scheduler idle in one
//: window takes another's work)
inline double floor_of(const HostLists& h, int form, bool dense, bool free) {
  const int boxes = h.width / kDigits;
  double total = 0;
  for (int box = 0; box < boxes; ++box) {
    long long all[4] = {};
    for (int w = 0; w < h.windows; ++w) {
      long long mmas[kWarps] = {};
      for (int p = 0; p < (dense ? 2 : 1); ++p)
        for (int d = 0; d < kDigits; ++d) {
          const int n = dense ? kSlots : entries_of(h, w, box * kDigits + d);
          mmas[d % kWarps] += (form == 0 ? (n + 15) / 16 * 4 : (n + 7) / 8 * 2) * (kChannels / 16);
        }
      long long busiest = 0;
      for (int s = 0; s < 4; ++s) {
        busiest = std::max(busiest, mmas[s] + mmas[s + 4]);
        all[s] += mmas[s] + mmas[s + 4];
      }
      total += free ? 0.0 : busiest * 32.0;
    }
    total += free ? *std::max_element(all, all + 4) * 32.0 : 0.0;
  }
  return total / (static_cast<double>(h.windows) * boxes);
}

//: THE CHECK's reference: a live (token, digit) add is the box's lists x 0.0625 x (-/+ 0.0625) x the coefficient, -/+
//: the lists' count a channel at the unit 256 (even channels negative), times m = 1 + ((slot ^ digit) & 1) for the
//: lists' coefficients (1 dense); a panel word holds channels (2k, 2k + 1) on M and (16 (k / 8) + k % 8, + 8) on N as
//: hi x 2^16 + lo, so a slot's word is its sum of m times one add's word, mod 2^32 -- OUT 1, a CTA's panel summed over
//: its windows; OUT 2, a window-pass's drain summed over the CTAs
struct Reference {
  const HostLists* h;
  int sms;
};

struct Verdict {
  long long bad, words;
  std::string detail;
};

inline uint32_t one_add(int form, int k) {
  const int lo = form == 0 ? 2 * k : 16 * (k >> 3) + (k & 7), hi = form == 0 ? lo + 1 : lo + 8;
  const auto v = [](int ch) { return static_cast<uint32_t>(ch & 1 ? kLists : -kLists); };
  return v(hi) * 65536u + v(lo);
}

template <int FORM, int OUT, int DENSE>
Verdict verify(const Reference& ref, const Lists& ls) {
  const HostLists& h = *ref.h;
  const int boxes = h.width / kDigits, windows = h.windows;
  const auto live = [&](int w, int box, std::vector<uint32_t>& m) {
    for (int d = 0; d < kDigits; ++d) {
      if (DENSE) {
        for (int s = 0; s < kSlots; ++s) m[s] += 1;
        continue;
      }
      const size_t row = static_cast<size_t>(w) * h.width + box * kDigits + d;
      const uint16_t* at = &h.at[row * kListAt];
      for (int e = at[0]; e < at[kListAt - 1]; ++e) {
        const int slot = h.slots[row * kListCap + e];
        m[slot] += 1 + ((slot ^ d) & 1);
      }
    }
  };
  std::vector<uint32_t> got;
  long long bad = 0, words = 0;
  const auto compare = [&](const std::vector<uint32_t>& m, const uint32_t* panel) {
    for (int s = 0; s < kSlots; ++s)
      for (int k = 0; k < kWords; ++k, ++words) bad += panel[s * kWords + k] != m[s] * one_add(FORM, k);
  };
  if constexpr (OUT == 1) {
    got.resize(static_cast<size_t>(ref.sms) * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < ref.sms; ++c) {
      std::vector<uint32_t> m(kSlots, 0);
      for (int w = 0; w < windows; ++w)
        for (int p = 0; p < (DENSE ? 2 : 1); ++p) live(w, c % (boxes * (32 / kLists)) / (32 / kLists), m);
      compare(m, &got[static_cast<size_t>(c) * kSlots * kWords]);
    }
  } else {
    got.resize(static_cast<size_t>(windows) * 2 * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < windows; ++w)
      for (int p = 0; p < 2; ++p) {
        std::vector<uint32_t> m(kSlots, 0);
        if (DENSE || p == 0)
          for (int c = 0; c < ref.sms; ++c) live(w, c % boxes, m);
        compare(m, &got[(static_cast<size_t>(w) * 2 + p) * kSlots * kWords]);
      }
  }
  return {bad, words,
          std::to_string(words - bad) + " of " + std::to_string(words) + " panel words exact" +
              (OUT == 1 ? " (each CTA's panel)" : " (each window-pass's drain)")};
}


//: THE INPUTS on the device: `read_image.inputs.py`'s files from `dir`, both forms' entries and heads (a zero window
//: past the last for the prefetch), the drain's target and the check's panel copy for `sms` CTAs
inline void load(const std::string& dir, int sms, HostLists& host, Lists& lm, Lists& ln) {
  host = read_lists(dir + "/read.bin");
  const size_t entries = static_cast<size_t>(host.windows) * host.width * kListCap * 8;
  const size_t heads = static_cast<size_t>(host.windows) * (host.width / kDigits) * kWarps * kHeadBytes;
  uint16_t* count = upload<uint16_t>(dir + "/read.bin", 16 + host.slots.size() * 2 + host.at.size() * 2 +
                                                            host.count.size() * 2, host.groups * 2);
  uint32_t *out = nullptr, *panel_out = nullptr;
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(host.windows) * 2 * kSlots * kWords * 4));
  LAB_CHECK(cudaMalloc(&panel_out, static_cast<size_t>(sms) * kSlots * kWords * 4));
  const uint16_t* counts = count + 8 + host.slots.size() + host.at.size();
  const size_t pad = heads / host.windows;
  lm = Lists{upload<uint2>(dir + "/entries_m.bin", entries), upload<uint8_t>(dir + "/heads_m.bin", heads, pad),
             counts, host.windows, host.width, host.groups, out, panel_out};
  ln = lm;
  ln.entries = upload<uint2>(dir + "/entries_n.bin", entries);
  ln.heads = upload<uint8_t>(dir + "/heads_n.bin", heads, pad);
}

}  // namespace read_image
