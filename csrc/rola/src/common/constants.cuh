// csrc/rola/src/common/constants.cuh -- the family's file constants, the launch
// table and the S-from-SMEM rule. See docs/internals/common/constants.md.
#pragma once

#include "common/arch_caps.cuh"
#include "common/geom.cuh"

namespace rola::carry {

//: W -- see docs/internals/common/constants.md#window
constexpr int kWindow = 512;

//: THE SEGMENT -- see docs/internals/common/constants.md#segment
constexpr int kSegmentTokens = 16;

//: see docs/internals/common/constants.md#warps-per-cta
constexpr int kWarpsPerCtaShipped = 8;
constexpr int kWarpsPerCtaDiagnostic = 4;
constexpr int kWarpsPerCta[2] = {kWarpsPerCtaShipped, kWarpsPerCtaDiagnostic};

constexpr bool is_warps_per_cta(int warps_per_cta) {
  return warps_per_cta == kWarpsPerCtaShipped || warps_per_cta == kWarpsPerCtaDiagnostic;
}

//: BC -- see docs/internals/common/constants.md#bc
constexpr int box_leaves(int dv, int warps_per_cta) { return leaves_per_warp(dv) * warps_per_cta; }

//: see docs/internals/common/constants.md#launch-table
struct LaunchRow {
  int cc;
  int ctas_per_sm;
  int warps_per_cta;
};

constexpr LaunchRow kLaunchTable[] = {
    {800, 1, kWarpsPerCtaShipped}, {860, 1, kWarpsPerCtaShipped}, {870, 1, kWarpsPerCtaShipped},
    {890, 1, kWarpsPerCtaShipped}, {900, 1, kWarpsPerCtaShipped},
};
constexpr int kLaunchTableRows = sizeof(kLaunchTable) / sizeof(kLaunchTable[0]);

constexpr int launch_table_index(int cc) {
  int idx = -1;
  for (int i = 0; i < kLaunchTableRows; ++i)
    if (kLaunchTable[i].cc == cc) idx = i;
  return idx;
}

constexpr bool launch_table_covers(int cc) { return launch_table_index(cc) >= 0; }

constexpr LaunchRow launch_row(int cc) {
  const int idx = launch_table_index(cc);
  return idx < 0 ? LaunchRow{cc, 0, 0} : kLaunchTable[idx];
}

//: see docs/internals/common/constants.md#launch-table (the cross-check with arch_caps.cuh)
constexpr bool launch_table_rows_are_all_tabulated() {
  for (int i = 0; i < kLaunchTableRows; ++i)
    if (!denseref::arch::tabulated(kLaunchTable[i].cc)) return false;
  return true;
}

static_assert(launch_table_rows_are_all_tabulated(),
              "a launch-table row names an architecture arch_caps.cuh does not tabulate");
static_assert(launch_table_covers(800), "arch_caps.cuh tabulates sm_80");
static_assert(launch_table_covers(860), "arch_caps.cuh tabulates sm_86");
static_assert(launch_table_covers(870), "arch_caps.cuh tabulates sm_87");
static_assert(launch_table_covers(890), "arch_caps.cuh tabulates sm_89");
static_assert(launch_table_covers(900), "arch_caps.cuh tabulates sm_90");
static_assert(kLaunchTableRows == 5,
              "a launch-table row was added or removed without a cross-check update");

//: see docs/internals/common/constants.md#launch-table (one CTA per SM on every row)
constexpr bool launch_table_is_one_cta_per_sm() {
  for (int i = 0; i < kLaunchTableRows; ++i)
    if (kLaunchTable[i].ctas_per_sm != 1 || kLaunchTable[i].warps_per_cta != warps_per_sm)
      return false;
  return true;
}

static_assert(launch_table_is_one_cta_per_sm(),
              "the CTA is the SM: every launch-table row runs one CTA holding warps_per_sm warps");

//: see docs/internals/common/constants.md#bc (the geom.cuh agreement)
static_assert(leaves_per_warp(64) == 32, "leaves per warp at DV <= 64");
static_assert(leaves_per_warp(128) == 16, "leaves per warp halves at DV = 128");
static_assert(warps_per_sm == 8, "KERNEL_STANDARDS.md §R15");

//: see docs/internals/common/constants.md#s-from-smem
constexpr int derive_stream_count(int per_stream_bytes, int residency_smem_bytes,
                                  int warps_per_cta) {
  for (int s = warps_per_cta; s >= 1; s >>= 1)
    if ((long long)s * (long long)per_stream_bytes <= (long long)residency_smem_bytes) return s;
  return 0;
}

}  // namespace rola::carry
