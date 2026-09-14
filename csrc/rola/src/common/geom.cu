// csrc/rola/src/common/geom.cu -- the addressing block's host TU: the ONE derivation's
// seam refusals, and the two entries that make the shipped block readable from Python.
// No kernel is instantiated here and none is launched.
// See docs/internals/common/geom_api.md

#include <torch/extension.h>

#include "common/geom.cuh"
#include "common/geom_api.cuh"

namespace rola::carry {

//: THE ADDRESSING BLOCK, DERIVED AND REFUSED AT THE SEAM. Every launch-time law the axis rule
//: names lives here; `min_l B_l < 16` is a CENSUS FACT and not one of them.
//: -- see docs/internals/common/geom_api.md#derive-block
static CarryGeomRT derive_block(const std::vector<int64_t>& widths, int64_t level_modes, int bc,
                                int dv, int nsr, int nsw) {
  int w[kGeomMaxLevels] = {1, 1, 1, 1};
  const int D = (int)widths.size();
  TORCH_CHECK(D >= 1 && D <= kGeomMaxLevels, "the routing depth is one to four levels, got ", D);
  for (int l = 0; l < D; ++l) w[l] = (int)widths[l];
  CarryGeomRT geo{};
  CarveOrder order{};
  carve_order_of_modes(D, (uint32_t)level_modes, order);
  const char* err = derive_carry_geom(D, w, bc, dv, order, nsr, nsw, geo);
  TORCH_CHECK(err == nullptr, "the addressing block refuses this call: ", err == nullptr ? "" : err,
              " (widths=", widths, ", BC=", bc, ", level_modes=", level_modes, ")");
  TORCH_CHECK(geo.wtot % 8 == 0,
              "the packed amplitude row must be a whole number of sixteen-byte granules: "
              "sum_l B_l = ",
              geo.wtot);
  for (int l = 0; l < D; ++l)
    TORCH_CHECK(
        geo.col_base[l] % geo.r.row_span[l] == 0 && geo.col_base[l] % geo.w.row_span[l] == 0,
        "level ", l, "'s column base ", geo.col_base[l],
        " is not a whole number of the runs read out of it (read ", geo.r.row_span[l], ", write ",
        geo.w.row_span[l], "): the level runs would not be aligned");
  return geo;
}

std::vector<int64_t> geometry(std::vector<int64_t> widths, int64_t level_modes, int64_t bc,
                              int64_t nsr, int64_t nsw, int64_t dv) {
  const CarryGeomRT g = derive_block(widths, level_modes, (int)bc, (int)dv, (int)nsr, (int)nsw);
  std::vector<int64_t> o;
  auto push_side = [&](const SideGeom& s) {
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.level_at[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.rank[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_span[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.grid[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.grid_div[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_g[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_prefix[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.sub_bits[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.sub_shift[l]);
    o.push_back(s.carves);
    o.push_back(s.rows);
    o.push_back(s.leaf);
    o.push_back(s.streams);
    o.push_back(s.assign);
  };

  o.push_back(g.D);
  o.push_back(g.bc);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.width[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.span[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.grid[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.col_base[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.run_shift[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.owner_div[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.weight[l]);
  o.push_back(g.wtot);
  o.push_back(g.owners);
  o.push_back(g.leaves);
  o.push_back(g.local_bits);
  o.push_back(g.min_width);
  o.push_back(g.identity_map);
  push_side(g.r);
  push_side(g.w);

  //: THE TRANSIT'S ROW MAP, BOTH SIDES, evaluated over the whole box: the model test
  //: asserts each side's map is a bijection of the region and that the two coincide
  //: exactly when `identity_map` says so.
  for (int w = 0; w < g.w.streams; ++w)
    for (int P = 0; P < g.w.leaf; ++P) o.push_back(geom_xch_row(g, g.w, w, P));
  for (int r = 0; r < g.r.streams; ++r)
    for (int P = 0; P < g.r.leaf; ++P) o.push_back(geom_xch_row(g, g.r, r, P));
  return o;
}

std::vector<std::vector<int64_t>> sub_box_set(int64_t depth, int64_t box_leaves, int64_t workers) {
  const int D = (int)depth;
  TORCH_CHECK(D >= 1 && D <= kGeomMaxLevels, "the shape set is generated to depth four, got ", D);
  TORCH_CHECK(geom_ilog2((int)box_leaves) >= 0 && geom_ilog2((int)workers) >= 0,
              "the box and the worker count are powers of two");
  const SubBoxSet set = sub_boxes(D, (int)box_leaves, (int)workers);
  std::vector<std::vector<int64_t>> out;
  for (int a = 0; a < set.count; ++a) {
    std::vector<int64_t> row;
    for (int l = 0; l < D; ++l) row.push_back(1 << set.bits[a][l]);
    out.push_back(row);
  }
  return out;
}

}  // namespace rola::carry
