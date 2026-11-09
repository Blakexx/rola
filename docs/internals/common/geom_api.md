# `geom.cu` / `geom_api.cuh` — the addressing block's host seam

Mirrors `csrc/rola/src/common/geom.cu` and `csrc/rola/src/common/geom_api.cuh`. The block's
arithmetic is header-only and lives in `geom.cuh` (`geom.md` is its page); this TU is the
part that has to exist somewhere ONCE — the launch-time refusals, and the two entries that
let a test read the SHIPPED derivation rather than a Python mirror of it.

**Why the entries exist at all.** Two derivations disagree exactly when one of them is
wrong (`matching-the-naive-antipattern`), and the carve has had FOUR derivation sites in
this repo's history. The model tests assert the law against the parent kernel's own
compile-time census, so they must measure the function the kernel will call — which means
a binding, not a mirror.

**No kernel is instantiated or launched here.** This TU is host code that includes one
header of pure `constexpr` arithmetic; it adds no entry point to the binary, which is what
lets the foundation land while the shipped body stays byte-identical.

<a id="derive-block"></a>
## `derive_block` — the seam's refusals

Everything `geom.cuh`'s `derive_carry_geom` refuses, re-raised as a torch error naming the
law that failed, plus the two the SEAM owns rather than the shape:

* the packed amplitude row is a whole number of sixteen-byte granules (`Σ_l B_l % 8 == 0`);
* every level's column base is a whole number of the runs read out of it, on BOTH sides —
  the wide loads' alignment, which is a property of the pair (widths, carve) and so cannot
  be asserted inside either side's own derivation.

`min_l B_l` is a CENSUS FACT and never a refusal: `geom.md#refusals` states why.

<a id="tall-block"></a>
## `tall_block` — the tall owner set for a kernel family

The one entry a kernel family's host seam calls for the tall owner set (`geom.md#tall-owner-set`):
`derive_block` with `bc` the tall leaf count, `dv` the channel group, both stream counts the
CTA's warps, run placement and no warp-level page floor; the mode word decides the spans. The
drift guard that keeps `derive_carry_geom` inside the geometry block is what this entry exists
for -- a family derives its block here and never re-spells the law in its own seam.

<a id="geometry"></a>
## `geometry(widths, level_modes, bc, nsr, nsw, dv, interleave, atom_floor)`

The block, flattened to one `int64` vector: the scalars and per-level arrays in
declaration order, then the read side's `SideGeom`, then the write side's, then BOTH
SIDES' TRANSIT ROW MAPS evaluated over the whole region (`streams × leaf` entries each).
The maps are in the vector because a bijection claim is about the map's whole image, and
the test must not re-implement `geom_xch_row` to make it.

The layout is positional and the reader is `rola/ops/carry.py::geometry`, which names every
field once; nothing else parses it. `interleave` (0 or 1) selects the stream placement at
every carved level (`geom.md#interleaved-placement`) and is echoed in each side's block;
`atom_floor` (0 or 1) is the warp-level page floor on the canonical innermost level, the shipped
body's `1` and the tall form's `0` (`geom.md#tall-owner-set`).

<a id="sub-box-set"></a>
## `sub_box_set(depth, box_leaves, workers, atom_floor)`

The GENERATED set of warp sub-box shapes — `geom.md#warp-box-set` — as spans rather than
bit counts, one row per member, in the generator's own order (which is what `assign`
indexes). This is layer 1 of R9's four (`docs/internals/common/structure_switch.md`): the
set a switch's cases come from is generated, and the test that asserts a member count
asserts it against THIS, never against a hand-written table.
