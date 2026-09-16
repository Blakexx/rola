"""THE ONE DERIVATION, against the PARENT KERNEL'S OWN COMPILE-TIME CENSUS.

`csrc/rola/src/common/geom.cuh` computes at runtime what the shipped carry bodies compute
in their templates (`box.cuh`'s `BoxPlan`/`ModeOrder`/`StreamSpans`). No kernel on this
branch consumes the runtime block -- the pipelined body is what will
(`docs/internals/common/geom.md#on-the-tip`) -- so THIS FILE IS THE PROOF that the block is
the same law: for every arm the parent builds, the derivation's own output is checked
against the census the parent binary reports off the device.

**WHY A GOLDEN TABLE, NOT A LIVE READ.** The golden table below is the parent's
census, transcribed, so the assertion survives the parent's deletion (the SKILLS
FEEDBACK 3: the A-side of any parity claim is a pinned tip that will not always exist) --
and C0 IS that deletion: `carry_build_stamp`/`carry_arms` are gone from the extension
entirely (not merely some arms unbuilt). The refill has since landed under the FINAL
`(D, DV, warps_per_cta)` key (`gen_shards.CARRY_ARMS == ((2, 64, 8),)`, one shipped row),
a different schema than the 13-field row this table's provenance names. There is
therefore no live binary left to re-read against, on this line, ever again (the parent
tip is never edited or retrofitted). `## C0`
found this file 36F/9P for exactly this reason (parameterized over the now-empty
`CARRY_ARMS`). The fix is what SKILLS FEEDBACK 3 anticipated: `ARM_PARAMS` below transcribes
the same 335c2b9 arm rows the parent built `PARENT_CENSUS` from, so `_arm()` no longer reads
`gen_shards.CARRY_ARMS` at all, and the one-time live cross-check
(`test_the_golden_table_is_the_binarys_own`) stays retired by design -- the table is
pinned-tip evidence, not a stand-in for an empty live set -- rather than left calling a
binding that cannot exist on this branch.

**WHAT IS ASSERTED AS THE LAW, NOT AS THE TABLE.** The parity rows are 11 of the parent's
13 arms. The other two are the `D = 4, B = 8` rows, which the RESTORED WIDTH FLOOR
(`B_l >= 16`, power of two) makes unlawful: a level narrower than the owner box's span
there is REFUSED BY NAME, and the deep-4 conformance topology moves to the lawful
`(16, 16, 16, 16)` at the same state size. The test names
that ruling in its own message so a reader meets the citation, not a surprise.

Symbols, each at first use: `D` routing levels; `B_l` level `l`'s digit width; `BC` the
leaves one owner (one CTA's box) holds; `DV` the value width; `MO` the per-level mode word
(two bits a level, read then write); `s(l)` the OWNER span; `g(l) = B_l/s(l)` the owner
grid; `sv(l)` a side's WARP SUB-BOX span; `NS` a side's stream count.
"""

from __future__ import annotations

import pytest

from rola.ops import carry as carry_ops

#: THE ARM ROWS THE PARENT BUILT, transcribed from `gen_shards.CARRY_ARMS` at
#: `k31/k35-final` 335c2b9 (the tip `PARENT_CENSUS` below was read off), in that commit's
#: 13-field schema `(D, B, K, M, DV, W, C, CV, MO, WPS, NSW, NSR, PK)`. C0 deleted
#: `CARRY_ARMS` to `()` on this line and G4 refills it under a DIFFERENT (three-field) key,
#: so this table -- not the live list -- is what `_arm()` reads: the parity claim's A-side
#: is a pinned tip, not a moving target.
ARM_PARAMS = {
    0: (2, 256, 4, 8, 64, 384, 32, 1, 0, 1, 1, 1, 0),
    1: (2, 256, 4, 8, 64, 384, 32, 1, 6, 1, 4, 4, 0),
    2: (2, 256, 4, 8, 64, 512, 32, 1, 0, 1, 1, 1, 0),
    3: (2, 256, 4, 8, 64, 512, 32, 1, 6, 1, 4, 4, 0),
    4: (2, 64, 4, 8, 64, 64, 32, 1, 0, 1, 1, 1, 0),
    5: (2, 64, 4, 8, 64, 64, 32, 1, 6, 1, 4, 4, 0),
    6: (3, 16, 2, 16, 64, 64, 32, 1, 0, 1, 1, 1, 0),
    7: (3, 16, 2, 16, 64, 64, 32, 1, 38, 1, 1, 1, 0),
    8: (4, 8, 2, 8, 64, 64, 32, 1, 0, 1, 1, 1, 0),
    9: (4, 8, 2, 8, 64, 64, 32, 1, 102, 1, 4, 4, 0),
    10: (2, 256, 4, 8, 64, 512, 32, 1, 1, 1, 1, 1, 0),
    11: (2, 256, 4, 8, 64, 512, 32, 1, 10, 1, 1, 1, 0),
    12: (2, 256, 4, 8, 64, 384, 32, 1, 3, 1, 4, 4, 0),
}

#: THE PARENT'S CENSUS, ARM BY ARM, read off `carry_build_stamp` on the all-arms binary at
#: `k31/k35-final` 335c2b9: `{arm index: (BC, owners, write spans, read spans, kXchBytes)}`.
#: The spans are `StreamSpans::s(l)` -- what ONE STREAM of that side holds -- padded past
#: `D` with `B` the way the stamp reports them; the runtime block's `row_span` is the same
#: quantity. `kXchBytes == 0` is the parent's compile-time TIED case.
PARENT_CENSUS = {
    0: (128, 512, [8, 16], [8, 16], 0),
    1: (128, 512, [2, 16], [8, 4], 18432),
    2: (128, 512, [8, 16], [8, 16], 0),
    3: (128, 512, [2, 16], [8, 4], 18432),
    4: (128, 32, [8, 16], [8, 16], 0),
    5: (128, 32, [2, 16], [8, 4], 18432),
    6: (128, 32, [2, 4, 16], [2, 4, 16], 0),
    7: (128, 32, [2, 4, 16], [2, 4, 16], 0),
    8: (128, 32, [2, 2, 4, 8], [2, 2, 4, 8], 0),
    9: (128, 32, [1, 2, 2, 8], [2, 1, 4, 4], 18432),
    10: (128, 512, [8, 16], [8, 16], 0),
    11: (128, 512, [8, 16], [8, 16], 0),
    12: (128, 512, [2, 16], [2, 16], 0),
}

#: the rows the restored floor removes: `B = 8` is narrower than the owner box's
#: innermost span (16, the page atom held whole), so the shape refuses them by name.
UNLAWFUL_UNDER_THE_FLOOR = (8, 9)

#: THE ROWS THE INNERMOST-DIGIT INVARIANT RE-CARVES. The carve never divides the
#: canonical innermost level below the page rectangle -- the fastest-varying digits are
#: the vector width, and a warp box narrower there owns a stride instead of a run -- so
#: the order passes over that level and divides the next one instead. The parent carved
#: its read side to `[8, 4]` at these arms; the same order under the invariant reaches
#: the same stream count as `[2, 16]`, which makes the two sides' carves coincide and the
#: transit an identity the parent had to move bytes for. The parent's own numbers are
#: kept below as the record of what changed.
RECARVED = {1: ([2, 16], [8, 4]), 3: ([2, 16], [8, 4]), 5: ([2, 16], [8, 4])}

#: THE ARCH CONSTANTS, mirrored from `geom.cuh` -- three scalars, and the only thing in
#: this file that is a mirror rather than a call. The SET they feed is the shipped
#: generator's, read through `carry_ops.sub_boxes`. -- docs/internals/common/geom.md#arch-constants
WARPS_PER_SM = 8

#: the page rectangle in leaves, the grain the innermost digits are never carved below.
PAGE_RECTANGLE = 16

#: the 2x4 ruling: `warps_per_cta` is a compile-time
#: axis with exactly this two-value set -- `(1, 8)` the shipped, one-CTA-per-SM shape,
#: `(2, 4)` the latency-diagnostic test arm. The per-side member set is
#: `f(D, leaves_per_warp, warps_per_cta)`, so the generated-and-deduplicated set is
#: enumerated at BOTH values, never just the one these arms happen to compile.
WARPS_PER_CTA = (8, 4)


def leaves_per_sm(dv: int) -> int:
    return 16384 // dv


def leaves_per_warp(dv: int) -> int:
    return leaves_per_sm(dv) // WARPS_PER_SM


def _arm(i):
    (D, B, K, M, DV, W, C, CV, MO, _WPS, NSW, NSR, _PK) = ARM_PARAMS[i]
    return dict(D=D, widths=[B] * D, k=K, m=M, dv=DV, window=W, chunk=C, carve=CV, mo=MO,
                nsw=NSW, nsr=NSR)


def _block(i):
    a = _arm(i)
    return carry_ops.geometry(a["widths"], level_modes=a["mo"], bc=PARENT_CENSUS[i][0],
                              nsr=a["nsr"], nsw=a["nsw"], d_v=a["dv"])


LAWFUL = [i for i in PARENT_CENSUS if i not in UNLAWFUL_UNDER_THE_FLOOR]


@pytest.mark.parametrize("arm", LAWFUL)
def test_the_derivation_reproduces_the_parents_census(arm):
    """PARITY, arm by arm: the runtime block IS the compile-time law it replaces."""
    bc, owners, w_spans, r_spans, xch = PARENT_CENSUS[arm]
    g = _block(arm)
    assert g["bc"] == bc
    assert g["owners"] == owners
    assert g["w"]["row_span"] == w_spans, "the WRITE side's stream span vector moved"
    want_r = RECARVED[arm][0] if arm in RECARVED else r_spans
    assert g["r"]["row_span"] == want_r, "the READ side's stream span vector moved"
    assert g["w"]["streams"] == _arm(arm)["nsw"]
    assert g["r"]["streams"] == _arm(arm)["nsr"]
    #: the parent ELIDES the transit at compile time exactly when the two carves
    #: coincide; the block reports the same fact instead of a second code path (K50
    #: deleted the elided body). An uncarved arm has no transit at all, so the parent's
    #: zero says nothing there and the claim is made only where it carves.
    if _arm(arm)["nsw"] > 1 and arm not in RECARVED:
        assert (xch == 0) == bool(g["identity_map"])
    if arm in RECARVED:
        assert bool(g["identity_map"]), (
            "under the innermost-digit invariant both sides reach this stream count on the "
            "outer level, so the carves coincide and the transit is the identity map -- "
            f"the parent moved {xch} bytes here because its read side carved "
            f"{RECARVED[arm][1]}, which the invariant does not admit")


@pytest.mark.parametrize("arm", UNLAWFUL_UNDER_THE_FLOOR)
def test_the_deep_four_rows_are_refused_by_the_restored_width_floor(arm):
    """RULED, not a regression: the restored width floor makes `B_l = 8` unlawful.

    The parent builds these two rows and carves them; the shape this block owns has no
    narrower slot layout to fall back to, so it REFUSES rather than rounding, and the
    deep-4 conformance topology is the lawful `(16, 16, 16, 16)` at the same state size.
    """
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        _block(arm)


def test_the_lawful_deep_four_topology_is_admitted_where_the_unlawful_one_is_not():
    g = carry_ops.geometry([16, 16, 16, 16], level_modes=102, bc=128, nsr=4, nsw=4, d_v=64)
    assert g["span"] == [2, 2, 2, 16], "the atom is held WHOLE innermost"
    assert g["leaves"] == 65536


@pytest.mark.parametrize("arm", LAWFUL)
def test_the_block_obeys_its_own_laws_not_just_its_table(arm):
    """The laws, restated as assertions: spans, grids, bases, weights, slots, streams."""
    g = _block(arm)
    D, bc = g["D"], g["bc"]
    prod = 1
    for s in g["span"]:
        prod *= s
    assert prod == bc, "prod_l s(l) = BC"
    for lvl in range(D):
        assert g["width"][lvl] % g["span"][lvl] == 0, "s(l) divides B_l"
        assert g["grid"][lvl] == g["width"][lvl] // g["span"][lvl], "g(l) = B_l / s(l)"
    assert g["col_base"] == [sum(g["width"][:lvl]) for lvl in range(D)], "prefix sum of B_l"
    weight = [1] * D
    for lvl in range(D - 2, -1, -1):
        weight[lvl] = weight[lvl + 1] * g["width"][lvl + 1]
    assert g["weight"] == weight, "weight(l) = prod_{l' > l} B_l'"
    assert g["wtot"] == sum(g["width"])
    for side in ("r", "w"):
        s = g[side]
        leaf, streams = 1, 1
        for lvl in range(D):
            leaf *= 1 << s["sub_bits"][lvl]
            streams *= s["grid"][lvl]
            assert s["row_span"][lvl] == (1 << s["sub_bits"][lvl]), "row_span IS sv(l)"
            assert s["grid"][lvl] * s["row_span"][lvl] == g["span"][lvl]
        assert leaf == s["leaf"], "prod_l sv(l) = the leaves one stream owns"
        assert streams == s["streams"], "prod_l s(l)/sv(l) = NS"
        assert leaf * streams == bc, "the streams partition the owner's box"
        #: the slot layout is a BIJECTION of the slot's bits: each level's digits occupy
        #: `sub_bits[l]` bits at `sub_shift[l]`, disjointly, covering `log2 leaf`.
        used = 0
        for lvl in range(D):
            mask = ((1 << s["sub_bits"][lvl]) - 1) << s["sub_shift"][lvl]
            assert used & mask == 0, "two levels claim the same slot bit"
            used |= mask
        assert used == leaf - 1, "the slot bits are exactly covered"
        assert sorted(s["rank"]) == list(range(D))
        assert [s["level_at"][s["rank"][lvl]] for lvl in range(D)] == list(range(D))


@pytest.mark.parametrize("arm", LAWFUL)
def test_each_sides_transit_row_map_is_a_bijection_of_the_region(arm):
    """THE ROW MAP. Every leaf of `[0, BC)` is reached exactly once, by each side."""
    g = _block(arm)
    for side in ("r", "w"):
        rows = [row for stream in g["xch"][side] for row in stream]
        assert sorted(rows) == list(range(g["bc"])), f"the {side} map is not a bijection"
    same = g["xch"]["r"] == g["xch"]["w"]
    assert same == bool(g["identity_map"]), (
        "`identity_map` is the FACT that the two sides' maps coincide, and it disagreed "
        "with the maps themselves")


def test_a_width_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(RuntimeError, match="power of two"):
        carry_ops.geometry([24, 256], level_modes=6, bc=128, nsr=4, nsw=4, d_v=64)


def test_a_topology_too_small_for_the_owner_box_is_refused():
    """`Pi_l B_l < BC` cannot hold a box; the refusal names the level that cannot."""
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        carry_ops.geometry([16, 4], level_modes=0, bc=128, nsr=1, nsw=1, d_v=64)


def test_a_stream_count_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(RuntimeError, match="power of two"):
        carry_ops.geometry([256, 256], level_modes=6, bc=128, nsr=3, nsw=4, d_v=64)


#: THE GENERATED SET -- the layer 1. `sub_boxes(D, box_leaves, workers)` runs the
#: top-down carve over EVERY level order and de-duplicates; these are the counts the
#: switch's `NA` is, and they are asked of the generator, never of a written list.
#: `f(D, leaves_per_warp, warps_per_cta)`: the member set is enumerated at BOTH of the #: two-value `warps_per_cta` set -- the
#: shipped `(1, 8)` shape and the `(2, 4)` test/diagnostic arm -- not only the one these
#: arms happen to compile.
#: THE INNERMOST-DIGIT INVARIANT IS WHAT MAKES THIS SET SMALL: the carve passes over the
#: canonical innermost level rather than dividing its digits below the page rectangle, so
#: every member ends in the same innermost span and the orders that differ only in how
#: they would have split it collapse onto one shape. That is asserted below as the law --
#: the counts are its consequence, not a transcription.
_SUB_BOX_COUNTS = {
    4: {1: 1, 2: 1, 3: 2, 4: 3},
    8: {1: 1, 2: 1, 3: 2, 4: 3},
}


@pytest.mark.parametrize("warps_per_cta", WARPS_PER_CTA)
@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_the_sub_box_set_is_generated_and_deduplicated(D, warps_per_cta):
    dv = 64
    box = leaves_per_warp(dv) * warps_per_cta  # the CTA's own box at this warp count
    members = carry_ops.sub_boxes(D, box, warps_per_cta)
    count = _SUB_BOX_COUNTS[warps_per_cta][D]
    assert len(members) == count, (
        f"the generated set at D={D}, warps_per_cta={warps_per_cta} is not {count} shapes")
    assert len(set(members)) == len(members), "the generator returned a duplicate shape"
    for shape in members:
        prod = 1
        for s in shape:
            prod *= s
        assert prod * warps_per_cta == box, "a member times the worker count IS the box"
        assert shape[-1] >= PAGE_RECTANGLE or shape[-1] == carry_ops.leaves_per_warp(dv), (
            "the innermost digits are the vector width: no member divides the canonical "
            "innermost level below the page rectangle")


def test_the_shape_set_is_the_arch_law_at_one_cta_per_sm():
    """`sb_matches_arch_law`, asserted where it is EVALUABLE (the own design point).

    The set is a property of `(D, leaves_per_warp, warps_per_cta)` with THE CTA'S BOX as
    the unit, and it equals the SM-grain set exactly when the CTA owns the SM
    (`warps_per_cta == warps_per_sm`, the one CTA per SM). At the `kWarps = 4` these
    arms compile it is the same law run on HALF the SM box, which is a different set --
    asserted here too, so the two grains cannot be confused for each other."""
    dv = 64
    sm_box = leaves_per_sm(dv)
    for D in (1, 2, 3, 4):
        at_sm = carry_ops.sub_boxes(D, sm_box, WARPS_PER_SM)
        by_cta = carry_ops.sub_boxes(D, leaves_per_warp(dv) * WARPS_PER_SM, WARPS_PER_SM)
        assert at_sm == by_cta, (
            "at one CTA per SM the CTA-box law and the SM-box law are the same set")
    #: the four-warp CTA runs the same law on HALF the box, and reaches the same SHAPE:
    #: its box is half as wide at level 0 and it carves half as many workers out of it,
    #: while the innermost level -- which the invariant never divides -- is the same run.
    #: The two sets are therefore equal by arithmetic, not by being the same derivation,
    #: and the box they are derived from is what differs.
    half = carry_ops.sub_boxes(2, leaves_per_warp(dv) * 4, 4)
    assert half == carry_ops.sub_boxes(2, sm_box, WARPS_PER_SM)
    assert leaves_per_warp(dv) * 4 != sm_box, "the two boxes are not the same box"


#: RETIRED (was `test_the_golden_table_is_the_binarys_own`): that test re-read the
#: golden rows off `carry_ops.build_stamp`/`carry_ops.arms`, i.e. `extension().
#: carry_build_stamp`/`carry_arms`. C0 removed BOTH bindings from `rola._C` entirely --
#: not "some arms unbuilt", the entry points themselves are gone -- so there
#: is no live parent binary left on this branch to re-derive against, and there will not
#: be one again (the parent tip is never edited or retrofitted). `ARM_PARAMS` above is the
#: transcription that was supposed to let the claim survive that deletion (the SKILLS
#: FEEDBACK 3); `PARENT_CENSUS`'s numbers are what the parity tests below check, is the
#: whole of what remains, and it is intentionally NOT re-derived from a binary anymore.
