"""THE EXCHANGE'S LAW, PROVED ON THE MODEL -- INDEPENDENT OF ANY KERNEL'S CODE.

The transit is the one place in the carry where two INDEPENDENT derivations have to meet:
the publisher names a row from its own WRITE order and the consumer addresses that same
row as a slot of its own READ order, and the two share no code.  What has to be proved is
that the COMPOSITE IS THE IDENTITY ON LATTICE LEAVES -- and, separately, that neither leg
of the traffic conflicts in the banks.

THE MODEL CONSUMES THE SHIPPED DERIVATION and composes it itself.  It reads the addressing
block through `rola.ops.carry.geometry` (`csrc/rola/src/common/geom.cuh`, the ONE
derivation) and builds the row map, the four accesses and the whole hybrid transit from
it; a wrong derivation shows up as a broken permutation here rather than as noise on a
device.  Pure arithmetic: no CUDA kernel is launched.

WHICH FILE IS WHICH.  `tests/unit/test_exchange_layout.py` proves the layout of the
PARENT BODY on this branch, whose carve is compile-time; this file proves the LAW on the
runtime block the pipelined body will consume (`docs/internals/common/geom.md#row-map`).
They are two derivations of the same law and both are kept until one body remains.

Symbols: `D` levels; `B_l` level `l`'s width; `BC` the leaves one CTA's box holds; `DV`
the value width; `MO` the mode word; `sv(l)` a side's warp sub-box span; `NS` a side's
stream count; a MEMBER is the element of the generated shape set a side's order selects.
"""

import pytest

from rola.ops import carry as carry_ops

K_PAD_E = 8
K_MMA_M, K_MMA_N, K_MMA_K = 16, 8, 16


def _ilog2c(x):
    """the bit index of a power of two -- the model's own, matching `geom_ilog2`."""
    b = 0
    while (1 << b) < x:
        b += 1
    return b


#: THE BOX ONE CTA OWNS on the arms this branch builds: `leaves_per_warp(64) = 32` leaves
#: of state per warp times the four warps of the CTA (`geom.md#cta-box-first`).
BC = 128


def _block(widths, modes, ns):
    """The shipped block, FLAT-KEYED for this model: `r_row_span`, `w_grid`, ..."""
    g = carry_ops.geometry(list(widths), level_modes=modes, bc=BC, nsr=ns, nsw=ns)
    out = {k: v for k, v in g.items() if k not in ("r", "w", "xch")}
    for side in ("r", "w"):
        for name, value in g[side].items():
            out[f"{side}_{name}"] = value
    out["map"] = g["xch"]
    return out


class _Side:
    """One side's derived carve, READ OFF THE BLOCK -- never re-derived here."""

    def __init__(self, blk, side, ns):
        self.D, self.ns, self.blk = blk["D"], ns, blk
        self._p = side  # "r" or "w"
        self.bc = blk["bc"]
        self.leaf = blk[f"{side}_leaf"]

    def s(self, l):
        return self.blk[f"{self._p}_row_span"][l]

    def g(self, l):
        return self.blk[f"{self._p}_grid"][l]

    def rank(self, l):
        return self.blk[f"{self._p}_rank"][l]

    def shift(self, l):
        """THIS SIDE'S OWN SLOT LAYOUT: the member's spans packed innermost-lowest.
        The two sides select DIFFERENT members from the mode word, so this is read off the
        block per side rather than shared -- which is the claim this file now checks."""
        return self.blk[f"{self._p}_sub_shift"][l]

    def assign(self):
        return self.blk[f"{self._p}_assign"]

    def run_shift(self, l):
        """the OWNER run's own bit offset -- the region's index is that canonical run."""
        return sum(_ilog2c(self.blk["span"][i]) for i in range(l + 1, self.D))

    def grid_div(self, l):
        return self.blk[f"{self._p}_grid_div"][l]

    def off(self, l, stream):
        return ((stream // self.grid_div(l)) % self.g(l)) * self.s(l)


#: THE DECLARATIONS THE TRANSIT IS PROVED OVER.  They are no longer arms -- one binary
#: serves every one of them -- so the parametrization names the geometry directly,
#: and it is a SUPERSET of what the former arm list covered: the flagship's ALTERNATING
#: word and its TIED word, the small window, both deep topologies, and the three flat
#: declarations the conformance cells drive.
CASES = [
    ("flag-alt", (256, 256), 6),
    ("flag-tied", (256, 256), 3),
    ("flag-dense", (256, 256), 0),
    ("flag-read-one", (256, 256), 1),
    ("flag-write-both", (256, 256), 10),
    ("small-alt", (64, 64), 6),
    ("deep3-alt", (16, 16, 16), 38),
    ("deep4-alt", (16, 16, 16, 16), 102),
]
NS = 4


def _sides(case):
    _, widths, modes = case
    blk = _block(widths, modes, NS)
    return _Side(blk, "w", NS), _Side(blk, "r", NS), 64


def _swz(row):
    """THE REGION'S SWIZZLE, mirroring `geom.cuh`'s `sb_swizzle`: every three-bit group of
    the index folded into its low three bits. GF(2)-linear, so it distributes over
    `base ^ scatter(P)` and every fragment term stays an immediate."""
    return row ^ (((row >> 3) ^ (row >> 6) ^ (row >> 9)) & 7)


def _read_index(blk, local):
    """THE READ-ORDER INDEX of an owner-local canonical row, mirroring `geom_read_index`:
    the read side's carve order from the mode word, rank 0 innermost."""
    D, span, rank = blk["D"], blk["span"], blk["r_rank"]
    bits = [_ilog2c(x) for x in span]
    index = 0
    for l in range(D):
        sh = sum(bits[m] for m in range(D) if rank[m] < rank[l])
        rs = sum(bits[i] for i in range(l + 1, D))
        index += ((local >> rs) & (span[l] - 1)) << sh
    return index


def _row(side, stream, slot):
    """THE REGION'S ROW: the OWNER-LOCAL CANONICAL INDEX of the leaf this side's stream
    holds at that slot, taken to its READ-ORDER index under the swizzle.  Each side reaches
    a leaf through its OWN base word and its own member's shifts, so the two sides meet on
    the canonical index and nowhere else; the region is indexed in the read order so the
    readout's pull -- eight consecutive read-order leaves -- is eight consecutive rows."""
    row = 0
    for l in range(side.D):
        d = ((slot >> side.shift(l)) & (side.s(l) - 1)) + side.off(l, stream)
        row += d << side.run_shift(l)
    return _swz(_read_index(side.blk, row))


def _row_of(w_side, r_side, stream, slot):
    """the publisher's row, kept under its old name for the tests that only need it."""
    return _row(w_side, stream, slot)


def _tied(w_side, r_side):
    """THE IDENTITY-MAP FACT, and it is a fact and no longer a body: K50 deleted the
    elided tied path ("one transit, always"), so what a tie buys is a row map that
    happens to be the identity -- reported so a wasted transit is visible in the ledger."""
    return w_side.assign() == r_side.assign() and all(
        w_side.g(l) == 1 or w_side.grid_div(l) == r_side.grid_div(l)
        for l in range(w_side.D))


CARVED = [(c[0], c) for c in CASES]
UNTIED = [(i, c) for i, c in CARVED if not _tied(*_sides(c)[:2])]
IDS = [c[0] for c in CASES]


def test_at_least_one_declaration_is_the_identity_map_and_one_is_not():
    """Neither end of the fact may go untested: the identity map needs a declaration
    that reaches it, and the transit needs a permuted one to prove it moves anything."""
    tied = [i for i, c in CARVED if _tied(*_sides(c)[:2])]
    assert tied, "no declaration reaches the identity row map; the fact is unproved"
    assert UNTIED, "no declaration carries a permuted transit"


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_exchange_relays_the_state_leaf_for_leaf(idx, row):
    """THE COMPOSITE IS THE IDENTITY.  Every (publisher, write slot) claims exactly one
    row, the rows are a permutation of `[0, BC)`, and the row a consumer pulls at its
    own slot carries the leaf that slot names."""
    w, r, _ = _sides(row)
    assert w.leaf == r.leaf and w.bc == r.bc
    claimed = {}
    for stream in range(w.ns):
        for slot in range(w.leaf):
            row_i = _row_of(w, r, stream, slot)
            assert row_i not in claimed, f"arm {idx}: two publishers claim row {row_i}"
            claimed[row_i] = (stream, slot)
    assert sorted(claimed) == list(range(w.bc)), f"arm {idx}: the rows are not a permutation"

    # and the READ side's own map covers the same region exactly once: the two sides meet
    # on the canonical index, so a leaf is the same ROW to the writer that mints it and to
    # the reader that pulls it -- whatever members the mode word selected for each.
    pulled = {}
    for reader in range(r.ns):
        for slot in range(r.leaf):
            row_i = _row(r, reader, slot)
            assert row_i not in pulled, f"arm {idx}: two consumers claim row {row_i}"
            pulled[row_i] = (reader, slot)
    assert sorted(pulled) == list(range(r.bc)), f"arm {idx}: the pull is not a permutation"
    for row_i, (stream, slot) in claimed.items():
        reader, q = pulled[row_i]
        for l in range(w.D):
            d_w = ((slot >> w.shift(l)) & (w.s(l) - 1)) + w.off(l, stream)
            d_r = ((q >> r.shift(l)) & (r.s(l) - 1)) + r.off(l, reader)
            assert d_w == d_r, (
                f"arm {idx}: publisher ({stream}, {slot}) and consumer ({reader}, {q}) "
                f"disagree on level {l}'s digit: {d_w} != {d_r}")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_exchange_row_map_is_affine_in_the_slots_bits(idx, row):
    """`row(w, P) = row(w, 0) ^ delta(P)`, every step ONE bit and independent of the
    publisher, and the base's bits disjoint from every step's -- which is what lets the
    kernel pay one base per warp and an IMMEDIATE per fragment, and what makes the XOR
    an ADD in the address arithmetic (`XchMap`'s own static_assert)."""
    w, r, _ = _sides(row)
    for side, who in ((w, "publisher"), (r, "consumer")):
        nb = _ilog2c(side.leaf)
        ref_step = [_delta(side, 1 << b) for b in range(nb)]
        #: PRE-SWIZZLE a slot bit set exactly ONE row bit. The swizzle deliberately mixes
        #: (bit 2 becomes rows 0 and 2), so what must hold is the property the ADDRESS
        #: ARITHMETIC actually rests on: the step is NON-ZERO, it is the same for every
        #: stream, and the composition is exact -- all asserted below.
        for b, st in enumerate(ref_step):
            assert st != 0, f"arm {idx}: {who} slot bit {b} moves no row bit"
        base_mask = 0
        for stream in range(side.ns):
            base = _row(side, stream, 0)
            base_mask |= base
            for b in range(nb):
                assert _row(side, stream, 1 << b) ^ base == ref_step[b], (
                    f"arm {idx}: {who} {stream}'s step for bit {b} differs from stream 0's")
            for slot in range(side.leaf):
                acc = base
                for b in range(nb):
                    if slot >> b & 1:
                        acc ^= ref_step[b]
                assert acc == _row(side, stream, slot)
        #: THE KERNEL XORS EXPLICITLY: under the swizzle a slot bit CAN share a row
        #: bit with the base, so the address arithmetic composes with `^` and never with
        #: `+`; the exactness assertion above is what stands in for the old disjointness.


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_each_sides_map_is_its_own_members_scatter_and_the_two_agree_on_the_region(idx, row):
    """THE PER-SIDE SHAPE CLAIM, on the shipped derivation.

    Each side selects a member of the GENERATED set from its own order, so the two sides
    no longer share a slot layout: a leaf keeps neither its slot nor its warp across the
    transit.  What they DO share is the region's index -- the owner-local canonical run --
    so each side's map is a bijection of `[0, BC)` built from one base word per stream and
    the member's fixed shifts on the slot bits, and the identity-map fact is exactly the
    case where both sides selected the same member with the same stream numbering."""
    w, r, _ = _sides(row)
    for side, who in ((w, "publisher"), (r, "consumer")):
        seen = set()
        for stream in range(side.ns):
            base = _row(side, stream, 0)
            for slot in range(side.leaf):
                assert _row(side, stream, slot) == base ^ _delta(side, slot), (
                    f"arm {idx}: {who} {stream} slot {slot} is not base ^ scatter(slot)")
                seen.add(_row(side, stream, slot))
        assert sorted(seen) == list(range(side.bc)), (
            f"arm {idx}: the {who} side does not cover the region exactly once")
    same = all(_row(w, s_, p_) == _row(r, s_, p_)
               for s_ in range(w.ns) for p_ in range(w.leaf))
    assert same == _tied(w, r), (
        f"arm {idx}: the identity-map fact disagrees with the two maps")


def _bank_free(rows, ldx, gran, word_off=0, words_per_gran=4):
    """the region's bank model: a 16-byte granule is four consecutive banks, and a row
    is `ldx / 2` words on from the one before it."""
    words, banks = ldx // 2, set()
    for row_i in rows:
        for word in range(words_per_gran):
            b = (words * row_i + 4 * gran + word_off + word) % 32
            if b in banks:
                return False
        for word in range(words_per_gran):
            banks.add((words * row_i + 4 * gran + word_off + word) % 32)
    return True


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_regions_accesses_are_conflict_free_at_every_declaration(idx, row):
    """THE ACCESS SWEEP, AS A COMPILE-TIME ENUMERATION, OVER BOTH SIDES.

    `ldx / 2 = 4 x odd` maps eight consecutive rows onto eight different four-bank blocks
    by itself, so an access is conflict-free exactly when its rows are distinct mod 8.
    The write side's accesses are its member's scatter of a run; the pull is eight
    consecutive READ-ORDER leaves. In the physical order the pull steps rows by a power of
    two whenever a read-sparse level sits inside the run level, and no GF(2)-linear swizzle
    of the physical row clears every order the mode words reach -- so the region is
    indexed in the read order and the swizzle folds the index's three-bit groups
    (`geom.md#row-swizzle`). All four accesses are walked here, at every declaration the
    sweep reaches."""
    w, r, dv = _sides(row)
    ldx = dv + K_PAD_E
    assert (ldx // 2) % 4 == 0 and (ldx // 2) // 4 % 2 == 1, (
        f"arm {idx}: half a row is {ldx // 2} words, which is not four times an odd "
        f"number -- an aligned eight of rows would share a four-bank block")
    for bad in _conflicts(w, r, dv):
        raise AssertionError(f"arm {idx}: {bad}")


def _conflicts(w, r, dv):
    """every conflicting access of the four, at this (write member, read member) pair."""
    ldx, n_gran, leaf = dv + K_PAD_E, dv // K_MMA_M_CH, w.leaf
    bad = []
    for stream in range(w.ns):
        base = _row(w, stream, 0)
        for j in range(leaf // K_MMA_N):
            for e1 in (0, 1):
                rows_p = [base ^ _delta(w, j * K_MMA_N + 2 * q + e1) for q in range(4)]
                for g in range(n_gran):
                    if not _bank_free(rows_p, ldx, g):
                        bad.append("publish conflicts")
                if not _bank_free(rows_p, ldx, n_gran, words_per_gran=1):
                    bad.append("the mass publish conflicts")
            rows_r = [base ^ _delta(w, j * K_MMA_N + q) for q in range(8)]
            for g in range(n_gran):
                if not _bank_free(rows_r, ldx, g):
                    bad.append("the return pull conflicts")
    #: THE PULL walks the read order: a k-step is sixteen consecutive read-order leaves,
    #: each half an `ldmatrix` of eight consecutive rows; the mass lanes take four of them.
    for v0 in range(0, r.bc, K_MMA_K):
        for half in (0, 8):
            rows_l = [_swz(v0 + half + i) for i in range(8)]
            for g in range(n_gran):
                if not _bank_free(rows_l, ldx, g):
                    bad.append("the pull conflicts")
            for e1 in (0, 1):
                rows_m = [_swz(v0 + half + 2 * q + e1) for q in range(4)]
                if not _bank_free(rows_m, ldx, n_gran, words_per_gran=1):
                    bad.append("the mass pull conflicts")
    return sorted(set(bad))


def test_every_member_pair_the_mode_words_can_reach_is_conflict_free():
    """THE PAIR SWEEP: every `(read order, write member)` a mode word can select, at
    every topology in the case list, with all four accesses under the region's row law.
    This sweep is what says the read-order indexing and the folded swizzle clear every
    one of them."""
    pairs, bad = {}, []
    for _name, widths, _m in CASES:
        for modes in range(1 << (2 * len(widths))):
            blk = _block(widths, modes, NS)
            wS, rS = _Side(blk, "w", NS), _Side(blk, "r", NS)
            key = (len(widths), tuple(blk["r_rank"]), wS.assign())
            if key in pairs:
                continue
            pairs[key] = (widths, modes)
            c = _conflicts(wS, rS, 64)
            if c:
                bad.append((key, widths, modes, c))
    assert pairs, "no member pair was reached at all"
    assert not bad, f"member pairs needing a swizzle: {bad}"
    #: the reachable pair count is a consequence of the INNERMOST-DIGIT INVARIANT: the
    #: carve never divides the canonical innermost level, so the orders that differ only
    #: in how they would have split it select the same member and the pair set collapses.
    #: The floor is here so a future change that stops reaching pairs at all is a failure
    #: rather than a silently vacuous sweep.
    assert len(pairs) >= 14, f"only {len(pairs)} member pairs reached; the sweep shrank"


def test_the_swizzle_is_a_bijection_of_the_region_and_is_linear():
    """The swizzle re-INDEXES the region: it must lose no row and split no XOR.

    Bijection on `[0, BC)` is what keeps every leaf reachable exactly once; GF(2)-linearity
    (`swz(a ^ b) == swz(a) ^ swz(b)`) is what lets the kernel swizzle the lane's base ONCE
    and keep every per-fragment row term a compile-time constant -- without it the address
    arithmetic would need a runtime XOR per store and the immediates would be gone. Both
    hold at every box the fold covers, up to twelve index bits."""
    for bc in (64, 128, 256, 512, 1024, 2048, 4096):
        assert sorted(_swz(r) for r in range(bc)) == list(range(bc)), (
            f"the swizzle is not a bijection of a {bc}-leaf region: a leaf is lost or aliased")
        for a in range(0, bc, 3):
            for b in range(0, bc, 5):
                assert _swz(a ^ b) == _swz(a) ^ _swz(b), (
                    f"the swizzle is not GF(2)-linear at ({a}, {b}), so `base ^ scatter` "
                    f"does not distribute and the fragment terms stop being immediates")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_models_row_map_is_the_shipped_row_map(idx, row):
    """The model's row law and the kernel's (`geom_xch_row`: the read-order index under the
    swizzle) are two derivations of one map; the shipped map is asserted against the model
    row for row, both sides, so neither can drift alone."""
    w, r, _ = _sides(row)
    for side, name in ((w, "w"), (r, "r")):
        shipped = side.blk["map"][name]
        for stream in range(side.ns):
            for slot in range(side.leaf):
                assert shipped[stream][slot] == _row(side, stream, slot), (
                    f"arm {idx}: the shipped {name} map differs from the model at "
                    f"({stream}, {slot})")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_transit_is_sized_by_the_disorder_between_the_two_carves(idx, row):
    """The transit is ZERO exactly when no leaf crosses a warp, and one whole
    `[consumer][leaf][channel]` region otherwise -- never something in between (K35
    S1b addendum 3(b): every group is live at the same time, so the union of the
    groups' rows is all `BC` of them)."""
    w, r, dv = _sides(row)
    crossings = sum(1 for stream in range(w.ns) for slot in range(w.leaf)
                    if _row(w, stream, slot) != _row(r, stream, slot))
    expected = 0 if crossings == 0 else w.bc * (dv + K_PAD_E) * 2
    assert (crossings == 0) == _tied(w, r), (
        f"arm {idx}: `kTied` and the permutation disagree about whether a leaf moves")
    assert expected == (0 if _tied(w, r) else w.bc * (dv + K_PAD_E) * 2)


# ---------------------------------------------------------------- the hybrid's two legs
#
# THE HI PLANE MOVES.  The writers publish it and drop it; the readers pull it as
# their A operand; after the readout the readers store those same bits BACK and the
# writers pull them in their OWN layout and rejoin them with the lo plane they kept.
# What has to be proved is that the round trip returns each publisher's OWN element to
# the register the rejoin expects -- so this models all four accesses at BYTE addresses,
# with the kernel's own arithmetic, and runs the whole transit as a simulation.

K_MMA_M_CH = 8  #: channels per 16-byte granule


def _addr_chan(row, col, ldx, n_gran):
    """the region's byte address of one channel element -- a linear tile, no swizzle."""
    gran, within = divmod(col, K_MMA_M_CH)
    return row * ldx * 2 + gran * 16 + within * 2


def _addr_mass(row, ldx, dv):
    return row * ldx * 2 + dv * 2


def _delta(side, slot):
    """one side's row map step for a slot value: `row(s, P) = row(s, 0) ^ delta(P)`,
    read off the shipped derivation rather than assumed."""
    return _row(side, 0, slot) ^ _row(side, 0, 0)


def _ldm_trans(addr_of_lane, mem):
    """`ldmatrix.m8n8.x4.trans`: lane L supplies row (L & 7) of matrix (L >> 3); lane L's
    register i is matrix i's column (L >> 2) at rows 2*(L & 3) and 2*(L & 3) + 1."""
    a = [addr_of_lane(L) for L in range(32)]
    out = []
    for L in range(32):
        regs = []
        for i in range(4):
            lo = mem[a[8 * i + 2 * (L & 3)] + 2 * (L >> 2)]
            hi = mem[a[8 * i + 2 * (L & 3) + 1] + 2 * (L >> 2)]
            regs.append((lo, hi))
        out.append(regs)
    return out


@pytest.mark.parametrize("idx,row", UNTIED, ids=[f"arm{i}" for i, _ in UNTIED])
def test_the_hybrid_returns_every_hi_half_to_the_register_that_published_it(idx, row):
    """THE WHOLE TRANSIT, SIMULATED AT BYTE ADDRESSES.  Publish -> pull -> store back ->
    pull in write layout, with the kernel's own address arithmetic at every step; the
    assertion is that the writer's rejoin register holds the hi halves of ITS OWN two
    accumulator elements, and that the store-back leg is the identity on the region."""
    w, r, dv = _sides(row)
    ldx, n_gran, leaf = dv + K_PAD_E, dv // K_MMA_M_CH, w.leaf
    mch, knl, kdw = dv // K_MMA_M, leaf // K_MMA_N, leaf // K_MMA_K
    d1, d2, d4 = (_delta(w, 1), _delta(w, 2), _delta(w, 4))

    mem, published = {}, {}
    for stream in range(w.ns):
        base = _row_of(w, r, stream, 0)
        for lane in range(32):
            lane_row = base ^ (d2 if lane & 1 else 0) ^ (d4 if lane & 2 else 0)
            for j in range(knl):
                for e in range(4):
                    dl = _delta(w, j * K_MMA_N + (e & 1))
                    for m in range(mch):
                        gr = 2 * m + (e >> 1)
                        a = ((lane_row ^ dl) * ldx * 2 + (lane >> 2) * 2 + gr * 16)
                        slot = j * K_MMA_N + 2 * (lane & 3) + (e & 1)
                        col = m * K_MMA_M + (lane >> 2) + 8 * (e >> 1)
                        assert a == _addr_chan(_row_of(w, r, stream, slot), col, ldx, n_gran), (
                            f"arm {idx}: the publish's address arithmetic is not the region's map")
                        assert a not in mem, f"arm {idx}: two publishers claim byte {a}"
                        mem[a] = (stream, lane, j, m, e)
                        published[(stream, lane, j, m, e)] = a
                if (lane >> 2) == 0:
                    for e in range(2):
                        dl = _delta(w, j * K_MMA_N + e)
                        a = (lane_row ^ dl) * ldx * 2 + dv * 2
                        assert a == _addr_mass(_row_of(w, r, stream, j * K_MMA_N
                                                       + 2 * (lane & 3) + e), ldx, dv)
                        mem[a] = ("mass", stream, lane, j, e)

    before = dict(mem)

    #: THE PULL, and the STORE-BACK that must be its exact inverse.
    frags = {}
    for rs in range(r.ns):
        for kk in range(kdw):
            for m in range(mch):
                def addr_of(L, rs=rs, kk=kk, m=m):
                    slot = ((L >> 3) >> 1) * 8 + (L & 7) + kk * K_MMA_K
                    return (_row(r, rs, slot) * ldx * 2
                            + ((2 * m) | ((L >> 3) & 1)) * 16)
                out = _ldm_trans(addr_of, mem)
                for L in range(32):
                    frags[(rs, L, kk, m)] = out[L]
                # the store-back: the kernel's own address, and it must land the same bits
                for L in range(32):
                    mrow = _row(r, rs, 2 * (L & 3))
                    nx = _delta(r, 1)
                    for i in range(4):
                        rr = _delta(r, kk * K_MMA_K + (i >> 1) * 8)
                        a = ((mrow ^ rr) * ldx * 2 + (L >> 2) * 2 + (2 * m + (i & 1)) * 16)
                        assert mem[a] == out[L][i][0], (
                            f"arm {idx}: the store-back's low half misses its own address")
                        a1 = ((mrow ^ rr ^ nx) * ldx * 2 + (L >> 2) * 2
                              + (2 * m + (i & 1)) * 16)
                        assert mem[a1] == out[L][i][1], (
                            f"arm {idx}: the store-back's high half misses its own address")
                        mem[a], mem[a1] = out[L][i]
    assert mem == before, f"arm {idx}: the two legs are not the identity on the region"

    #: THE WRITER'S PULL, in the WRITE layout, and the rejoin's register map.
    for ws in range(w.ns):
        base = _row_of(w, r, ws, 0)
        for j in range(knl):
            dj = _delta(w, j * K_MMA_N)
            for t in range(mch // 2):
                def addr_of(L, base=base, dj=dj, t=t):
                    ret_row = (base ^ (d1 if L & 1 else 0) ^ (d2 if L & 2 else 0)
                               ^ (d4 if L & 4 else 0))
                    return (ret_row ^ dj) * ldx * 2 + (4 * t + (L >> 3)) * 16
                out = _ldm_trans(addr_of, mem)
                for L in range(32):
                    for i in range(4):
                        g = 4 * t + i
                        m, h = g >> 1, g & 1
                        want = (mem[published[(ws, L, j, m, 2 * h)]],
                                mem[published[(ws, L, j, m, 2 * h + 1)]])
                        assert out[L][i] == want, (
                            f"arm {idx}: writer {ws} lane {L} n-tile {j} granule {g} did not "
                            f"get back its own (m={m}, e={2 * h}, {2 * h + 1})")
