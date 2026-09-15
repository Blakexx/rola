"""CONFORMANCE FAMILIES — the data-regime box, as DATA.

The tier-1/tier-2 suites answer *is the kernel the oracle's function* and *is it
the same function at every launch geometry*. This tier answers the remaining
question: **is it that function over the whole distribution space the model can
put through it?** Routing is a moving distribution (the gain knob alone sweeps
candidate fraction ~0.96 -> ~0.09 in training), so any single-density fixture
set is an artifact; the family scheme makes the covered regimes DECLARED and
the uncovered ones LISTED.

WHY AXIS-DEFINED GENERATION AND NOT NAIVE RANDOM (ratified design, 2026-08-05):
independent per-token sampling provably manufactures ANTI-structure — synthetic
cells are anti-coincident exactly where realized producer routing is
coincident — so a "random" fixture set silently occupies one corner of the box
while reporting itself as general. Every fixture here therefore states its
coordinates on the regime axes, and an in-test NON-VACUITY assertion proves the
realized tensors actually sit there (`generators.assert_nonvacuous`): a fixture
that cannot demonstrate its claimed regime is treated as ABSENT, which is the
token-0 vacuity lesson generalized.

THREE SOURCES over the box, each a ``Family.source`` value:

* ``stratified`` — random generation confined to one stratum per axis.
* ``producer``   — the REAL entmax producer at swept gains, because some
  correlations only realized routing produces and no sampler imitates them.
* ``adversarial``— named corners, this week's bug list FROZEN PERMANENT:
  cold read, all-read-only, K = 0 fold, ragged tail, chunk boundary. A corner,
  once named, never leaves this file; a future bug-discovered axis adds one.

**This module is pure data and pure CPU** (the matrix.py doctrine): it imports
no torch, so the census join (`test_conformance_census.py`) runs in the CPU CI
job. Generation lives in ``generators.py``; execution in ``test_families.py``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tests.oracle.instantiation import (
    BUILT_TOPOLOGIES,
)

# ---------------------------------------------------------------------------
# The regime axes — the census join's NEW dimension
# ---------------------------------------------------------------------------

#: Every family states a value for EVERY axis. ``basis`` says what the axis is
#: and why it is an axis (a bug class or a distribution fact, never a vibe).
REGIME_AXES: Mapping[str, tuple[str, ...]] = {
    #: Support density. The gain knob sweeps it in training, so a single-density
    #: verdict is an artifact by standing rule; ``swept`` is the producer-realized
    #: sweep itself.
    "density": ("dense", "sparse", "swept"),
    #: Temporal coherence of the routing. Independent per-token draws are the
    #: ANTI-structured corner; realized routing is coherent over spans, and
    #: coherence is what makes owner tiles hot or cold for whole segments.
    "coherence": ("iid", "coherent"),
    #: Read-write support correlation. ``tied`` (read == write), ``independent``
    #: (overlapping but distinct), ``anti`` (disjoint at leaf level — every read
    #: lands on an unwritten leaf).
    "rw_correlation": ("tied", "independent", "anti"),
    #: Mass concentration. ``concentrated`` puts most of a level's amplitude on
    #: one digit; ``cold_read`` names the regime where read mass sits on leaves
    #: no write ever touched (the carve-out's data-side twin).
    "mass": ("spread", "concentrated", "cold_read"),
    #: The token tail. ``ragged`` is T not a multiple of the chunk length (and,
    #: under a split, segments of unequal tile count) — the off-by-one surface.
    "tail": ("divisible", "ragged"),
    #: Per-token support cardinality stratum: exactly one digit, some, all.
    "support": ("singleton", "partial", "full"),
}


# ---------------------------------------------------------------------------
# A family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    name: str
    source: str                      # "stratified" | "producer" | "adversarial"
    #: The claimed cell of the regime box: one value per REGIME_AXES key, all
    #: keys present. `__post_init__` enforces completeness so a family cannot
    #: silently opt out of an axis.
    regime: Mapping[str, str]
    #: The instantiation the family runs at. Keys: widths, d_v, B, T, H,
    #: norm, decay, paging, seed -- plus generator parameters under `params`.
    #: The census derives the axis values from these, and a CUDA meta-row
    #: asserts the DECLARED executor matches what the engine's arm fold says,
    #: so a declaration cannot lie (`test_family_meta.py`). The endgame executor dropped `BT`
    #: and `mma`; P67 D2-b dropped `BC` -- the chunk arm's `BC` is a slot tile
    #: inside a built arm, fixed by the manifest and not host-selectable, so it
    #: is not a coordinate a family can pick.
    config: Mapping[str, Any]
    basis: str
    #: The generator `generators.py` dispatches on; defaults to the source name.
    generator: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.basis, f"family {self.name!r} declares no basis"
        assert self.source in ("stratified", "producer", "adversarial"), self.source
        assert set(self.regime) == set(REGIME_AXES), (
            f"family {self.name!r} must state every regime axis; missing "
            f"{set(REGIME_AXES) - set(self.regime)}, extra {set(self.regime) - set(REGIME_AXES)}")
        for axis, value in self.regime.items():
            assert value in REGIME_AXES[axis], (
                f"family {self.name!r}: {value!r} is not a declared value of regime "
                f"axis {axis!r}")
        for key in ("widths", "d_v", "T", "norm", "decay", "paging", "seed"):
            assert key in self.config, f"family {self.name!r} config lacks {key!r}"

    @property
    def id(self) -> str:
        return self.name


def _cfg(widths=(8, 8), d_v=64, B=1, T=128, H=2, norm="global",
         decay=False, paging=False, seed=0, m0=False,
         p_read=1.0, p_write=1.0):
    return dict(widths=widths, d_v=d_v, B=B, T=T, H=H, norm=norm,
                decay=decay, paging=paging, seed=seed, m0=m0,
                p_read=p_read, p_write=p_write)


_MID = dict(density="sparse", coherence="iid", rw_correlation="independent",
            mass="spread", tail="divisible", support="partial")


# ---------------------------------------------------------------------------
# Source 1 — stratified random, one family per stratum
# ---------------------------------------------------------------------------

_STRATIFIED: tuple[Family, ...] = (
    Family(
        "stratified/dense-full", "stratified",
        dict(density="dense", coherence="iid", rw_correlation="independent",
             mass="spread", tail="divisible", support="full"),
        _cfg(seed=101, decay=True, norm="global"),
        basis="the FA-like end of the box: every digit live on every token. The "
              "kernel's dense fast path must be the same function, and this is the "
              "cell the gain knob's cf~0.96 end realizes."),
    Family(
        "stratified/sparse-partial", "stratified",
        dict(**_MID),
        _cfg(seed=102, p_read=0.4, p_write=0.4, m0=True),
        basis="the mid-box cell: Bernoulli-thinned supports with a carried state, "
              "the regime most training steps live in mid-sweep."),
    Family(
        "stratified/singleton", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="concentrated", tail="divisible", support="singleton"),
        _cfg(seed=103, m0=True),
        params=dict(one_hot=True),
        basis="exactly one live digit per level per token — the sparsest reachable "
              "support and the K = 1 end of the compaction. All mass on one leaf "
              "per token is also the concentrated-mass extreme."),
    Family(
        "stratified/tied", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="tied",
             mass="spread", tail="divisible", support="partial"),
        _cfg(seed=104, p_read=0.5, p_write=0.5),
        params=dict(tied=True),
        basis="read == write bitwise — the self-attention-like correlation a "
              "shared-stream producer realizes, and the cell where a read/write "
              "index swap in the kernel is INVISIBLE. It is in the box so the "
              "census records that visibility limit rather than hiding it; the "
              "anti cell below is where such a swap is maximally visible."),
    Family(
        "stratified/anti", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="anti",
             mass="cold_read", tail="divisible", support="partial"),
        _cfg(seed=105, m0=True),
        params=dict(anti=True),
        basis="read and write supports DISJOINT at leaf level: every read lands "
              "on a leaf no write ever touches, so y is a function of the carried "
              "state alone. Synthetic anti-coincidence is exactly what independent "
              "sampling manufactures by accident; here it is deliberate, named, "
              "and paired with a nonzero M_0 so the reads are observable."),
    Family(
        "stratified/coherent", "stratified",
        dict(density="sparse", coherence="coherent", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(seed=106, decay=True, p_read=0.5, p_write=0.5),
        params=dict(block=16),
        basis="blockwise-constant routing (16-token spans share one draw): the "
              "temporal structure realized routing has and iid draws do not. "
              "Coherence keeps whole owner tiles hot or cold for entire tiles, "
              "which is the regime the schedule's candidate masks actually see."),
    Family(
        "stratified/concentrated", "stratified",
        dict(density="dense", coherence="iid", rw_correlation="independent",
             mass="concentrated", tail="divisible", support="full"),
        _cfg(seed=107, decay=True, norm="global"),
        params=dict(peak=0.9),
        basis="~90% of every level's amplitude on one digit, the rest spread thin: "
              "full support with extreme mass skew, the cell where an accumulation-"
              "order defect is amplified rather than averaged away."),
    Family(
        "stratified/imbalanced-heads", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(seed=108, H=4, m0=True),
        params=dict(head_densities=(1.0, 0.5, 0.15, "one_hot")),
        basis="four heads at four densities (dense, half, thin, one-hot): heads "
              "share a launch but not a distribution, so per-head anything that "
              "leans on a batch-uniform density assumption breaks exactly here.",
        generator="imbalanced_heads"),
    Family(
        "stratified/ragged", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="spread", tail="ragged", support="partial"),
        _cfg(seed=109, T=83, p_read=0.6, p_write=0.6, norm="global"),
        basis="T = 83, not a multiple of the 32-token chunk (`TOKEN_WORD`): the "
              "consumer's last super-chunk closes short. The stratified twin of "
              "the adversarial ragged corner, at mid-box density."),
    #: ARM VARIANTS. The named cells above run at one decay arm each; these
    #: close the near-default census holes — each variant exists because the
    #: join listed its (DECAY x regime) pair as empty and the pair is CHEAP to
    #: cover, which the design prefers to expecting it empty. `norm` is no
    #: longer a second axis here -- `"raw"` is dead and every family runs
    #: `"global"` -- so these variants now cover DECAY alone (their names keep
    #: the historical "-raw-"/"-global-" tokens for stability; they no longer
    #: name a live arm).
    Family(
        "stratified/dense-full-raw-plain", "stratified",
        dict(density="dense", coherence="iid", rw_correlation="independent",
             mass="spread", tail="divisible", support="full"),
        _cfg(seed=111),
        basis="the dense-full cell at the no-decay arm."),
    Family(
        "stratified/tied-global-decay", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="tied",
             mass="spread", tail="divisible", support="partial"),
        _cfg(seed=114, p_read=0.5, p_write=0.5, norm="global", decay=True),
        params=dict(tied=True),
        basis="the tied cell at the (global, decay) arm."),
    Family(
        "stratified/anti-global-decay", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="anti",
             mass="cold_read", tail="divisible", support="partial"),
        _cfg(seed=115, m0=True, norm="global", decay=True),
        params=dict(anti=True),
        basis="the anti cell at the (global, decay) arm — decayed state under "
              "cold reads is where a clock that wrongly ticks on READS would "
              "show (the clock is the STORED WRITE product)."),
    Family(
        "stratified/singleton-global-decay", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="concentrated", tail="divisible", support="singleton"),
        _cfg(seed=113, m0=True, norm="global", decay=True),
        params=dict(one_hot=True),
        basis="the singleton cell at the (global, decay) arm."),
    Family(
        "stratified/ragged-raw-decay", "stratified",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="spread", tail="ragged", support="partial"),
        _cfg(seed=119, T=83, p_read=0.6, p_write=0.6, decay=True),
        basis="the ragged cell at the decay arm — a partial word is where a "
              "decay clock accumulated per full word would slip."),
)


# ---------------------------------------------------------------------------
# Source 2 — the realized producer, at swept gains
# ---------------------------------------------------------------------------

_PRODUCER: tuple[Family, ...] = (
    Family(
        "producer/swept-gain", "producer",
        dict(density="swept", coherence="coherent", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(widths=(64,), d_v=64, T=64, H=4, seed=201, norm="global"),
        params=dict(gains=(0.25, 1.0, 4.0, 16.0)),
        basis="the REAL entmax producer over one hidden stream at four input "
              "gains: realized routing carries correlations (token coherence from "
              "the hidden state, read-write coincidence from the shared stream) "
              "that no sampler imitates, and the gain is the knob that sweeps its "
              "density in training. Each gain's bundle runs both arms of rola_op "
              "and they must agree at the exactness tolerance. FLAT ROUTING: "
              "`D = 1` at the manifest\'s flat-baseline width, so this sweep is "
              "the one family carrying realized routing onto the depth-1 arm.",
        generator="swept_gain"),
    Family(
        "producer/swept-gain-d64", "producer",
        dict(density="swept", coherence="coherent", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(widths=(64, 64), d_v=64, T=128, H=2, seed=202, norm="global"),
        params=dict(gains=(0.25, 1.0, 4.0, 16.0)),
        basis="the gain sweep INSIDE the chunk consumer envelope (uniform "
              "two-level, d_v=64, global): each gain bundle runs both arms "
              "of rola_op and the OUTPUTS must agree at the exactness "
              "tolerance -- the cross-arm gate the decoupled write gain "
              "earned.  Final-state agreement joins when the paged state "
              "I/O lands.",
        generator="swept_gain"),
    Family(
        "producer/swept-gain-d2", "producer",
        dict(density="swept", coherence="coherent", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(widths=(8, 8), d_v=64, T=64, H=4, seed=203),
        params=dict(gains=(0.25, 1.0, 4.0, 16.0)),
        basis="the gain sweep at the DEFAULT instantiation (D = 2, B = 8): the "
              "cell every named regime family runs at, carrying REALIZED routing "
              "onto it rather than a sampler, so the mid-box cell is covered by "
              "the producer as well as by generation.",
        generator="swept_gain"),
)


# ---------------------------------------------------------------------------
# Source 3 — named adversarial corners, FROZEN PERMANENT
# ---------------------------------------------------------------------------
#
# This week's bug list, kept forever. Deleting a corner is forbidden the way
# deleting a regression test is; a new bug-discovered axis ADDS one.

_ADVERSARIAL: tuple[Family, ...] = (
    Family(
        "corner/cold-read", "adversarial",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="cold_read", tail="divisible", support="partial"),
        _cfg(seed=301, m0=True),
        params=dict(write_live_digits=2),
        basis="level 0's write support confined to 2 of 8 digits while reads are "
              "dense: 48 of 64 leaves are read but never written. The "
              "carve-out's data regime, and the self-normalizing-readout fact "
              "(the self-normalizing readout's dead-leaf ruling) made checkable: "
              "reading never-written state must be exact, not approximately dead."),
    Family(
        "corner/cold-read-global-decay", "adversarial",
        dict(density="sparse", coherence="iid", rw_correlation="independent",
             mass="cold_read", tail="divisible", support="partial"),
        _cfg(seed=306, m0=True, norm="global", decay=True),
        params=dict(write_live_digits=2),
        basis="the cold-read corner at the (global, decay) arm: a decayed, "
              "normalized readout over never-written leaves — the denominator "
              "must come from the carried mass column alone, and the clock must "
              "not tick where no write lands."),
    Family(
        "corner/all-read-only", "adversarial",
        dict(density="sparse", coherence="coherent", rw_correlation="anti",
             mass="cold_read", tail="divisible", support="singleton"),
        _cfg(seed=302, m0=True),
        params=dict(),
        basis="the degenerate end of cold read: ONE leaf ever written (a constant "
              "one-hot write), every other leaf read-only for the whole sequence. "
              "The written column exercises the fold alone; the other N-1 rows "
              "must be exactly the carried state.",
        generator="all_read_only"),
    Family(
        "corner/chunk-boundary", "adversarial",
        dict(density="sparse", coherence="coherent", rw_correlation="independent",
             mass="spread", tail="divisible", support="partial"),
        _cfg(seed=305, T=192, m0=True),
        params=dict(boundary_tiles=(2, 5), live_before=4, live_after=2),
        basis="the write support FLIPS exactly at chunk boundaries (tokens 64 and "
              "160 under the operator\'s 32-token chunk, `rola.engine.rules.arm."
              "CHUNK_TOKENS`; `T = 192` so 6 chunks exist to hold both boundary "
              "chunks): the support-change-at-a-chunk-boundary corner, where an "
              "inclusive/exclusive chunk-range error moves which chunk last "
              "touches a leaf -- a support that changes mid-chunk tests a "
              "different (weaker) thing. The tiled consumer's retirement took the DECAY off this "
              "corner: decay is retired from the operator, so a decayed corner "
              "could only ever assert the refusal and the boundary claim -- the "
              "bug class the corner is frozen for -- would never reach a kernel.",
        generator="chunk_boundary"),
)


# ---------------------------------------------------------------------------
# The instantiation sweep — one mid-regime fixture across the shipped grid
# ---------------------------------------------------------------------------
#
# The named families above pin the DEFAULT instantiation (D = 2, B = 8,
# unpaged); this sweep carries the SAME mid-box regime across every built chunk
# arm and both backings, so the census join's instantiation dimension is covered
# by measurement rather than by extrapolation from one cell.

def _sweep() -> tuple[Family, ...]:
    """One mid-box fixture per BUILT chunk arm, plus the paged backing.

    P67 D2-b re-anchored this sweep. It used to cross the TILED consumer's
    template grid (`BC` x `d_v` x `decay`); every one of those three axes is
    gone -- `BC` belongs to the built arm, `d_v = 64` is the whole matrix,
    decay is retired from the operator -- so the sweep now walks the axes that
    remain: the manifest's `(D, B)` topologies, taken from
    `tests.oracle.instantiation.BUILT_TOPOLOGIES` rather than written out,
    so a widened matrix demands a family here on the day it lands.
    """
    rows: list[Family] = []
    # deterministic, order-derived -- never hash(), which is salted per process
    for seed, (D, B) in enumerate(sorted(BUILT_TOPOLOGIES), start=501):
        rows.append(Family(
            f"sweep/D{D}-B{B}", "stratified", dict(**_MID),
            _cfg(widths=(B,) * D, d_v=64, T=64, m0=True,
                 p_read=0.5, p_write=0.5, seed=seed),
            basis="the built manifest at the mid-box regime: the same claimed "
                  "cell at every (D, B) arm the dispatch can reach, so the "
                  "instantiation dimension is covered by measurement rather "
                  "than by extrapolation from one topology."))
    rows.append(Family(
        "sweep/paged", "stratified", dict(**_MID),
        _cfg(widths=(16, 16, 16), d_v=64, T=64, paging=True,
             p_read=0.5, p_write=0.5, seed=420),
        basis="the paging runtime axis at the mid-box regime, on a built arm "
              "(D = 3, B = 16). Residency is planned from the kernel's own "
              "exact write bitmap; the partially-resident composition and the "
              "bit-identity claim are tier 2's rows."))
    return tuple(rows)


FAMILIES: tuple[Family, ...] = _STRATIFIED + _PRODUCER + _ADVERSARIAL + _sweep()

_names = [f.name for f in FAMILIES]
assert len(_names) == len(set(_names)), "duplicate family names"

FAMILY_BY_NAME: Mapping[str, Family] = {f.name: f for f in FAMILIES}
