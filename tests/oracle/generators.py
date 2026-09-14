"""Generation and non-vacuity for the conformance families.

``families.py`` is the pure-data declaration of the box; this module realizes a
declaration as tensors (:func:`build_case`) and PROVES the realization sits in
its claimed cell (:func:`assert_nonvacuous`). The proof is keyed by
``(axis, value)`` — one checker per declared regime value, applied for every
axis a family states — so a family cannot claim a regime without the
corresponding check running, and a new regime value cannot be declared without
writing its checker (an undeclared pair is a KeyError at test time, not a
silent pass). ``test_family_meta.py`` proves each checker has teeth.
"""
from __future__ import annotations

import math

import torch

from rola.engine.facts.call import arm_refusal
from rola.engine.rules.arm import CHUNK_TOKENS
from rola.ops.naive import naive_rola
from rola.routing.factors import RouteFactors
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
)
from tests.oracle.families import Family

#: The chunk arm's chunk length, re-exported so a family fixture and the
#: checkers that read a "word" of tokens both name ONE number. It is the
#: MAXIMUM a super-chunk gathers, not a quantum the caller must land on: `T` is
#: arbitrary and a short final super-chunk is the compaction's ordinary
#: outcome, so a family is free to state a ragged `T` and still run the kernel.
TOKEN_WORD = CHUNK_TOKENS

_ROUTING = IndependentRouting(
    width=1, read=SoftmaxActivation(), write=SoftmaxActivation())


def _topology(widths):
    return Topology(levels=tuple(_ROUTING.at(w) for w in widths))


def _simplex(shape, p_nonzero, gen, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=gen) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return x / x.sum(-1, keepdim=True)


def _masked_simplex(shape, mask, gen, device):
    mask = mask.to(device=device, dtype=torch.float64).expand(shape)
    assert bool((mask.sum(-1) > 0).all()), "every row must have a nonempty support"
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen) * mask
    x = torch.where(x.sum(-1, keepdim=True) == 0, mask, x)
    return x / x.sum(-1, keepdim=True)


def _one_hot(shape, gen, device):
    B, T, H, W = shape
    digits = torch.randint(0, W, (B, T, H), device=device, generator=gen)
    out = torch.zeros(shape, device=device, dtype=torch.float64)
    return out.scatter_(-1, digits.unsqueeze(-1), 1.0)


def _decay_dials(widths, H, gen, device):
    return tuple(
        0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=gen)
        for w in widths)


def _leaf_product(levels, widths):
    """``[B,T,H,N]`` — every checker that talks about LEAVES computes them the
    oracle's way (MSB-first mixed radix), independently of production."""
    N = math.prod(widths)
    strides = [math.prod(widths[l + 1:]) for l in range(len(widths))]
    idx = torch.arange(N, device=levels[0].device)
    out = None
    for level, stride, width in zip(levels, strides, widths):
        sel = level.index_select(-1, (idx // stride) % width)
        out = sel if out is None else out * sel
    return out


class Case:
    """One realized family fixture: tensors, the routing bundle, and both runs."""

    def __init__(self, family: Family):
        c = dict(family.config)
        p = dict(family.params)
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(c["seed"])
        self.family = family
        self.widths, self.d_v = tuple(c["widths"]), c["d_v"]
        self.B, self.T, self.H = c["B"], c["T"], c["H"]
        # `BT` is the chunk length the tail and chunk-boundary geometry below are
        # written against ("one word"). It is not a config axis and never was a
        # template parameter.
        self.BT = CHUNK_TOKENS
        self.norm, self.paging = c["norm"], c["paging"]
        self.topology = _topology(self.widths)
        self.N = self.topology.N
        self.cols = self.d_v + (1 if self.norm == "global" else 0)
        shape = (self.B, self.T, self.H)

        builder = getattr(self, "_" + (family.generator or "stratified"))
        self.read_levels, self.write_levels = builder(gen, device, c, p)

        self.g_write = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
        self.v = torch.randn(*shape, self.d_v, device=device, dtype=torch.float64, generator=gen)
        self.decay = (LeafMassDecay(dials=_decay_dials(self.widths, self.H, gen, device))
                      if c["decay"] else None)
        self.initial_state = None
        if c.get("m0"):
            assert not self.paging, "an implicit dense initial state is forbidden under paging"
            #: CANONICAL leaf order, like everything the oracle touches. A consumer
            #: whose plane is lattice-ordered crosses at its own seam (`rola.ops.lattice`).
            m0 = 0.1 * torch.randn(self.B, self.H, self.N, self.cols, device=device,
                                   dtype=torch.float64, generator=gen)
            if self.norm == "global":
                m0[..., self.d_v] = m0[..., self.d_v].abs()
            self.initial_state = m0

        self._f32 = dict(
            v=self.v.to(torch.float32),
            read=tuple(t.to(torch.float32) for t in self.read_levels),
            write=tuple(t.to(torch.float32) for t in self.write_levels),
            g_write=self.g_write.to(torch.float32))
        self.bundle = RouteFactors(
            topology=self.topology, read=self._f32["read"],
            write=self._f32["write"],
            g_write=self._f32["g_write"])
        #: WHY the kernel arm refuses this family, or None. Read from the
        #: PRODUCTION predicate, not from the family's pure-CPU mirror: the
        #: mirror exists so `families.py` stays device-free, and this is the
        #: line that makes a drift between them observable
        #: (`test_family_meta.py` asserts the two agree for every family).
        self.refusal = arm_refusal(self.bundle, self.decay, d_v=self.d_v)

    # -- generators ---------------------------------------------------------

    def _stratified(self, gen, device, c, p):
        shape4 = lambda w: (self.B, self.T, self.H, w)  # noqa: E731
        if p.get("one_hot"):
            make = lambda w, _p: _one_hot(shape4(w), gen, device)  # noqa: E731
        elif p.get("peak"):
            def make(w, _p, peak=p["peak"]):
                hot = torch.randint(0, w, (1, 1, self.H), device=device, generator=gen)
                spread = _simplex(shape4(w), 1.0, gen, device)
                out = (1 - peak) * spread
                return out.scatter_add_(
                    -1, hot.expand(self.B, self.T, self.H).unsqueeze(-1),
                    torch.full((self.B, self.T, self.H, 1), peak, device=device,
                               dtype=torch.float64))
        else:
            make = lambda w, pd: _simplex(shape4(w), pd, gen, device)  # noqa: E731

        if p.get("block"):
            blocks = -(-self.T // p["block"])

            def blockwise(w, pd):
                draw = _simplex((self.B, blocks, self.H, w), pd, gen, device)
                return draw.repeat_interleave(p["block"], dim=1)[:, :self.T]
            make = blockwise
        if p.get("anti"):
            w0 = self.widths[0]
            half = torch.zeros(1, 1, 1, w0)
            read_mask, write_mask = half.clone(), half.clone()
            read_mask[..., :w0 // 2], write_mask[..., w0 // 2:] = 1.0, 1.0
            read = (_masked_simplex(shape4(w0), read_mask, gen, device),) + tuple(
                make(w, c["p_read"]) for w in self.widths[1:])
            write = (_masked_simplex(shape4(w0), write_mask, gen, device),) + tuple(
                make(w, c["p_write"]) for w in self.widths[1:])
            return read, write
        if p.get("write_live_digits"):  # the cold-read corner
            w0 = self.widths[0]
            mask = torch.zeros(1, 1, 1, w0)
            mask[..., :p["write_live_digits"]] = 1.0
            write = (_masked_simplex(shape4(w0), mask, gen, device),) + tuple(
                make(w, c["p_write"]) for w in self.widths[1:])
            return tuple(make(w, c["p_read"]) for w in self.widths), write
        read = tuple(make(w, c["p_read"]) for w in self.widths)
        if p.get("tied"):
            return read, tuple(t.clone() for t in read)
        return read, tuple(make(w, c["p_write"]) for w in self.widths)

    def _imbalanced_heads(self, gen, device, c, p):
        def side():
            levels = []
            for w in self.widths:
                per_head = []
                for density in p["head_densities"]:
                    shape = (self.B, self.T, 1, w)
                    per_head.append(_one_hot(shape, gen, device) if density == "one_hot"
                                    else _simplex(shape, float(density), gen, device))
                levels.append(torch.cat(per_head, dim=2))
            return tuple(levels)
        assert len(p["head_densities"]) == self.H
        return side(), side()

    def _all_read_only(self, gen, device, c, p):
        #: one constant one-hot write per level -> exactly leaf 0 is ever written;
        #: reads exclude level-0 digit 0, so EVERY read leaf is read-only (anti).
        write = tuple(
            torch.zeros(self.B, self.T, self.H, w, device=device, dtype=torch.float64)
            .index_fill_(-1, torch.tensor([0], device=device), 1.0)
            for w in self.widths)
        w0 = self.widths[0]
        read_mask = torch.ones(1, 1, 1, w0)
        read_mask[..., 0] = 0.0
        read = (_masked_simplex((self.B, self.T, self.H, w0), read_mask, gen, device),) + tuple(
            _simplex((self.B, self.T, self.H, w), 1.0, gen, device) for w in self.widths[1:])
        return read, write

    def _chunk_boundary(self, gen, device, c, p):
        """Write support flips EXACTLY at tile boundaries, and the write draws
        are blockwise-constant per tile — the fixture is temporally coherent by
        construction, which is also what makes the flip a step rather than
        noise riding on iid draws."""
        t1, t2 = (t * self.BT for t in p["boundary_tiles"])
        n_tiles = self.T // self.BT
        w0 = self.widths[0]
        tile_mask = torch.zeros(1, n_tiles, 1, w0)
        tile_mask[:, :p["boundary_tiles"][0], :, :p["live_before"]] = 1.0
        tile_mask[:, p["boundary_tiles"][0]:p["boundary_tiles"][1], :, :p["live_after"]] = 1.0
        tile_mask[:, p["boundary_tiles"][1]:, :, 2:2 + p["live_before"]] = 1.0

        def per_tile(w, mask):
            draw = _masked_simplex((self.B, n_tiles, self.H, w), mask, gen, device)
            return draw.repeat_interleave(self.BT, dim=1)

        write = (per_tile(w0, tile_mask),) + tuple(
            per_tile(w, torch.ones(1, 1, 1, w)) for w in self.widths[1:])
        read = tuple(_simplex((self.B, self.T, self.H, w), c["p_read"], gen, device)
                     for w in self.widths)
        self._flip_tokens = (t1, t2)
        return read, write

    # -- runs ---------------------------------------------------------------

    def oracle(self):
        return naive_rola(
            self.v, self.read_levels, self.write_levels, self.g_write,
            self.topology, self.decay,
            initial_state=self.initial_state, output_final_state=True)


def build_case(family: Family) -> Case:
    return Case(family)


def relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference.double()).abs().max()) / scale


# ---------------------------------------------------------------------------
# Non-vacuity — one checker per declared regime value
# ---------------------------------------------------------------------------

def _support_sizes(levels):
    """Per-(token, level) exact-support cardinalities, flattened per level."""
    return [(t != 0).sum(dim=-1) for t in levels]


def _sides(case):
    return (("read", case.read_levels), ("write", case.write_levels))


def _check_dense(case):
    for side, levels in _sides(case):
        for level in levels:
            assert not bool((level == 0).any()), (
                f"claimed dense, but the {side} side has exact zeros")


def _check_sparse(case):
    zeros = any(bool((t == 0).any()) for _, ls in _sides(case) for t in ls)
    assert zeros, "claimed sparse, but no exact zero exists on either side"


def _coherence_ratios(case) -> dict[str, float]:
    """Per SIDE: mean |p_t - p_{t-1}| over mean |p_t - p_shuffled| (pooled over
    that side's levels). ~1 for independent per-token draws, << 1 for temporally
    coherent routing, 0 for a constant side. Value-based rather than
    support-based, so it discriminates on DENSE fixtures too. Per side, because
    a fixture may legitimately pair one coherent side with one iid side (the
    all-read-only corner: a constant write against random reads)."""
    ratios = {}
    for side, levels in _sides(case):
        adjacent, shuffled = 0.0, 0.0
        for t in levels:
            adjacent += float((t[:, 1:] - t[:, :-1]).abs().mean())
            perm = torch.randperm(t.shape[1], generator=torch.Generator().manual_seed(7))
            shuffled += float((t - t[:, perm]).abs().mean())
        ratios[side] = adjacent / max(shuffled, 1e-30)
    return ratios


def _check_iid(case):
    ratios = _coherence_ratios(case)
    assert all(r > 0.7 for r in ratios.values()), (
        f"claimed iid, but a side's adjacent-token difference is only {ratios} of "
        "the shuffled baseline — the fixture is temporally structured")


def _check_coherent(case):
    ratios = _coherence_ratios(case)
    assert any(r < 0.5 for r in ratios.values()), (
        f"claimed coherent, but every side's adjacent-token difference is {ratios} "
        "of the shuffled baseline — the fixture is temporally unstructured")


def _check_tied(case):
    for r, w in zip(case.read_levels, case.write_levels):
        assert torch.equal(r, w), "claimed tied, but read != write"


def _leaf_supports(case):
    R = _leaf_product(case.read_levels, case.widths) != 0
    W = _leaf_product(case.write_levels, case.widths) != 0
    return R, W


def _check_independent(case):
    tied = all(torch.equal(r, w) for r, w in zip(case.read_levels, case.write_levels))
    assert not tied, "claimed independent, but the sides are bitwise tied"
    R, W = _leaf_supports(case)
    assert bool((R & W).any()), (
        "claimed independent, but read and write leaf supports are DISJOINT — "
        "that is the anti cell")


def _check_anti(case):
    R, W = _leaf_supports(case)
    assert not bool((R & W).any()), (
        "claimed anti, but some token reads a leaf it also writes")
    assert bool(R.any()) and bool(W.any()), "an empty side is vacuous, not anti"


def _concentration(case) -> float:
    """Mean WIDTH-NORMALIZED largest digit share, pooled over sides and levels:
    ``(max_share - 1/width) / (1 - 1/width)`` — 0 for a uniform row, 1 for a
    one-hot row, comparable across level widths (a raw 0.5 share means nothing
    at width 4 and a lot at width 16). MEASURED over the committed families
    (2026-08-06): every spread-claiming family sits in 0.12..0.59 (worst: the
    thinned width-4 sweep topologies), the concentrated claimers at 0.90/1.00 —
    so the 0.65 / 0.75 cuts below are calibrated bands with margin, not tuned
    pass thresholds."""
    values = []
    for _, levels in _sides(case):
        for t in levels:
            width = t.shape[-1]
            m = t.max(dim=-1).values.reshape(-1)
            values.append((m - 1.0 / width) / (1 - 1.0 / width))
    return float(torch.cat(values).mean())


def _check_spread(case):
    mean = _concentration(case)
    assert mean <= 0.65, (
        f"claimed spread mass, but the width-normalized concentration is {mean:.2f} "
        "(> 0.65) — the fixture is concentrated in aggregate")


def _check_concentrated(case):
    mean = _concentration(case)
    assert mean >= 0.75, (
        f"claimed concentrated, but the width-normalized concentration is {mean:.2f} "
        "(< 0.75)")


def _check_cold_read(case):
    R, W = _leaf_supports(case)
    ever_written = W.any(dim=1, keepdim=True)          # [B, 1, H, N]
    cold = R & ~ever_written
    assert bool(cold.any()), (
        "claimed cold_read, but every read leaf is written somewhere in the sequence")
    assert case.initial_state is not None, (
        "a cold read against a zero initial state is unobservable; the family "
        "must carry M_0 != 0")


def _check_divisible(case):
    assert case.T % case.BT == 0, f"claimed divisible, but T={case.T} % BT={case.BT} != 0"


def _check_ragged(case):
    assert case.T % case.BT != 0, (
        f"claimed ragged, but T={case.T} tiles evenly under BT={case.BT}")


def _check_singleton(case):
    for side, levels in _sides(case):
        if all(bool((s == 1).all()) for s in _support_sizes(levels)):
            return
    raise AssertionError("claimed singleton, but no side has exactly one live "
                         "digit per token on every level")


def _check_partial(case):
    for _, levels in _sides(case):
        for t, s in zip(levels, _support_sizes(levels)):
            width = t.shape[-1]
            if bool(((s > 1) & (s < width)).any()):
                return
    raise AssertionError("claimed partial support, but every row is singleton or full")


def _check_full(case):
    for side, levels in _sides(case):
        for t, s in zip(levels, _support_sizes(levels)):
            assert bool((s == t.shape[-1]).all()), (
                f"claimed full support, but the {side} side has a thinned row")


CHECKERS = {
    ("density", "dense"): _check_dense,
    ("density", "sparse"): _check_sparse,
    # "swept" is a property of a SWEEP, not of one tensor set; the producer
    # tests prove it with `assert_swept` over the realized densities.
    ("density", "swept"): lambda case: None,
    ("coherence", "iid"): _check_iid,
    ("coherence", "coherent"): _check_coherent,
    ("rw_correlation", "tied"): _check_tied,
    ("rw_correlation", "independent"): _check_independent,
    ("rw_correlation", "anti"): _check_anti,
    ("mass", "spread"): _check_spread,
    ("mass", "concentrated"): _check_concentrated,
    ("mass", "cold_read"): _check_cold_read,
    ("tail", "divisible"): _check_divisible,
    ("tail", "ragged"): _check_ragged,
    ("support", "singleton"): _check_singleton,
    ("support", "partial"): _check_partial,
    ("support", "full"): _check_full,
}


# -- corner-specific structure, beyond the shared regime axes ----------------

def _corner_all_read_only(case):
    #: CANONICAL leaf indices, from `_leaf_product`. Leaf 0 is the all-zero digit tuple,
    #: which the lattice also numbers 0, so the corner reads the same either way.
    _, W = _leaf_supports(case)
    written = W.reshape(-1, W.shape[-1]).any(dim=0)
    assert bool(written[0]) and int(written.sum()) == 1, (
        "the all-read-only corner requires EXACTLY leaf 0 ever written; got "
        f"{int(written.sum())} written leaves")


def _corner_chunk_boundary(case):
    t1, t2 = case._flip_tokens
    assert t1 % case.BT == 0 and t2 % case.BT == 0, (
        f"the support flips at tokens {t1}/{t2}, which are not tile boundaries "
        f"under BT={case.BT} — a mid-chunk flip tests a weaker thing")
    support = (case.write_levels[0] != 0)
    for t in (t1, t2):
        assert not torch.equal(support[:, t - 1], support[:, t]), (
            f"the write support does not actually change across token {t}")
        assert torch.equal(support[:, t], support[:, min(t + case.BT, case.T) - 1]), (
            f"the support is not constant within the tile after token {t}")


CORNER_CHECKS = {
    "corner/all-read-only": _corner_all_read_only,
    "corner/chunk-boundary": _corner_chunk_boundary,
    # corner/cold-read: fully expressed by the shared (mass, cold_read) checker.
    # corner/k0-fold: its K = 0 witness needs a LAUNCH (ws_count introspection),
    # so it lives in test_families.py next to the run it inspects.
}


def assert_nonvacuous(family: Family, case) -> None:
    """Prove the realized fixture sits in its declared regime cell.

    A missing checker is a KeyError — a new regime value cannot land without
    one — and a failing checker means the family is ABSENT from its cell, which
    is a test failure of the family itself, before any kernel comparison runs.
    ``case`` is duck-typed (read_levels/write_levels/widths/T/BT/P/
    initial_state), so the producer tests can run the same checkers over a
    realized bundle.
    """
    for axis, value in family.regime.items():
        CHECKERS[(axis, value)](case)
    corner = CORNER_CHECKS.get(family.name)
    if corner is not None:
        corner(case)


def assert_swept(densities) -> None:
    """The `density = swept` non-vacuity: the realized support fractions must
    actually sweep (max/min >= 4x) and move monotonically with the gain."""
    lo, hi = min(densities), max(densities)
    assert hi / max(lo, 1e-9) >= 4.0, (
        f"claimed a density sweep, but realized densities {densities} span less "
        "than 4x — the gain knob is not doing anything")
    ordered = all(a >= b for a, b in zip(densities, densities[1:]))
    assert ordered, (
        f"realized densities {densities} are not monotone in the gain; the sweep "
        "is not the gain sweep it claims to be")
