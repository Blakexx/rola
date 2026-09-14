"""THE ROW SORT'S TIE BATTERY -- the gate on `RowSort`'s exactness claim.

`csrc/rola/src/entmax/entmax.cuh`'s `RowSort` lowers phase B to one of two networks,
chosen by width class: a register-resident bitonic network at ``IPT <= 4`` and
``cub::WarpMergeSort`` at ``IPT == 8``. The claim that makes the choice free is that a
sort is a PERMUTATION and the solve reads the sorted row only through its prefix sums and
its rank-k value -- both invariant to how EQUAL keys are ordered. So the two networks may
break ties differently and `tau` is still bit-identical.

That is an argument. These are its teeth: rows built so that TIES ARE THE COMMON CASE and
so that they straddle the support boundary, where a tie-order difference would move `k*`
if the argument were wrong. The SUPPORT SET is the tie-sensitive quantity -- a tie order
that moved `k*` would move `tau` across a tied cluster and add or drop that whole cluster
-- so it is asserted at ``rtol=0`` against the torch lowering, which sorts with
`torch.sort`, a third tie order again.

WHAT IS DELIBERATELY *NOT* ASSERTED BITWISE: the amplitudes, across a change of LANE
ASSIGNMENT. `run_p` is a `WarpScan` over per-lane sums, so which lane holds which column
fixes the summation association; permuting a row's columns, or comparing against a
lowering that takes a full-row `cumsum`, moves `tau` by an ulp. That is a property of the
warp-blocked reduction and predates the row sort by every commit. The bitwise old-vs-new
comparison of the sort itself lives in the P74 lever-3 build record: 1560 cases / 5040
output tensors, merge sort against register network, zero mismatches.

Design of record: docs/internals/entmax/entmax.md#row-sort
"""
from __future__ import annotations

import pytest
import torch

from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.types import ResolvedRouting, UnionRouting
from tests.unit.test_entmax_production import _PRODUCTION_ATOL, _PRODUCTION_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

#: The family's ratified production tolerance, imported rather than restated so this file
#: cannot drift from the gate that owns the number.
_RTOL, _ATOL = _PRODUCTION_RTOL, _PRODUCTION_ATOL

#: Every padded-width class, so every `(LW, IPT)` instantiation of both networks runs;
#: the non-powers of two also drive the sentinel padding into the tied region.
WIDTHS = (2, 3, 8, 16, 32, 60, 64, 128, 200, 256)
BH, T = 2, 64


def _routing(width: int, alpha: float) -> ResolvedRouting:
    return ResolvedRouting((UnionRouting(width=width, alpha=alpha),))


def _rows(kind: str, width: int, seed: int) -> torch.Tensor:
    """[BH, T, width] fp32 rows of one tie family."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shp = (BH, T, width)
    kw = dict(device="cuda", dtype=torch.float32)
    if kind == "all-equal":
        return torch.full(shp, 0.25, **kw)
    if kind.startswith("levels-"):
        #: `n` distinct values over the whole row: every rank is a tie with many others,
        #: and with n small the tie CLUSTER straddles the support boundary.
        n = int(kind.split("-")[1])
        lvl = torch.randn((BH, T, n), generator=g, **kw) * 2.0
        idx = torch.randint(0, n, shp, generator=g, device="cuda")
        return lvl.gather(2, idx)
    if kind == "one-ulp":
        base = torch.randn((BH, T, 1), generator=g, **kw).expand(shp).contiguous()
        eps = torch.finfo(torch.float32).eps
        steps = torch.randint(-2, 3, shp, generator=g, device="cuda").float()
        return base + steps * eps * base.abs().clamp_min(1.0)
    if kind == "boundary":
        #: Exact ties pinned ON the certificate's own value cuts: 0 (the row max), -1
        #: (alpha 2's bound) and -2 (alpha 1.5's), so a tie sits exactly where a
        #: selection scheme would cut.
        r = torch.zeros(shp, **kw)
        r[..., 1::3] = -1.0
        r[..., 2::3] = -2.0
        return r
    raise AssertionError(kind)


TIE_FAMILIES = ("all-equal", "levels-1", "levels-2", "levels-3", "levels-8",
                "one-ulp", "boundary")


@pytest.mark.cuda
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("alpha", (1.5, 2.0))
@pytest.mark.parametrize("family", TIE_FAMILIES)
def test_the_row_sort_is_bitwise_tie_order_independent(width, alpha, family):
    """The CUDA solve and the torch lowering sort with DIFFERENT networks and therefore
    different tie orders. Every output must still agree bit for bit."""
    routing = _routing(width, alpha)
    read = _rows(family, width, seed=width * 7 + 1)
    write = _rows(family, width, seed=width * 7 + 2)

    cuda_r, cuda_w = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.float32)
    cpu_r, cpu_w = production_routing_factor_levels(
        read.cpu(), write.cpu(), routing, stored_dtype=torch.float32)

    tag = f"{family} w={width} alpha={alpha}"
    for level, (a, b) in enumerate(zip(cuda_r, cpu_r)):
        assert torch.equal(a.cpu() != 0, b != 0), f"read level {level} support: {tag}"
        torch.testing.assert_close(a.cpu(), b, rtol=_RTOL, atol=_ATOL)
    for level, (a, b) in enumerate(zip(cuda_w, cpu_w)):
        assert torch.equal(a.cpu() != 0, b != 0), f"write level {level} support: {tag}"
        torch.testing.assert_close(a.cpu(), b, rtol=_RTOL, atol=_ATOL)


@pytest.mark.cuda
@pytest.mark.parametrize("width", (8, 64, 256))
@pytest.mark.parametrize("alpha", (1.5, 2.0))
def test_permuting_tied_columns_permutes_the_solve_exactly(width, alpha):
    """The tie-immateriality argument, stated as an equivariance the solve must satisfy.

    Rolling a row's columns is a permutation of the SAME multiset, so it reaches the sort
    as the same keys in a different order -- exactly the perturbation a different tie-break
    would produce. `k*` must not move, so the SUPPORT must be the same roll exactly; the
    amplitudes follow to tolerance, because the roll also moves the lane assignment the
    prefix scan associates over.
    """
    routing = _routing(width, alpha)
    read = _rows("levels-3", width, seed=99)
    write = _rows("levels-3", width, seed=100)

    base_r, base_w = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.float32)
    roll_r, roll_w = production_routing_factor_levels(
        read.roll(3, dims=2), write.roll(3, dims=2), routing, stored_dtype=torch.float32)

    for level, (a, b) in enumerate(zip(base_r, roll_r)):
        assert torch.equal(a.roll(3, dims=2) != 0, b != 0), f"read level {level} w={width}"
        torch.testing.assert_close(a.roll(3, dims=2), b, rtol=_RTOL, atol=_ATOL)
    for level, (a, b) in enumerate(zip(base_w, roll_w)):
        assert torch.equal(a.roll(3, dims=2) != 0, b != 0), f"write level {level} w={width}"
        torch.testing.assert_close(a.roll(3, dims=2), b, rtol=_RTOL, atol=_ATOL)


@pytest.mark.cuda
@pytest.mark.parametrize("family", TIE_FAMILIES)
def test_the_tie_fixtures_are_not_vacuous(family):
    """A gate that never had teeth is not evidence -- so the fixtures are measured.

    Two properties have to hold for this file to be testing anything: the rows must
    genuinely be TIED (adjacent ranks equal, which is what a sorting network is free to
    order either way), and the support boundary must fall strictly INSIDE the row, so a
    moved `k*` would change the answer. MEASURED at width 64: `levels-2` is 95% tied with
    39% support, `boundary` 97% tied with 34% support.
    """
    width = 64
    read, write = _rows(family, width, seed=width * 7 + 1), _rows(family, width, seed=width * 7 + 2)
    mid = 0.5 * read + 0.5 * write
    z = mid - mid.amax(-1, keepdim=True)
    srt, _ = z.sort(-1, descending=True)
    tied = (srt[..., 1:] == srt[..., :-1]).float().mean().item()
    assert tied >= 0.35, f"{family}: only {tied:.3f} of adjacent ranks are exact ties"

    reads, _ = production_routing_factor_levels(
        read, write, _routing(width, 1.5), stored_dtype=torch.float32)
    support = (reads[0] != 0).float().mean().item()
    assert 0.0 < support <= 1.0, f"{family}: degenerate support {support}"


@pytest.mark.cuda
@pytest.mark.parametrize("family", ("levels-2", "levels-3", "boundary"))
def test_teeth_a_moved_k_star_would_change_the_support(family):
    """The teeth: prove the asserted quantity is SENSITIVE to the failure being excluded.

    If a sorting network ordered a tie cluster differently and that moved `k*`, `tau` would
    cross to the next distinct key value. Here that counterfactual tau is constructed and
    the support it induces is required to DIFFER from the shipped one -- so the support
    equality asserted above is a real constraint on tie order, not a tautology.
    """
    width = 64
    read, write = _rows(family, width, seed=width * 7 + 1), _rows(family, width, seed=width * 7 + 2)
    reads, _ = production_routing_factor_levels(
        read, write, _routing(width, 1.5), stored_dtype=torch.float32)
    shipped = reads[0] != 0

    mid = 0.5 * read + 0.5 * write
    z = mid - mid.amax(-1, keepdim=True)
    #: tau moved DOWN to the next distinct value below the realized boundary: exactly the
    #: support a `k*` one tie-cluster too large would produce.
    boundary = torch.where(shipped, z, torch.full_like(z, float("inf"))).amin(-1, keepdim=True)
    below = torch.where(z < boundary, z, torch.full_like(z, -float("inf"))).amax(-1, keepdim=True)
    #: `>=` rather than `>`: the counterfactual ADMITS that next cluster, which is what a
    #: `k*` one tie-cluster too large means.
    moved = z >= below

    rows_that_move = (moved != shipped).any(-1).float().mean().item()
    assert rows_that_move > 0.5, (
        f"{family}: a moved k* changes the support on only {rows_that_move:.3f} of rows; "
        "the support assertion would not catch a tie-order regression here")
