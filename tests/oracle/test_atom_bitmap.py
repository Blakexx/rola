"""The stats pass emits the EXACT atom-grain write bitmap.

This is the gate the paging work's first stage exists to pass, and it is the one
that decides the shape of everything after it. Residency is planned from this
bitmap, so if the bitmap is a SUPERSET the arena's `resident == realized`
accounting stops being true and the "planned-exact footprint" contract becomes a
bound instead of a fact. Exactness is therefore asserted as `torch.equal`
against two independent references, never as a containment.

Vocabulary, defined here and used unqualified below: `D` = level count;
`B` = one uniform level width; `N = B ** D` = leaves; `L`/`T` = tokens;
`BH` = batch * heads; **atom** = the 16-leaf page granule
(`rola.ops.paging.MMA_K_QUANTUM`); `atoms = N // 16`; the **bitmap** is
`[BH, atoms]`, true where this routing's WRITE side touches the atom.

**THE REFERENCE, and why it is an independent one.** The bitmap is checked
against an fp64 recomputation built from the ORACLE's own leaf materialization
(`rola.ops.naive._leaf_product` over `rola.ops.naive._radix_strides`), which
derives leaf addresses separately from production on purpose. Validating a device
kernel against a FRESH reference written beside it would prove only
self-consistency -- the failure mode this repository has already been bitten by --
so the reference has to be one that already existed for another reason, and the
oracle's address map is exactly that.

P67 D2-b (ruling R-4) removed a SECOND reference, `atom_bitmap_from_schedule`,
the host condensation the tiled arena admitted pages from. It takes a `Schedule`
and retires with it. Its removal does not reopen the matching-the-naive risk,
which is the reason the ruling turned on: the surviving reference is the
independent fp64 one, not a re-derivation of the kernel. What it does remove is a
drop-in-equality claim about a production path that no longer exists.

**The adversarial cell is the point of the file.** Every looser derivation of
residency -- OR over tokens first, AND over levels second, or the same
conjunction evaluated on the `BC` rectangle -- agrees with the exact one on
ordinary routing. It disagrees on two tokens whose live digits differ per level,
where the loose form admits CROSS-PRODUCT leaves that were never live at any
single token. `test_two_token_cross_product_is_excluded` is that cell, with the
excluded atoms named explicitly.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.engine.facts import planes
from rola.ops._ext import extension
from rola.ops.naive import _leaf_product, _radix_strides
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    SoftmaxActivation,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the stats pass is a CUDA kernel"),
]

#: The built stats arms: `(D, B)`, the projection of the chunk arm list that
#: `tools/gen_shards.py`'s `chunk_atom_arms()` keys the bitmap's reduction to --
#: residency is BC-free, so the list is shorter than the consumer's. Depths 2-4 are the sweep
#: the design asks for; the two `D = 1` arms are the flat-routing baselines,
#: carried because they are built and because `D = 1` is the shape with no upper
#: prefix at all -- the reduction's degenerate carve.
_ARMS = ((1, 64), (1, 256), (2, 8), (2, 16), (2, 64), (3, 16), (4, 8))

#: The support sweep. `1.0` is the dense limit, where every atom is live and the
#: reduction's early exit is the path taken.
_SUPPORTS = {"sparse": 0.35, "mixed": 0.7, "dense": 1.0}

_ROUTING = IndependentRouting(
    width=1, read=SoftmaxActivation(), write=EntmaxActivation(1.5))

_B, _T, _H, _DV = 2, 48, 2, 64


def _simplex(shape, p_nonzero, generator):
    """Rows on the simplex with EXACT support zeros -- entmax's own output shape.

    A fixture with no exact zero tests only the dense limit
    (sparsity comes from entmax, not from a floor).
    """
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device="cuda", dtype=torch.float64,
                          generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _pack(write):
    """`[B,T,H,w] per level -> [B,H,T,sum(widths)]` bf16 -- the seam's packing."""
    return torch.cat(write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()


def _kernel_bitmap(write, widths):
    packed = _pack(write)
    #: THE FIXTURE MUST SURVIVE ITS OWN CAST. The kernel reads bf16, the
    #: references read fp32/fp64; an amplitude that flushed to a bf16 zero would
    #: make the two disagree about the fixture rather than about the derivation.
    flat = torch.cat(write, -1)
    assert torch.equal(flat != 0, flat.to(torch.bfloat16) != 0), (
        "fixture amplitudes flushed to zero in bf16; the comparison would be "
        "about the cast, not about the bitmap")
    return planes.written_atoms(planes.atom_bits(packed, widths))


def _oracle_bitmap(write, widths):
    """THE REFERENCE: the leaf is written iff at SOME token its fp64 leaf product
    is nonzero, condensed to atoms. Built from the oracle's address map."""
    B, T, H = write[0].shape[:3]
    N = prod(widths)
    levels64 = tuple(level.to(torch.float64) for level in write)
    leaves = _leaf_product(levels64, _radix_strides(tuple(widths)), N)  # [B,T,H,N]
    live = (leaves != 0).permute(0, 2, 1, 3).reshape(B * H, T, N).any(dim=1)
    return live.view(B * H, N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(dim=-1)


def _fixture(widths, p_nonzero, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    write = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    g_write = torch.rand((_B, _T, _H), device="cuda", generator=g)
    return read, write, g_write


@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"d{a[0]}b{a[1]}")
def test_atom_bitmap_is_exact(arm, support):
    """Exact tensor equality against the independent reference, on every built arm."""
    D, B = arm
    widths = (B,) * D
    _read, write, _g_write = _fixture(widths, _SUPPORTS[support], seed=D * 1000 + B)
    bits = _kernel_bitmap(write, widths)
    oracle = _oracle_bitmap(write, widths)
    assert bits.shape == (_B * _H, prod(widths) // MMA_K_QUANTUM)
    assert torch.equal(bits, oracle), (
        f"kernel and the fp64 leaf-product reference disagree on "
        f"{(bits ^ oracle).sum().item()} atoms")


def test_two_token_cross_product_is_excluded():
    """THE ADVERSARIAL CELL: two tokens, disjoint live digits per level.

    At `D = 2, B = 64` leaf `s = d0 * 64 + d1`. Token 0 writes only `(0, 0)`
    (leaf 0, atom 0); token 1 writes only `(1, 32)` (leaf 96, atom 6). The exact
    answer is `{0, 6}`. A derivation that ORs over tokens before ANDing over
    levels also admits `(0, 32)` -> leaf 32 -> atom 2 and `(1, 0)` -> leaf 64 ->
    atom 4: leaves that were never live at any single token. Those two atoms are
    named here so the assertion states the trap rather than merely avoiding it.
    """
    widths = (64, 64)
    write = tuple(torch.zeros((1, 2, 1, w), device="cuda", dtype=torch.float32) for w in widths)
    write[0][0, 0, 0, 0] = 1.0
    write[1][0, 0, 0, 0] = 1.0
    write[0][0, 1, 0, 1] = 1.0
    write[1][0, 1, 0, 32] = 1.0

    bits = _kernel_bitmap(write, widths)
    live = set(torch.nonzero(bits[0]).flatten().tolist())
    assert live == {0, 6}, f"expected atoms {{0, 6}}, got {sorted(live)}"
    assert 2 not in live and 4 not in live, (
        "the cross-product atoms are present: the reduction ORed over tokens "
        "before ANDing over levels")
    assert torch.equal(bits, _oracle_bitmap(write, widths))


def test_the_bitmap_is_bc_free():
    """THE PUBLISHED FACT CARRIES NO BLOCK GRAIN, at every built `BC` of a topology.

    The pass that emits the atom rows is templated on `BC` -- it emits the owner
    block's rows in the same launch -- so the 16-leaf keying of the atom rows is a
    property of the CODE and this is what holds it: one routing through every built
    arm, `torch.equal` on the bitmap. A bitmap that moved with `BC` would move
    residency with the arm, which is the overlap the plan is built to keep.
    """
    from rola.engine.facts import manifest as census
    from rola.engine.types import Mask

    D, B = 2, 64
    widths = (B,) * D
    _, write, _ = _fixture(widths, 0.35, seed=4242)
    packed = _pack(write)
    bcs = sorted({int(bc) for (c, bc, d, b) in census.chunk_arms() if d == D and b == B})
    assert len(bcs) > 1, "this topology has one built BC; the claim needs two to vary"
    first = None
    for bc in bcs:
        c = min(int(c) for (c, bc_, d, b) in census.chunk_arms()
                if d == D and b == B and bc_ == bc)
        bits = extension().chunk_facts(packed, packed, c, bc, list(widths), Mask.M2,
                                       True)[1]
        if first is None:
            first, first_bc = bits, bc
            continue
        assert torch.equal(bits, first), (
            f"the atom bitmap differs between BC={first_bc} and BC={bc}: the "
            f"published residency fact has acquired the owner block's grain")
    assert torch.equal(first.bool(), _oracle_bitmap(write, widths))


def test_the_stateless_call_class_cannot_be_asked_for_the_bitmap():
    """PLANTED: the class and the outputs must agree, or a call gets an unwritten table.

    The stateless variant compiles no atom-grain emission, so asking it for the bitmap
    is a refusal and never a silently absent fact.
    """
    from rola.engine.facts import manifest as census
    from rola.engine.types import Mask

    D, B = 2, 64
    widths = (B,) * D
    _, write, _ = _fixture(widths, 0.35, seed=17)
    packed = _pack(write)
    c, bc = next((int(c), int(bc)) for (c, bc, d, b) in census.chunk_arms()
                 if d == D and b == B)
    with pytest.raises(RuntimeError, match="stateful call's fact"):
        extension().chunk_facts(packed, packed, c, bc, list(widths), Mask.M1, True)


def test_stats_through_the_host_seam():
    """`rola.engine.facts.planes.atom_bits` on the packed write plane -- the facade's own path.

    A real producer's `RouteFactors` rather than hand-built tensors, because the
    permute-and-cast between the bundle's layout and the kernel's is exactly the
    step a hand-built fixture would skip.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from conftest import build_routes

    from rola.routing.producer import union_routing

    torch.manual_seed(0)
    producer = build_routes(hidden_size=64, num_heads=_H, widths=(16, 16),
                           routing=union_routing(1)).cuda()
    x = torch.randn(_B, _T, 64, device="cuda")
    routes = producer(x)
    bits = planes.atom_bits(planes.pack_side(routes.write), routes.topology.widths)
    assert bits.dtype == torch.uint8 and bits.device.type == "cuda"
    assert torch.equal(planes.written_atoms(bits), _oracle_bitmap(routes.write, (16, 16)))
