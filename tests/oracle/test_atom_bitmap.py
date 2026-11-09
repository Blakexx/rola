"""The facts pass's words, folded to atoms, are the EXACT atom-grain write bitmap.

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

from rola.engine.facts import liveness as lv
from rola.engine.facts import planes
from rola.ops import facts as facts_ops
from rola.ops.naive import _leaf_product, _radix_strides
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.factors import RouteFactors
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    OpaqueRouting,
    SoftmaxActivation,
    Topology,
)
from rola.values import Values
from tests.oracle.fixtures import with_gain

#: the lists' form the facts pass is called in (sm_86's even carry arm); this file reads its words and unions only
_LISTS = facts_ops.ListsForm(read_row=64, read_list_group=32, write_list_group=32)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the stats pass is a CUDA kernel"),
]

#: The topologies checked, `(D, B)` at one uniform width. The facts pass takes `D` and
#: `B_l` at runtime, so every topology is one. Depths 2-4 are the sweep the design asks
#: for; the two `D = 1` rows are the flat-routing baselines, the shape with no upper prefix
#: at all -- the reduction's degenerate carve.
_ARMS = ((1, 64), (1, 256), (2, 16), (2, 64), (3, 16), (4, 16))

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


def _alternating_sparse(D):
    """The ALTERNATING declaration (docs/api.md section 2.1), ``(read, write)`` sparse flags a level: read-dense on the
    even levels, write-dense on the odd ones and, at depth one, on the one level -- every side has a dense level."""
    return (tuple(level % 2 == 1 for level in range(D)), tuple(level % 2 == 0 and D > 1 for level in range(D)))


def _bundle(write, sparse=None):
    """The fixture's write levels in the format (both sides the same levels): the rows the facts pass reads, each
    side's levels sparse or dense as ``sparse`` declares (default the alternating split): a dense level is floored by
    the quantizer, never zero. The reference reads the bundle's OWN levels (`dequantized`), so a value that rounded to a
    zero digit (or to the floor) is the same on both legs and the comparison is about the derivation, not the
    rounding."""
    B, T, H = write[0].shape[:3]
    topology = Topology(levels=tuple(OpaqueRouting(width=level.shape[-1]) for level in write))
    ones = torch.ones((B, T, H), device=write[0].device)
    sparse = _alternating_sparse(len(write)) if sparse is None else sparse
    return (RouteFactors.from_levels(topology, write, with_gain(write, ones), sparse=sparse),
            Values.from_projection(torch.ones((B, T, H, 8), device=write[0].device)))


def _kernel_bitmap(routes, values):
    """The written atoms of the facts pass's words, every level voted from its rows."""
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    words = facts_ops.facts(routes, values, layout, *statics, _LISTS).words
    return planes.written_atoms(lv.page_bits(words, layout, *statics))


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
    return read, write


@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"d{a[0]}b{a[1]}")
def test_atom_bitmap_is_exact(arm, support):
    """Exact tensor equality against the independent reference, on every built arm."""
    D, B = arm
    widths = (B,) * D
    _read, write = _fixture(widths, _SUPPORTS[support], seed=D * 1000 + B)
    routes, values = _bundle(write)
    bits = _kernel_bitmap(routes, values)
    oracle = _oracle_bitmap(routes.dequantized()[1], widths)
    assert bits.shape == (_B * _H, prod(widths) // MMA_K_QUANTUM)
    assert torch.equal(bits, oracle), (
        f"kernel and the fp64 leaf-product reference disagree on "
        f"{(bits ^ oracle).sum().item()} atoms")


def test_two_token_cross_product_is_excluded():
    """THE ADVERSARIAL CELL: two tokens, disjoint live digits per sparse level.

    Every side keeps a dense level, so the write side's two SPARSE levels are levels 0 and 1 of a depth-three call and
    level 2 is the dense one (every digit live, so a leaf's atom is its two sparse digits' and the 16 leaves under
    them). At `B = (64, 64, 16)` leaf `s = d0 * 1024 + d1 * 16 + d2`. Token 0 writes only `(0, 0)` (atom 0); token 1
    writes only `(1, 32)` (leaf 1536, atom 96). The exact answer is `{0, 96}`. A derivation that ORs over tokens before
    ANDing over levels also admits `(0, 32)` -> leaf 512 -> atom 32 and `(1, 0)` -> leaf 1024 -> atom 64: leaves that
    were never live at any single token. Those two atoms are named here so the assertion states the trap rather than
    merely avoiding it.
    """
    widths = (64, 64, 16)
    write = tuple(torch.zeros((1, 2, 1, w), device="cuda", dtype=torch.float32) for w in widths)
    write[0][0, 0, 0, 0] = 1.0
    write[1][0, 0, 0, 0] = 1.0
    write[0][0, 1, 0, 1] = 1.0
    write[1][0, 1, 0, 32] = 1.0
    write[2][:] = 1.0 / 16

    bits = _kernel_bitmap(*_bundle(write, sparse=((False, True, False), (True, True, False))))
    live = set(torch.nonzero(bits[0]).flatten().tolist())
    assert live == {0, 96}, f"expected atoms {{0, 96}}, got {sorted(live)}"
    assert 32 not in live and 64 not in live, (
        "the cross-product atoms are present: the reduction ORed over tokens "
        "before ANDing over levels")
    assert torch.equal(bits, _oracle_bitmap(write, widths))


def test_stats_through_the_host_seam():
    """The prefill's own path -- the layout and statics the topology declares, the facts pass, the page fold -- on a
    real producer's bundle, whose packed rows are exactly the views a hand-built fixture would not make.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from conftest import alternating_routing, build_layer

    torch.manual_seed(0)
    layer = build_layer(hidden_size=64, num_heads=_H, widths=(16, 16), routing=alternating_routing(2)).cuda().eval()
    x = torch.randn(_B, _T, 64, device="cuda")
    with torch.no_grad():
        routes, values = layer._produce(x)
    _, write = routes.dequantized()
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    words = facts_ops.facts(routes, values, layout, *statics, _LISTS).words
    bits = lv.page_bits(words, layout, *statics)
    assert bits.dtype == torch.uint8 and bits.device.type == "cuda"
    assert torch.equal(planes.written_atoms(bits), _oracle_bitmap(write, (16, 16)))
