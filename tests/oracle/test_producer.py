"""Producer tests -- the feature map's own stage.

Validates `rola.routing.producer` against the ONE canonical
ground truth this branch authorizes: the Stage 1 fp64 oracle
(`rola.ops.naive.naive_rola`). Per the "matching-the-naive
anti-pattern" standing rule, nothing here is checked against a second,
freshly written reference -- every assertion either (a) feeds the producer's
output through the oracle and checks numerical properties the oracle itself
guarantees (finite output, differentiability), or (b) checks an algebraic
identity of the producer's own contract (mass-fold correctness, simplex
normalization) that is checkable without an independent implementation.

The conditioning surface this file used to also cover
(a temperature actuator, a budget embedding, a usage-conditioned bias) is
DESCOPED -- see `docs/internals/DELETIONS.md`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import produce_routing, raw_side_gain, resolved_routing_from_topology

from rola.ops.naive import naive_rola
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
)

#: the fixtures' value width; it is the LAYER's fact, so a topology no longer
#: carries one and these gates state their own.
_D_V = 5


def _topology(mixed: bool = False, tied: bool = False) -> Topology:
    levels = [
        IndependentRouting(width=4, read=SoftmaxActivation(), write=SoftmaxActivation()),
        UnionRouting(width=6, alpha=1.5),
    ]
    if mixed:
        levels.append(IndependentRouting(
            width=3, read=SoftmaxActivation(), write=SoftmaxActivation()))
    if tied:
        levels.append(TiedRouting(width=3, op=EntmaxActivation(1.5)))
    return Topology(levels=tuple(levels))


def _params(topology: Topology, *, B, L, H, dm, dtype, seed=0, gain_sides=2):
    torch.manual_seed(seed)
    resolved = resolved_routing_from_topology(topology)
    x = torch.randn(B, L, dm, dtype=dtype, requires_grad=True)
    route_W = torch.randn(H, dm, resolved.packed_router_width, dtype=dtype, requires_grad=True)
    gain_W = torch.randn(H, gain_sides, dm, dtype=dtype, requires_grad=True)
    gain_bias = torch.zeros(H, gain_sides, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, _D_V, dtype=dtype)
    return x, route_W, gain_W, gain_bias, v


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_producer_output_satisfies_oracle_contract_and_differentiates(dtype):
    """Item 4: run the Stage 1 oracle end-to-end on the producer's output, fp64
    and fp32, and check gradients reach every producer input (value stream,
    router weights, side gains)."""
    topology = _topology(mixed=True)
    B, L, H, dm = 2, 9, 2, 7
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=1)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology)
    assert len(out.read_levels) == topology.D
    assert len(out.write_levels) == topology.D
    for level, width in enumerate(topology.widths):
        assert out.read_levels[level].shape == (B, L, H, width)
        assert out.write_levels[level].shape == (B, L, H, width)
        assert torch.allclose(
            out.read_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)
        assert torch.allclose(
            out.write_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)

    assert not hasattr(out, "g_read")
    assert out.g_write.shape == (B, L, H)
    assert torch.all(out.g_write > 0)

    y, _ = naive_rola(v, out.read_levels, out.write_levels, out.g_write, topology, None)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for name, tensor in [("x", x), ("route_W", route_W), ("gain_W", gain_W), ("gain_bias", gain_bias)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_producer_output_satisfies_oracle_contract_with_a_tied_routing_level(dtype):
    """Item 4's gate, repeated with a `TiedRouting` level in the mix: the producer's
    packed projection, solve and mass-fold all admit it end to end, with no code path
    that only knows `IndependentRouting`/`UnionRouting`."""
    topology = _topology(mixed=True, tied=True)
    B, L, H, dm = 2, 9, 2, 7
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=1)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology)
    assert len(out.read_levels) == len(out.write_levels) == topology.D
    tied_index = topology.D - 1
    torch.testing.assert_close(
        out.read_levels[tied_index], out.write_levels[tied_index], rtol=0, atol=0)
    for level, width in enumerate(topology.widths):
        assert out.read_levels[level].shape == (B, L, H, width)
        assert torch.allclose(
            out.read_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)
        assert torch.allclose(
            out.write_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)

    y, _ = naive_rola(v, out.read_levels, out.write_levels, out.g_write, topology, None)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for name, tensor in [("x", x), ("route_W", route_W), ("gain_W", gain_W), ("gain_bias", gain_bias)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"


def test_softmax_level_mass_is_identically_one():
    """`IndependentRouting`/softmax levels are unit-sum by construction;
    the producer's mass-fold (`_split_level_mass`) must be a no-op for them, so
    an all-softmax topology's g_write must equal its raw projected gain exactly
    (up to floating point), not scaled by any level mass."""
    levels = (IndependentRouting(width=4, read=SoftmaxActivation(), write=SoftmaxActivation()),)
    topology = Topology(levels=levels)
    B, L, H, dm = 1, 4, 1, 5
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology)
    raw = raw_side_gain(x, route_W, gain_W, gain_bias, resolved_routing_from_topology(topology))
    torch.testing.assert_close(out.g_write, raw["g_write"], rtol=1e-10, atol=1e-10)


def test_union_mass_fold_is_an_exact_identity_at_fp64():
    """The fold must MOVE the per-level mass, not approximate it (
    "multiplicative per-level masses collapse into one side-wide product").

    Required identity for every leaf `s`, on the write side -- the side that carries
    a gain under the shipped `global` arm:
        g_write_raw * prod_l level_l_raw[d_l(s)]  ==  g_write_out * prod_l p_l_out[d_l(s)]
    where `level_l_raw` is `routing_factor_levels`' own (possibly unnormalized, on
    UnionRouting's side) output and `p_l_out` is the producer's simplex.
    Checked against the machinery the producer wraps -- NOT against a second
    implementation of routing (the "matching-the-naive" standing rule).
    """
    from rola.routing.projection import _packed_router_logits
    from rola.routing.reference import routing_factor_levels

    topology = Topology(
        levels=(
            UnionRouting(width=4, alpha=1.5),
            IndependentRouting(width=5, read=SoftmaxActivation(), write=SoftmaxActivation()),
        ),
    )
    B, L, H, dm = 2, 6, 3, 7
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    resolved = resolved_routing_from_topology(topology)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology)

    read_logits, write_logits, _ = _packed_router_logits(
        x, route_W, resolved, route_bias=None, h_w=None, projection_dtype=torch.float64)
    reads_raw, writes_raw = routing_factor_levels(read_logits, write_logits, resolved)
    gains_raw = raw_side_gain(x, route_W, gain_W, gain_bias, resolved)

    def unfold(tensor):
        T, width = tensor.shape[1], tensor.shape[2]
        return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)

    # The union side must be genuinely unnormalized here, or the identity is vacuous.
    read_mass = unfold(reads_raw[0]).sum(-1)
    assert float((read_mass - 1).abs().max()) > 1e-3

    for side, raw_levels, raw_gain, out_levels, out_gain in (
        ("write", writes_raw, gains_raw["g_write"], out.write_levels, out.g_write),
    ):
        leaf_raw = raw_gain[..., None, None] * (
            unfold(raw_levels[0])[..., :, None] * unfold(raw_levels[1])[..., None, :])
        leaf_out = out_gain[..., None, None] * (
            out_levels[0][..., :, None] * out_levels[1][..., None, :])
        torch.testing.assert_close(
            leaf_out, leaf_raw, rtol=1e-13, atol=1e-13,
            msg=f"{side}-side mass fold is not an identity")
