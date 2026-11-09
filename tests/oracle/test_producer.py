"""Producer tests -- the feature map's own stage.

Validates `rola.routing.producer` against the ONE canonical
ground truth this branch authorizes: the Stage 1 fp64 oracle
(`rola.ops.naive.naive_rola`). Per the "matching-the-naive
anti-pattern" standing rule, nothing here is checked against a second,
freshly written reference -- every assertion either (a) feeds the producer's
output through the oracle and checks numerical properties the oracle itself
guarantees (finite output, differentiability), or (b) checks an algebraic
identity of the producer's own contract (each level stored as its solve produced
it, the write gain folded into write level 0: docs/CONSTITUTION.md#r23) that is
checkable without an independent implementation.

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

#: THE FORMAT'S ROUNDING BUDGET on a row's sum: each digit is within half a u8 step of its row max, so a
#: width-w row's sum is within w / 510 of its own sum times its max (`rola.routing.quant`).
_ROW_BUDGET = 16 / 510 + 1e-4
#: the fixtures' value width; it is the LAYER's fact, so a topology no longer carries one and these gates state their
#: own.
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

    out = produce_routing(x, v, route_W, None, gain_W, gain_bias, topology)
    assert len(out.read_levels) == topology.D
    assert len(out.write_levels) == topology.D
    for level, width in enumerate(topology.widths):
        for side in (out.read_levels, out.write_levels):
            assert side[level].shape == (B, L, H, width)
            assert torch.isfinite(side[level]).all() and bool((side[level] >= 0).all())

    y, _ = naive_rola(v, out.read_levels, out.write_levels, topology, None)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for name, tensor in [("x", x), ("route_W", route_W), ("gain_W", gain_W), ("gain_bias", gain_bias)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_producer_output_satisfies_oracle_contract_with_a_tied_routing_level(dtype):
    """Item 4's gate, repeated with a `TiedRouting` level in the mix: the producer's
    packed projection, solve and storage all admit it end to end, with no code path
    that only knows `IndependentRouting`/`UnionRouting`."""
    topology = _topology(mixed=True, tied=True)
    B, L, H, dm = 2, 9, 2, 7
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=1)

    out = produce_routing(x, v, route_W, None, gain_W, gain_bias, topology)
    assert len(out.read_levels) == len(out.write_levels) == topology.D
    tied_index = topology.D - 1
    torch.testing.assert_close(
        out.read_levels[tied_index], out.write_levels[tied_index], rtol=0, atol=0)
    for level, width in enumerate(topology.widths):
        assert out.read_levels[level].shape == (B, L, H, width)
        assert torch.isfinite(out.read_levels[level]).all() and bool((out.read_levels[level] >= 0).all())

    y, _ = naive_rola(v, out.read_levels, out.write_levels, topology, None)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for name, tensor in [("x", x), ("route_W", route_W), ("gain_W", gain_W), ("gain_bias", gain_bias)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"


def test_the_write_gain_is_write_level_0s_magnitude():
    """THE GAIN IS FOLDED INTO WRITE LEVEL 0 and nowhere else (docs/CONSTITUTION.md#r23): on an all-softmax topology
    every level is unit-sum by its solve, so write level 0's row sum IS the producer's write gain -- to the format's
    rounding of that row -- and every other level, and every read level, sums to one."""
    levels = (IndependentRouting(width=4, read=SoftmaxActivation(), write=SoftmaxActivation()),
              IndependentRouting(width=3, read=SoftmaxActivation(), write=SoftmaxActivation()))
    topology = Topology(levels=levels)
    B, L, H, dm = 1, 4, 1, 5
    x, route_W, gain_W, gain_bias, v = (t.detach() for t in _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1))
    out = produce_routing(x, v, route_W, None, gain_W, gain_bias, topology)
    gain = raw_side_gain(x, route_W, gain_W, gain_bias, resolved_routing_from_topology(topology))
    assert float((gain - 1).abs().max()) > 1e-2, "a gain of one would make this gate vacuous"
    ones = torch.ones(B, L, H, dtype=torch.float64)
    for level in range(topology.D):
        torch.testing.assert_close(out.read_levels[level].sum(-1).double(), ones, rtol=0, atol=_ROW_BUDGET)
        want = gain if level == 0 else ones
        torch.testing.assert_close(out.write_levels[level].sum(-1).double(), want, rtol=_ROW_BUDGET, atol=0)


def test_the_levels_are_stored_as_their_solve_produced_them_at_fp64():
    """THE PRODUCER STORES WHAT ITS SOLVE PRODUCED (docs/CONSTITUTION.md#r23): no level is divided by its sum, and the
    write gain is folded into write level 0. Checked on a union level, whose read half does not sum to one:

        read:   out_l[d]                        ==  level_l_raw[d]
        write:  out_0[d_0(s)] * out_1[d_1(s)]   ==  g_raw * level_0_raw[d_0(s)] * level_1_raw[d_1(s)]

    where ``level_l_raw`` is `routing_factor_levels`' own output and ``g_raw`` the producer's gain through its own
    pieces (`raw_side_gain`). Checked against the machinery the producer wraps -- NOT against a second
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
    x, route_W, gain_W, gain_bias, v = (t.detach() for t in _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1))
    resolved = resolved_routing_from_topology(topology)

    out = produce_routing(x, v, route_W, None, gain_W, gain_bias, topology)

    read_logits, write_logits, _ = _packed_router_logits(
        x, route_W, resolved, route_bias=None, projection_dtype=torch.float64)
    reads_raw, writes_raw = routing_factor_levels(read_logits, write_logits, resolved)
    gain_raw = raw_side_gain(x, route_W, gain_W, gain_bias, resolved)

    def unfold(tensor):
        T, width = tensor.shape[1], tensor.shape[2]
        return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)

    #: The union read must be genuinely unnormalized here, or the read identity is vacuous.
    read_mass = unfold(reads_raw[0]).sum(-1)
    assert float((read_mass - 1).abs().max()) > 1e-3

    #: THE FORMAT ROUNDS EACH LEVEL'S ROW against its max (`rola.routing.quant`): |delta| <= m / 510 a digit, so a
    #: level holds to that and a leaf product to m_0 * m_1 / 255 (+ the second order term).
    for level in range(topology.D):
        raw = unfold(reads_raw[level])
        bound = raw.abs().amax(-1, keepdim=True) / 510.0 * 1.01 + 1e-13
        excess = ((out.read_levels[level] - raw).abs() - bound).clamp_min(0)
        assert float(excess.max()) == 0.0, (
            f"read level {level} is not its solve's output: worst excess {float(excess.max()):.3e}")
    leaf_raw = gain_raw[..., None, None] * (
        unfold(writes_raw[0])[..., :, None] * unfold(writes_raw[1])[..., None, :])
    leaf_out = out.write_levels[0][..., :, None] * out.write_levels[1][..., None, :]
    m0, m1 = out.write_levels[0].abs().amax(-1), out.write_levels[1].abs().amax(-1)
    bound = (m0 * m1 / 255.0 * 1.01 + 1e-13)[..., None, None]
    excess = ((leaf_out - leaf_raw).abs() - bound).clamp_min(0)
    assert float(excess.max()) == 0.0, (
        f"the write leaves are not the gain times the solve's: worst excess {float(excess.max()):.3e}")
