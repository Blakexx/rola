# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The gain surface (user ruling, 2026-08-01): one public flag, no dead knobs.

The design claim being gated is not "the gains compute the right numbers" -- the
tier-1 gates do that. It is that **every reachable setting of the configuration is
connected to real behaviour**: a disabled gain is a missing projection column rather
than a pinned parameter, and the one combination that would optimize a parameter with
an identically zero gradient is refused at construction.

The gain's WEIGHT is columns of the router GEMM (`route_W`'s trailing span), so
"a disabled gain is a missing column" is now literally a narrower parameter, and the
rows below read it off a real `RouteProducer`.
"""

from __future__ import annotations

import pytest
import torch

from rola.layer import RoLA
from rola.routing.producer import RouteProducer, uniform, union_routing
from rola.routing.side_gain import GainConfig, SideGain

_H, _DV = 2, 32
_HIDDEN = _H * _DV


def _producer(*, gain: bool):
    return RouteProducer(uniform(2, union_routing(4, 1.5)), hidden_size=_HIDDEN,
                         num_heads=_H, gain=gain).double()

# --- the public flag ---------------------------------------------------------


def test_the_public_flag_maps_onto_the_one_internal_boolean():
    """`gain` enables the WRITE gain, and there is no second side to enable:
    the readout is a ratio, which fixes `g_read := 1`."""
    assert GainConfig.from_public(True) == GainConfig(write_gain=True)
    assert GainConfig.from_public(False) == GainConfig(write_gain=False)


def test_the_public_flag_is_the_only_gain_parameter_on_the_route_producer():
    """A tri-state, or a pair of booleans, would put a research surface in the public
    signature. The fine-grained combinations stay reachable via `gain_config`.

    MIGRATED (API-contract break, 2026-08-05): `gain` moved off `RoLA.__init__`
    onto `RouteProducer.__init__` -- it is a fact about the feature map's side
    gains, not about the layer wrapping it (`RoLA`'s own signature carries no gain
    knob at all any more)."""
    import inspect

    from rola.routing.producer import RouteProducer

    layer_parameters = inspect.signature(RoLA.__init__).parameters
    assert not any(name in layer_parameters for name in ("gain", "write_gain", "read_gain"))

    parameters = inspect.signature(RouteProducer.__init__).parameters
    assert "gain" in parameters and parameters["gain"].default is True
    assert not any(name in parameters for name in ("write_gain", "read_gain"))


def test_the_public_flag_rejects_a_non_bool_rather_than_coercing_it():
    with pytest.raises(TypeError, match="gain must be a bool"):
        GainConfig.from_public("auto")


# --- what the engine actually receives ---------------------------------------

@pytest.mark.parametrize("gain,write_is_one", [(True, False), (False, True)])
def test_a_missing_column_becomes_the_constant_one_and_there_is_never_a_read_gain(
        gain, write_is_one):
    """The engine's contract does not change shape when the gain is ablated: an absent
    write gain is the CONSTANT 1, and there is no read-gain column at all in either
    case -- the ratio readout fixes it to 1 identically."""
    torch.manual_seed(5)
    producer = _producer(gain=gain)
    factors = producer(torch.randn(2, 6, _HIDDEN, dtype=torch.float64))
    assert not hasattr(factors, "g_read")
    assert factors.g_write.shape == (2, 6, _H)
    assert torch.equal(
        factors.g_write, torch.ones_like(factors.g_write)) is write_is_one


def test_the_disabled_gain_removes_the_column_rather_than_pinning_it():
    """`gain=False` REMOVES the projection column: a parameter whose value is fixed is
    a parameter whose gradient is wasted, and a missing column is a constant."""
    routed = _producer(gain=False).routing.packed_router_width
    assert _producer(gain=False).route_W.shape[-1] == routed
    assert _producer(gain=True).route_W.shape[-1] == routed + 1
    assert _producer(gain=False).side_gain.columns == 0
    assert _producer(gain=True).side_gain.columns == 1


def test_the_gain_rides_the_router_gemm_and_is_not_a_second_projection():
    """THE FUSION, stated where it can fail: the gain's weight is `route_W`'s trailing
    span and there is no second weight anywhere on the module. Perturbing exactly those
    columns must move the gain and nothing else."""
    producer = _producer(gain=True)
    names = {name for name, _ in producer.named_parameters()}
    assert names == {"route_W", "side_gain.gain_bias"}, (
        f"the gain kept a projection of its own: {sorted(names)}")

    x = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    before = producer(x)
    with torch.no_grad():
        producer.route_W[:, :, -1] += 1.0
    after = producer(x)
    assert not torch.allclose(before.g_write, after.g_write), (
        "the trailing column does not reach the gain")
    for side in ("read", "write"):
        for level, (b, a) in enumerate(zip(getattr(before, side), getattr(after, side))):
            torch.testing.assert_close(b, a, rtol=0, atol=0,
                                       msg=f"the gain column moved {side} level {level}")


def test_the_side_gain_bias_init_places_the_zero_logit_gain():
    """`bias_init` is the flagged model-level init decision, inverted through
    `softplus` so an all-zero-logit gain starts where the caller asked."""
    module = SideGain(num_heads=2, config=GainConfig.from_public(True), bias_init=1.0)
    assert module.layout == ("write",) and module.gain_bias.shape == (2, 1)
    gains = module(torch.zeros(1, 3, 2, 1), batch=1, tokens=3, dtype=torch.float32,
                   device="cpu")
    torch.testing.assert_close(gains["g_write"], torch.ones(1, 3, 2), rtol=1e-6, atol=1e-6)
