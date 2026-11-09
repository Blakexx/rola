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
rows below read it off a real layer built from a declared `Routing`.
"""

from __future__ import annotations

import pytest
import torch

from rola.layer import RoLA
from rola.routing.producer import Routing, split_routing
from rola.routing.side_gain import GainConfig

_H, _DV = 2, 32
_HIDDEN = _H * _DV


def _layer(*, gain: bool, **kwargs) -> RoLA:
    #: the alternating scheme: every side has a dense level (docs/api.md section 2.1)
    levels = [split_routing(4, 1.5, sparse_duty="write"), split_routing(4, 1.5, sparse_duty="read")]
    return RoLA(Routing(levels, gain=gain, **kwargs), hidden_size=_HIDDEN,
                num_heads=_H, d_v=_DV).double()


def _factors(layer: RoLA, x: torch.Tensor):
    return layer._produce(x)[0]


def _gain(layer: RoLA, x: torch.Tensor):
    """``(gain, with, without)``: the layer's bundle, its gain-free twin's (the same routing columns, the gain column
    removed) and the write gain between them -- write level 0's scale over the twin's, since the producer folds its
    gain into that scale and nowhere else (docs/CONSTITUTION.md#r23)."""
    twin = _layer(gain=False)
    with torch.no_grad():
        twin.route_W.copy_(layer.route_W[..., :twin.route_W.shape[-1]])
    with_gain, without = _factors(layer, x), _factors(twin, x)
    return with_gain.scale_write[0] / without.scale_write[0], with_gain, without

# --- the public flag ---------------------------------------------------------


def test_the_public_flag_maps_onto_the_one_internal_boolean():
    """`gain` enables the WRITE gain, and there is no second side to enable:
    the readout is a ratio, which fixes `g_read := 1`."""
    assert GainConfig.from_public(True) == GainConfig(write_gain=True)
    assert GainConfig.from_public(False) == GainConfig(write_gain=False)


def test_the_public_flag_is_the_only_gain_parameter_on_the_routing():
    """A tri-state, or a pair of booleans, would put a research surface in the public
    signature. The fine-grained combinations stay reachable via `gain_config`.

    The gain is a fact about the routing's side gains, not about the layer wrapping
    it, so it sits on the declared `Routing` (`RoLA`'s own signature carries no gain
    knob at all)."""
    import dataclasses

    layer_parameters = __import__("inspect").signature(RoLA.__init__).parameters
    assert not any(name in layer_parameters for name in ("gain", "write_gain", "read_gain"))

    fields = {f.name: f for f in dataclasses.fields(Routing)}
    assert "gain" in fields and fields["gain"].default is True
    assert not any(name in fields for name in ("write_gain", "read_gain"))


def test_the_public_flag_rejects_a_non_bool_rather_than_coercing_it():
    with pytest.raises(TypeError, match="gain must be a bool"):
        GainConfig.from_public("auto")


# --- what the engine actually receives ---------------------------------------

def test_the_gain_is_write_level_0s_scale_and_there_is_never_a_read_gain():
    """The engine's contract has no gain field: the producer's write gain is a factor of write level 0's scale, and
    every other scale, every read scale and every level's rows are the gain-free layer's -- the ratio readout fixes a
    read gain to 1 identically, so there is none."""
    torch.manual_seed(5)
    x = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    layer = _layer(gain=True)
    gain, with_gain, without = _gain(layer, x)
    assert not hasattr(with_gain, "g_write") and not hasattr(with_gain, "g_read")
    assert gain.shape == (2, 6, _H) and bool((gain > 0).all()) and float((gain - 1).abs().max()) > 1e-2
    for name in ("amp_read", "amp_write", "scale_read"):
        for level, (a, b) in enumerate(zip(getattr(with_gain, name), getattr(without, name))):
            assert torch.equal(a, b), f"the gain reached {name}[{level}]"
    for level in range(1, len(with_gain.scale_write)):
        assert torch.equal(with_gain.scale_write[level], without.scale_write[level]), f"the gain reached level {level}"


def test_the_disabled_gain_removes_the_column_rather_than_pinning_it():
    """`gain=False` REMOVES the projection column: a parameter whose value is fixed is
    a parameter whose gradient is wasted, and a missing column is a constant."""
    routed = _layer(gain=False).routing.resolved.packed_router_width
    assert _layer(gain=False).route_W.shape[-1] == routed
    assert _layer(gain=True).route_W.shape[-1] == routed + 1
    assert _layer(gain=False).gain_bias.shape[-1] == 0
    assert _layer(gain=True).gain_bias.shape[-1] == 1


def test_the_gain_rides_the_router_gemm_and_is_not_a_second_projection():
    """THE FUSION, stated where it can fail: the gain's weight is `route_W`'s trailing
    span and there is no second weight anywhere on the module. Perturbing exactly those
    columns must move the gain and nothing else."""
    layer = _layer(gain=True)
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"route_W", "gain_bias", "v_proj.weight", "o_proj.weight"}, (
        f"the gain kept a projection of its own: {sorted(names)}")

    x = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    before = _factors(layer, x)
    with torch.no_grad():
        layer.route_W[:, :, -1] += 1.0
    after = _factors(layer, x)
    assert not torch.allclose(before.scale_write[0], after.scale_write[0]), (
        "the trailing column does not reach the gain")
    for name in ("amp_read", "amp_write", "scale_read", "scale_write"):
        for level, (b, a) in enumerate(zip(getattr(before, name), getattr(after, name))):
            if (name, level) != ("scale_write", 0):
                assert torch.equal(b, a), f"the gain column moved {name}[{level}]"


def test_the_side_gain_bias_init_places_the_zero_logit_gain():
    """`gain_bias_init` is the flagged model-level init decision, inverted through
    `softplus` so an all-zero-logit gain starts where the caller asked."""
    layer = _layer(gain=True, gain_bias_init=1.0)
    assert layer.routing.gain_layout.layout == ("write",) and layer.gain_bias.shape == (_H, 1)
    with torch.no_grad():
        layer.route_W.zero_()
    gain, _, _ = _gain(layer, torch.randn(1, 3, _HIDDEN, dtype=torch.float64))
    torch.testing.assert_close(gain, torch.ones(1, 3, _H), rtol=1e-6, atol=1e-6)
