# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facade's stateful half: the SURFACE that survives the K31 deletion.

Until the K31 batch (baseline = tag `baseline/pre-k31`) this file walked the
whole stateful surface through the prefill op -- bind-then-continue, chained
halves vs one whole call, clone independence, dense-vs-paged agreement, the BC
flip over a live arena. Those claims are claims about the PREFILL BINDING PATH,
which is deleted, and they return with it; what remains gated
here is the state object's own contract -- construction refusals, the engine's
state-type refusal, and the decode continuation's carrying requirement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import build_routes  # noqa: E402

import rola  # noqa: E402
from rola.routing.producer import union_routing  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode arm is a CUDA kernel"),
]

_HIDDEN, _H, _DV, _WIDTHS, _T = 64, 2, 64, (16, 16), 128


def _producer(seed=0, widths=_WIDTHS):
    torch.manual_seed(seed)
    return build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=widths,
                        routing=union_routing(1)).cuda()


def _inputs(seed=0, T=_T, B=2):
    torch.manual_seed(seed + 1000)
    return (torch.randn(B, T, _HIDDEN, device="cuda"),
            torch.randn(B, T, _H, _DV, device="cuda"))


def test_a_raw_tensor_is_not_a_state():
    """The engine's fast-fail refuses a raw tensor where a RoLAState belongs.

    Gated at `validate_context` (the seam the rebuilt kernel re-fronts): `rola_op` itself is
    the deletion refusal and speaks before it reads its arguments.
    """
    from rola.engine.dags.chunk_dag import ChunkContext, validate_context

    producer, (x, v) = _producer(), _inputs()
    with torch.no_grad(), pytest.raises(TypeError, match="rola.RoLAState"):
        validate_context(ChunkContext(routes=producer(x), v=v,
                                      state=torch.zeros(1), want_state=True))


def test_an_unbuilt_strategy_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="cow"):
        rola.state("cow")
    with pytest.raises(ValueError, match="strategy must be"):
        rola.state("mutable-ish")


def test_a_fresh_state_carries_nothing_and_decode_refuses_it():
    """`carries` is contents, not binding, and a decode step CONTINUES: with the
    prefill arm deleted there is nothing in production to charge a fresh state,
    and the decode entry's refusal must still say so by name."""
    state = rola.state()
    assert not state.bound and not state.carries
    with pytest.raises(RuntimeError, match="carries nothing"):
        state._require_carrying()
