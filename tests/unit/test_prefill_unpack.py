"""The prefill's unpack of the carry's ``num`` (`rola.ops.prefill.unpack_num`): its multiply over the f16 halves is the
oracle's channel map (`tests/oracle/readout_reference.unpack_num`, the definition: masks, casts and a stack) times the
token's factor, BIT FOR BIT, for the f16-pair words and for an fp32 ``num``.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.prefill import unpack_num
from tests.oracle.readout_reference import unpack_num as reference_unpack


def _words(BH: int, L: int, words: int, seed: int) -> torch.Tensor:
    """Random finite f16 pairs as the drain's int32 words: each half a random f16 of either sign, never inf or NaN."""
    g = torch.Generator().manual_seed(seed)
    halves = (torch.randn(BH, L, 2 * words, generator=g) * 1e3).to(torch.float16)
    return halves.view(torch.int32).reshape(BH, L, words)


@pytest.mark.parametrize("words", [16, 32, 64])
def test_the_pairs_unpack_scaled_as_the_reference_maps_them(words):
    num = _words(3, 40, words, seed=words)
    factor = torch.exp2(torch.randint(-8, 8, (3, 40)).float() + 1.0) * 0.73
    got = unpack_num(num, factor)
    want = reference_unpack(num) * factor.unsqueeze(-1)
    assert got.shape == want.shape and got.dtype == torch.float32 and got.is_contiguous()
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    assert torch.equal(unpack_num(num, 0.5).view(torch.int32), (reference_unpack(num) * 0.5).view(torch.int32))


def test_an_fp32_num_is_scaled_in_place_order():
    num = torch.randn(2, 16, 64)
    assert torch.equal(unpack_num(num, 0.5), num * 0.5)
    factor = torch.rand(2, 16)
    assert torch.equal(unpack_num(num, factor), num * factor.unsqueeze(-1))
