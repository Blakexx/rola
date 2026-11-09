"""A parameter's no-grad copy (rola/_inference_copy.py): equal to a fresh cast bit for bit, reused while the parameter
is unchanged, rebuilt by an in-place update, and never read by a step a gradient flows through.

See docs/api.md section 2.3 (projection-weight).
"""
from __future__ import annotations

import torch
from torch import nn

from rola._inference_copy import InferenceCopy, reads_copy
from rola.ops.produce import hidden_contiguous


def test_the_copy_is_the_cast_and_is_reused_until_the_weight_moves():
    W = nn.Parameter(torch.randn(4, 8, 6))
    copy = InferenceCopy(lay=hidden_contiguous)
    first = copy(W, torch.bfloat16)
    assert torch.equal(first, W.detach().to(torch.bfloat16))
    assert copy(W, torch.bfloat16) is first
    with torch.no_grad():
        W.add_(1.0)
    moved = copy(W, torch.bfloat16)
    assert moved is not first and torch.equal(moved, W.detach().to(torch.bfloat16))
    assert copy(W, torch.float16).dtype is torch.float16


def test_a_step_a_gradient_reaches_never_reads_the_copy():
    W = nn.Parameter(torch.randn(3, 3))
    assert not reads_copy(W)
    with torch.no_grad():
        assert reads_copy(W)
    assert reads_copy(W.detach())
