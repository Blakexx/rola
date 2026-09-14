# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A CROWN-JEWELS GATE ON THE REFERENCE: the oracle's own autograd vs finite differences.

This is the ONLY place ``torch.autograd.gradcheck`` belongs. Finite differences need
perturbations far below bf16 resolution and the kernels' envelope forbids the shapes
gradcheck wants, so the kernels are gated by a PER-INPUT COTANGENT COMPARISON against
this reference instead -- and that comparison is only worth something because this
test certifies the reference it compares to.
"""
from __future__ import annotations

import torch

from rola.ops.naive import naive_rola
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _topology(width, D):
    level = IndependentRouting(width=width, read=SoftmaxActivation(),
                              write=SoftmaxActivation())
    return Topology(levels=(level,) * D)


def _simplex(shape, gen):
    x = torch.rand(*shape, device=DEV, generator=gen, dtype=torch.float64) + 0.05
    return (x / x.sum(-1, keepdim=True)).requires_grad_()


def test_the_oracles_autograd_equals_finite_differences():
    T, width, D, d_v = 4, 4, 2, 8
    topo = _topology(width, D)
    gen = torch.Generator(device=DEV).manual_seed(1)

    def y_of(*flat):
        read = list(flat[:D])
        write = list(flat[D:2 * D])
        gw, v = flat[2 * D], flat[2 * D + 1]
        return naive_rola(v, read, write, gw, topo, None)[0]

    flat = tuple([_simplex((1, T, 1, width), gen) for _ in range(2 * D)]
                 + [(torch.rand(1, T, 1, device=DEV, generator=gen,
                                dtype=torch.float64) + 0.5).requires_grad_(),
                    torch.randn(1, T, 1, d_v, device=DEV, generator=gen,
                                dtype=torch.float64).requires_grad_()])
    assert torch.autograd.gradcheck(y_of, flat, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_the_oracles_final_state_autograd_equals_finite_differences():
    """The chained cotangent path: ``state_out``'s gradient is what seeds Scan B."""
    T, width, D, d_v = 3, 4, 2, 4
    topo = _topology(width, D)
    gen = torch.Generator(device=DEV).manual_seed(2)

    def state_of(*flat):
        read = list(flat[:D])
        write = list(flat[D:2 * D])
        gw, v, s0 = flat[2 * D], flat[2 * D + 1], flat[2 * D + 2]
        return naive_rola(v, read, write, gw, topo, None,
                          initial_state=s0, output_final_state=True)[1]

    N = width ** D
    flat = tuple([_simplex((1, T, 1, width), gen) for _ in range(2 * D)]
                 + [(torch.rand(1, T, 1, device=DEV, generator=gen,
                                dtype=torch.float64) + 0.5).requires_grad_(),
                    torch.randn(1, T, 1, d_v, device=DEV, generator=gen,
                                dtype=torch.float64).requires_grad_(),
                    (0.1 * torch.randn(1, 1, N, d_v + 1, device=DEV, generator=gen,
                                       dtype=torch.float64)).requires_grad_()])
    assert torch.autograd.gradcheck(state_of, flat, eps=1e-6, atol=1e-7, rtol=1e-4)
