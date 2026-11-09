# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The oracle's input contract: everything `naive_rola` refuses, and why.

The oracle is read, not just run. It is the definition every Tier 1 gate is
anchored to, so `rola/ops/naive.py` reads like the mathematics and its shape and
domain checks -- none of them math, all of them a boundary -- live here, called
in one line. The domain is nonnegative, finite levels: no level need sum to 1
(docs/CONSTITUTION.md#r23), and the op is defined on their products
(docs/api.md section 2.2).
"""

from __future__ import annotations

import torch

from rola.routing.types import DecayConfig, LeafMassDecay, Topology


def validate_levels(name: str, levels: tuple[torch.Tensor, ...]) -> None:
    """Each level is nonnegative and finite: the op is defined on the levels' products alone, and no level need sum to 1
    (docs/CONSTITUTION.md#r23)."""
    for level, factors in enumerate(levels):
        if torch.any(factors < 0):
            raise ValueError(f"{name}[{level}] must be nonnegative")
        if not torch.isfinite(factors).all():
            raise ValueError(f"{name}[{level}] must be finite")


def validate_inputs(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    topology: Topology,
    decay: DecayConfig,
) -> tuple[int, int, int, int, int]:
    """Shapes and domains. Returns `(B, T, H, H_w, d_v)`: `H` the query heads (the read side's), `H_w` the write
    heads (the write side's, the values' and the decay's), `H_w` dividing `H`."""
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B,T,H_w,d_v], got {tuple(v.shape)}")
    B, T, H_w, d_v = v.shape
    read_levels = tuple(read_levels)
    H = read_levels[0].shape[2] if read_levels and isinstance(read_levels[0], torch.Tensor) else H_w
    if H % H_w:
        raise ValueError(f"the write heads H_w={H_w} must divide the query heads H={H}: each write head is read "
                         "by H / H_w consecutive query heads")
    widths = topology.widths
    for name, levels, heads in (("read_levels", read_levels, H), ("write_levels", write_levels, H_w)):
        levels = tuple(levels)
        if len(levels) != topology.D:
            raise ValueError(f"{name} must contain one tensor per level (D={topology.D}), got {len(levels)}")
        for level, (factors, width) in enumerate(zip(levels, widths)):
            if not isinstance(factors, torch.Tensor):
                raise TypeError(f"{name}[{level}] must be a torch.Tensor, got {type(factors).__name__}")
            if tuple(factors.shape) != (B, T, heads, width):
                raise ValueError(
                    f"{name}[{level}] must have shape [B,T,heads,width_l]=({B},{T},{heads},{width}), "
                    f"got {tuple(factors.shape)}"
                )
    if decay is not None:
        if not isinstance(decay, LeafMassDecay):
            raise TypeError(f"decay must be None or LeafMassDecay, got {type(decay).__name__}")
        decay.validate_against(topology)
        for level, dial in enumerate(decay.dials):
            if dial.shape[0] != H_w:
                raise ValueError(
                    f"decay.dials[{level}] head dim {dial.shape[0]} must match the write heads H_w={H_w}"
                )
    return B, T, H, H_w, d_v


def initial_cache(
    initial_state: torch.Tensor | None, *, B: int, H: int, N: int, d_v: int, device
) -> torch.Tensor:
    """The carried state a run starts from, `[B,H,N,d_v+1]` fp64 (the mass column), `H` the write heads."""
    expected = (B, H, N, d_v + 1)
    if initial_state is None:
        return torch.zeros(expected, device=device, dtype=torch.float64)
    if not isinstance(initial_state, torch.Tensor) or tuple(initial_state.shape) != expected:
        shape = getattr(initial_state, "shape", None)
        raise ValueError(f"initial_state must have shape {expected}, got {shape}")
    return initial_state.to(device=device, dtype=torch.float64)


__all__ = ["initial_cache", "validate_inputs", "validate_levels"]
