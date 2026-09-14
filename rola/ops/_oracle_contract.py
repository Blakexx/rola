# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The oracle's input contract: everything `naive_rola` refuses, and why.

The oracle is read, not just run. It is the definition every Tier 1 gate is
anchored to, so the literal-clarity standard says the file should read like the
mathematics -- and it did not: 69 lines of shape and domain checking stood
between the reader and 40 lines of recurrence. None of it is
math. All of it is a boundary.

So the boundary lives here and `rola/ops/naive.py` calls it in one line. Nothing
is weakened by the move: every check below is the one that was there, and the
one semantic slack among them -- the simplex tolerance -- is now a named
constant with its derivation attached rather than a keyword default nobody had
to justify.

`ROLA_V3_SPEC.md` section references are the contract's authority; each check
cites the clause it enforces.
"""

from __future__ import annotations

import torch

from rola.routing.types import DecayConfig, LeafMassDecay, Topology

#: How far a routing level's mass may sit from 1 and still be accepted as a
#: simplex.
#:
#: THIS IS A SEMANTIC SLACK, NOT A FLOATING-POINT ONE, which is why it is named
#: here instead of defaulting quietly inside the check. The oracle is fed the STORED
#: form, which is bf16, and bf16
#: carries about 8 mantissa bits, so a width-256 simplex's per-entry rounding
#: accumulates to a few parts in 1e3 across the sum. `5e-3` covers that and
#: nothing more: an input that is genuinely unnormalized (a missing softmax, a
#: level summing to its width) misses it by orders of magnitude.
SIMPLEX_ATOL = 5e-3


def validate_simplex(
    name: str, levels: tuple[torch.Tensor, ...], *, atol: float = SIMPLEX_ATOL
) -> None:
    """Each level is nonnegative and sums to 1 over its width (spec section 4.1)."""
    for level, factors in enumerate(levels):
        if torch.any(factors < -atol):
            raise ValueError(f"{name}[{level}] must be nonnegative (p >= 0)")
        mass = factors.sum(dim=-1)
        if not torch.allclose(mass, torch.ones_like(mass), atol=atol):
            raise ValueError(
                f"{name}[{level}] must sum to 1 over its width (sum_d p = 1); "
                f"max deviation {float((mass - 1).abs().max())}"
            )


def validate_inputs(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_write: torch.Tensor,
    topology: Topology,
    decay: DecayConfig,
) -> tuple[int, int, int, int]:
    """Shapes and domains. Returns `(B, T, H, d_v)`."""
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B,T,H,d_v], got {tuple(v.shape)}")
    B, T, H, d_v = v.shape
    widths = topology.widths
    for name, levels in (("read_levels", read_levels), ("write_levels", write_levels)):
        levels = tuple(levels)
        if len(levels) != topology.D:
            raise ValueError(f"{name} must contain one tensor per level (D={topology.D}), got {len(levels)}")
        for level, (factors, width) in enumerate(zip(levels, widths)):
            if not isinstance(factors, torch.Tensor):
                raise TypeError(f"{name}[{level}] must be a torch.Tensor, got {type(factors).__name__}")
            if tuple(factors.shape) != (B, T, H, width):
                raise ValueError(
                    f"{name}[{level}] must have shape [B,T,H,width_l]=({B},{T},{H},{width}), "
                    f"got {tuple(factors.shape)}"
                )
    if tuple(g_write.shape) != (B, T, H):
        raise ValueError(f"g_write must have shape [B,T,H]={(B, T, H)}, got {tuple(g_write.shape)}")
    if torch.any(g_write <= 0):
        raise ValueError("g_write must be strictly positive: g_write[t] > 0")
    if decay is not None:
        if not isinstance(decay, LeafMassDecay):
            raise TypeError(f"decay must be None or LeafMassDecay, got {type(decay).__name__}")
        decay.validate_against(topology)
        for level, dial in enumerate(decay.dials):
            if dial.shape[0] != H:
                raise ValueError(
                    f"decay.dials[{level}] head dim {dial.shape[0]} must match H={H}"
                )
    return B, T, H, d_v


def initial_cache(
    initial_state: torch.Tensor | None, *, B: int, H: int, N: int, d_v: int, device
) -> torch.Tensor:
    """The carried state a run starts from, `[B,H,N,d_v+1]` fp64 (the mass column)."""
    expected = (B, H, N, d_v + 1)
    if initial_state is None:
        return torch.zeros(expected, device=device, dtype=torch.float64)
    if not isinstance(initial_state, torch.Tensor) or tuple(initial_state.shape) != expected:
        shape = getattr(initial_state, "shape", None)
        raise ValueError(f"initial_state must have shape {expected}, got {shape}")
    return initial_state.to(device=device, dtype=torch.float64)


__all__ = ["SIMPLEX_ATOL", "initial_cache", "validate_inputs", "validate_simplex"]
