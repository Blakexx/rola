"""Canonical fp64 routing reference helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rola.routing.entmax.reference import entmax, entmax_forward
from rola.routing.types import (
    READ_SUPPORT,
    ROUTE_TILE_SIZE,
    WRITE_SUPPORT,
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    RouteDescriptor,
    RoutingActivation,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
    _validated_alpha,
)


def _pack_token_support(support: torch.Tensor) -> torch.Tensor:
    """Pack semantic ``[..., L, branch]`` support into signed uint32-bit words."""
    tokens, branches = support.shape[-2:]
    n_words = (tokens + 31) // 32
    padding = n_words * 32 - tokens
    if padding:
        tail = torch.zeros(
            (*support.shape[:-2], padding, branches), dtype=torch.bool, device=support.device)
        support = torch.cat((support, tail), dim=-2)
    bits = (1 << torch.arange(32, device=support.device, dtype=torch.int64)).view(
        *((1,) * (support.ndim - 2)), 1, 32, 1)
    return (support.to(torch.int64).reshape(*support.shape[:-2], n_words, 32, branches) * bits).sum(
        dim=-2).to(torch.int32).contiguous()


def _activation_factors(
    logits: torch.Tensor,
    activation: RoutingActivation,
    *,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(activation, SoftmaxActivation):
        masked = logits if mask is None else logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(masked, dim=-1), None
    if isinstance(activation, EntmaxActivation):
        values = entmax(logits, activation.alpha, dim=-1, mask=mask)
        _, support, _ = entmax_forward(logits, activation.alpha, dim=-1, mask=mask)
        return values, _pack_token_support(support)
    raise TypeError(f"activation must be a routing activation, got {activation!r}")


def _union_split_with_support(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    *,
    alpha: float = 1.5,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use entmax support before splitting read/write roles."""
    alpha = _validated_alpha(alpha, "union_split")
    combined_logits = 0.5 * read_logits + 0.5 * write_logits
    support_values = entmax(combined_logits, alpha, dim=-1, mask=mask)
    _, support, _ = entmax_forward(combined_logits, alpha, dim=-1, mask=mask)
    finfo = torch.finfo(support_values.dtype)
    # Preserve representable deltas and give subtraction overflow the clamp's zero direct adjoint.
    delta = (read_logits - write_logits).clamp(min=-finfo.max, max=finfo.max)
    read = support_values * torch.sigmoid(delta)
    safe_values = torch.where(support, support_values, torch.ones_like(support_values))
    log_write = (safe_values.log() + F.logsigmoid(-delta)).masked_fill(~support, float("-inf"))
    has_support = support.any(dim=-1, keepdim=True)
    write = torch.softmax(torch.where(has_support, log_write, torch.zeros_like(log_write)), dim=-1)
    return read, torch.where(has_support & support, write, torch.zeros_like(write)), _pack_token_support(support)


def union_split(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    *,
    alpha: float = 1.5,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use entmax support before splitting read/write roles."""
    read, write, _ = _union_split_with_support(
        read_logits, write_logits, alpha=alpha, mask=mask)
    return read, write


def _tile_major_arena(levels: list[torch.Tensor], token_count: int) -> torch.Tensor:
    """Pack compact level widths into the descriptor's physical tile-major arena."""
    values = torch.cat(levels, dim=-1).to(torch.bfloat16)
    tiles = (token_count + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE
    padded = torch.nn.functional.pad(values, (0, 0, 0, tiles * ROUTE_TILE_SIZE - token_count))
    return padded.reshape(values.shape[0], tiles, ROUTE_TILE_SIZE, values.shape[-1]).permute(
        0, 1, 3, 2).contiguous()


def routing_factor_levels(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Solve reference semantic read/write factors without materializing a descriptor."""
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if read_logits.ndim != 3 or read_logits.shape[-1] != routing.packed_logit_width:
        raise ValueError(
            "routing logits must be [BH, T, packed_logit_width] with packed_logit_width="
            f"{routing.packed_logit_width}, got {tuple(read_logits.shape)}")

    reads, writes = [], []

    for index, level in enumerate(routing.levels):
        span = routing.level_logit_slice(index)
        read_level = read_logits[..., span]
        write_level = write_logits[..., span]
        if isinstance(level, UnionRouting):
            read_factor, write_factor, _ = _union_split_with_support(
                read_level, write_level, alpha=level.alpha, mask=None)
        else:
            assert isinstance(level, (IndependentRouting, TiedRouting))
            if isinstance(level, TiedRouting):
                write_factor, _ = _activation_factors(
                    write_level, level.activation("write"), mask=None)
                read_factor = write_factor
            else:
                read_factor, _ = _activation_factors(
                    read_level, level.read, mask=None)
                write_factor, _ = _activation_factors(
                    write_level, level.write, mask=None)
        reads.append(read_factor)
        writes.append(write_factor)

    return tuple(reads), tuple(writes)


def routing_factors(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
    *,
    token_origin: int = 0,
) -> RouteDescriptor:
    """Pack reference factors into the production RouteDescriptor layout."""
    if type(token_origin) is not int or token_origin < 0:
        raise ValueError("token_origin must be a nonnegative exact int")
    reads, writes = routing_factor_levels(read_logits, write_logits, routing)

    token_count = read_logits.shape[1]
    read_values = _tile_major_arena(reads, token_count)
    write_values = read_values if all(read is write for read, write in zip(reads, writes)) else _tile_major_arena(
        writes, token_count)
    support_chunks = []
    for index, (offset, flag) in enumerate(zip(routing.support_offsets, routing.support_side_flags)):
        if offset == -1:
            continue
        support = torch.zeros_like(reads[index], dtype=torch.bool)
        if flag & READ_SUPPORT:
            support |= reads[index].to(torch.bfloat16) != 0
        if flag & WRITE_SUPPORT:
            support |= writes[index].to(torch.bfloat16) != 0
        support_chunks.append(_pack_token_support(support))
    words = (token_count + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE
    support_words = torch.cat(support_chunks, dim=-1).contiguous() if support_chunks else torch.empty(
        (read_logits.shape[0], words, 0), dtype=torch.int32, device=read_logits.device)
    return RouteDescriptor(
        read_values, write_values, support_words,
        routing.level_offsets, routing.support_offsets, routing.branches,
        routing.radix_strides, routing.support_side_flags,
        token_origin, token_count, routing.topology_stamp,
    )


__all__ = ["routing_factor_levels", "routing_factors", "union_split"]
