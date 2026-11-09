# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRODUCER: one call from a stretch of tokens to the bundle -- the routes and the values, in the internal format.

    routes, values = produce(x, plan, weights)

The layer declares WHAT it routes (:class:`rola.routing.producer.Routing`), owns the weights, and compiles a
:class:`ProducerPlan` once; this is the one primitive that computes the bundle from them. It is the whole job --
the projection, every level's solve, the write gain (folded into write level 0's scale), the quantization into
unit-max rows and the value digits -- because every step reads the same declaration, and it is one call so that one fused kernel stands
behind it (docs/api.md section 2): on CUDA with no gradient, the producer kernel's one launch
(docs/internals/produce/produce.md). The torch composition is the reference -- fp64, the host -- and, until the
native backward has the kernel write the levels a gradient reads, the path a gradient takes. The bundles it
returns are the ones the ops take; a caller with its own levels builds them with :meth:`RouteFactors.from_levels` and
:meth:`Values.from_projection`, the same format utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange

from rola.engine.facts.operands import requires_backward
from rola.routing import quant
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.factors import RouteFactors, default_sparse
from rola.routing.producer import Routing
from rola.routing.projection import _packed_router_logits
from rola.routing.side_gain import WRITE_GAIN, softplus_side_gain
from rola.routing.types import IndependentRouting, TiedRouting, UnionRouting
from rola.values import Values

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "ProducerPlan", "ProducerWeights", "hidden_contiguous", "job_table",
           "ledger_bind", "produce", "projection_dtype", "route_logits", "row_layout", "run_kernel"]


@dataclass(frozen=True)
class ProducerPlan:
    """What one layer's producer computes, compiled once from its declaration: the routing, the heads, the widths
    in and out. Every per-call decision is read off this, never off the data."""

    routing: Routing
    num_heads: int
    hidden_size: int
    d_v: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "hidden_size", "d_v"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact int, got {value!r}")


@dataclass(frozen=True)
class ProducerWeights:
    """The tensors a producer call reads, as the call reads them: ``route_W [H, hidden, columns]`` in the projection's
    dtype (``projection_dtype``), ``route_bias [H, columns]`` or ``None``, ``gain_bias [H, gain columns]``, and
    ``v_W [H * d_v, hidden]`` (a linear's weight) in bf16. The kernel reads both weights along the hidden axis, so it
    takes them with that axis contiguous (:func:`hidden_contiguous`; a linear's weight already is)."""

    route_W: torch.Tensor
    route_bias: torch.Tensor | None
    gain_bias: torch.Tensor
    v_W: torch.Tensor


def projection_dtype(x: torch.Tensor) -> torch.dtype:
    """The routing projection's arithmetic: fp64 in, fp64 out (the oracle-equivalence gate runs there); on CUDA bf16
    operands with fp32 accumulate, the tensor-core class; fp32 on the host, which has no tensor cores to reach."""
    if x.dtype == torch.float64:
        return torch.float64
    return torch.bfloat16 if x.is_cuda else torch.float32


def hidden_contiguous(route_W: torch.Tensor) -> torch.Tensor:
    """``route_W [H, hidden, columns]`` stored with its hidden axis contiguous and viewed back: the kernel loads a
    column's whole hidden run in 16-byte chunks, straight into its MMA fragments."""
    return route_W.transpose(1, 2).contiguous().transpose(1, 2)


#: THE JOB TABLE's kinds and solves, mirrored from csrc/rola/src/produce/produce.cuh (`JobKind`, `SolveKind`).
_ROUTE, _VALUE = 0, 1
_SOLVES = {("softmax", None): 0, ("entmax", 1.5): 1, ("entmax", 2.0): 2, ("union", 1.5): 3, ("union", 2.0): 4}
#: the kernel's widest job (a union level's two rows, or a value job's channels) and widest solve row
_MAX_COLUMNS = 512
_MAX_WIDTH = 256


def _solve(activation) -> int:
    """The kernel's solve for an activation, by its registry tag."""
    if activation.tag not in _SOLVES:
        raise ValueError(f"the producer kernel has no solve for the activation {activation.tag}")
    return _SOLVES[activation.tag]


def row_layout(plan: ProducerPlan, side: str = "read", dense: bool = False) -> tuple[tuple[int, ...], int]:
    """``(offsets, width)``: each level's slot in one of a side's packed row planes and the plane's width, in elements --
    the plane's levels in order, each at its bytes rounded up to the row alignment law's grain
    (`rola.routing.quant.row_slot`). Each side is a u8 plane of its sparse levels and (``dense``) an f16 plane of its
    dense ones. A level outside the plane takes no slot (offset 0)."""
    members = dense_levels(plan, side) if dense else (
        frozenset(range(plan.routing.topology.D)) - dense_levels(plan, side))
    size = 2 if dense else 1
    offsets, end = [], 0
    for level, width in enumerate(plan.routing.resolved.branches):
        offsets.append(end if level in members else 0)
        end += quant.row_slot(size * width) // size if level in members else 0
    return tuple(offsets), end


def dense_levels(plan: ProducerPlan, side: str) -> frozenset[int]:
    """The plan's dense levels on ``side``: their rows are f16 (`rola.routing.quant.DENSE_DTYPE`)."""
    return frozenset(level for level, sparse in enumerate(default_sparse(plan.routing.topology)[side == "write"]) if not sparse)


def job_table(plan: ProducerPlan) -> list[tuple[int, ...]]:
    """The plan compiled into the kernel's job table (docs/internals/produce/produce.md
    #jobs). A route job is one head's one solve -- a union level's two logit rows, a tied level's one row filling both
    planes, or one side of an independent level; a value job is the heads whose channels fit one job."""
    resolved = plan.routing.resolved
    slots, read_slot = resolved.packed_slot_offsets, resolved.packed_read_slot_lookup

    def offsets(side: str) -> tuple[int, ...]:
        dense, sparse_at, dense_at = (dense_levels(plan, side), row_layout(plan, side)[0],
                                      row_layout(plan, side, dense=True)[0])
        return tuple(dense_at[level] if level in dense else sparse_at[level] for level in range(len(sparse_at)))

    out, out_w = offsets("read"), offsets("write")
    per_head: list[tuple[int, ...]] = []
    for level, spec in enumerate(resolved.levels):
        width = resolved.branches[level]
        if width > _MAX_WIDTH:
            raise ValueError(f"the producer solves rows of at most {_MAX_WIDTH} leaves; level {level} is {width} wide")
        write_col, read_col = slots[level], slots[read_slot[level]]
        if isinstance(spec, UnionRouting):
            per_head.append((_ROUTE, 0, 1, _SOLVES[("union", spec.alpha)], width, read_col, write_col, out[level],
                             out_w[level], level))
        elif isinstance(spec, TiedRouting):
            per_head.append((_ROUTE, 0, 1, _solve(spec.op), width, write_col, -1, out[level], out_w[level], level))
        elif isinstance(spec, IndependentRouting):
            per_head.append((_ROUTE, 0, 1, _solve(spec.read), width, read_col, -1, out[level], -1, level))
            per_head.append((_ROUTE, 0, 1, _solve(spec.write), width, write_col, -1, -1, out_w[level], level))
    jobs = [(job[0], head, *job[2:]) for head in range(plan.num_heads) for job in per_head]
    if plan.d_v not in (32, 64, 128, 256):
        raise ValueError(f"the producer's value job reads heads of 32, 64, 128 or 256 channels, not d_v={plan.d_v}")
    if plan.d_v > _MAX_COLUMNS:
        raise ValueError(f"a value job holds at most {_MAX_COLUMNS} channels, one head's d_v={plan.d_v} does not fit")
    heads_per = max(1, _MAX_COLUMNS // 2 // plan.d_v)
    for head in range(0, plan.num_heads, heads_per):
        jobs.append((_VALUE, head, min(heads_per, plan.num_heads - head), 0, 0, -1, -1, -1, -1, 0))
    return jobs


#: a warp's accumulator columns: 16 column tiles of 16 (produce.cuh `kWarpColumns`)
_WARP_COLUMNS = 256


def _shape(job: tuple[int, ...], plan: ProducerPlan) -> int:
    """A job's control as the kernel reads it, one int16 word in the launch's parameters, so every branch on it is
    warp-uniform to ptxas: ``dense_write << 13 | dense_read << 12 | pair << 11 | tiles << 8 | writes << 7 |
    reads << 6 | kind << 5 | rows << 3 | solve`` -- ``rows`` the solve row's class
    (widths to 32, 64, 128, 256), ``tiles`` the projection's column tiles as a power of two's exponent, ``pair`` a wide
    union's warp-pair form, ``dense_read``/``dense_write`` whether the side's level is dense (an f16 row),
    ``reads``/``writes`` the sides a route job writes (produce.cuh `Job`)."""
    kind, _head, heads, solve, width, _in0, in1, out_read, out_write, level = job
    d_v = plan.d_v
    #: a value job's `rows` is its head's channel class, `d_v = 32 << rows`: a lane's value's head is then an immediate
    rows = ({32: 0, 64: 1, 128: 2, 256: 3}[d_v] if kind == _VALUE
            else 0 if width <= 32 else 1 if width <= 64 else 2 if width <= 128 else 3)
    union = kind != _VALUE and in1 >= 0
    #: a narrow union's write row at its width class's tile offset (32 << rows columns), so a column's two logits share
    #: a lane; a wider one is a warp pair's
    columns = heads * d_v if kind == _VALUE else 2 * (32 << rows) if union else width
    pair = columns > _WARP_COLUMNS
    count = 16 if pair else -(-columns // 16)
    #: the projection's classes are 4 to 16 column tiles: a narrower job's stage is its x rows' copies, under the
    #: MMAs' rate on sm_80, until narrow rows are packed into one job
    tiles = next(c for c in range(2, 5) if count <= 1 << c)
    dense = [kind != _VALUE and level in dense_levels(plan, side) for side in ("read", "write")]
    return (int(dense[1]) << 13 | int(dense[0]) << 12 | int(pair) << 11 | tiles << 8 | int(out_write >= 0) << 7
            | int(out_read >= 0) << 6 | kind << 5 | rows << 3 | solve)


@lru_cache(maxsize=64)
def _job_tensor(plan: ProducerPlan, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(table, shapes)``: the job table's addressing on the device (the head, the heads, the
    width, the columns in and out, the level) and its control words on the host."""
    jobs = job_table(plan)
    table = torch.tensor([job[1:3] + job[4:10] for job in jobs], dtype=torch.int32, device=device)
    return table, torch.tensor([_shape(job, plan) for job in jobs], dtype=torch.int16)


#: THE PHASE LEDGER'S SHAPE: ``[jobs][warps][phases]``, the kernel's own `ProducePhase` order, booked over the tiles.
LEDGER_PHASES = ("job", "project", "threshold", "levels", "bundle", "values")
LEDGER_WARPS = 4


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/produce/ampere/produce_kernel.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[jobs, 4, 6]`` int64 cycles, or ``None``."""
    from rola.ops._ext import extension

    extension().rola_produce_ledger_bind(ledger)


def run_kernel(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> dict[str, torch.Tensor]:
    """The fused producer's launch over bf16 CUDA operands: every output ``[B, T, H, ...]``, a view of the head-major
    ``[B, H, T, ...]`` storage the kernel writes -- a head's tokens contiguous, which every prefill reader walks, and
    the same bytes as token-major at one token (docs/internals/produce/ampere/produce_kernel.md#bundle)."""
    from rola.ops._ext import extension

    B, T, hidden = x.shape
    if hidden % 8:
        raise ValueError(f"the producer kernel reads hidden runs in 16-byte chunks: the hidden size must be a multiple "
                         f"of 8, got {hidden}")
    H, D = plan.num_heads, plan.routing.topology.D
    rows_read, rows_write = row_layout(plan, "read")[1], row_layout(plan, "write")[1]
    dense = {side: dense_levels(plan, side) for side in ("read", "write")}
    device = x.device

    def empty(*shape, dtype=torch.float32):
        return torch.empty(B, H, T, *shape, dtype=dtype, device=device)

    out = {"amp_read": empty(rows_read, dtype=torch.uint8), "amp_write": empty(rows_write, dtype=torch.uint8),
           **{f"dense_{side}": empty(row_layout(plan, side, dense=True)[1], dtype=quant.DENSE_DTYPE)
              if dense[side] else None for side in ("read", "write")},
           "scale_read": empty(D), "scale_write": empty(D), "v": empty(plan.d_v, dtype=torch.float16),
           "v_scale": empty()}
    jobs, shapes = _job_tensor(plan, str(device))
    route_bias = None if weights.route_bias is None else weights.route_bias.float().contiguous()
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).to(torch.bfloat16).contiguous(), weights.route_W, route_bias,
        weights.gain_bias.float().contiguous(), weights.v_W, jobs,
        *(out[name] for name in ("amp_read", "amp_write", "dense_read", "dense_write", "scale_read", "scale_write",
                                 "v", "v_scale")),
        None, None, shapes, D,
        *(sum(1 << level for level in dense[side]) for side in ("read", "write")))
    return {name: None if t is None else t.transpose(1, 2) for name, t in out.items()}


def produce(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """``x [B, T, hidden]`` -> ``(RouteFactors, Values)``, the bundle the ops take: the kernel's on CUDA with no
    gradient, the composition's otherwise."""
    if x.ndim != 3 or x.shape[-1] != plan.hidden_size:
        raise ValueError(f"x must be [B, T, {plan.hidden_size}], got {tuple(x.shape)}")
    operands = (x, weights.route_W, weights.route_bias, weights.gain_bias, weights.v_W)
    if x.is_cuda and x.dtype != torch.float64 and not requires_backward(*(t for t in operands if t is not None)):
        return _kernel_bundle(x, plan, weights)
    return _composition(x, plan, weights)


def _kernel_bundle(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """The kernel's outputs as the bundle: each level's rows and scale a view of the packed planes the kernel
    writes (the ops read them by address), the write gain in write level 0's scale."""
    out = run_kernel(x, plan, weights)
    resolved, D = plan.routing.resolved, plan.routing.topology.D

    def rows(plane: torch.Tensor, side: str) -> tuple[torch.Tensor, ...]:
        dense, dense_at = dense_levels(plan, side), row_layout(plan, side, dense=True)[0]
        spans = zip(row_layout(plan, side)[0], resolved.branches, strict=True)
        return tuple(out[f"dense_{side}"][..., dense_at[level]:dense_at[level] + width]
                     if level in dense else plane[..., offset:offset + width]
                     for level, (offset, width) in enumerate(spans))

    def scales(plane: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(plane[..., level] for level in range(D))

    routes = RouteFactors(topology=plan.routing.topology, amp_read=rows(out["amp_read"], "read"),
                          amp_write=rows(out["amp_write"], "write"), scale_read=scales(out["scale_read"]),
                          scale_write=scales(out["scale_write"]), sparse=default_sparse(plan.routing.topology))
    return routes, Values(v=out["v"], scale=out["v_scale"])


def _composition(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """The producer in torch: the fp64 and host reference, and the path a gradient takes."""
    B, T, _ = x.shape
    H = plan.num_heads
    routing = plan.routing
    read_logits, write_logits, gain_logits = route_logits(x, plan, weights)
    read_bh, write_bh = production_routing_factor_levels(read_logits, write_logits, routing.resolved)

    #: THE LEVELS AS THEY ARE: no level need sum to 1 (docs/CONSTITUTION.md#r23). The producer's write gain is its
    #: own, folded into write level 0, so its scale carries it as the kernel's level-0 job stores it.
    read_levels, write_levels = _public(read_bh, B=B, H=H), _public(write_bh, B=B, H=H)
    g_write = _write_gain(gain_logits, weights.gain_bias, B=B, T=T, H=H, dtype=read_levels[0].dtype,
                          device=x.device)
    write_levels = (write_levels[0] * g_write.unsqueeze(-1), *write_levels[1:])
    routes = RouteFactors.from_levels(routing.topology, read_levels, write_levels)

    #: `v` FLOWS bf16 into the format (docs/api.md section 4.1); the fp64 reference computes it in fp64
    v_dtype = torch.float64 if x.dtype == torch.float64 else torch.bfloat16
    v = F.linear(x.to(v_dtype), weights.v_W.to(v_dtype))
    values = Values.from_projection(rearrange(v, "b l (h d) -> b l h d", d=plan.d_v))
    return routes, values


def route_logits(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights):
    """The projection step alone: ``(read_logits, write_logits, gain_logits)`` as the solve reads them.

    TOKEN-MAJOR ``[B, T, H, W]`` on the bf16 (CUDA) path -- the GEMM's own output order, so the head axis is free
    (docs/internals/entmax/entmax.md#stream-split); the fp64 and host paths read the flat plane. Named so a gate that
    needs the logits the producer computes reads this step rather than reconstructing it.
    """
    dtype = projection_dtype(x)
    return _packed_router_logits(
        x, weights.route_W, plan.routing.resolved, route_bias=weights.route_bias, projection_dtype=dtype,
        token_major=dtype is torch.bfloat16, gain_columns=plan.routing.gain_columns)


def _write_gain(gain_logits, gain_bias, *, B, T, H, dtype, device) -> torch.Tensor:
    """``g_write [B, T, H]``: ``softplus(logit + bias)``, or the constant 1 when the gain column is ablated."""
    if gain_logits is None or gain_logits.shape[-1] == 0:
        return torch.ones(B, T, H, dtype=dtype, device=device)
    biased = gain_logits + gain_bias.to(gain_logits.dtype)[None, None, :, :]
    return softplus_side_gain(biased[..., WRITE_GAIN]).to(dtype)


def _public(levels: tuple[torch.Tensor, ...], *, B: int, H: int) -> tuple[torch.Tensor, ...]:
    """Each level from the solve's ``[BH, T, width]`` to the public ``[B, T, H, width]`` (a permute of it)."""
    return tuple(level.reshape(B, H, -1, level.shape[-1]).permute(0, 2, 1, 3) for level in levels)
