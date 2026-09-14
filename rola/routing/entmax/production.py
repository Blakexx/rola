"""Production routing factors: the exact sorted alpha=1.5/2.0 solve and its analytic VJP.

This module is deliberately separate from ``routing.entmax.reference``. The reference is an fp64
bisection oracle; this module owns the value path every runtime caller takes, and its analytic
fixed-alpha VJP. It offers the same solve through two plan-level surfaces:

* ``production_routing_factor_levels`` -- dense per-level ``[BH, T, width_l]`` factors, the contract
  the consumer and its fp64 oracle both consume. This is what the producer calls.
* ``production_routing_factors`` -- the bf16 tile-major ``RouteDescriptor`` arena ABI.

and through two lowerings of one semantics: hand-CUDA on fp32 CUDA tensors, and the identical
sorted closed form in torch for the domains the kernels do not cover (CPU, fp64). Neither surface
ever calls the reference; it is the test oracle, not a fallback.

THE KERNELS ARE `csrc/rola/src/entmax/` -- `entmax.cu` for the union split, `factor.cu` for the
independent per-side solves and the softmax pair. There is no Triton in this package and no torch
fallback on CUDA: `rola.ops._ext` raises loudly if the wheel is missing
(docs/internals/entmax/factor.md).
"""

from __future__ import annotations

from math import prod

import torch

from rola.routing.topology import MAX_BRANCH_WIDTH
from rola.routing.types import (
    ROUTE_TILE_SIZE as _ROUTE_TILE_SIZE_VALUE,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    RouteDescriptor,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
)

_SPARSE_ACTIVATIONS = {"entmax", "sparsemax"}

#: THE LOGIT PLANE'S ADMISSIBLE DTYPES. `bfloat16` is what the shipped projection emits
#: (bf16 operands on the tensor cores, fp32 accumulate, bf16 out); `float32` is the gate
#: domain that wants the solve's arithmetic without the projection's output quantization.
#: The kernels take either through one `LogitT` template and widen in-register
#: (docs/internals/entmax/factor.md#logit-dtype).
_SOLVE_LOGIT_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_PRODUCTION_ALPHAS = (1.5, 2.0)

#: The support word's token quantum, and the tile-major arena's lane count: one word is
#: one tile, which is what lets a level's support and its values share a tile index.
_WORD_TOKENS = 32


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _production_alpha(alpha: float) -> float:
    """Validate the two exact production entmax specializations."""
    alpha = float(alpha)
    if alpha not in _SUPPORTED_PRODUCTION_ALPHAS:
        raise ValueError(
            "production routing supports exactly alpha=1.5 (entmax) and alpha=2.0 "
            f"(sparsemax), got {alpha!r}")
    return alpha


def _extension():
    # NO FALLBACK, by policy: `rola.ops._ext` raises loudly if the wheel is missing.
    from rola.ops._ext import extension

    return extension()


def _one(width: int) -> list[int]:
    """The single-level table a one-level launch takes: offsets are already in the
    tensors' own `data_ptr`, because every caller here hands a level SLICE."""
    return [width]


def _launch_fwd(z, mask, alpha, *, tokens, n_streams):
    rows, cols = z.shape
    p = torch.empty_like(z)
    support_words = torch.empty(
        (n_streams, _cdiv(tokens, _WORD_TOKENS), cols), dtype=torch.int32, device=z.device)
    if rows == 0:
        return p, support_words
    _extension().entmax_factor_forward(
        z.view(n_streams, tokens, cols), mask.view(n_streams, tokens, cols),
        p.view(n_streams, tokens, cols), None, support_words,
        [0], [0], [0], _one(cols), 1, float(alpha))
    return p, support_words


class _ProductionFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, alpha, tokens, n_streams):
        p, support_words = _launch_fwd(
            z, mask, alpha, tokens=tokens, n_streams=n_streams)
        ctx.save_for_backward(p, support_words, mask)
        ctx.alpha = float(alpha)
        ctx.tokens = int(tokens)
        ctx.n_streams = int(n_streams)
        ctx.mark_non_differentiable(support_words)
        return p, support_words

    @staticmethod
    def backward(ctx, dp, _dsupport_words):
        if dp is None:
            return None, None, None, None, None
        p, support_words, mask = ctx.saved_tensors
        dp = dp.contiguous()
        rows, cols = p.shape
        dz = torch.empty_like(p)
        if rows == 0:
            return dz, None, None, None, None
        shape = (ctx.n_streams, ctx.tokens, cols)
        _extension().entmax_factor_backward(
            p.view(shape), support_words, mask.view(shape), dp.view(shape), None, dz.view(shape),
            [0], [0], [0], _one(cols), False, 1, ctx.alpha)
        return dz, None, None, None, None


def production_factor(z, activation="entmax", alpha=1.5, mask=None):
    """Return ``(values, packed_support_words)`` for an alpha=1.5 or alpha=2 CUDA routing solve.

    Logits and output factors are fp32. ``mask`` is a detached boolean candidate mask and is broadcast
    over leading dimensions. Support is packed over the penultimate (token) axis into signed ``int32``
    words whose bit pattern is uint32. Packed support describes factors that remain nonzero after the
    native bf16 route-storage cast; the returned fp32 factors retain the analytic solve values and are
    reused directly by the VJP.
    """
    if not z.is_cuda:
        raise RuntimeError("production_factor is CUDA-only; use the fp64 naive oracle on CPU")
    if z.dtype != torch.float32:
        raise TypeError(f"production_factor requires fp32 logits, got {z.dtype}")
    if z.dim() < 1 or z.shape[-1] < 1:
        raise ValueError(f"production_factor expects a non-empty branch axis, got {tuple(z.shape)}")
    if z.shape[-1] > MAX_BRANCH_WIDTH:
        raise ValueError(
            f"production_factor branch width must be <= {MAX_BRANCH_WIDTH}; "
            f"got {z.shape[-1]}")
    if activation not in _SPARSE_ACTIVATIONS:
        raise ValueError(f"production_factor requires entmax/sparsemax, got {activation!r}")
    alpha = _production_alpha(alpha)

    cols = z.shape[-1]
    if z.dim() == 1:
        tokens, n_streams, support_shape = 1, 1, (1, cols)
    else:
        tokens = z.shape[-2]
        n_streams = prod(z.shape[:-2])
        support_shape = (*z.shape[:-2], _cdiv(tokens, _WORD_TOKENS), cols)
    z_rows = z.reshape(n_streams * tokens, cols).contiguous()
    if mask is None:
        mask_rows = torch.ones_like(z_rows, dtype=torch.bool)
    else:
        mask_rows = mask.to(device=z.device, dtype=torch.bool).expand_as(z).reshape(-1, cols).contiguous()
    p, support_words = _ProductionFactorFn.apply(
        z_rows, mask_rows, float(alpha), tokens, n_streams)
    return p.reshape_as(z), support_words.reshape(support_shape)


def _arena_shape(logits: torch.Tensor) -> tuple[int, int]:
    """``(streams, tokens)`` for either logit layout.

    Rank 3 is the flat ``[BH, T, W]`` plane; rank 4 is the TOKEN-MAJOR ``[B, T, H, W]``
    the packed router GEMM writes directly, whose stream ``b*H + h`` is two axes. Which
    one arrived is `_heads`' question, and the solve carries the answer as a stride pair
    rather than as a copy.
    """
    if logits.ndim == 4:
        return logits.shape[0] * logits.shape[2], logits.shape[1]
    if logits.ndim != 3:
        raise ValueError(
            f"production routing logits must be [BH, T, W] or [B, T, H, W], got "
            f"{tuple(logits.shape)}")
    return logits.shape[0], logits.shape[1]


def _heads(logits: torch.Tensor) -> int:
    """The STREAM SPLIT the launch runs under: ``H`` for token-major logits, 1 for flat.

    NEVER 0, and that is the kernel's constraint rather than a convention: the split is
    unconditional there because guarding it cost the family its register budget, so
    ``heads = 1`` -- one head, ``sb = stream``, ``sh = 0`` -- is how a flat plane says it
    has no head axis (docs/internals/entmax/entmax.md#stream-split).
    """
    return logits.shape[2] if logits.ndim == 4 else 1


# ---------------------------------------------------------------------------
# THE LAUNCH PLAN -- one launch per (operator, alpha, width class, operand set),
# which is what makes `producer.py`'s "one batched solve" true rather than aspirational.
# ---------------------------------------------------------------------------

#: The kernel's own bound on one launch's level table (`entmax.cuh::MAX_BATCHED_LEVELS`).
#: A topology deeper than this issues a second launch of the same arm, not a wider table.
_MAX_BATCHED_LEVELS = 16


def _width_class(width: int) -> int:
    """The PADDED width, which is what selects the template arm.

    Two levels may share a launch only if they share this. `LW`/`IPT` set how the sort
    permutes and how the prefix scan associates, so running a narrow level in a wider
    arm would round `tau` differently and could move a support boundary -- the batch has
    to be the loop, not an approximation of it.
    """
    return max(1 << (int(width) - 1).bit_length(), 2)


class _Launch:
    """The levels of ONE kernel launch: one operator, one alpha, one width class, one
    set of base tensors. Offsets are COLUMN indices into those bases, which is the unit
    the kernel's address expression adds them in."""

    __slots__ = ("kind", "alpha", "operands", "heads",
                 "logit", "value", "midpoint", "support", "widths")

    def __init__(self, kind, alpha, operands, heads):
        self.kind, self.alpha, self.operands, self.heads = kind, alpha, operands, heads
        self.logit, self.value, self.midpoint, self.support, self.widths = [], [], [], [], []

    def full(self) -> bool:
        return len(self.widths) == _MAX_BATCHED_LEVELS

    def add(self, width, logit_off, value_off, midpoint_off, support_off) -> None:
        self.widths.append(int(width))
        self.logit.append(int(logit_off))
        self.value.append(int(value_off))
        self.midpoint.append(int(midpoint_off))
        self.support.append(int(support_off))

    def issue(self) -> None:
        extension = _extension()
        if self.kind == "union":
            read_logits, write_logits, midpoint, support, read_values, write_values = self.operands
            extension.entmax_union_forward(
                read_logits, write_logits, midpoint, support, read_values, write_values,
                self.logit, self.value, self.midpoint, self.support, self.widths, self.heads,
                self.alpha)
        elif self.kind == "entmax":
            logits, values, values_second, support = self.operands
            extension.entmax_factor_forward(
                logits, None, values, values_second, support,
                self.logit, self.value, self.support, self.widths, self.heads, self.alpha)
        else:
            logits, values, values_second = self.operands
            extension.routing_softmax_forward(
                logits, values, values_second, self.logit, self.value, self.widths, self.heads)


class _Plan:
    """The forward walk's launches, grouped by arm and issued in FIRST-APPEARANCE order.

    That order is what makes the grouping safe rather than merely faster. In the forward
    every level owns disjoint output columns, so there is no write hazard to preserve at
    all; the order is kept because a reader should be able to read the launch sequence
    off the plan walk.
    """

    def __init__(self):
        self._open: dict = {}
        self._issued: list[_Launch] = []

    def add(self, kind, alpha, operands, width, *, logit_off,
            value_off=0, midpoint_off=0, support_off=0, heads=1):
        key = (kind, alpha, _width_class(width), heads, *(id(t) for t in operands))
        launch = self._open.get(key)
        if launch is None or launch.full():
            launch = _Launch(kind, alpha, operands, heads)
            self._open[key] = launch
            self._issued.append(launch)
        launch.add(width, logit_off, value_off, midpoint_off, support_off)

    def issue(self) -> int:
        for launch in self._issued:
            launch.issue()
        return len(self._issued)


# ---------------------------------------------------------------------------
# THE BACKWARD stays one launch per level-side, and that is a MEASURED shape, not an
# oversight: autograd delivers D separate cotangent tensors with no common base, so
# batching them would mean copying D of them into one plane to save D-1 launches.
# ---------------------------------------------------------------------------


def _launch_factor_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    support_words: torch.Tensor,
    d_logits: torch.Tensor,
    alpha: float,
    *,
    accumulate: bool,
) -> None:
    _extension().entmax_factor_backward(
        output, support_words, None, d_output, d_output_second, d_logits,
        [0], [0], [0], _one(d_logits.shape[-1]), bool(accumulate), _heads(d_logits),
        float(alpha))


def _launch_union_bwd(
    read_logits_level: torch.Tensor,
    write_logits_level: torch.Tensor,
    midpoint: torch.Tensor,
    support_words: torch.Tensor,
    read_output: torch.Tensor,
    write_output: torch.Tensor,
    d_read_output: torch.Tensor,
    d_write_output: torch.Tensor,
    d_read_level: torch.Tensor,
    d_write_level: torch.Tensor,
    alpha: float,
) -> None:
    width = _one(d_read_level.shape[-1])
    _extension().entmax_union_backward(
        read_logits_level, write_logits_level, midpoint, support_words,
        read_output, write_output, d_read_output, d_write_output,
        d_read_level, d_write_level, [0], [0], [0], [0], width, _heads(d_read_level),
        float(alpha))


def _launch_softmax_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    d_logits: torch.Tensor,
    *,
    accumulate: bool,
) -> None:
    _extension().routing_softmax_backward(
        output, d_output, d_output_second, d_logits,
        [0], [0], _one(d_logits.shape[-1]), bool(accumulate), _heads(d_logits))


class _ProductionRoutingArenaFn(torch.autograd.Function):
    """Own final bf16 arenas and retain their output references for routing VJPs.

    Union alone retains compact fp32 midpoint values and its original logits.
    """

    @staticmethod
    def forward(ctx, read_logits, write_logits, routing):
        n_streams, tokens = _arena_shape(read_logits)
        if read_logits.shape != write_logits.shape:
            raise ValueError("read/write routing logits must have identical shapes")
        if read_logits.dtype not in _SOLVE_LOGIT_DTYPES or write_logits.dtype != read_logits.dtype:
            raise TypeError(
                f"production routing arenas require {_SOLVE_LOGIT_DTYPES} logits of one dtype")
        if not read_logits.is_cuda or not write_logits.is_cuda:
            raise RuntimeError("production routing arenas are CUDA-only")
        if read_logits.device != write_logits.device:
            raise ValueError("read/write routing logits must share one CUDA device")
        if read_logits.ndim != 3 or read_logits.shape[-1] != routing.packed_logit_width:
            raise ValueError("routing logits do not match the resolved routing layout")

        fully_shared = all(isinstance(level, TiedRouting) for level in routing.levels)
        arena_shape = (
            n_streams,
            _cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
            routing.level_offsets[-1],
            _ROUTE_TILE_SIZE_VALUE,
        )
        read_arena = torch.zeros(arena_shape, dtype=torch.bfloat16, device=read_logits.device)
        write_arena = read_arena if fully_shared else torch.zeros_like(read_arena)
        support_words = torch.empty(
            (n_streams, _cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
             sum(width for width, offset in zip(routing.branches, routing.support_offsets) if offset >= 0)),
            dtype=torch.int32,
            device=read_logits.device,
        )
        empty_launch = n_streams == 0 or tokens == 0
        heads = _heads(read_logits)
        #: ONE midpoint plane, packed over the UNION levels alone -- the only levels that
        #: have one. It is saved whole and sliced in the backward.
        union_widths = [routing.branches[i] for i, level in enumerate(routing.levels)
                        if isinstance(level, UnionRouting)]
        midpoints = torch.empty(
            (n_streams, tokens, sum(union_widths)), dtype=torch.float32,
            device=read_logits.device)
        plan = _Plan()
        midpoint_offset = 0
        midpoint_offsets: list[int] = []

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            logit_offset = routing.level_logit_slice(index).start
            offset = routing.level_offsets[index]
            support_offset = routing.support_offsets[index]

            if isinstance(level, UnionRouting):
                midpoint_offsets.append(midpoint_offset)
                if not empty_launch:
                    plan.add("union", level.alpha,
                             (read_logits, write_logits, midpoints, support_words,
                              read_arena, write_arena),
                             width, logit_off=logit_offset, value_off=offset,
                             midpoint_off=midpoint_offset, support_off=support_offset, heads=heads)
                midpoint_offset += width
                continue

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if empty_launch:
                continue
            if isinstance(level, TiedRouting):
                second = None if fully_shared else write_arena
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    plan.add("entmax", shared_activation.alpha,
                             (write_logits, read_arena, second, support_words), width,
                             logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (write_logits, read_arena, second), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)
                continue

            for activation, source, output in (
                (level.read, read_logits, read_arena),
                (level.write, write_logits, write_arena),
            ):
                if isinstance(activation, EntmaxActivation):
                    plan.add("entmax", activation.alpha, (source, output, None, support_words),
                             width, logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (source, output, None), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)

        plan.issue()

        union_state = (read_logits, write_logits, midpoints) if midpoint_offsets else ()
        ctx.save_for_backward(read_arena, write_arena, support_words, *union_state)
        ctx.routing = routing
        ctx.logits_shape = read_logits.shape
        ctx.logits_dtype = read_logits.dtype
        ctx.midpoint_offsets = midpoint_offsets
        ctx.mark_non_differentiable(support_words)
        return read_arena, write_arena, support_words

    @staticmethod
    def backward(ctx, d_read, d_write, _d_support_words):
        saved = ctx.saved_tensors
        read_arena, write_arena, support_words = saved[:3]
        if ctx.midpoint_offsets:
            read_logits, write_logits, midpoints = saved[3:6]
        else:
            read_logits = write_logits = midpoints = None
        routing = ctx.routing
        if d_read is None:
            d_read = torch.zeros_like(read_arena)
        if d_write is None:
            d_write = torch.zeros_like(write_arena)
        d_read_logits = torch.zeros(
            ctx.logits_shape, dtype=ctx.logits_dtype, device=read_arena.device)
        d_write_logits = torch.zeros_like(d_read_logits)
        union_index = 0
        n_streams, tokens = ctx.logits_shape[:2]
        if n_streams == 0 or tokens == 0:
            return d_read_logits, d_write_logits, None

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            offset = routing.level_offsets[index]
            read_output = read_arena[:, :, offset:offset + width, :]
            write_output = write_arena[:, :, offset:offset + width, :]
            d_read_output = d_read[:, :, offset:offset + width, :]
            d_write_output = d_write[:, :, offset:offset + width, :]
            span = routing.level_logit_slice(index)
            d_read_level = d_read_logits[..., span]
            d_write_level = d_write_logits[..., span]
            support_offset = routing.support_offsets[index]

            if isinstance(level, UnionRouting):
                assert read_logits is not None and write_logits is not None
                midpoint_offset = ctx.midpoint_offsets[union_index]
                midpoint = midpoints[:, :, midpoint_offset:midpoint_offset + width]
                union_index += 1
                words = support_words[:, :, support_offset:support_offset + width]
                _launch_union_bwd(
                    read_logits[..., span], write_logits[..., span],
                    midpoint, words, read_output, write_output,
                    d_read_output, d_write_output, d_read_level, d_write_level, level.alpha)
                continue

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if isinstance(level, TiedRouting):
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        read_output, d_read_output, d_write_output,
                        support_words[:, :, support_offset:support_offset + width], d_write_level,
                        shared_activation.alpha, accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        read_output, d_read_output, d_write_output, d_write_level, accumulate=False)
                continue

            for activation, output, d_output, target in (
                (level.read, read_output, d_read_output, d_read_level),
                (level.write, write_output, d_write_output, d_write_level),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        output, d_output, None,
                        support_words[:, :, support_offset:support_offset + width], target,
                        activation.alpha, accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        output, d_output, None, target, accumulate=False)
        return d_read_logits, d_write_logits, None


class _ProductionLevelsFn(torch.autograd.Function):
    """Solve a routing plan into per-level dense ``[BH, T, width_l]`` factors.

    Structurally the same plan walk as ``_ProductionRoutingArenaFn`` -- the same
    kernels, the same tied/union dispatch, the same packed support -- against one
    dense plane per side rather than the tile-major arena the earlier chunk ABI took.

    ``stored_dtype`` is the STORED form (docs/internals/entmax/factor.md#stored-form):
    ``bfloat16`` is what the consumer reads and therefore what the producer emits, and
    the solve writes it directly rather than through a narrowing pass. ``float32``
    exists for a gate that wants the solve's own arithmetic without the storage
    quantization on top of it.

    Read and write are always distinct allocations, even where one solve serves both
    (a ``TiedRouting`` level): the kernels already write a second output and accumulate
    a second incoming gradient, and returning one tensor twice from a single
    ``autograd.Function`` would alias two graph outputs.
    """

    @staticmethod
    def forward(ctx, read_logits, write_logits, routing, stored_dtype):
        n_streams, tokens = _arena_shape(read_logits)
        device = read_logits.device
        support_words = torch.empty(
            (n_streams, _cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
             sum(width for width, offset in zip(routing.branches, routing.support_offsets) if offset >= 0)),
            dtype=torch.int32,
            device=device,
        )
        empty_launch = n_streams == 0 or tokens == 0
        heads = _heads(read_logits)
        #: ONE PLANE PER SIDE, sliced into levels. Each level is still a distinct graph
        #: output, at its own columns of one allocation -- which is what lets the levels
        #: of a width class share a launch
        #: (docs/internals/entmax/factor.md#batched-levels), and what makes the whole
        #: side ONE contiguous tensor in the layout the consumer reads.
        #:
        #: UNINITIALIZED: a dense level is exactly the tokens, and every solve writes
        #: every element of one (unlike the tile-major arena, whose tail lanes past
        #: `tokens` no kernel touches).
        packed_width = routing.level_offsets[-1]
        read_plane = torch.empty(
            (n_streams, tokens, packed_width), dtype=stored_dtype, device=device)
        write_plane = torch.empty_like(read_plane)
        union_widths = [routing.branches[i] for i, level in enumerate(routing.levels)
                        if isinstance(level, UnionRouting)]
        midpoints = torch.empty(
            (n_streams, tokens, sum(union_widths)), dtype=torch.float32, device=device)

        read_levels: list[torch.Tensor] = []
        write_levels: list[torch.Tensor] = []
        midpoint_offsets: list[int] = []
        midpoint_offset = 0
        plan = _Plan()

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            logit_offset = routing.level_logit_slice(index).start
            offset = routing.level_offsets[index]
            read_levels.append(read_plane[:, :, offset:offset + width])
            write_levels.append(write_plane[:, :, offset:offset + width])
            support_offset = routing.support_offsets[index]

            if isinstance(level, UnionRouting):
                midpoint_offsets.append(midpoint_offset)
                if not empty_launch:
                    plan.add("union", level.alpha,
                             (read_logits, write_logits, midpoints, support_words,
                              read_plane, write_plane),
                             width, logit_off=logit_offset, value_off=offset,
                             midpoint_off=midpoint_offset, support_off=support_offset, heads=heads)
                midpoint_offset += width
                continue

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if empty_launch:
                continue
            if isinstance(level, TiedRouting):
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    plan.add("entmax", shared_activation.alpha,
                             (write_logits, read_plane, write_plane, support_words), width,
                             logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (write_logits, read_plane, write_plane), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)
                continue

            for activation, source, output in (
                (level.read, read_logits, read_plane),
                (level.write, write_logits, write_plane),
            ):
                if isinstance(activation, EntmaxActivation):
                    plan.add("entmax", activation.alpha, (source, output, None, support_words),
                             width, logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (source, output, None), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)

        plan.issue()

        union_state = (read_logits, write_logits, midpoints) if midpoint_offsets else ()
        ctx.save_for_backward(*read_levels, *write_levels, support_words, *union_state)
        ctx.routing = routing
        ctx.logits_shape = read_logits.shape
        ctx.logits_dtype = read_logits.dtype
        ctx.midpoint_offsets = midpoint_offsets
        return (*read_levels, *write_levels)

    @staticmethod
    def backward(ctx, *grads):
        routing = ctx.routing
        D = routing.D
        saved = ctx.saved_tensors
        read_levels = saved[:D]
        write_levels = saved[D:2 * D]
        support_words = saved[2 * D]
        if ctx.midpoint_offsets:
            read_logits, write_logits, midpoints = saved[2 * D + 1:2 * D + 4]
        else:
            read_logits = write_logits = midpoints = None

        d_read_logits = torch.zeros(
            ctx.logits_shape, dtype=ctx.logits_dtype, device=support_words.device)
        d_write_logits = torch.zeros_like(d_read_logits)
        n_streams, tokens = ctx.logits_shape[:2]
        if n_streams == 0 or tokens == 0:
            return d_read_logits, d_write_logits, None, None

        union_index = 0
        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            read_output = read_levels[index]
            write_output = write_levels[index]
            d_read_output = grads[index]
            d_write_output = grads[D + index]
            if d_read_output is None:
                d_read_output = torch.zeros_like(read_output)
            if d_write_output is None:
                d_write_output = torch.zeros_like(write_output)
            d_read_output = d_read_output.contiguous()
            d_write_output = d_write_output.contiguous()
            span = routing.level_logit_slice(index)
            d_read_level = d_read_logits[..., span]
            d_write_level = d_write_logits[..., span]
            support_offset = routing.support_offsets[index]
            words = None if support_offset < 0 else support_words[
                :, :, support_offset:support_offset + width]

            if isinstance(level, UnionRouting):
                assert read_logits is not None and write_logits is not None
                midpoint_offset = ctx.midpoint_offsets[union_index]
                midpoint = midpoints[:, :, midpoint_offset:midpoint_offset + width]
                union_index += 1
                _launch_union_bwd(
                    read_logits[..., span], write_logits[..., span],
                    midpoint, words, read_output, write_output,
                    d_read_output, d_write_output, d_read_level, d_write_level, level.alpha)
                continue

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if isinstance(level, TiedRouting):
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        read_output, d_read_output, d_write_output, words, d_write_level,
                        shared_activation.alpha, accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        read_output, d_read_output, d_write_output, d_write_level, accumulate=False)
                continue

            for activation, output, d_output, target in (
                (level.read, read_output, d_read_output, d_read_level),
                (level.write, write_output, d_write_output, d_write_level),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        output, d_output, None, words, target, activation.alpha,
                        accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        output, d_output, None, target, accumulate=False)
        return d_read_logits, d_write_logits, None, None


# ---------------------------------------------------------------------------
# Torch lowering of the SAME exact solve, for the domains the kernels do not cover
# (CPU, and fp64 -- the oracle-equivalence gate projects fp64 logits end to end).
#
# This is a second *lowering* of the production semantics, not a fallback to the
# reference: it computes the identical sorted closed form the CUDA kernels do
# (row-max shift, descending sort, prefix/prefix-square scans, greatest valid k)
# with the identical analytic fixed-alpha VJP. The fp64 bisection in
# ``entmax/reference.py`` is reachable from no runtime path at all -- it is the
# test oracle, and the point of an oracle is that nothing in production shares
# its code.
# ---------------------------------------------------------------------------


def _sorted_solve(z: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``(values, support)`` for alpha=1.5/2.0 by the sorted closed form."""
    width = z.shape[-1]
    z = z - z.amax(dim=-1, keepdim=True)
    sorted_z, _ = torch.sort(z, dim=-1, descending=True)
    prefix = sorted_z.cumsum(dim=-1)
    k = torch.arange(1, width + 1, device=z.device, dtype=z.dtype)
    if alpha == 2.0:
        tau_candidates = (prefix - 1.0) / k
        candidate = sorted_z > tau_candidates
    else:
        prefix_sq = (sorted_z * sorted_z).cumsum(dim=-1)
        discriminant = prefix * prefix - k * (prefix_sq - 4.0)
        tau_candidates = (prefix - discriminant.clamp(min=0.0).sqrt()) / k
        candidate = (discriminant >= 0.0) & (sorted_z > tau_candidates)
    columns = torch.arange(width, device=z.device)
    chosen = torch.where(candidate, columns, torch.full_like(columns, -1)).amax(dim=-1, keepdim=True)
    tau = torch.gather(tau_candidates, -1, chosen.clamp(min=0))
    tau = torch.where(chosen >= 0, tau, torch.zeros_like(tau))
    support = z > tau
    base = ((alpha - 1.0) * (z - tau)).clamp(min=0.0)
    values = torch.where(support, base if alpha == 2.0 else base * base, torch.zeros_like(base))
    return values, support


class _SortedEntmaxFn(torch.autograd.Function):
    """Sorted exact solve with the analytic sparse fixed-alpha Jacobian-vector product."""

    @staticmethod
    def forward(ctx, z, alpha):
        values, support = _sorted_solve(z, float(alpha))
        s = torch.where(
            support, values.clamp(min=0.0).pow(2.0 - float(alpha)), torch.zeros_like(values))
        ctx.save_for_backward(s)
        return values

    @staticmethod
    def backward(ctx, dp):
        (s,) = ctx.saved_tensors
        num = (s * dp).sum(dim=-1, keepdim=True)
        den = s.sum(dim=-1, keepdim=True)
        den = torch.where(den > 0, den, torch.ones_like(den))
        return s * (dp - num / den), None


def _overflow_free_delta(read_logits: torch.Tensor, write_logits: torch.Tensor) -> torch.Tensor:
    """``read - write`` with the union kernel's saturating magnitude arithmetic.

    Opposite-sign finite extrema overflow a raw subtraction while their midpoint stays
    representable, so the kernel clamps the magnitude instead. Mirrored here so the two
    lowerings agree on the same near-max fixtures the CUDA gate already covers.
    """
    finfo = torch.finfo(read_logits.dtype)
    read_negative = read_logits < 0.0
    opposite_sign = read_negative != (write_logits < 0.0)
    read_magnitude, write_magnitude = read_logits.abs(), write_logits.abs()
    hi = torch.maximum(read_magnitude, write_magnitude)
    lo = torch.minimum(read_magnitude, write_magnitude)
    opposite_magnitude = hi + torch.minimum(lo, finfo.max - hi)
    opposite_delta = torch.where(read_negative, -opposite_magnitude, opposite_magnitude)
    zero = torch.zeros_like(read_logits)
    same_delta = torch.where(opposite_sign, zero, read_logits) - torch.where(
        opposite_sign, zero, write_logits)
    return torch.where(opposite_sign, opposite_delta, same_delta)


def _torch_union_split(
    read_logits: torch.Tensor, write_logits: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = 0.5 * read_logits + 0.5 * write_logits
    values = _SortedEntmaxFn.apply(z, alpha)
    # Support comes from the same `z > tau` bits `_sorted_solve` and the CUDA kernels'
    # `semantic_support = valid & (z > tau)` compute, not from `values > 0`: the
    # two conventions agree everywhere except a genuinely-supported member whose amplitude
    # underflows fp32 to exactly 0 (alpha=1.5 squares `(z-tau)/2`, so `z-tau ~ 1e-25`
    # suffices) -- adversarial review Claim 2. Recomputed directly (no grad needed; support
    # is a discrete, non-differentiable mask) rather than threaded out of the autograd
    # Function, so the two lowerings share one support definition instead of two that
    # happen to usually agree.
    with torch.no_grad():
        _, support = _sorted_solve(z.detach(), alpha)
    delta = _overflow_free_delta(read_logits, write_logits)
    read = values * torch.sigmoid(delta)
    safe = torch.where(support, values, torch.ones_like(values))
    log_write = (safe.log() + torch.nn.functional.logsigmoid(-delta)).masked_fill(
        ~support, float("-inf"))
    has_support = support.any(dim=-1, keepdim=True)
    write = torch.softmax(
        torch.where(has_support, log_write, torch.zeros_like(log_write)), dim=-1)
    return read, torch.where(has_support & support, write, torch.zeros_like(write))


def _torch_activation(logits: torch.Tensor, activation) -> torch.Tensor:
    if isinstance(activation, SoftmaxActivation):
        return torch.softmax(logits, dim=-1)
    return _SortedEntmaxFn.apply(logits, activation.alpha)


def _torch_routing_factor_levels(
    read_logits: torch.Tensor, write_logits: torch.Tensor, routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    reads, writes = [], []
    for index, level in enumerate(routing.levels):
        span = routing.level_logit_slice(index)
        read_level = read_logits[..., span]
        write_level = write_logits[..., span]
        if isinstance(level, UnionRouting):
            read_factor, write_factor = _torch_union_split(read_level, write_level, level.alpha)
        elif isinstance(level, TiedRouting):
            # TiedRouting names ONE activation for its one solve through the same
            # `.activation('write')` accessor it answers `'read'` with; it carries no
            # separate `.write` attribute.
            write_factor = _torch_activation(write_level, level.activation("write"))
            read_factor = write_factor
        else:
            read_factor = _torch_activation(read_level, level.read)
            write_factor = _torch_activation(write_level, level.write)
        reads.append(read_factor)
        writes.append(write_factor)
    return tuple(reads), tuple(writes)


def production_routing_factor_levels(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
    *,
    stored_dtype: torch.dtype = torch.bfloat16,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Exact per-level ``(read, write)`` routing factors on the production solve.

    Returns one ``[BH, T, width_l]`` tensor per level-side -- the shape the producer
    hands ``naive_rola`` and the consumer. Lowered to the hand-CUDA solve
    on CUDA fp32 and to the identical sorted closed form in torch elsewhere;
    the two agree by construction rather than by tolerance, since the threshold is a
    closed form and not an iterate.

    LOGITS ARRIVE IN EITHER DTYPE AND EITHER LAYOUT. ``bfloat16`` is what the shipped
    projection emits -- bf16 operands on the tensor cores, fp32 accumulate, bf16 out,
    which is the only shape cuBLAS offers and 2.76x the fp32 SIMT GEMM at the pinned cell
    -- and ``float32`` is the gate domain that wants the solve's arithmetic without the
    projection's output quantization. The kernels take either through one ``LogitT``
    template and widen each logit IN REGISTER, so no pass over the plane exists in either
    direction (docs/internals/entmax/factor.md#logit-dtype).

    LAYOUT: flat ``[BH, T, packed_logit_width]``, or the
    TOKEN-MAJOR ``[B, T, H, packed_logit_width]`` the packed router GEMM writes without
    a transpose. The OUTPUT layout is the same either way. Only the CUDA solve reads the
    second one as it lies (it addresses ``b`` and ``h`` as two strides); the torch
    lowering flattens it first, which is the transpose this seam exists to avoid paying
    on the shipped path and does not mind paying on the oracle's.

    ``stored_dtype`` selects the CUDA solve's output format, and its default is the
    STORED form: the consumer reads bf16 and nothing else reads these at all, so the
    kernel writes bf16 and no narrowing pass exists. ``float32`` is for a gate that
    wants the solve's arithmetic without the storage quantization on it. It does not
    reach the torch lowering, whose domains (CPU, fp64) are the ones where the stored
    form is not the question.
    """
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if read_logits.ndim not in (3, 4) or read_logits.shape[-1] != routing.packed_logit_width:
        raise ValueError(
            "routing logits must be [BH, T, packed_logit_width] or "
            f"[B, T, H, packed_logit_width] with packed_logit_width="
            f"{routing.packed_logit_width}, got {tuple(read_logits.shape)}")
    if read_logits.device != write_logits.device or read_logits.dtype != write_logits.dtype:
        raise ValueError("read/write logits must share one device and dtype")

    if read_logits.is_cuda and read_logits.dtype in _SOLVE_LOGIT_DTYPES:
        #: NOT `.contiguous()`: every solve entry below reads the logits through an
        #: explicit stride tuple, so a packed projection's half-slice -- and its head
        #: axis -- are addressable as they stand, and materializing either is a whole
        #: extra pass over the side.
        outputs = _ProductionLevelsFn.apply(read_logits, write_logits, routing, stored_dtype)
        return tuple(outputs[:routing.D]), tuple(outputs[routing.D:])
    return _torch_routing_factor_levels(_flat(read_logits), _flat(write_logits), routing)


def _flat(logits: torch.Tensor) -> torch.Tensor:
    """Token-major ``[B, T, H, W]`` as the flat ``[BH, T, W]`` plane, or itself.

    This IS the transpose the token-major layout exists to avoid; it is here because the
    torch lowering has no stride tuple to read the head axis through, and it runs on the
    fp64 oracle path where a pass over the plane is not the cost that matters.
    """
    if logits.ndim != 4:
        return logits
    return logits.permute(0, 2, 1, 3).reshape(
        logits.shape[0] * logits.shape[2], logits.shape[1], logits.shape[3])


def production_routing_factors(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
    *,
    token_origin: int = 0,
) -> RouteDescriptor:
    """Solve a canonical routing plan directly into the RouteDescriptor ABI."""
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if type(token_origin) is not int or token_origin < 0:
        raise ValueError("token_origin must be a nonnegative exact int")
    if read_logits.ndim != 3 or read_logits.shape[-1] != routing.packed_logit_width:
        raise ValueError(
            "routing logits must be [BH, T, packed_logit_width] with packed_logit_width="
            f"{routing.packed_logit_width}, got {tuple(read_logits.shape)}")

    outputs = _ProductionRoutingArenaFn.apply(read_logits, write_logits, routing)
    read_values, write_values, support_words = outputs
    return RouteDescriptor(
        read_values, write_values, support_words,
        routing.level_offsets, routing.support_offsets, routing.branches,
        routing.radix_strides, routing.support_side_flags,
        token_origin, read_logits.shape[1], routing.topology_stamp,
    )


__all__ = [
    "production_factor",
    "production_routing_factor_levels",
    "production_routing_factors",
]
