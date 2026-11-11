# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The routing bundle: the op's first operand, and the package's ONE duck boundary.

Linear attention's op takes ``(q, k, v)``. RoLA's takes ``(routes, v)``, where ``routes``
is the ``(q, k)`` pair AFTER the feature map -- the read-side and write-side simplices
and their side gains. That bundle is :class:`RouteFactors`, and it is the whole producer
contract: **any callable that returns a well-formed one is a producer.** There is no
producer base class to subclass, no per-level tier and no assembly gateway, because none
of them was a boundary anything checked -- the bundle's own invariants are, and they are
structural rather than provenance-based.

The bundle carries tensors and nothing else, which is why it is constructible rather
than sealed. What its invariants are, and why unit-sum-ness is not among them:
``docs/api.md`` section 2.1. The layer's producer (:func:`rola.ops.produce.produce`) builds it from a declared
:class:`~rola.routing.producer.Routing`; :meth:`RouteFactors.from_levels` builds it from any levels, equally checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant
from rola.routing.types import Topology

__all__ = ["RouteFactors", "carry_level_s"]

SIDES = ("read", "write")


def carry_level_s(read_sparse) -> int:
    """THE CARRY'S LEVEL S, from the read side's sparse flags: at depth two the read-sparse level (level 1 when it is
    read-sparse or level 0 is not, else level 0) -- the readout's digits, its read rows the coefficients; ``-1`` at
    depth one (the list level alone) and deeper (the carry is not built)."""
    if len(read_sparse) != 2:
        return -1
    return 1 if read_sparse[1] or not read_sparse[0] else 0


def _aligned(rows: torch.Tensor) -> bool:
    """THE ROW ALIGNMENT LAW on one level's ``[B, T, H, width]`` rows: the first row on the grain, and every stride of a
    dimension longer than one a multiple of it, in bytes."""
    grain = quant.ROW_ALIGN
    size = rows.element_size()
    return rows.data_ptr() % grain == 0 and all(
        rows.stride(d) * size % grain == 0 for d in range(3) if rows.shape[d] > 1)


def _requires_grad(levels) -> bool:
    return torch.is_grad_enabled() and any(level.requires_grad for level in levels)


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides in THE INTERNAL FORMAT: q and k as scaled blocks. Tensors, and nothing else.

    ``amp_read``/``amp_write``: one row set per level, ``[B, T, H, width_l]`` read and ``[B, T, H_w, width_l]``
    write (``H_w`` write heads, each read by ``H / H_w`` consecutive query heads), each row UNIT-MAX -- u8 with its
    largest digit at 255, at every level, a zero a zero. An all-zero row is a dead token at that level. ``sparse``: ``(read, write)``, a bool a level -- the
    levels whose support the facts pass words (they may hold exact zeros worth skipping); set at production, read
    by every consumer, never derived from the activations. All dense by default. ``scale_read``/``scale_write``: the rows' maxima,
    fp32 per level at its side's prefix -- whatever magnitude a level carries, a producer's gain included, is in its
    scale; no level need sum to 1 (docs/CONSTITUTION.md#r23). A side's shared scale is the product of its levels'.
    ``source``: the producer's floating ``(read, write)`` levels, held ONLY when they require a
    gradient -- the straight-through gradient's anchor for :meth:`dequantized`; ``None`` otherwise, so an
    inference bundle holds the rows and nothing else.

    The format is written by the framework's producers and by nothing else (ruled 2026-09-26): a new
    production is a producer class here. Direct construction is legal for a fixture; ``__post_init__``
    holds it to the format's invariants (``docs/api.md`` section 2.1).
    """

    topology: Topology
    amp_read: tuple[torch.Tensor, ...]
    amp_write: tuple[torch.Tensor, ...]
    scale_read: tuple[torch.Tensor, ...]
    scale_write: tuple[torch.Tensor, ...]
    source: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None = None
    sparse: tuple[tuple[bool, ...], tuple[bool, ...]] | None = None

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        for name in ("amp_read", "amp_write", "scale_read", "scale_write"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D = self.topology.D
        flags = ((False,) * D, (False,) * D) if self.sparse is None else tuple(tuple(map(bool, f)) for f in self.sparse)
        if len(flags) != 2 or any(len(f) != D for f in flags):
            raise ValueError(f"RouteFactors.sparse is (read, write), a bool a level (D={D}); got {self.sparse!r}")
        for side, side_flags in zip(SIDES, flags):
            if all(side_flags):
                raise ValueError(f"RouteFactors.sparse declares every {side} level sparse ({side_flags}); every side "
                                 "has a dense level (docs/api.md section 2.1)")
        object.__setattr__(self, "sparse", flags)
        prefixes = self._check_rows()
        labels = {"read": "[B, T, H]", "write": "[B, T, H_w]"}
        for name in ("scale_read", "scale_write"):
            scales, prefix, label = getattr(self, name), prefixes[name[6:]], labels[name[6:]]
            if len(scales) != D:
                raise ValueError(f"RouteFactors.{name} holds one tensor per level (D={D}), got {len(scales)}")
            for level, scale in enumerate(scales):
                if (not isinstance(scale, torch.Tensor) or scale.dtype is not torch.float32
                        or tuple(scale.shape) != prefix):
                    raise ValueError(f"RouteFactors.{name}[{level}] is the rows' maxima, fp32 {label}={prefix}; "
                                     f"got {getattr(scale, 'dtype', type(scale).__name__)} "
                                     f"{tuple(getattr(scale, 'shape', ()))}")
        if self.source is not None:
            for side, levels, rows in (("read", self.source[0], self.amp_read),
                                       ("write", self.source[1], self.amp_write)):
                if len(levels) != D or any(tuple(a.shape) != tuple(r.shape) for a, r in zip(levels, rows)):
                    raise ValueError(f"RouteFactors.source's {side} levels are the rows' floating originals")

    def _check_rows(self) -> dict[str, tuple[int, ...]]:
        """Every row set: one tensor a level, ``[B, T, H, width_l]`` read and ``[B, T, H_w, width_l]`` write, a
        prefix a side (one B and T, ``H_w`` dividing ``H``), one device and dtype. Returns each side's prefix."""
        D, widths = self.topology.D, self.topology.widths
        prefixes, first = {}, None
        for name in ("amp_read", "amp_write"):
            rows, prefix = getattr(self, name), None
            if len(rows) != D:
                raise ValueError(f"RouteFactors.{name} must hold one tensor per routing level (D={D}), got {len(rows)}")
            for level, tensor in enumerate(rows):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    shape = getattr(tensor, "shape", type(tensor).__name__)
                    raise TypeError(f"RouteFactors.{name}[{level}] must be a [B, T, H, width_l] tensor, got {shape}")
                if tensor.shape[-1] != widths[level]:
                    raise ValueError(f"RouteFactors.{name}[{level}] is {tensor.shape[-1]} wide but the topology's "
                                     f"level {level} is {widths[level]}")
                if prefix is None:
                    prefix = tuple(tensor.shape[:3])
                elif tuple(tensor.shape[:3]) != prefix:
                    raise ValueError(f"a side's routing levels share one prefix; {name}[{level}] is "
                                     f"{tuple(tensor.shape[:3])} against {prefix}")
                first = tensor if first is None else first
                if tensor.device != first.device:
                    raise ValueError(f"every routing level lives on one device; {name}[{level}] is on "
                                     f"{tensor.device} against {first.device}")
                want = quant.ROWS.dtype
                if tensor.dtype is not want:
                    raise TypeError(f"RouteFactors.{name}[{level}] is stored as {want}, got {tensor.dtype}")
                if not _aligned(tensor):
                    raise ValueError(f"RouteFactors.{name}[{level}] breaks the row alignment law: its rows start on "
                                     f"and step by multiples of {quant.ROW_ALIGN} bytes (strides "
                                     f"{tuple(tensor.stride())}, address {tensor.data_ptr() % quant.ROW_ALIGN} past the "
                                     f"grain; docs/internals/facts/liveness_contract.md#alignment)")
            prefixes[name[4:]] = prefix
        (b, t, h), (bw, tw, hw) = prefixes["read"], prefixes["write"]
        if (b, t) != (bw, tw) or hw < 1 or h % hw:
            raise ValueError(f"the write side's [B, T, H_w]={prefixes['write']} must share the read side's B and T "
                             f"and its H_w divide H ({prefixes['read']}): each write head is read by H / H_w "
                             "consecutive query heads (docs/api.md section 2.1)")
        return prefixes

    @classmethod
    def from_levels(cls, topology: Topology, read, write, *, sparse=None) -> RouteFactors:
        """THE FACTORY: any nonnegative, finite floating levels in, the format out -- each level quantized into its
        the rows' form against its own row max (`rola.routing.quant.ROWS`: u8), refusing an entry the form does not hold -- and the ``sparse`` flags (:meth:`from_rows`). Whatever
        magnitude a level carries lands in its scale; there is nothing else to pass."""
        read, write = tuple(read), tuple(write)
        sparse = default_sparse(topology) if sparse is None else tuple(tuple(map(bool, f)) for f in sparse)

        quant.refuse_unheld((quant.ROWS, level) for level in read + write)

        def rows(levels):
            return zip(*(quant.ROWS.quantize(level, checked=True) for level in levels))

        (amp_read, scale_read), (amp_write, scale_write) = rows(read), rows(write)
        source = (read, write) if _requires_grad(read + write) else None
        return cls.from_rows(topology, amp_read, amp_write, scale_read, scale_write, source=source, sparse=sparse)

    @classmethod
    def from_rows(cls, topology: Topology, amp_read, amp_write, scale_read, scale_write, source=None, *,
                  sparse=None) -> RouteFactors:
        """The bundle over rows already in the format (unit-max u8, ``[B, T, H, width_l]`` a level a side).
        ``sparse`` is ``(read, write)``, a bool a level; by default the production's own rule, a side whose
        activation has exact zeros (an opaque level, which may have zeros, included)."""
        if sparse is None:
            sparse = default_sparse(topology)
        return cls(topology=topology, amp_read=amp_read, amp_write=amp_write, scale_read=scale_read,
                   scale_write=scale_write, source=source, sparse=sparse)

    @property
    def batch(self) -> int:
        return self.amp_read[0].shape[0]

    @property
    def tokens(self) -> int:
        return self.amp_read[0].shape[1]

    @property
    def heads(self) -> int:
        """The query heads ``H``: the read side's."""
        return self.amp_read[0].shape[2]

    @property
    def write_heads(self) -> int:
        """The write heads ``H_w``: the write side's, the values' and the state's."""
        return self.amp_write[0].shape[2]

    @property
    def group(self) -> int:
        """The query heads reading each write head, ``H / H_w`` (1 for an ungrouped layer)."""
        return self.heads // self.write_heads

    def refuse_grouped(self, consumer: str) -> None:
        """THE SEAM'S REFUSAL for a consumer that reads one head count: a grouped bundle (``group`` > 1) is refused by
        name rather than read at the wrong heads (docs/api.md section 2.1)."""
        if self.group != 1:
            raise NotImplementedError(f"{consumer} reads one head count, and this bundle groups {self.heads} query "
                                      f"heads over {self.write_heads} write heads (docs/api.md section 2.1)")

    @property
    def device(self) -> torch.device:
        return self.amp_read[0].device

    @property
    def read_rows(self) -> tuple[torch.Tensor, ...]:
        """Every read level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "read") for level in range(self.topology.D))

    @property
    def write_rows(self) -> tuple[torch.Tensor, ...]:
        """Every write level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "write") for level in range(self.topology.D))

    def rows(self, level: int, side: str) -> torch.Tensor:
        """``level``'s rows ``[B, T, H, width_l]`` on ``side``."""
        return getattr(self, f"amp_{side}")[level]

    def dequantized(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``(read, write)``: the levels as fp32 values, ``rows x scale``, computed at the call. With a ``source``
        the producer's gradient steps straight through the rounding. A consumer still computing on floating
        levels calls this at its seam, and the call goes when that consumer reads the rows."""
        sides = []
        for side, name in enumerate(SIDES):
            scales = getattr(self, f"scale_{name}")
            levels = tuple(quant.ROWS.dequantize(self.rows(level, name), scale)
                           for level, scale in enumerate(scales))
            if self.source is not None:
                levels = tuple(quant.straight_through(exact, level) for exact, level in zip(self.source[side], levels))
            sides.append(levels)
        return sides[0], sides[1]

    def live(self, side: str) -> tuple[torch.Tensor, ...]:
        """THE LIVE DIGITS of one side, a bool ``[B, T, H, width_l]`` a level: its nonzero digits, at every level. The
        facts pass words a dense level by its width instead, a superset of this (docs/internals/facts/facts.md)."""
        return tuple(self.rows(level, side) != 0 for level in range(self.topology.D))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        source = () if self.source is None else (*self.source[0], *self.source[1])
        return (*self.amp_read, *self.amp_write, *self.scale_read, *self.scale_write, *source)


def default_sparse(topology: Topology) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """THE PRODUCTION'S SPARSE RULE, ``(read, write)`` a bool a level: a side's level whose activation has exact zeros
    (an opaque level included) -- the producer's flags, the factory's default and the layer's working set."""
    return tuple(tuple(topology.activation_properties(level, side).exact_zeros for level in range(topology.D))
                 for side in SIDES)
