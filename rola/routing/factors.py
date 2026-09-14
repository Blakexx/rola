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
``docs/api.md`` section 2.1. :class:`~rola.routing.producer.RouteProducer` is the one
this package ships; hand construction is equally legal and equally checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing.types import Topology

__all__ = ["RouteFactors"]


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides and their side gains. Tensors, and nothing else.

    ``read``/``write``: one ``[B, T, H, width_l]`` level per topology level; ``write``
    is a simplex, ``read`` is a simplex SCALED by ``read_mass``.
    ``g_write``: ``[B, T, H]`` write-side gain. There is no read-side gain column at
    all: the readout is a ratio, which fixes it to 1 identically.
    ``read_mass``: the ``[B, T, H]`` per-token scale the read levels carry, or ``None``
    for the unscaled case. :meth:`read_simplex` divides it back out.

    Direct construction is legal; ``__post_init__`` holds it to the same invariants a
    derived bundle satisfies (``docs/api.md`` §2.1).
    """

    topology: Topology
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    g_write: torch.Tensor
    read_mass: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "write", tuple(self.write))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D, widths = self.topology.D, self.topology.widths

        for name, levels in (("read", self.read), ("write", self.write)):
            if len(levels) != D:
                raise ValueError(
                    f"RouteFactors.{name} must hold one tensor per routing level "
                    f"(D={D}), got {len(levels)}")

        prefix = None
        for name, levels in (("read", self.read), ("write", self.write)):
            for level, tensor in enumerate(levels):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    shape = getattr(tensor, "shape", type(tensor).__name__)
                    raise TypeError(
                        f"RouteFactors.{name}[{level}] must be a [B, T, H, width_l] "
                        f"tensor, got {shape}")
                if tensor.shape[-1] != widths[level]:
                    raise ValueError(
                        f"RouteFactors.{name}[{level}] is {tensor.shape[-1]} wide but the "
                        f"topology's level {level} is {widths[level]}")
                if prefix is None:
                    prefix = tuple(tensor.shape[:3])
                elif tuple(tensor.shape[:3]) != prefix:
                    raise ValueError(
                        f"every routing level shares one [B, T, H] prefix; "
                        f"{name}[{level}] is {tuple(tensor.shape[:3])} against {prefix}")
                if tensor.device != self.read[0].device:
                    raise ValueError(
                        f"every routing level lives on one device; {name}[{level}] is on "
                        f"{tensor.device} against {self.read[0].device}")
                if tensor.dtype != self.read[0].dtype:
                    raise ValueError(
                        f"every routing level shares one dtype; {name}[{level}] is "
                        f"{tensor.dtype} against {self.read[0].dtype}")
        for name, column in (("g_write", self.g_write),
                             ("read_mass", self.read_mass)):
            if column is None:
                continue
            if not isinstance(column, torch.Tensor) or tuple(column.shape) != prefix:
                shape = getattr(column, "shape", type(column).__name__)
                raise ValueError(
                    f"RouteFactors.{name} must be a [B, T, H]={prefix} tensor, got {shape}")

    @property
    def batch(self) -> int:
        return self.read[0].shape[0]

    @property
    def tokens(self) -> int:
        return self.read[0].shape[1]

    @property
    def heads(self) -> int:
        return self.read[0].shape[2]

    @property
    def device(self) -> torch.device:
        return self.read[0].device

    def read_needs_normalization(self) -> tuple[bool, ...]:
        """Per read level: is it still off the simplex? THE PREDICATE, stated once.

        :meth:`read_simplex` divides exactly these levels and the decode fold normalizes
        exactly these levels inside its kernel; two sites folding one question from the
        same facts is one divergence waiting to happen.
        """
        if self.read_mass is None:
            return (False,) * len(self.read)
        return tuple(not self.topology.duty_simplex_output(index, "read")
                     for index in range(len(self.read)))

    def read_simplex(self) -> tuple[torch.Tensor, ...]:
        """The read levels on the simplex -- ``read`` with :attr:`read_mass` divided out.

        The consumer a bundle is built for reads ``(read, read_mass)`` unscaled; this is
        for every consumer that is defined on the simplex instead (the fp64 oracle and
        its contract, the decode arm, the reference recurrence).
        """
        if self.read_mass is None:
            return self.read
        #: PER LEVEL, not by the product: `read_mass` is the product the readout's ratio
        #: takes, but a simplex is a per-level claim, and the levels the topology already
        #: declares on it are returned untouched so this is exactly the bundle a
        #: mass-carrying one would have been.
        #:
        #: THE QUOTIENT IS AT LEAST fp32: a bf16 quotient of a bf16 sum of bf16 addends
        #: is three roundings where one is forced, and every consumer of this widens
        #: anyway. A level already wider than fp32 keeps its own width.

        def divided(level: torch.Tensor) -> torch.Tensor:
            wide = torch.promote_types(level.dtype, torch.float32)
            return level.to(wide) / level.sum(dim=-1, keepdim=True, dtype=wide)

        return tuple(divided(level) if needs else level
                     for level, needs in zip(self.read, self.read_needs_normalization(),
                                             strict=True))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        mass = () if self.read_mass is None else (self.read_mass,)
        return (*self.read, *self.write, self.g_write, *mass)
