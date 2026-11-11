# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEVICE'S CARD as the binary selects it (``csrc/rola/src/common/select.cuh``) -- its compute capability,
product, build variant, family and window W -- for the readers that need a hardware fact the kernel path cannot serve:
the fp64 oracle building its reference windows, the gates and the measurement tools. No kernel call is shaped by it:
every entry reads its own card (docs/architecture/families.md#the-entry-and-the-card).
"""
from __future__ import annotations

import dataclasses
import functools

from rola.ops._ext import device_index, extension

__all__ = ["Card", "card_of", "window_of"]

#: the families in `Family`'s order (`common/hardware.cuh`)
FAMILIES = ("ampere", "hopper", "blackwell")


@dataclasses.dataclass(frozen=True)
class Card:
    """A device's card: its compute capability (x10), its product row's index in the generated `kProducts` (-1: none),
    the build variant it runs, its family and its window W."""

    cc: int
    product: int
    variant: int
    family: str
    window: int


@functools.cache
def _card_of(index: int) -> Card:
    cc, product, variant, family, window = (int(x) for x in extension().device_card(index))
    return Card(cc, product, variant, FAMILIES[family], window)


def card_of() -> Card:
    """The current device's card; refused by name where the binary carries no build for it."""
    return _card_of(device_index(None))


def window_of() -> int:
    """The current device's window in tokens."""
    return card_of().window
