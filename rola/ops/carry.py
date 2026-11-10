# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY, as a caller reaches it: one launch over a call's windows, from the bundle's rows and values, the facts
pass's lists, the den pass's units and the carve (docs/internals/carry/carry_api.md).

Symbols, each at first use: ``B`` batch, ``T`` = ``L`` tokens, ``H`` heads, ``BH`` = ``B H``, ``d_v`` the value
width, ``W`` the window (the device's card's, which the entry reads). The carve (:class:`rola.engine.types.Carve`) names the box -- an extent a
level and the channel extent -- and each side's list levels and operand levels; its arm key
(:func:`arm_key`) selects the compiled body, and the box's extents are runtime addressing. The readout (a value carve:
each warp its channels over every leaf of the box) adds each CTA's partials at each token's scale byte (``scale``, the
scale pass's) into ``num`` ``[BH, L, d_v / 2]`` int32 f16 pairs in channel order, or fp32 ``[BH, L, d_v]``.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import lists as list_model
from rola.engine.types import Carve
from rola.ops import facts as facts_ops
from rola.ops._ext import extension

__all__ = [
    "LEDGER_PHASES",
    "LEDGER_WARPS",
    "REDUCE_F16_MAX_CTAS",
    "Form",
    "arm_key",
    "arms",
    "build_stamp",
    "capture_bind",
    "capture_bytes",
    "carry",
    "census",
    "forced_word",
    "form",
    "ledger_bind",
    "plan_words",
    "probe_bind",
    "readout_accumulate",
    "reduce_ctas",
    "reduce_dtype_for",
    "same_split",
    "sm_clock_ghz",
]


#: THE PHASE LEDGER'S SHAPE: ``[ctas][warps][phases]``, the kernel's own `CarryPhase` order.
LEDGER_PHASES = ("landed", "maxima", "store", "readout", "drain", "fill", "fold", "wait")
LEDGER_WARPS = 8


def _outer(plan) -> set[int]:
    """A side's levels outside its dense rows: its lists, or a side with none its coefficient levels."""
    return set(plan.lists if plan.lists else plan.operand[1:])


def same_split(carve: Carve) -> int:
    """Whether the fold's lists are the readout's digits (1: both sides list the same levels) or its K (0) -- the
    hand-off's form (docs/internals/carry/handoff.md#store); a box whose sides share part of a level set has none."""
    read = carve.readout
    fold, read_rows = _outer(carve.fold), set(read.operand if read.lists else read.operand[:1])
    if fold == _outer(read):
        return 1
    if fold == read_rows:
        return 0
    raise ValueError(f"the fold's lists {sorted(fold)} are neither the readout's digits nor its K "
                     f"{sorted(read_rows)}: no hand-off form carries them")


def arm_key(carve: Carve) -> tuple[int, ...]:
    """The compiled body a carve selects: ``(fold_rows, fold_factors, read_rows, read_factors, channels,
    same_split)`` -- each side's dense rows (the fold's warp tile, the readout's K) and its operand's level count, the
    channel extent and the hand-off's form (:func:`same_split`; ``tools/gen_shards.py``'s ``ARM_KEY``)."""
    return (carve.fold.tile_rows, len(carve.fold.operand), carve.readout.tile_rows, len(carve.readout.operand),
            carve.channels, same_split(carve))


def _list_level(plan) -> int:
    return plan.lists[0] if plan.lists else -1


#: THE CROSS-CTA SUM's auto rule: f16 pairs while their error, ~0.012% x sqrt(n) a token for ``n`` CTAs adding into a
#: slot, stays under the bf16 output store's 0.16% (docs/internals/carry/carry_api.md#carry-reduce)
REDUCE_F16_MAX_CTAS = 177


def reduce_ctas(carve: Carve) -> int:
    """The most CTAs any output slot sums: the read side's boxes along every level (a listed level counted whole, a
    bound until the declared cap reaches the carve)."""
    n = 1
    for width, extent in zip(carve.widths, carve.extents):
        n *= width // extent
    return n


def reduce_dtype_for(carve: Carve, reduce_dtype: torch.dtype | None = None) -> torch.dtype:
    """The call's cross-CTA sum dtype: ``reduce_dtype`` as given, else f16 pairs while :func:`reduce_ctas` stays within
    :data:`REDUCE_F16_MAX_CTAS`, else fp32."""
    if reduce_dtype is not None:
        if reduce_dtype not in (torch.float16, torch.float32):
            raise ValueError(f"reduce_dtype is torch.float16, torch.float32 or None (auto), got {reduce_dtype}")
        return reduce_dtype
    return torch.float16 if reduce_ctas(carve) <= REDUCE_F16_MAX_CTAS else torch.float32


def carry(routes, values, facts, carve: Carve, scale: torch.Tensor, accumulate: str | None = None,
          forced: str | None = None,
          reduce_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """THE LAUNCH: ``(num, den, check, plans)`` -- ``num`` the readout's partials summed across CTAs: int32
    ``[BH, T, d_v / 2]`` f16 pairs in channel order at each token's ``scale`` byte (``reduce_dtype`` f16), or fp32
    ``[BH, T, d_v]`` (fp32; :func:`reduce_dtype_for`'s rule where None) -- ``den`` each token's den, fp32 ``[BH, T]``,
    the v+1 channel (`handoff.md#den`) -- ``check`` the skeleton's fp32 sum of the final state -- ``plans`` the plan
    pass's records the entry allocated at the device's window, int32 ``[BH_w, windows, box classes, words]``, for a gate
    to read (docs/internals/carry/carry_api.md#carry-forward). ``facts`` is the facts pass's over this bundle, its read entries' offsets
    in this arm's read pool rows (:func:`form`). The readout accumulates in the device's form
    (:func:`readout_accumulate`); ``accumulate`` (``"fp32"`` or ``"fp16"``) is reserved: the gates' choice of form, so
    both forms are gated on any card (``tests/oracle/test_carry_readout.py``). Each window's plan is the planner rule's
    (docs/internals/carry/schedule.md#planner); ``forced`` (:func:`forced_word`) is reserved: the gates' and the
    measurements' choice of the plan a window takes where every chunk of it fits. A grouped bundle's CTAs are its write heads', each reading its query
    heads' rows; ``num`` and ``scale`` are a query head's."""
    BH = routes.batch * routes.heads
    d_v = values.v.shape[-1]
    dtype = reduce_dtype_for(carve, reduce_dtype)
    num = (torch.zeros((BH, routes.tokens, d_v // 2), dtype=torch.int32, device=routes.device)
           if dtype == torch.float16 else
           torch.zeros((BH, routes.tokens, d_v), dtype=torch.float32, device=routes.device))
    den = torch.zeros((BH, routes.tokens), dtype=torch.float32, device=routes.device)
    check = torch.zeros(1, dtype=torch.float32, device=routes.device)
    plans = extension().carry_forward(
        list(routes.amp_read), list(routes.amp_write), values.v, list(facts_ops.side_lists(facts, "write")),
        facts.list_exp, facts.fm, facts.gains_exp, list(facts_ops.side_lists(facts, "read")), scale, num, den, check,
        list(carve.widths),
        list(carve.extents), carve.channels, _list_level(carve.fold), list(carve.fold.operand),
        _list_level(carve.readout), list(carve.readout.operand), list(arm_key(carve)),
        _ACCUMULATE[accumulate] if accumulate else -1, _REDUCE[dtype], forced_word(forced))
    return num, den, check, plans


def plan_words(heads: int, w: int) -> int:
    """A window's plan record's 32-bit words at ``heads`` query heads a write head and a window of ``w`` tokens
    (`carry.cuh` `plan_words`): its plan word, then a 16-bit step word a step -- its read chunks (W / 64 a head), its
    fold chunks (W / 64), the retiring step -- in whole 16-byte copies."""
    steps = list_model.list_at(w) * heads + list_model.list_at(w) + 1
    return (1 + (steps + 1) // 2 + 3) // 4 * 4


_ACCUMULATE = {"fp32": 0, "fp16": 1}
_REDUCE = {torch.float16: 0, torch.float32: 1}
_MODES = {"rule": 0, "whole": 1, "halves": 2}


def forced_word(forced: str | None) -> int:
    """A FORCED PLAN FORM as the plan pass takes it (docs/internals/carry/schedule.md#planner): ``"mode/read/fold"``,
    the mode ``rule``, ``whole`` or ``halves`` and each chunk ``rule``, ``union`` or a multiple of 64 slots -- a window
    takes it where every chunk of it fits, else the rule's plan; ``None`` the rule's everywhere."""
    if forced is None:
        return -1
    mode, read, fold = forced.split("/")

    def chunk(name: str, union: int) -> int:
        if name in ("rule", "union"):
            return 0 if name == "rule" else union
        if int(name) % 64 or not 0 < int(name) // 64 < union:
            raise ValueError(f"a forced chunk is rule, union or a multiple of 64 slots below {64 * union}: {name}")
        return int(name) // 64

    return _MODES[mode] | chunk(read, 0xFF) << 2 | chunk(fold, 0xF) << 10


def readout_accumulate() -> str:
    """The readout's MMA accumulation on the current device, as the entries resolve it when no form is named -- for a
    gate or a tool naming the form: ``"fp16"`` where its fp32-accumulate MMA runs slower than its fp16-accumulate one by
    its rates (a GeForce card), else ``"fp32"`` (docs/internals/carry/carry_api.md#carry-readout-accumulate)."""
    return "fp16" if extension().carry_readout_accumulate(torch.cuda.current_device()) else "fp32"


def arms() -> list[tuple[int, ...]]:
    """The arms THIS binary carries, each its :func:`arm_key`: the binary's own answer, never the source tree's."""
    return [tuple(row) for row in extension().carry_arms()]


def census() -> list[tuple[int, ...]]:
    """Each built arm's device census: its index and key, then registers, local bytes, shared bytes, the block's
    thread limit, the binary version, and its form on this card (the area its plans place buffers in, then its pool
    rows: :class:`Form`)."""
    return [tuple(row) for row in extension().carry_census()]


class Form(NamedTuple):
    """An arm's form on the current card (docs/internals/carry/plan.md#plan): its shared memory and the area its plans
    place buffers in (one head's tables below the top), then a read and a write pool row's bytes -- what the facts pass
    addresses the read entries' rows in -- and each side's list group, the digits a union of the lists it reads spans
    (plan.md#list-groups)."""

    smem: int
    area: int
    read_row: int
    write_row: int
    read_list_group: int
    write_list_group: int


#: a census row's fields: the index, the key's six, then registers, local bytes, shared bytes, the thread limit, the
#: binary version, the area, the two pool rows, the two list groups
_KEY = slice(1, 7)


def form(key: tuple[int, ...]) -> Form:
    """The form of the built arm ``key`` (:func:`arm_key`) on the current card, refusing a key no built arm carries."""
    for row in census():
        if tuple(row[_KEY]) == tuple(key):
            return Form(row[9], row[12], row[13], row[14], row[15], row[16])
    raise ValueError(f"no built carry arm has the key {tuple(key)}; this binary carries {arms()}")


def lists_form(key: tuple[int, ...]) -> facts_ops.ListsForm:
    """The lists the built arm ``key`` reads, the facts pass's call: its read pool row and each side's list group."""
    f = form(key)
    return facts_ops.ListsForm(f.read_row, f.read_list_group, f.write_list_group)


def build_stamp() -> int:
    """The carry family's DEVICE build stamp, a fact only a built kernel produces (the window, the capability digest,
    the state's values and a list's boundary granularity, read back off a kernel this binary compiled)."""
    return extension().carry_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/carry/carry_kernel.md#phases)
    """Bind the phase ledger the next launches book into, ``[ctas, LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles,
    or ``None``."""
    extension().carry_ledger_bind(ledger)


def capture_bind(out: torch.Tensor | None) -> None:  #: reserved: the lab's window on a real call (docs/internals/carry/carry_api.md#carry-capture-bind)
    """Bind the host u8 tensor (:func:`capture_bytes` long) the next launch copies its operand block and grid into, or
    ``None``."""
    extension().carry_capture_bind(out)


def capture_bytes() -> int:
    """A capture's bytes in this binary: its byte count, boxes and batch-heads (int64 each), then the operand
    block."""
    return int(extension().carry_capture_bytes())


def probe_bind(probe: torch.Tensor | None, den: torch.Tensor | None = None) -> None:  #: reserved: the fold's oracle instrument (docs/internals/carry/fold.md#probe)
    """Bind the probes the next launches write their final state into, absolute fp32 ``[BH, N, d_v]`` and the den state
    ``[BH, N]`` by global leaf in level order, or ``None``."""
    extension().carry_probe_bind(probe, den)


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
