# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CELL REGISTRY: the carry family's cells as DATA, with one draw behind them.

`carry_cells.json` holds the records; this module reads them, derives each cell's
STATE DESCRIPTOR and LAUNCH SHAPE from its fields, realizes its routing draw, and builds
`carry_call` -- one cell's WHOLE call. ONE definition, two readers: the oracle tier
and the benches both reach the kernel through it, so a cell's
operands are the same tensors in a correctness run and in a measured one.

The registry's other half is `layer.py` (`layer_cells.json`): a LAYER cell declares a
CONSTRUCTOR -- a producer, a routing template, a gain -- from which the amplitudes are
PRODUCED, rather than a draw that puts them on the simplex directly. The two answer
different questions and neither stands in for the other, so both live here under one set
of names.

A record DECLARES A DRAW AND A SHAPE and never an arm: the arm key is
``(D, DV, warps_per_cta)`` and `rola.ops.carry` derives it. Nothing in a record names
a window, a token tile, a stream count or a ``(k, m)`` box -- the axis law made those
kernel-internal, derived or constant, so a cell that could name one would be declaring
something the kernel does not read.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``k_tok`` = a token's nonzero digits per sparse
level (a property of the DRAW, never told to the kernel).
"""
from __future__ import annotations

import dataclasses
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

_REGISTRY = Path(__file__).resolve().parent / "carry_cells.json"

DRAWS = ("dense", "alt", "both", "cohort", "dead", "deposit",
         "tied", "anti", "cold", "readonly", "concentrated", "onehot", "flip")
#: the draws with no support to sample, which state it themselves and declare no regime
DEGENERATE = ("dead", "deposit")
BACKINGS = ("dense", "paged")
STATES = ("none", "fresh", "carried")
TIERS = ("oracle", "probe", "both")


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One record, validated. ``seed`` is derived from ``name`` and never stored.

    A seed that is a PROPERTY OF THE CELL is the point: `hash(str)` is salted per
    process, so a cell seeded from it would be a different draw on every run and a
    failure could not be reproduced from its own name.
    """

    name: str
    widths: tuple[int, ...]
    dv: int
    tokens: int
    warps_per_cta: int
    draw: str
    k_tok: int | None
    cohort: int | None
    support: float
    backing: str
    state: str
    tier: str
    #: ``((axis, value), ...)`` on every axis of `benchmarks.cells.regimes.REGIME_AXES`; None for a degenerate draw
    regime: tuple[tuple[str, str], ...] | None

    @property
    def seed(self) -> int:
        return zlib.crc32(self.name.encode()) & 0xFFFF

    @property
    def D(self) -> int:
        return len(self.widths)

    @property
    def arm(self) -> tuple[int, int, int]:
        """The carry arm this cell launches: ``(D, DV, warps_per_cta)``, the key a binary's arms are listed by
        (`rola.ops.carry.arms()`) and the shipped set is declared in (`tools/manifests/shipped_set.json`)."""
        return (self.D, self.dv, self.warps_per_cta)

    @property
    def N(self) -> int:
        n = 1
        for width in self.widths:
            n *= width
        return n

    def descriptor(self, BH: int = 1):
        """This cell's state format descriptor -- the one shape every entry reads."""
        from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
        return StateFormat(D=self.D, B=tuple(self.widths), order=CANONICAL_ORDER,
                           page_bits=PAGE_RECTANGLE_BITS, DV=self.dv,
                           dtype=SPLIT_PLANE_DTYPE,
                           ids=BH * (self.N >> PAGE_RECTANGLE_BITS))

    def launch(self):
        from rola.ops.carry import LaunchShape
        return LaunchShape(warps_per_cta=self.warps_per_cta)

    @property
    def level_modes(self) -> int:
        """The per-level support the draw actually has, in the kernel's mode word: two
        bits a level, bit 0 the read side sparse there, bit 1 the write side. A dense or
        degenerate draw declares nothing; the alternation declares the read side sparse at
        the odd levels and the write side at the even ones; ``both`` declares level zero
        sparse on both sides. The carve order is derived from this, so a cell that leaves
        it at zero runs the canonical carve whatever its draw."""
        from rola.ops.carry import SIDE_SPARSE
        if self.draw not in ("alt", "both", "cohort") or self.k_tok is None:
            return 0
        modes = 0
        for level in range(self.D):
            if self.draw == "both":
                if level == 0:
                    modes |= (SIDE_SPARSE[0] | SIDE_SPARSE[1]) << (2 * level)
            elif level % 2:
                modes |= SIDE_SPARSE[0] << (2 * level)
            else:
                modes |= SIDE_SPARSE[1] << (2 * level)
        return modes


def _validate(record: dict) -> CellSpec:
    from benchmarks.cells.regimes import REGIME_AXES

    regime = record["regime"]
    spec = CellSpec(name=record["name"], widths=tuple(record["widths"]), dv=record["dv"],
                    tokens=record["tokens"], warps_per_cta=record["warps_per_cta"],
                    draw=record["draw"], k_tok=record["k_tok"], cohort=record["cohort"],
                    support=record["support"], backing=record["backing"],
                    state=record["state"], tier=record["tier"],
                    regime=None if regime is None else tuple((axis, regime[axis]) for axis in REGIME_AXES
                                                             if axis in regime))
    if (spec.draw in DEGENERATE) != (regime is None):
        raise ValueError(f"{spec.name}: a {spec.draw} draw {'declares no' if spec.draw in DEGENERATE else 'states its'} "
                         f"regime")
    if regime is not None:
        if set(regime) != set(REGIME_AXES):
            raise ValueError(f"{spec.name}: the regime states {sorted(regime)}, not every axis of {list(REGIME_AXES)}")
        for axis, value in regime.items():
            if value not in REGIME_AXES[axis]:
                raise ValueError(f"{spec.name}: {value!r} is not a value of the {axis} axis {REGIME_AXES[axis]}")
    if spec.draw not in DRAWS:
        raise ValueError(f"{spec.name}: draw {spec.draw!r} is not one of {DRAWS}")
    if spec.backing not in BACKINGS:
        raise ValueError(f"{spec.name}: backing {spec.backing!r} is not one of {BACKINGS}")
    if spec.state not in STATES:
        raise ValueError(f"{spec.name}: state {spec.state!r} is not one of {STATES}")
    if spec.tier not in TIERS:
        raise ValueError(f"{spec.name}: tier {spec.tier!r} is not one of {TIERS}")
    if (spec.draw == "cohort") != (spec.cohort is not None):
        raise ValueError(f"{spec.name}: a cohort draw carries a cohort and nothing else does")
    if spec.draw in ("alt", "both", "cohort", "tied") and spec.k_tok is None:
        raise ValueError(f"{spec.name}: a sparse draw states its k_tok")
    if spec.draw not in ("alt", "both", "cohort", "tied") and spec.k_tok is not None:
        raise ValueError(f"{spec.name}: a {spec.draw} draw has no k_tok -- its support is "
                         f"stated by the draw itself, not by a count")
    return spec


def carry_cell(name: str, **params) -> CellSpec:
    """A carry cell's data provider (`carry_cells.json` names it): the record's parameters, validated."""
    return _validate({"name": name, **params})


def carry_cells(tier: str | None = None) -> tuple[CellSpec, ...]:
    """The registry, validated. ``tier`` selects the readers a cell is declared for."""
    records = json.loads(_REGISTRY.read_text())["cells"]
    cells = tuple(_validate(record) for record in records)
    names = [cell.name for cell in cells]
    if len(set(names)) != len(names):
        raise ValueError("two cells share a name; a cell's name IS its seed")
    if tier is None:
        return cells
    return tuple(cell for cell in cells if cell.tier in (tier, "both"))


def by_name(name: str) -> CellSpec:
    for cell in carry_cells():
        if cell.name == name:
            return cell
    raise KeyError(f"no cell named {name!r} in {_REGISTRY}")


# ------------------------------------------------------------------ the draw

def simplex(shape, k_tok, gen, device, live=None):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense.

    ``live`` truncates a row to its FIRST ``live`` digits -- STRUCTURED support, as
    against ``k_tok``'s unstructured one. The two are different questions: unstructured
    sparsity at long ``L`` still reaches every page, so only a truncation can produce a
    page the routing never touches (the idle-resident cell). It is applied after both
    draws so a cell's RNG stream does not depend on it.
    """
    import torch

    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    if live is not None and live < shape[-1]:
        x[..., live:] = 0.0
    return x / x.sum(-1, keepdim=True)


def clustered(shape, k_tok, cohort, gen, device):
    """A simplex draw whose support is ONE contiguous digit window per run of ``cohort``
    consecutive tokens -- the structure the whole-window skip exists to exploit, and the
    one that leaves an owner with a handful of live tokens per window."""
    import torch

    B, T, H, width = shape
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    starts = torch.randint(0, width, (B, T // cohort, 1, 1), device=device, generator=gen)
    idx = (starts.expand(B, T // cohort, cohort, 1).reshape(B, T, 1, 1)
           + torch.arange(k_tok, device=device).view(1, 1, 1, k_tok)) % width
    keep = torch.zeros(shape, device=device, dtype=torch.float64)
    keep.scatter_(-1, idx.expand(B, T, H, k_tok), 1.0)
    x = x * keep
    return x / x.sum(-1, keepdim=True)


#: THE ONE TOKEN a ``deposit`` draw makes live. Not the window's first: a token index the
#: kernel's segment layout has to place correctly is what makes the cell a test of the
#: layout rather than of its origin.
DEPOSIT_TOKEN = 5


def _degenerate(spec, shape, gen, device):
    """The two draws with NO support to sample: nothing live, and exactly one deposit.

    ``dead`` puts zero amplitude on every digit of every level on both sides, so no token
    reads and no token writes -- the recurrence's identity, whose whole content is that
    the state comes out as it went in. ``deposit`` is the single-tile point of the fold:
    one token, one-hot on digit zero of every write level, so exactly one leaf receives
    exactly one deposit, and the read side is dead so the readout is the zero it would be
    against an entry state a single window never reads.
    """
    import torch

    B, T, H = shape
    zeros = tuple(torch.zeros(B, T, H, w, device=device, dtype=torch.float64)
                  for w in spec.widths)
    write = zeros
    if spec.draw == "deposit":
        write = tuple(torch.zeros(B, T, H, w, device=device, dtype=torch.float64)
                      for w in spec.widths)
        for level in write:
            level[:, DEPOSIT_TOKEN, :, 0] = 1.0
    gain = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(B, T, H, spec.dv, device=device, dtype=torch.float64, generator=gen)
    return RealizedCell(spec=spec,
                        read=tuple(x.to(torch.bfloat16) for x in zeros),
                        write=tuple(x.to(torch.bfloat16) for x in write),
                        gain=gain.to(torch.bfloat16), v=v.to(torch.bfloat16))


@dataclass(frozen=True, slots=True)
class RealizedCell:
    """A drawn cell: the bf16 operands the kernel gets, and their fp64 doubles.

    The reference reads the ROUNDED operands, never the raw fp64 draws: comparing an
    fp64 oracle of the fp64 draws against a kernel fed their bf16 images would measure
    the CAST, which is not what is under test.
    """

    spec: CellSpec
    read: tuple
    write: tuple
    gain: object
    v: object
    #: ``[BH, N, DV + 1]`` fp32, CANONICAL leaf order: the entry state a ``carried`` cell binds (value columns
    #: ``0.1 * N(0, 1)``, the mass column their magnitudes), or None. Drawn after every operand, so binding one moves
    #: no other tensor of the cell.
    entry: object = None

    def doubles(self):
        return (tuple(x.double() for x in self.read), tuple(x.double() for x in self.write),
                self.gain.double(), self.v.double())


def _masked(shape, mask, gen, device):
    """A simplex draw confined to the digits ``mask`` marks (a ``[..., width]`` 0/1 tensor broadcast over ``shape``)."""
    import torch

    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen) * mask
    return x / x.sum(-1, keepdim=True)


def _one_hot(shape, gen, device):
    import torch

    digit = torch.randint(0, shape[-1], shape[:-1] + (1,), device=device, generator=gen)
    return torch.zeros(shape, device=device, dtype=torch.float64).scatter_(-1, digit, 1.0)


def _corner(spec, shape, gen, device):
    """``(read, write)`` for a CORNER draw -- a named region of the distribution a random draw does not reach.

    ``tied``: one ``k_tok`` draw per level is both sides. ``anti``: level zero reads its first half of digits and
    writes its second half, so no token reads a leaf it writes and every read leaf is cold. ``cold``: level zero's
    write side is confined to its leading ``support`` fraction under dense reads. ``readonly``: every token writes
    leaf zero and nothing else. ``concentrated``: every row carries 0.9 of its mass on one digit. ``onehot``: one live
    digit per row on both sides. ``flip``: level zero's write support is its first half of digits in even windows
    and its second half in odd ones, so which window last wrote a leaf changes exactly at a window line.
    """
    import torch

    from rola.ops.carry import WINDOW

    def dense(width):
        return simplex(shape + (width,), None, gen, device)

    def half(width, second):
        mask = torch.zeros(width, device=device, dtype=torch.float64)
        mask[width // 2:] = float(second)
        mask[:width // 2] = float(not second)
        return mask

    widths = spec.widths
    if spec.draw == "tied":
        read = tuple(simplex(shape + (w,), spec.k_tok, gen, device) for w in widths)
        return read, read
    if spec.draw == "onehot":
        return (tuple(_one_hot(shape + (w,), gen, device) for w in widths),
                tuple(_one_hot(shape + (w,), gen, device) for w in widths))
    if spec.draw == "concentrated":
        def peaked(width):
            x = torch.rand(shape + (width,), device=device, dtype=torch.float64, generator=gen)
            return 0.1 * x / x.sum(-1, keepdim=True) + 0.9 * _one_hot(shape + (width,), gen, device)
        return tuple(peaked(w) for w in widths), tuple(peaked(w) for w in widths)
    read = [dense(w) for w in widths]
    write = [dense(w) for w in widths]
    w0 = widths[0]
    if spec.draw == "anti":
        read[0] = _masked(shape + (w0,), half(w0, second=False), gen, device)
        write[0] = _masked(shape + (w0,), half(w0, second=True), gen, device)
    elif spec.draw == "cold":
        mask = torch.zeros(w0, device=device, dtype=torch.float64)
        mask[:max(1, int(w0 * spec.support))] = 1.0
        write[0] = _masked(shape + (w0,), mask, gen, device)
    elif spec.draw == "readonly":
        write = [torch.zeros(shape + (w,), device=device, dtype=torch.float64) for w in widths]
        for level in write:
            level[..., 0] = 1.0
    elif spec.draw == "flip":
        odd = (torch.arange(shape[1], device=device) // WINDOW) % 2 == 1
        mask = torch.where(odd.view(1, -1, 1, 1), half(w0, second=True), half(w0, second=False))
        write[0] = _masked(shape + (w0,), mask, gen, device)
    return tuple(read), tuple(write)


def realize(spec: CellSpec, B: int = 1, H: int = 1, device: str = "cuda") -> RealizedCell:
    """Draw ``spec``: ``[B, L, H, B_l]`` per level, ``[B, L, H]`` gain, ``[B, L, H, DV]`` v.

    THE ALTERNATION (Blake, 2026-08-19): per level at most ONE sparse side, and the
    sparse side alternates -- EVEN levels write-sparse / read-dense, ODD levels
    read-sparse / write-dense -- so the two sides' clause sets are DIFFERENT level
    subsets, which is what the two-sided schedule exists for. ``both`` is sparse on both
    sides at level zero and dense above it; the two sides are drawn INDEPENDENTLY there
    on purpose, because the kernel never assumes they agree.
    """
    import torch

    gen = torch.Generator(device=device).manual_seed(spec.seed)
    shape = (B, spec.tokens, H)
    live = tuple(max(1, int(w * spec.support)) for w in spec.widths)

    def draw(width, kt, lv):
        if kt is None or kt >= width:
            return simplex(shape + (width,), None, gen, device, live=lv)
        if spec.draw == "cohort":
            return clustered(shape + (width,), kt, spec.cohort, gen, device)
        return simplex(shape + (width,), kt, gen, device, live=lv)

    if spec.draw in DEGENERATE:
        drawn = _degenerate(spec, shape, gen, device)
    else:
        kt = spec.k_tok
        if spec.draw in ("dense", "alt", "both", "cohort"):
            if spec.draw == "dense":
                read_k = write_k = [None] * spec.D
            elif spec.draw == "both":
                read_k = write_k = [kt if l == 0 else None for l in range(spec.D)]
            else:
                read_k = [kt if l % 2 else None for l in range(spec.D)]
                write_k = [None if l % 2 else kt for l in range(spec.D)]
            read = tuple(draw(w, read_k[l], live[l]) for l, w in enumerate(spec.widths))
            write = tuple(draw(w, write_k[l], live[l]) for l, w in enumerate(spec.widths))
        else:
            read, write = _corner(spec, shape, gen, device)
        gain = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
        v = torch.randn(*shape, spec.dv, device=device, dtype=torch.float64, generator=gen)
        drawn = RealizedCell(spec=spec,
                             read=tuple(x.to(torch.bfloat16) for x in read),
                             write=tuple(x.to(torch.bfloat16) for x in write),
                             gain=gain.to(torch.bfloat16), v=v.to(torch.bfloat16))
    if spec.state == "carried":
        entry = 0.1 * torch.randn(B * H, spec.N, spec.dv + 1, device=device, dtype=torch.float64, generator=gen)
        entry[..., spec.dv] = entry[..., spec.dv].abs()
        drawn = dataclasses.replace(drawn, entry=entry.to(torch.float32))
    if spec.regime is not None:
        from benchmarks.cells.regimes import prove

        prove(drawn)
    return drawn


def liveness_words(realized: RealizedCell, descriptor):
    """THE CELL'S OWN STATEMENT of its draw's support, as the class-1 words.

    A cell knows its support because it DREW it, so the words here come from the
    contract's pure-torch MODEL and never from the pass: the device pass reads the
    amplitudes itself and the two legs are then a cross-check, which is the only shape
    in which two derivations of one fact are allowed to exist. The LAYOUT is the one
    contract both legs share: ``[BH, 2, sum_l B_l, ceil(L / 32)]`` int32.
    """
    from rola.engine.facts import liveness as lv
    from rola.ops.carry import LivenessWords, pack_side

    read, write = pack_side(realized.read), pack_side(realized.write)
    layout = lv.LivenessLayout(D=len(descriptor.B), B=tuple(descriptor.B), L=read.shape[-2])
    statics = lv.side_statics(layout, (False,) * layout.D, layout.B)
    return LivenessWords(words=lv.liveness_words(read, write, layout, statics, statics))


def carry_call(spec: CellSpec, bh: int = 1, device: str = "cuda"):
    """``(drawn, kwargs)`` -- one cell's WHOLE carry call, built once from its record.

    The kwargs are exactly `rola.ops.carry.carry_forward`'s keyword operands, plus the
    two positional ones under ``routes`` and ``v``; the state planes and the page table
    are the CALLER's, because which backing a cell binds is what a caller varies.

    THIS IS THE ONE BINDING. The oracle tier and the benches
    both reach the kernel through it, so a cell's operands are the same tensors in a
    correctness run and in a measured one -- a bench that packed its own would be
    measuring a second definition of the cell.
    """
    from rola.ops import carry as carry_ops

    drawn = realize(spec, device=device)
    descriptor = spec.descriptor(BH=bh)
    launch = spec.launch()
    return drawn, dict(
        routes=carry_ops.route_planes(drawn.read, drawn.write, drawn.gain),
        v=drawn.v.permute(0, 2, 1, 3).reshape(bh, spec.tokens, spec.dv).contiguous(),
        descriptor=descriptor,
        geometry=carry_ops.geometry_block(descriptor, launch, level_modes=spec.level_modes),
        liveness=liveness_words(drawn, descriptor),
        activity=conservative_activity(descriptor, bh, device),
        launch=launch)


def conservative_activity(descriptor, BH: int = 1, device: str = "cuda"):
    """Every page READ and WRITTEN -- the truth a caller with no facts pass may state.

    It asserts nothing about the draw and therefore skips nothing, which is exactly the
    pre-facts behaviour. A cell may tighten to the liveness pass's own bits;
    until then this is the honest conservative statement rather than an invented one.
    """
    import torch

    from rola.ops.carry import ACTIVITY_READ, ACTIVITY_WRITTEN, PAGE_LEAVES

    pages = descriptor.N // PAGE_LEAVES
    return torch.full((BH, pages), ACTIVITY_READ | ACTIVITY_WRITTEN,
                      dtype=torch.uint8, device=device)


__all__ = [
    "BACKINGS",
    "DEPOSIT_TOKEN",
    "DRAWS",
    "STATES",
    "TIERS",
    "CellSpec",
    "RealizedCell",
    "by_name",
    "carry_call",
    "carry_cells",
    "clustered",
    "conservative_activity",
    "liveness_words",
    "realize",
    "simplex",
]
