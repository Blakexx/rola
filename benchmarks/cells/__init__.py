# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CELL REGISTRY: the carry family's cells as DATA, with one draw behind them.

`carry_cells.json` holds the records; this module reads them, derives each cell's
STATE DESCRIPTOR and LAUNCH SHAPE from its fields, realizes its routing draw, and builds
`carry_call` -- one cell's WHOLE call. ONE definition, three readers: the oracle tier,
`tools/probe_cells.py` and the benches all reach the kernel through it, so a cell's
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

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

_REGISTRY = Path(__file__).resolve().parent / "carry_cells.json"

DRAWS = ("dense", "alt", "both", "cohort", "dead", "deposit")
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

    @property
    def seed(self) -> int:
        return zlib.crc32(self.name.encode()) & 0xFFFF

    @property
    def D(self) -> int:
        return len(self.widths)

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
        if self.draw in ("dense", "dead", "deposit") or self.k_tok is None:
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
    spec = CellSpec(name=record["name"], widths=tuple(record["widths"]), dv=record["dv"],
                    tokens=record["tokens"], warps_per_cta=record["warps_per_cta"],
                    draw=record["draw"], k_tok=record["k_tok"], cohort=record["cohort"],
                    support=record["support"], backing=record["backing"],
                    state=record["state"], tier=record["tier"])
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
    if spec.draw in ("alt", "both", "cohort") and spec.k_tok is None:
        raise ValueError(f"{spec.name}: a sparse draw states its k_tok")
    if spec.draw in ("dead", "deposit") and spec.k_tok is not None:
        raise ValueError(f"{spec.name}: a {spec.draw} draw has no k_tok -- its support is "
                         f"stated by the draw itself, not by a count")
    return spec


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

    def doubles(self):
        return (tuple(x.double() for x in self.read), tuple(x.double() for x in self.write),
                self.gain.double(), self.v.double())


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

    if spec.draw in ("dead", "deposit"):
        return _degenerate(spec, shape, gen, device)

    kt = spec.k_tok
    if spec.draw == "dense":
        read_k = write_k = [None] * spec.D
    elif spec.draw == "both":
        read_k = write_k = [kt if l == 0 else None for l in range(spec.D)]
    else:
        read_k = [kt if l % 2 else None for l in range(spec.D)]
        write_k = [None if l % 2 else kt for l in range(spec.D)]

    read = tuple(draw(w, read_k[l], live[l]) for l, w in enumerate(spec.widths))
    write = tuple(draw(w, write_k[l], live[l]) for l, w in enumerate(spec.widths))
    gain = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(*shape, spec.dv, device=device, dtype=torch.float64, generator=gen)
    return RealizedCell(spec=spec,
                        read=tuple(x.to(torch.bfloat16) for x in read),
                        write=tuple(x.to(torch.bfloat16) for x in write),
                        gain=gain.to(torch.bfloat16), v=v.to(torch.bfloat16))


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

    THIS IS THE ONE BINDING. The oracle tier, the benches and `tools/probe_cells.py`
    all reach the kernel through it, so a cell's operands are the same tensors in a
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
