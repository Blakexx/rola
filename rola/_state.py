# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carried recurrent state: A PAGED TENSOR, and nothing more.

Storage (an arena or a dense plane), a page table, the logical shape
``[B, H, N, cols]``, a device and a dtype, and the storage-mechanics stamp
(``strategy``). Everything else that used to ride here failed one question --
*would a paged tensor know this?* -- and lives where it belongs now
(docs/internals/state.md).

    s = rola.state()                     # unbound; the first call binds the shape
    y1, s = rola.rola_op(routes1, v1, state=s)
    y2, s = rola.rola_op(routes2, v2, state=s)

``state=None`` is the stateless call and returns ``(y, None)``. That predicate --
``state is not None`` -- is the whole engagement signal at the op and at the layer.

**A fixed-size recurrent state is not a KV cache.** A KV cache grows with the
sequence; this shape does not mention ``T`` at all, which is why nothing here borrows
``transformers``' cache vocabulary (concatenation, eviction, window rolling).

Using one across a branch, or against a call it was not bound by, is the CALLER's
business exactly as it is for a ``torch`` tensor: a disagreement fails as a shape
error where the bytes are addressed, and nothing here polices it in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.engine.types import MMA_K_QUANTUM
from rola.ops.paging import AtomKeying, PageArena

__all__ = ["RoLAState", "StateFormat", "state"]

#: The page rectangle, as a bit count: an atom is the trailing
#: ``log2(MMA_K_QUANTUM)`` canonical leaf bits, and the page IS that rectangle.
PAGE_RECTANGLE_BITS = MMA_K_QUANTUM.bit_length() - 1

#: The one canonical leaf order a descriptor may name today. It is a FIELD rather
#: than an assumption so a second order is a refusal at the seam and not a silent
#: reinterpretation of somebody's bytes.
CANONICAL_ORDER = "mixed-radix-msb-first"

#: The stored form of an fp32 state value: the two halves of the word live in two
#: separate bf16 planes of the page, and ``hi || lo`` is the fp32 word exactly.
SPLIT_PLANE_DTYPE = "fp32 as split bf16 planes"

#: The per-level width floor (``B_l`` a power of two at or above 16 = ``K_max``), which
#: is what makes every span fit inside one level's run and the atom stay inside it.
MIN_LEVEL_WIDTH = 16


@dataclass(frozen=True, slots=True)
class StateFormat:
    """THE STATE OWNS ITS FORMAT: how a kernel reads these bytes.

    Bound from the FIRST call that receives the state and CHECKED against every later
    one, which is what replaces the old "later calls are NOT checked" clause: a state
    handed to a call it does not fit is refused at the seam, by name, instead of
    mis-addressing bytes somewhere inside a kernel.

    ``dtype`` is the LOGICAL element type -- fp32 -- and the split-plane form is how
    that word is laid out inside a page, not a narrowing of it
    (docs/internals/state.md#format).
    """

    D: int
    B: tuple[int, ...]
    order: str
    page_bits: int
    DV: int
    dtype: str
    ids: int

    def __post_init__(self) -> None:
        if len(self.B) != self.D:
            raise ValueError(f"a descriptor names D={self.D} levels and {len(self.B)} widths")
        for level, width in enumerate(self.B):
            if width & (width - 1) or width < MIN_LEVEL_WIDTH:
                raise ValueError(
                    f"B_l is a power of two at or above {MIN_LEVEL_WIDTH} (the axis law: the "
                    f"floor is K_max, which is what keeps every span inside one level's run and "
                    f"the atom inside one level); level {level} is {width}")
        if self.order != CANONICAL_ORDER:
            raise ValueError(f"the only leaf order this build addresses is {CANONICAL_ORDER!r}")
        if self.page_bits != PAGE_RECTANGLE_BITS:
            raise ValueError(
                f"the page rectangle is the trailing {PAGE_RECTANGLE_BITS} canonical bits")
        if self.DV % 2:
            raise ValueError(
                f"the split planes address a PAIR of value columns per 32-bit access, so DV is "
                f"even; got {self.DV}")
        if self.N % (1 << self.page_bits):
            raise ValueError(
                f"THERE IS NO RAGGED STATE: N = {self.N} is not a whole number of "
                f"{1 << self.page_bits}-leaf pages. Under the law above N = prod_l B_l with "
                f"every B_l a power of two at or above 16, so a lawful state is page aligned "
                f"by construction; a ragged one is refused here rather than padded or split "
                f"across a tail page (docs/internals/state.md#format)")

    @property
    def N(self) -> int:
        """Leaf capacity -- the mixed-radix product, derived and never configured."""
        n = 1
        for width in self.B:
            n *= width
        return n

    @property
    def cols(self) -> int:
        """``DV`` plus the mass column: the page's LOGICAL width, not its layout."""
        return self.DV + 1

    @property
    def atoms_per_bh(self) -> int:
        return self.N >> self.page_bits

    @property
    def page_bytes(self) -> int:
        """``[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]`` -- the same bytes an
        fp32 page held, re-laid out so a hi row is one line."""
        return (1 << self.page_bits) * self.cols * 4

    def check(self, other: StateFormat) -> None:
        """Refuse ``other`` against the bound format, naming the field that differs."""
        for field in ("D", "B", "order", "page_bits", "DV", "dtype", "ids"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ValueError(
                    f"this state was bound with {field}={mine!r} and the call presents "
                    f"{field}={theirs!r}. A state carries its FORMAT: its leaf order, its "
                    f"page rectangle and its value width are what every kernel derives its "
                    f"addressing from, so a call that disagrees is refused here rather than "
                    f"reading somebody else's bytes. Bind a fresh rola.state() for this shape.")


def _format_of(routes, d_v: int, BH: int) -> StateFormat:
    """The descriptor a CALL presents: the routes' widths and the value stream's DV."""
    widths = tuple(routes.topology.widths)
    n = 1
    for width in widths:
        n *= width
    return StateFormat(D=len(widths), B=widths, order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=int(d_v), dtype=SPLIT_PLANE_DTYPE,
                       ids=BH * (n >> PAGE_RECTANGLE_BITS))


#: THE ONE PAGING STRATEGY THIS VERSION BUILDS. It is a CONSTANT and not a list: one
#: value is not an axis, and a list of one is a list that reads as a choice nobody has.
_BUILT_STRATEGY = "mutable"

#: The next one, named in the design and refused BY NAME rather than by a generic
#: message, so a caller who asks for it learns it is scheduled rather than misspelled.
_DESIGNED_STRATEGY = "cow"


def state(strategy: str = "mutable", pool_slack: float = 0.0) -> RoLAState:
    """A fresh UNBOUND state. Everything but the two knobs binds at the first call.

    ``pool_slack`` is STORAGE MECHANICS, which is why it lives here and not on the op: a
    fraction of full residency, reserved per ``bh`` as physical slots a decode step can
    admit into by itself. ``0.0`` -- the default -- is exact commitment, and every growth
    step asks the host, exactly as it always did (docs/internals/state.md#the-slack-pool).
    """
    return RoLAState(strategy=strategy, pool_slack=pool_slack)


class RoLAState:
    """One sequence's carried state, mutated in place and handed back.

    Construct through :func:`rola.state`. The first stateful call binds the shape, the
    device and the backing from the call itself -- binding is SHAPE ACQUISITION, not a
    fingerprint, and there is no later compatibility interview.

    ``strategy`` is a storage-mechanics fact, the same class as a tensor's layout: it
    governs what a write does to this object's own storage. ``"mutable"`` is the built
    one -- the call writes in place and returns THIS object. Changing it is
    :meth:`clone`, never a mode flip on a live object.
    TWO INDEPENDENT AXES, never conflated. ``strategy`` is the WRITE-SEMANTICS
    contract -- what an advance does to existing references ("mutable": in
    place, old references alias the result; "cow", when built: writes never
    overwrite, every returned state stays valid, resubmission is branching).
    The BACKING (dense plane vs paged arena) is storage mechanics -- chosen by
    the plan at bind, priced in footprint, and semantically invisible (the
    dense-vs-paged torch.equal gate is that claim). All four combinations are
    coherent; dense x cow is legal-but-degenerate (no table -> a full-plane
    copy per advance).
    """

    __slots__ = ("strategy", "pool_slack", "_shape", "_device", "_paging", "_arena",
                 "_plane", "_format", "__weakref__")

    def __init__(self, *, strategy: str = "mutable", pool_slack: float = 0.0) -> None:
        if strategy == _DESIGNED_STRATEGY:
            raise NotImplementedError(
                f"strategy={strategy!r} is designed and NOT BUILT: copy-on-write state "
                "branching is named future work (s0_design.md), and its slot refcounts "
                f"and per-branch tables do not exist yet. The built strategy is "
                f"{_BUILT_STRATEGY!r}.")
        if strategy != _BUILT_STRATEGY:
            raise ValueError(
                f"strategy must be {_BUILT_STRATEGY!r}, got {strategy!r}")
        self.strategy = strategy
        self.pool_slack = float(pool_slack)
        self._shape: tuple[int, int, int, int] | None = None
        self._device: torch.device | None = None
        self._paging = True
        self._arena: PageArena | None = None
        self._plane: torch.Tensor | None = None
        self._format: StateFormat | None = None

    # -- what a caller may ask -------------------------------------------------

    def __repr__(self) -> str:
        if not self.bound:
            return f"RoLAState(strategy={self.strategy!r}, unbound)"
        return (f"RoLAState(strategy={self.strategy!r}, shape={self._shape}, "
                f"backing={'paged' if self.paged else 'dense'}, device={self._device})")

    @property
    def format(self) -> StateFormat | None:
        """This state's FORMAT DESCRIPTOR, or ``None`` while unbound.

        What every kernel derives its addressing from, and what a later call is
        checked against (docs/internals/state.md#format).
        """
        return self._format

    @property
    def bound(self) -> bool:
        """Whether a call has bound this state's shape, device and backing yet."""
        return self._shape is not None

    @property
    def shape(self) -> tuple[int, int, int, int] | None:
        """``[B, H, N, cols]``, or ``None`` while unbound."""
        return self._shape

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        """The STORAGE's dtype, or ``None`` while nothing is carried."""
        if self._arena is not None:
            return self._arena.state.dtype
        return None if self._plane is None else self._plane.dtype

    @property
    def paged(self) -> bool:
        """Whether the CURRENT backing is an arena. Not a mode: a footprint."""
        return self._arena is not None

    @property
    def carries(self) -> bool:
        """Whether anything is STORED yet. A bound state can still be empty.

        Binding fixes the shape and the backing kind; contents arrive with the first
        call that finishes. The two are separate because a call may bind a state and
        then have nothing to carry (a fresh sequence's entry state is no state at all).
        """
        return self._arena is not None or self._plane is not None

    @property
    def page_table(self) -> torch.Tensor | None:
        """The arena's ``[BH, N/16]`` table, or ``None`` for a dense backing."""
        return None if self._arena is None else self._arena.page_table

    def materialize(self) -> torch.Tensor | None:
        """`[B, H, N, cols]`, or `None` while nothing is carried. ORACLE AND BRIDGE ONLY.

        Allocating `[B, H, N, cols]` is precisely what paging exists to avoid, so
        this is for the fp64 oracle, for a cross-arm hand-off and for a test -- never
        on a measured path.
        """
        if not self.carries:
            return None
        if self._arena is not None:
            return self._arena.materialize().view(self._shape)
        return self._plane.view(self._shape)

    def clone(self, strategy: str | None = None) -> RoLAState:
        """An EAGER deep copy, and the only sanctioned strategy change.

        `clone()` returns an independent sequence: either handle may be passed into any
        op with no effect on the other, ever. The strategy chooses the MECHANISM
        (dense: plane copy; paged: referenced-slot copy; cow, future: table copy +
        refcounts) -- the contract does not vary.

        The copy shares nothing: a paged state's copy gets its own arena with the
        same residency, a dense one's its own plane. Changing strategy crosses a
        clone by rule (`s0_design.md` amendment 10 revised) -- a live ancestor is a
        standing alias onto shared slots, so a cheaper transition would hand a
        descendant a guarantee its ancestor can break.
        """
        out = RoLAState(strategy=self.strategy if strategy is None else strategy,
                        pool_slack=self.pool_slack)
        if not self.bound:
            return out
        out._shape, out._device, out._paging = self._shape, self._device, self._paging
        out._format = self._format
        if self._arena is None:
            if self._plane is not None:
                out._plane = self._plane.clone()
            return out
        arena = out._new_arena(self._arena.widths, self._arena.BC)
        arena.plan_exact(self._arena.page_table >= 0, fresh=True)
        arena.wait()
        source, destination = self._arena.page_table, arena.page_table
        present = source >= 0
        arena.state.index_copy_(
            0, destination[present].long(),
            self._arena.state.index_select(0, source[present].long()))
        out._arena = arena
        return out

    # -- the facade's seam (rola/interface.py and rola/ops/decode.py) -----------

    @property
    def _keying(self) -> AtomKeying:
        B, H, N, cols = self._shape
        return AtomKeying(N=N, cols=cols, BH=B * H)

    def _wants_pages(self, paging: bool) -> bool:
        """Whether this call needs an atom bitmap -- i.e. whether it will PAGE.

        Asked by the facade before the stats pass, so a dense-backed state does not
        pay for a bitmap nothing reads. Once bound, the state's own backing preference
        answers it; the argument is only consulted on the call that binds.
        """
        return self._paging if self.bound else bool(paging)

    def _kernel_entry(self, routes, atom_bits: torch.Tensor | None, *, d_v: int,
                      paging: bool, BC: int):
        """`(s_in, arena)` for the kernel arm, with this call's pages committed.

        The chunk DAG has already run the atom-bitmap pass, so the only thing that
        happens here is `plan_exact` on the arena's side stream -- one device-to-host
        size read, no other synchronization. The DAG's Join is what puts the
        admission before the block bitmap and the launch
        (docs/internals/engine/chunk_dag.md#the-join).

        `BC` arrives FROM THE CALL and is never stored: it is allocator policy (the
        extent default is one owner's atoms), and every ADDRESS is `BC`-invariant by
        the atom re-key, so a `BC` flip between calls neither rebuilds the arena nor
        moves a byte.
        """
        presented = _format_of(routes, d_v, routes.batch * routes.heads)
        if not self.bound:
            self._shape = (routes.batch, routes.heads, routes.topology.N, d_v + 1)
            self._device = routes.device
            self._paging = bool(paging)
            self._format = presented
        else:
            self._format.check(presented)
        from rola.engine.facts import planes

        #: RESIDENCY IS KEYED ON THE WRITE SET: the byte carries the READ bit
        #: beside it, and admitting a read-only atom would make exact commitment a
        #: superset (`rola.engine.facts.planes.written_atoms`).
        bitmap = (None if atom_bits is None else planes.written_atoms(
            atom_bits.reshape(self._keying.BH, self._keying.atoms_per_bh)))

        if self._arena is not None:
            #: a continuation: residency becomes the UNION, which is what continuing
            #: a sequence means, and the atoms already committed keep their contents.
            self._arena.plan_exact(bitmap, fresh=False)
            return self._arena.state, self._arena
        if self._plane is not None:
            return self._plane, None
        if not self._paging:
            return None, None
        arena = self._new_arena(routes.topology.widths, BC)
        arena.plan_exact(bitmap, fresh=True)
        self._arena = arena
        return None, arena

    def _kernel_commit(self, plane: torch.Tensor) -> None:
        """Take a FRESH dense plane; a continuation has nothing to take.

        Paged states already own the arena's plane, and a dense CONTINUATION launched
        with `s_out` aliased onto its own plane, so the update landed in place. The one
        call that moves a reference here is the one that BOUND a dense state
        (docs/internals/state.md#the-dense-continuation).
        """
        if self._arena is None and self._plane is None:
            self._plane = plane

    def _require_carrying(self) -> None:
        """A decode step CONTINUES; refuse a state with nothing to continue from.

        Asked by the decode family's `validate_context` before that step's first
        launch, and again at the seam below for a caller that reaches it directly.
        """
        if not self.carries:
            raise RuntimeError(
                "a decode step continues a carried state, and this one carries nothing: "
                "at T = 1 there is no prefill to run and no entry state to read. Run the "
                "sequence's first tokens through rola_op and step from the state it "
                "returns.")

    def _decode_entry(self, routes):
        """`(plane, page_table, arena)` for one step, over the residency it already has.

        A step CONTINUES a sequence, so unlike :meth:`_kernel_entry` there is no fresh
        case to serve and nothing to admit here: the step asks its own residency question
        on the device, admits from the slack pool where there is one, and where there is
        not, the admission its verdict calls for is :meth:`_decode_entry_arena`, between
        two walks of the step (docs/internals/engine/decode_dag.md#the-two-masks). The
        arena rides along for its pool buffers alone; a dense state has none.

        Dense states hand back a VIEW of their own plane, which is what makes the
        kernel's in-place update the state's update.

        The step's routing is CHECKED against the bound format first: a decode step
        derives its lattice and its unit from the descriptor, so a routing that
        disagrees is refused here rather than walked (docs/internals/state.md#format).
        """
        self._require_carrying()
        self._format.check(_format_of(routes, self._format.DV,
                                      self._shape[0] * self._shape[1]))
        if self._arena is not None:
            return self._arena.state, self._arena.page_table, self._arena
        return self._plane.view(self._shape), None, None

    def _decode_entry_arena(self, bitmap: torch.Tensor):
        """Admit `bitmap`'s atoms into the carried arena and hand it back.

        The growth half of :meth:`_decode_entry`: the gated step has already run as a
        no-op, so admission is all that is left before it is replayed
        (docs/internals/decode/decode.md#capture). The arena's own plane and table are
        untouched objects -- the base never moves and the table is written in place --
        which is what keeps a CUDA graph captured over them valid across an admission.
        """
        if self._arena is None:
            raise RuntimeError("a growth admission wants a paged state; this one is dense")
        self._arena.plan_exact(
            bitmap.reshape(self._keying.BH, self._keying.atoms_per_bh).bool(), fresh=False)
        return self._arena

    # -- backing internals -----------------------------------------------------

    def _new_arena(self, widths: tuple[int, ...], BC: int) -> PageArena:
        keying = self._keying
        return PageArena(widths=widths, BC=int(BC), cols=keying.cols, BH=keying.BH,
                         device=self._device, pool_slack=self.pool_slack)
