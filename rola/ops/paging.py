"""Page commitment, KEYED TO THE MMA ATOM (docs/internals/paging/).

What this module is for, in one sentence: make the state a layer actually keeps
resident scale with the atoms its routing REALIZES, instead of with the ``N`` its
topology PROVISIONS.

Vocabulary, defined here and used unqualified: ``N`` = leaf count; ``BC`` = leaves per
owner block (the owner rectangle); ``cols`` = ``d_v`` plus one mass column (the ratio
readout's denominator); ``BH`` = ``batch * heads``; ``MMA_K_QUANTUM = 16``;
**atom** = one ``MMA_K_QUANTUM``-leaf block in canonical leaf order;
``QPO = BC / MMA_K_QUANTUM`` = atoms per owner.

The atom re-key rationale, the address law, planned-exact-async allocation,
extent policy and scope live in docs/internals/paging/paging.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

import torch

from rola.engine.rules.paging import admission
from rola.engine.types import MMA_K_QUANTUM

#: ``log2(MMA_K_QUANTUM)``, DERIVED. THE ADDRESS INVARIANT: the address expression is
#: ``(slot << _LOG2_ATOM_LEAVES | row)``, and the shift must be an expression in the
#: named constant rather than the literal 4 -- otherwise a future atom size passes every
#: instantiated test and silently mis-addresses.
_LOG2_ATOM_LEAVES = MMA_K_QUANTUM.bit_length() - 1
assert 1 << _LOG2_ATOM_LEAVES == MMA_K_QUANTUM, (
    "MMA_K_QUANTUM must be a power of two for the shift form of the page address")

#: Mask for the intra-atom row, DERIVED for the same reason.
_ATOM_ROW_MASK = MMA_K_QUANTUM - 1

_ABSENT_PAGE = -1


def to_split_planes(logical: torch.Tensor) -> torch.Tensor:
    """A logical ``[..., 16, cols]`` fp32 page block in the STORED split-plane form.

    THE PAGE IS ``[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]``:
    the same fp32 bits, re-laid out so a value row's hi plane is one 128 B line at
    ``DV = 64`` and an atom a call only READS moves half the bytes. The container stays
    fp32 because the LOGICAL element is fp32 -- ``hi || lo`` is the word exactly -- and
    the state's descriptor is what declares the layout (docs/internals/state.md#format).
    """
    _refuse_ragged(logical)
    _refuse_dtype(logical)
    pages = logical.reshape(-1, MMA_K_QUANTUM, logical.shape[-1]).contiguous()
    d_v = logical.shape[-1] - 1
    words = pages.view(torch.int32)
    hi, lo = (words >> 16).to(torch.int16), words.to(torch.int16)
    out = torch.empty_like(pages)
    stored = out.view(torch.int16).reshape(pages.shape[0], -1)
    rows = MMA_K_QUANTUM * d_v
    stored[:, :rows] = hi[:, :, :d_v].reshape(-1, rows)
    stored[:, rows:2 * rows] = lo[:, :, :d_v].reshape(-1, rows)
    stored[:, 2 * rows:2 * rows + MMA_K_QUANTUM] = hi[:, :, d_v]
    stored[:, 2 * rows + MMA_K_QUANTUM:] = lo[:, :, d_v]
    return out.reshape(logical.shape)


def _refuse_dtype(state: torch.Tensor) -> None:
    """The stored form is 32-bit words; a wider or narrower element is not a page."""
    if state.dtype is not torch.float32:
        raise ValueError(
            f"a state page is fp32 words split into bf16 planes, so the pack and the "
            f"gather run on float32; got {state.dtype}. Convert AFTER the gather, never "
            f"before it (docs/internals/state.md#split-planes)")


def _refuse_ragged(state: torch.Tensor) -> None:
    """Refuse a state whose leaf axis is not a whole number of pages.

    THERE IS NO RAGGED STATE (Blake, 2026-08-29). A lawful state has
    ``N = prod_l B_l`` with every ``B_l`` a power of two at or above 16, so ``N`` is
    page-aligned by construction; a leaf axis that is not is a state no descriptor can
    name, and it is refused here rather than padded, split across a tail page, or
    served by a second layout (docs/internals/state.md#format).
    """
    n = state.shape[-2] if state.dim() >= 2 else 0
    if n % MMA_K_QUANTUM:
        raise ValueError(
            f"a state's leaf axis is a whole number of {MMA_K_QUANTUM}-leaf pages; got "
            f"{n}. N = prod_l B_l with each B_l a power of two at or above 16, so a "
            f"lawful state is page-aligned by construction (docs/internals/state.md#format)")


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte identity of two STORED page blocks -- the split format's own comparison.

    ``torch.equal`` is a FLOAT comparison, and a stored page is not a float: its 32-bit
    words are two neighbouring elements' halves, so a page pair that is byte-identical
    can contain a word whose float reading is NaN and compare UNEQUAL. The claim the
    backings make is about bytes, so it is asserted on the integer view -- which is
    also strictly stronger than the fp32 ``torch.equal`` it replaces.
    """
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32),
                                              b.contiguous().view(torch.int32))


def from_split_planes(stored: torch.Tensor) -> torch.Tensor:
    """The hi||lo GATHER: stored pages back as the logical ``[..., 16, cols]`` fp32.

    The inverse of :func:`to_split_planes`, and exactly what ``materialize()`` is.
    """
    _refuse_dtype(stored)
    pages = stored.reshape(-1, MMA_K_QUANTUM, stored.shape[-1]).contiguous()
    d_v = stored.shape[-1] - 1
    halves = pages.view(torch.int16).reshape(pages.shape[0], -1).to(torch.int32) & 0xFFFF
    rows = MMA_K_QUANTUM * d_v
    words = torch.empty((pages.shape[0], MMA_K_QUANTUM, d_v + 1), dtype=torch.int32,
                        device=pages.device)
    words[:, :, :d_v] = ((halves[:, :rows] << 16)
                         | halves[:, rows:2 * rows]).reshape(-1, MMA_K_QUANTUM, d_v)
    words[:, :, d_v] = ((halves[:, 2 * rows:2 * rows + MMA_K_QUANTUM] << 16)
                        | halves[:, 2 * rows + MMA_K_QUANTUM:])
    return words.view(torch.float32).reshape(stored.shape)


#: The driver-allocation unit the VMM backing commits in, in bytes. ``None`` resolves it
#: to the device's own ``cuMemGetAllocationGranularity`` MINIMUM, which is the finest
#: commitment the hardware admits: a smaller target would be rounded back up to it and a
#: larger one would over-commit for no reason. It is a knob only so a measurement can
#: coarsen it deliberately.
C_VMM_CHUNK_BYTES: int | None = None

#: ``C_EXTENT_ATOMS``, a CALIBRATION PLACEHOLDER. The pool is carved into runs of this many
#: CONSECUTIVE slots, and atoms admitted together take consecutive ids inside a run.
#: FILLED BY: the PCIe/HBM batch-efficiency curve.
#:
#: DEFAULT: ``None``, resolved per-arena to ``QPO`` -- one owner's worth of atoms. The
#: stated default, and it is the value at which the policy degrades to today's
#: behaviour exactly: an owner's atoms land contiguous, which is what a ``BC``-keyed
#: page gave for free, so a wrong placeholder cannot be worse than the thing this
#: replaces.
C_EXTENT_ATOMS: int | None = None


_VMM_PROBE: dict[int, dict] = {}


def _vmm_probe(device: torch.device) -> dict:
    """The driver's own capability verdict for one device, proven ONCE.

    ``rola_vmm_probe`` does not query an attribute and believe it: it reserves, maps,
    grants access to, unmaps and releases one granularity-sized allocation in this
    context, and reports ``supported`` only if every step returned success. Caching it
    per device index is what keeps a per-arena question off the arena's constructor.
    """
    from rola.ops._ext import extension

    index = device.index if device.index is not None else torch.cuda.current_device()
    probe = _VMM_PROBE.get(index)
    if probe is None:
        probe = extension().rola_vmm_probe(device=index)
        _VMM_PROBE[index] = probe
    return probe


def vmm_supported(device: torch.device | str) -> bool:
    """Whether this device's driver can serve the VMM backing."""
    device = torch.device(device)
    return device.type == "cuda" and bool(_vmm_probe(device)["supported"])


def _resolve_backing(backing: str, device: torch.device) -> str:
    if backing not in ("auto", "vmm", "dense"):
        raise ValueError(f"backing must be 'auto', 'vmm' or 'dense', got {backing!r}")
    if backing == "dense":
        return "dense"
    if device.type != "cuda":
        if backing == "vmm":
            raise ValueError(
                f"the VMM backing is CUDA-driver memory and this arena is on {device}; "
                "a host arena has one backing and it is the dense allocation")
        return "dense"
    probe = _vmm_probe(device)
    if not probe["supported"]:
        raise RuntimeError(
            f"the page arena's VMM backing is unavailable on {device}: {probe['reason']}. "
            "Paging's physical-memory claim IS the driver commitment, so the arena "
            "refuses rather than falling back to a dense allocation that would save "
            "nothing and still be reported as paged. Pass backing='dense' to ask for the "
            "unpaged footprint deliberately.")
    return "vmm"


@dataclass(frozen=True, slots=True)
class AtomKeying:
    """The paged addressing arithmetic, as ONE object so it has ONE derivation.

    Every consumer of the page table -- this module, the admission bitmap, the residency
    report, and (once it lands) the CTA's startup slot resolution -- computes the same
    three expressions. Carrying them here rather than inlining them three times is what
    makes the ``BC``-invariance claim below checkable instead of asserted.

    **THE PROPERTY THAT MATTERS, and it is why the re-key exists:** every field of this
    object is invariant in ``BC``. ``atoms_per_bh``, ``atom_of``, ``row_of`` and
    ``address`` mention only ``N``, ``cols`` and ``MMA_K_QUANTUM``. So two launches of
    the same sequence at different ``BC`` address the SAME bytes of the SAME slots, and
    a ``BC`` flip between them is a no-op for residency -- which is exactly what
    ``BC``-per-launch needs and what a ``BC``-keyed page could not give.
    """

    N: int
    cols: int
    BH: int

    def __post_init__(self) -> None:
        if self.N % MMA_K_QUANTUM:
            raise ValueError(
                f"N={self.N} must be a whole number of {MMA_K_QUANTUM}-leaf atoms "
                "(the page granule is geometry-owned and fixed across every "
                "config forever, so a topology whose leaf count is not a multiple of it "
                "has no page keying at all)")

    @property
    def atoms_per_bh(self) -> int:
        return self.N // MMA_K_QUANTUM

    @property
    def logical_atoms(self) -> int:
        return self.BH * self.atoms_per_bh

    def atom_of(self, bh: int, leaf: int) -> int:
        """``bh * (N / MMA_K_QUANTUM) + leaf // MMA_K_QUANTUM`` -- the atom key."""
        return bh * self.atoms_per_bh + (leaf >> _LOG2_ATOM_LEAVES)

    def row_of(self, leaf: int) -> int:
        """``leaf & (MMA_K_QUANTUM - 1)`` -- the leaf's row inside its atom."""
        return leaf & _ATOM_ROW_MASK

    def address(self, slot: int, leaf: int, v: int) -> int:
        """``((slot << LOG2_ATOM | row) * cols + v)`` -- the value address.

        The ``|`` is an OR and not an ADD because ``row < MMA_K_QUANTUM`` by
        construction, so the low bits of the shifted slot are provably clear; writing it
        as an OR is what makes that a stated invariant rather than an accident.
        """
        return ((slot << _LOG2_ATOM_LEAVES) | self.row_of(leaf)) * self.cols + v

    def atoms_per_owner(self, BC: int) -> int:
        """``QPO``. The ONE place ``BC`` meets the keying, and it meets it as a COUNT."""
        if BC % MMA_K_QUANTUM:
            raise ValueError(
                f"BC={BC} must be a whole number of {MMA_K_QUANTUM}-leaf atoms; every "
                "legal BC is by construction")
        return BC // MMA_K_QUANTUM


@dataclass(frozen=True, slots=True)
class PagingResult:
    """One call's committed-state accounting -- the reported page-commitment quantities."""

    #: Slots the allocator has handed out after this call's publication -- the CURSOR,
    #: which is the residency PLUS whatever alignment gap :class:`ExtentAllocator` opened
    #: (a placement tendency the class refuses to promise as an invariant). It is what
    #: sizes the physical commitment, and it is a host integer. The EXACT residency is
    #: the page table's popcount, :attr:`PageArena.resident_pages`.
    allocated_pages: int
    #: High-water mark of :attr:`allocated_pages` over the arena's lifetime.
    peak_pages: int
    #: Logical atoms this call's routing actually realized on the write side -- the
    #: popcount of the atom condensation, which is EXACT rather than a bound.
    realized_owners: int
    #: ``atoms_per_bh``: what a dense ``[N, cols]`` state would provision per ``bh``.
    owners_total: int
    #: Bytes of fp32 state the allocator has handed out.
    allocated_bytes: int
    #: Bytes a dense ``[BH, N, cols]`` fp32 state would occupy -- the counterfactual the
    #: feature exists to beat.
    dense_bytes: int
    #: PHYSICAL bytes the CUDA driver has mapped. This is the receipt: under the VMM
    #: backing it is :attr:`allocated_bytes` rounded up to the owner's chunk granularity
    #: (plus its one-chunk lookahead), and under the dense bridge it is
    #: :attr:`dense_bytes`, because the bridge's one allocation IS the dense limit.
    committed_bytes: int

    @property
    def allocated_fraction(self) -> float:
        return self.allocated_bytes / self.dense_bytes if self.dense_bytes else 0.0

    @property
    def committed_fraction(self) -> float:
        """The savings claim, as the driver measures it."""
        return self.committed_bytes / self.dense_bytes if self.dense_bytes else 0.0


class ExtentAllocator:
    """The slot policy: consecutive runs, and NOTHING the table or kernel can see.

    The pool is a sequence of EXTENTS of ``extent_atoms`` consecutive slots. A batch of
    atoms admitted together is placed into as few extents as possible, in ascending
    order, so the common case -- an owner's ``QPO`` atoms admitted in one call -- lands
    contiguous and the CTA resolves ONE base instead of ``QPO`` independent lookups.

    **INVISIBILITY IS THE CONTRACT.** The table format is ``atom -> slot`` either way,
    and no consumer may infer contiguity: a caller that read slot ids and assumed a run
    would break the moment a partially-filled extent forces a split. What the policy
    promises is a TENDENCY, priced by the transfer batching, not an invariant.
    """

    def __init__(self, capacity: int, extent_atoms: int) -> None:
        if extent_atoms < 1:
            raise ValueError(f"extent_atoms must be >= 1, got {extent_atoms}")
        self.capacity = int(capacity)
        self.extent_atoms = int(extent_atoms)
        self._next = 0

    @property
    def allocated(self) -> int:
        return self._next

    @property
    def free(self) -> int:
        return self.capacity - self._next

    def reset(self) -> None:
        self._next = 0

    def take(self, count: int) -> int:
        """Advance the cursor over the run :func:`rola.engine.rules.paging.admission` placed.

        The PLACEMENT -- the extent alignment and the all-or-nothing ceiling -- is the
        rule; what is left here is the one mutation, which is why this class holds a
        cursor and no policy (docs/internals/engine/rules.md).
        """
        first = admission(self._next, self.capacity, self.extent_atoms, count)
        if count > 0:
            self._next = first + count
        return first


class PageArena:
    """A pooled, per-layer arena over the value panels, keyed by the MMA ATOM.

    **Pooled, not per-(batch, head)** -- the shape, and the reason the savings math
    works (docs/internals/paging/paging.md). One arena backs every ``(batch, head)`` pair, so it tracks the live
    fraction of the WHOLE layer; a per-pair arena would round each pair's tiny live
    fraction up to its own extent and cap savings at the extent-to-footprint ratio.

    **THERE IS NO CEILING PARAMETER.** The arena's allocator capacity IS the dense
    limit (``BH * atoms_per_bh``) and it COMMITS only what :meth:`plan_exact` admits,
    which is the routing's exact write set. A cap over a planned-exact allocator is
    either never-binding or a self-inflicted refusal, and the storage it would have
    sized is the storage paging exists not to touch.

    **THE SLACK POOL (``pool_slack``, a FRACTION of full residency, default ``0.0``).**
    ``ceil(pool_slack * atoms_per_bh)`` slots are reserved PER ``bh`` up front, committed
    and zeroed, and a decode step admits into them ITSELF, on the device, with no host
    round trip. They are ordinary slots from this same allocator, so an atom admitted
    from the pool is indistinguishable from one :meth:`plan_exact` admitted -- which is
    what keeps the dense-vs-paged equality a claim about addressing and not about who
    allocated. ``0.0`` is EXACT COMMITMENT: no pool exists, and every growth step goes to
    the host exactly as it always did. Exhaustion degrades to that same path
    (docs/internals/paging/paging.md#the-slack-pool).

    **THE TWO BACKINGS, AND WHY NEITHER IS A FALLBACK.** ``backing='vmm'`` reserves the
    dense limit as ADDRESS SPACE (free) and asks the CUDA driver for physical memory only
    as the allocator hands out slots -- which is the entire physical-memory promise of
    paging, and the reason :attr:`committed_bytes` is a receipt rather than a restatement
    of the dense limit. ``backing='dense'`` is one ``torch.zeros`` at the dense limit: it
    is the CPU backing (a host device has no driver VMM at all) and the instrument the
    equivalence gates compare against, never a rescue path. ``backing='auto'`` -- the
    default -- resolves to ``'vmm'`` on CUDA and ``'dense'`` off it, and a CUDA device
    whose driver cannot serve VMM RAISES rather than silently degrading (the closed-world
    rule: a measurement must not be able to quote the bridge as the feature). The
    capability is proven once per device by :func:`vmm_supported`, which runs the
    driver's full reserve/map/access round trip, not an attribute query.

    **NOTE ON ``BC``.** The arena takes it only to resolve the default
    ``extent_atoms`` (``C_EXTENT_ATOMS`` defaults to ``QPO``) and to report the
    owner geometry a caller reasons about. Nothing it STORES and nothing the KERNEL
    ADDRESSES is ``BC``-keyed after the paired re-key, so two calls at different ``BC``
    share slots byte for byte and an arena is never rebuilt on a ``BC`` flip. That is
    the property ``BC``-per-launch needs and the reason the re-key exists;
    :attr:`atoms_per_owner`, :attr:`owners_total` and :attr:`extent_atoms` are
    allocator POLICY and reporting, evaluated at the construction ``BC``, and a launch
    at a different ``BC`` neither reads nor invalidates them.
    """

    def __init__(self, *, widths: tuple[int, ...], BC: int, cols: int, BH: int,
                 device: torch.device, extent_atoms: int | None = None,
                 backing: str = "auto", pool_slack: float = 0.0) -> None:
        N = prod(widths)
        self.widths = tuple(widths)
        self.BC = int(BC)
        self.cols = int(cols)
        self.BH = int(BH)
        self.N = int(N)
        self.keying = AtomKeying(N=self.N, cols=self.cols, BH=self.BH)
        self.atoms_per_owner = self.keying.atoms_per_owner(self.BC)
        self.owners_total = N // self.BC
        self.device = torch.device(device)
        self.logical_pages = self.keying.logical_atoms

        resolved_extent = (self.atoms_per_owner if extent_atoms is None
                           else int(extent_atoms))
        if C_EXTENT_ATOMS is not None and extent_atoms is None:
            resolved_extent = int(C_EXTENT_ATOMS)
        self.extent_atoms = resolved_extent
        self._allocator = ExtentAllocator(self.logical_pages, self.extent_atoms)

        if not 0.0 <= float(pool_slack) <= 1.0:
            raise ValueError(
                f"pool_slack is a FRACTION of full residency and must lie in [0, 1], got "
                f"{pool_slack!r}")
        self.pool_slack = float(pool_slack)
        #: Per ``bh``, because the device claim is per ``bh``: every CTA of a batch-head
        #: derives its slots from that batch-head's own region and cursor, which is what
        #: removes the cross-CTA communication a shared cursor would need
        #: (docs/internals/paging/paging.md#the-slack-pool).
        self.pool_capacity = ceil(self.pool_slack * self.keying.atoms_per_bh)

        self.backing = _resolve_backing(backing, self.device)

        self.peak_pages = 0
        #: The most recent :class:`PagingResult`, or ``None``.
        self.last_result: PagingResult | None = None
        #: The arena's side stream and the ONE event the consumer launch waits on.
        self._stream = (torch.cuda.Stream(device=self.device)
                        if self.device.type == "cuda" else None)
        self._ready: torch.cuda.Event | None = None
        self._owner = None
        self._plane_view: torch.Tensor | None = None
        self._allocate()

    # -- storage ----------------------------------------------------------

    def _allocate(self) -> None:
        self._page_table = torch.full((self.BH, self.keying.atoms_per_bh), _ABSENT_PAGE,
                                      dtype=torch.int32, device=self.device)
        #: ``[slots, MMA_K_QUANTUM, cols]`` -- exactly the shape
        #: :meth:`AtomKeying.address` indexes, so the address expression and the
        #: allocation cannot disagree about what a slot is. Under the VMM backing the
        #: view spans the whole RESERVED extent and :attr:`state` bounds it to the mapped
        #: prefix; the base pointer is the reservation's and never moves.
        if self.backing == "vmm":
            from rola.ops._ext import extension

            self._owner = extension().rola_vmm_create(
                device=self.device.index if self.device.index is not None
                else torch.cuda.current_device(),
                dense_limit_pages=self.logical_pages,
                page_rows=MMA_K_QUANTUM,
                page_cols=self.cols,
                target_chunk_bytes=(C_VMM_CHUNK_BYTES if C_VMM_CHUNK_BYTES is not None
                                    else _vmm_probe(self.device)["allocation_granularity"]))
            self._value_base = self._owner.base()
        else:
            self._value_base = torch.zeros(
                (self.logical_pages, MMA_K_QUANTUM, self.cols),
                dtype=torch.float32, device=self.device)
        self._pool_slots = torch.empty((self.BH, self.pool_capacity), dtype=torch.int32,
                                       device=self.device)
        self._pool_cursor = torch.zeros(self.BH, dtype=torch.int32, device=self.device)
        #: The device's own claim, read by the walk until the step folds it into the
        #: table. Sized like the table because that is what it indexes.
        self._pool_map = torch.full((self.BH, self.keying.atoms_per_bh), _ABSENT_PAGE,
                                    dtype=torch.int32, device=self.device)
        self._cursor_host = torch.empty(self.BH, dtype=torch.int32,
                                        pin_memory=self.device.type == "cuda")
        self._reserve_pool()

    def _reserve_pool(self) -> None:
        """Hand the pool `pool_capacity` slots per `bh`, COMMITTED AND ZEROED.

        A pool slot is an ORDINARY arena slot -- taken from the same allocator, physically
        present before it is named, and holding zeros -- so an atom the device admits into
        one is indistinguishable from an atom `plan_exact` admitted, and no migration or
        copy exists anywhere in the design.
        """
        if not self.pool_capacity:
            return
        total = self.BH * self.pool_capacity
        first = self._allocator.take(total)
        self._commit(self._allocator.allocated)
        self._value_base[first:first + total].zero_()
        self._pool_slots.copy_(
            torch.arange(first, first + total, dtype=torch.int32,
                         device=self.device).view(self.BH, self.pool_capacity))
        self._pool_cursor.zero_()
        self.peak_pages = max(self.peak_pages, self._allocator.allocated)

    def _commit(self, pages: int) -> None:
        """Make ``pages`` slots PHYSICALLY exist. The bridge's already do.

        Called before every :meth:`_admit_pages`, because admission's zero-init is the
        first write those slots see. The owner grows to a chunk boundary and never
        shrinks here -- a base pointer that moved would invalidate every launch the state
        has in flight, and the reservation is what makes never-moving free.
        """
        if self._owner is not None:
            self._owner.grow(pages)

    # -- properties -------------------------------------------------------

    @property
    def page_table(self) -> torch.Tensor:
        """``[BH, atoms_per_bh]`` int32; ``-1`` means the atom has no slot."""
        return self._page_table

    @property
    def state(self) -> torch.Tensor:
        """``[slots, MMA_K_QUANTUM, cols]`` -- what the consumer indexes.

        Bounded to :attr:`mapped_pages`, which is the whole pool under the dense bridge
        and the driver's mapped prefix under the VMM backing. The leading extent is a
        CAPACITY and nothing addresses through it: every slot a table names is below the
        allocator's cursor, which is below the mapped prefix by construction. Bounding it
        is what makes reaching past the commitment an index error rather than an
        unmapped-address fault.

        The view is CACHED and rebuilt only when the commitment moves, so a caller may
        hold it across steps and compare it by identity -- which is what "the kernel's
        in-place update IS the state's update" means at the seam.
        """
        mapped = self.mapped_pages
        if self._plane_view is None or self._plane_view.shape[0] != mapped:
            self._plane_view = self._value_base[:mapped]
        return self._plane_view

    @property
    def mapped_pages(self) -> int:
        """Slots that PHYSICALLY exist. The dense bridge's are all of them."""
        return (self.logical_pages if self._owner is None
                else int(self._owner.mapped_capacity_pages))

    @property
    def allocated_pages(self) -> int:
        """The allocator's CURSOR -- slots handed out, alignment gaps included.

        HOST-SIDE and therefore free, which is why it and not the residency is what
        :meth:`plan_exact` reports and what sizes the physical commitment. It is an upper
        bound on the residency, never equal to it in general.
        """
        return self._allocator.allocated

    @property
    def resident_pages(self) -> int:
        """The EXACT residency: atoms the page table maps.

        A device reduction and a host read, so this is a REPORTING surface and never
        appears on a measured path -- :attr:`allocated_pages` is what the arena's own
        logic runs on.
        """
        return int((self._page_table >= 0).sum())

    def dense_bytes(self) -> int:
        return self.BH * self.N * self.cols * 4

    def allocated_bytes(self) -> int:
        return self.allocated_pages * MMA_K_QUANTUM * self.cols * 4

    @property
    def committed_bytes(self) -> int:
        """PHYSICAL bytes the driver has mapped -- the arena's own accounting.

        The receipt the savings claim rests on, taken from the owner rather than from a
        process-wide instrument: ``nvidia-smi`` reports the caching allocator's
        reservations too and cannot separate this arena from them.
        """
        return (self.dense_bytes() if self._owner is None
                else int(self._owner.committed_bytes))

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Drop every committed atom. The arena is reusable afterwards.

        The page STORAGE is deliberately not zeroed here. Dropping the table makes every
        atom unmapped, so the next admission sees each of them as NEW and
        :meth:`_admit_pages` zeroes it before any read can reach it. Blanket-zeroing
        the whole pool on every reset would touch the DENSE footprint -- exactly the allocation paging exists to avoid -- to redo work
        admission already does on the realized set only.
        """
        # JOIN THE SIDE STREAM FIRST. The fills below run on the CALLER's stream, and
        # the previous call published the page table and zeroed its slots on the side
        # stream. Two plans back to back with no consumer launch between them -- which
        # the arena gates do -- would otherwise have this reset race that publication.
        # A stream-to-stream wait, so it costs the host nothing.
        self.wait()
        self._page_table.fill_(_ABSENT_PAGE)
        self._allocator.reset()
        self.peak_pages = 0
        self._ready = None
        #: The pool's slots came from the allocator that was just rewound, so they are
        #: re-taken here rather than kept: a pool naming slots the allocator may hand out
        #: again is the one way a device claim could alias a host admission.
        self._reserve_pool()

    # -- THE ONE DATA-MOVEMENT BOUNDARY -----------------------------------

    def _admit_pages(self, atom_ids: torch.Tensor, first_slot: int) -> None:
        """THE narrow data-movement boundary. Everything that MOVES BYTES lives here.

        ---- CONTRACT -------------------------------------------------------

        INPUTS
          ``atom_ids``  ``[k]`` int64, device-resident, STRICTLY ASCENDING logical atom
                        ids (``bh * atoms_per_bh + leaf // MMA_K_QUANTUM``), each one
                        currently ABSENT from the table. Ascending order is what makes
                        admission order observable in the slot ids, and it is the
                        caller's to establish.
          ``first_slot`` the physical slot the run starts at, already reserved from
                        :class:`ExtentAllocator`. Slots ``[first_slot, first_slot + k)``
                        are guaranteed free and inside the ceiling.

        EFFECTS, and exactly these two:
          1. ``table[atom_ids] = first_slot + arange(k)`` (the gather-scatter);
          2. ``state[first_slot : first_slot + k] = 0`` (the zero-init, which is what
             makes a fresh atom read as a zero state rather than as a stale one).

        There is no third effect any more. The admit mask used to be set here, one bit
        per admitted atom; with the capped mode deleted its bit is set for exactly the
        atoms the table maps, so it carried no information the table did not and it was
        removed end-to-end (arena, binding and kernel).

        STREAM / EVENT DISCIPLINE
          Both run on the arena's SIDE STREAM, which waits on the current stream
          first (the atom bitmap it was handed is produced there). The caller then
          records ONE event and the consumer launch waits on it. Nothing here
          synchronizes the host: there is no ``.item()``, no ``.cpu()``, no
          ``nonzero()`` whose length the host must learn.

        NO HOST SYNCS -- and that is why ``k`` is a HOST integer parameter rather than
        something read off a device tensor. The caller learns it from the planned-exact
        popcount it already performed for sizing.

        ---- WHY IT IS ONE FUNCTION -----------------------------------------

        These two operations are torch ops today and want to be ONE custom CUDA
        gather-scatter tomorrow: a single kernel that writes the table and zeroes the
        slots in one pass over ``atom_ids``, with the zeroing
        vectorized over the ``[k, MMA_K_QUANTUM, cols]` run. Confining them behind one
        boundary with a stated contract is what makes that a DROP-IN replacement rather
        than an archaeology exercise -- the same relationship the summary kernel has to
        its torch reference. Scattering these calls through the arena logic would make
        the eventual kernel a rewrite of the arena.
        """
        stream = self._stream
        context = (torch.cuda.stream(stream) if stream is not None
                   else _NullContext())
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.device))
        with context:
            count = int(atom_ids.numel())
            if count:
                if stream is not None:
                    # ``atom_ids`` was allocated on the CALLER's stream and is read
                    # here on the side stream. Without this the caching allocator may
                    # hand its bytes to another tensor the moment the caller's stream
                    # drops the reference -- while the `index_copy_` below is still
                    # reading them -- and the indices become garbage. That is not a
                    # theoretical hazard: it fired as `index_copy_(): index out of
                    # bounds` from this exact line. `record_stream` is the documented
                    # cross-stream lifetime contract and is the whole fix.
                    atom_ids.record_stream(stream)
                slots = torch.arange(first_slot, first_slot + count,
                                     dtype=torch.int32, device=self.device)
                self._page_table.view(-1).index_copy_(0, atom_ids, slots)
                self._value_base[first_slot:first_slot + count].zero_()

    # -- planning ---------------------------------------------------------

    def plan_exact(self, atom_bitmap: torch.Tensor, *,
                   fresh: bool = True) -> PagingResult:
        """The PLANNED-EXACT-ASYNC admission, from the atom condensation.

        ``atom_bitmap`` is ``[BH, atoms_per_bh]`` bool -- the per-leaf write set
        condensed to the atom quantum, as :func:`rola.engine.facts.planes.atom_bits` publishes it. It
        is the EXACT set of atoms this call touches on the write side, so the pool is
        SIZED rather than grown and there is nothing for an allocator to discover.

        ``fresh`` (default ``True``): reset first, so this call is call-scoped -- it
        computes as if nothing preceded it, matching the unpaged path's "allocate zeros
        every call" semantics. ``fresh=False`` is the explicit, named opt-in for a
        legitimate continuation (successive chunks of one sequence). There is no
        implicit third option.

        **ADMISSION IS TOTAL.** Atoms are admitted in ascending logical id -- which is
        ascending ``(bh, leaf)``, and therefore ascending first-write within a ``bh``
        under the canonical leaf order -- and EVERY one of them is admitted. If the
        ceiling cannot cover the demand the allocator RAISES; there is no prefix grant
        and no denied atom, because the capped mode that produced them was deleted.
        """
        if fresh:
            self.reset()
        expected = (self.BH, self.keying.atoms_per_bh)
        if tuple(atom_bitmap.shape) != expected:
            raise ValueError(
                f"the atom condensation must be {expected} (the derived atom "
                f"granularity); got {tuple(atom_bitmap.shape)}")

        # The atoms that are touched and not already resident. `nonzero` costs one
        # device->host size read; it is the ONE the planned-exact contract admits,
        # because the count is also what sizes the allocation.
        wanted = atom_bitmap.reshape(-1) & (self._page_table.reshape(-1) < 0)
        atom_ids = wanted.nonzero(as_tuple=True)[0]
        realized = int(atom_bitmap.sum())
        first_slot = self._allocator.take(int(atom_ids.numel()))
        #: PHYSICAL COMMITMENT FOLLOWS THE PLAN, and it precedes admission because
        #: admission's zero-init is the first write those slots see.
        self._commit(self._allocator.allocated)
        self._admit_pages(atom_ids, first_slot)
        if self._stream is not None:
            self._ready = torch.cuda.Event()
            self._ready.record(self._stream)

        self.peak_pages = max(self.peak_pages, self._allocator.allocated)
        self.last_result = PagingResult(
            allocated_pages=self._allocator.allocated, peak_pages=self.peak_pages,
            realized_owners=realized,
            owners_total=self.keying.atoms_per_bh,
            allocated_bytes=self.allocated_bytes(), dense_bytes=self.dense_bytes(),
            committed_bytes=self.committed_bytes)
        return self.last_result

    # -- the slack pool ---------------------------------------------------

    @property
    def pool_slots(self) -> torch.Tensor | None:
        """``[BH, pool_capacity]`` int32 slot ids, or ``None`` under exact commitment."""
        return self._pool_slots if self.pool_capacity else None

    @property
    def pool_cursor(self) -> torch.Tensor | None:
        """``[BH]`` int32 -- pool slots each batch-head has spent."""
        return self._pool_cursor if self.pool_capacity else None

    @property
    def pool_map(self) -> torch.Tensor | None:
        """``[BH, atoms_per_bh]`` int32 -- the device's claim, before the fold."""
        return self._pool_map if self.pool_capacity else None

    def pool_headroom(self) -> int:
        """Pool slots the THINNEST batch-head still has. A host read, at leisure.

        The minimum and not the mean, because the pool is spent per ``bh`` and the first
        batch-head to run out is the one that sends the step back to the host. It is what
        a serving loop polls to decide when to :meth:`refill_pool`, and it is deliberately
        NOT on the step's path: a step that exhausts the pool degrades to the host
        admission it always had.
        """
        if not self.pool_capacity:
            return 0
        self._cursor_host.copy_(self._pool_cursor, non_blocking=False)
        return self.pool_capacity - int(self._cursor_host.max())

    def refill_pool(self) -> int:
        """Replace the slots the device claimed. Returns how many. HOST, OFF THE PATH.

        A claimed slot is ordinary residency now -- the step folded it into the table --
        so a refill does not reclaim anything: it takes fresh slots for the spent
        positions, commits and zeroes them, and rewinds the cursors. Between two decode
        steps, on whatever stream the caller is on.
        """
        if not self.pool_capacity:
            return 0
        self._cursor_host.copy_(self._pool_cursor, non_blocking=False)
        spent = self._cursor_host.tolist()
        total = sum(spent)
        if not total:
            return 0
        first = self._allocator.take(total)
        self._commit(self._allocator.allocated)
        self._value_base[first:first + total].zero_()
        fresh = torch.arange(first, first + total, dtype=torch.int32, device=self.device)
        at = 0
        for bh, count in enumerate(spent):
            if count:
                self._pool_slots[bh, :count] = fresh[at:at + count]
                at += count
        self._pool_cursor.zero_()
        self.peak_pages = max(self.peak_pages, self._allocator.allocated)
        return total

    def wait(self) -> None:
        """The ONE event-wait before the consumer launch.

        Allocation and zero-init ran on the side stream while the launch's operands
        were being assembled; this is where the consumer's stream joins them. It is a stream-to-stream
        wait, not a host synchronization -- the host never learns when the zeroing
        finished, only that the consumer will not start before it did.
        """
        if self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)

    # -- the dense bridge OUT, crossed explicitly (docs/internals/paging/paging.md#backings) ------------

    def materialize(self) -> torch.Tensor:
        """Gather the committed atoms back into a dense ``[BH, N, cols]`` state.

        The explicit bridge OUT, and the hi||lo GATHER as well: the pages hold
        split bf16 planes, so the logical fp32 view is rejoined here and nowhere on a
        measured path. Allocating ``[BH, N, cols]`` is precisely what paging exists to
        avoid, so this is for oracle comparison and for the existing dense cache
        contract only.
        """
        present = self._page_table >= 0
        index = self._page_table.long().clamp_min(0).reshape(-1)
        gathered = self._value_base.index_select(0, index).view(
            self.BH, self.keying.atoms_per_bh, MMA_K_QUANTUM, self.cols)
        dense = torch.zeros_like(gathered)
        dense[present] = gathered[present]
        return from_split_planes(dense).reshape(self.BH, self.N, self.cols)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = [
    "C_EXTENT_ATOMS",
    "C_VMM_CHUNK_BYTES",
    "MMA_K_QUANTUM",
    "AtomKeying",
    "ExtentAllocator",
    "PageArena",
    "PagingResult",
    "bytes_equal",
    "from_split_planes",
    "to_split_planes",
    "vmm_supported",
]
