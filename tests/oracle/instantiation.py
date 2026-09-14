"""THE CENSUS'S LEFT-HAND SIDE — the chunk consumer's instantiation axes, as DATA.

The conformance census (`test_conformance_census.py`) joins the INSTANTIATION
axes with the data-regime axes (`families.REGIME_AXES`). Until P67 D2 the
left-hand side was the retired geometry matrix's `AXIS_DECLARATIONS`, the TILED
consumer's template parameter list `<BC, DV, D, DECAY>` crossed with the runtime
`paging` flag. That consumer retires, and four of those five axes retire with
it — `BC` is a slot tile inside a built chunk arm and is not host-selectable,
`DV` is single-valued (`d_v = 64` is the whole built matrix), `DECAY` is
single-valued (decay is retired from the operator, its semantics kept by the
fp64 oracle). So the census keeps its shape and changes its axes, which is
exactly the move P28 made when an axis left before.

**The surviving axes are the ones a CALLER can move.** A topology is a level
count `D` and, since the built manifest is uniform-width, one width `B`; the
state's backing is dense or paged. Those three are what a family's config picks
and what the kernel envelope is stated in, so those three are the census's
left-hand side.

**Declared here, as the TESTS' OWN STATEMENT of what the family must cover.**
`CHUNK_ARMS` used to be read live from `tools/gen_shards.py`'s `FACTS_ARMS`, so
that it could not disagree with the binary the way a hand-kept list can. C0
deleted the stats passes and that list with them (card C,
`development/queue/C_CLEAN_SLATE.md`), so the rows move HERE, unchanged, and
change meaning with the move: they are no longer a mirror of a built matrix but
the `(C, BC, D, B)` vocabulary the rebuilt family is REQUIRED to reach, which is
what a test-driven line wants a test module to hold. When the liveness pass and
the arm table exist, this list is what they are checked against — and if the
final key admits a different vocabulary, this list is the thing to argue with,
not a comment to update.

(The conformance census that asserted the projections went with the chunk
consumer in the K31 deletion batch; baseline = tag `baseline/pre-k31`.)

**Pure data and pure CPU.** No torch, no device, no extension: the census runs
in the CPU CI job, and so does the kernel-envelope predicate below.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: `(C, BC, D, B)` -- the dispatch vocabulary the family is stated over. Held here
#: since C0; see the module docstring for why it is a declaration and not a mirror.
CHUNK_ARMS: tuple[tuple[int, int, int, int], ...] = (
    (16, 64, 2, 8), (32, 64, 2, 8), (64, 64, 2, 8),
    (16, 64, 2, 64), (32, 64, 2, 64), (64, 64, 2, 64),
    (32, 32, 2, 64), (64, 32, 2, 64),
    (16, 128, 2, 64), (32, 128, 2, 64), (64, 128, 2, 64),
    (32, 256, 2, 64),
    (16, 64, 2, 16), (32, 64, 2, 16), (64, 64, 2, 16),
    (32, 128, 3, 16), (32, 128, 4, 8), (32, 32, 4, 8),
    # the FLAT-ROUTING baseline arms (D = 1; N = B <= 256 by the manifest):
    (32, 64, 1, 64), (32, 64, 1, 256), (16, 64, 1, 256),
    # the SCALE rung (N = 65536):
    (32, 128, 2, 256),
)

#: The `(D, B)` topologies the dispatch can reach. The engine's arm fold pins
#: `C = CHUNK_TOKENS` and picks `BC` from the manifest, so an arm at any other
#: `C` is built but unreachable from the facade, and the census must be stated
#: over what the DISPATCH reaches or it would claim coverage nobody can run.
CHUNK_TOKENS = 32
BUILT_TOPOLOGIES: frozenset[tuple[int, int]] = frozenset(
    (D, B) for (C, _BC, D, B) in CHUNK_ARMS if C == CHUNK_TOKENS)

#: The value width the whole built matrix is compiled at (`rola.engine.rules.envelope.arm_envelope`).
BUILT_DV = 64


@dataclass(frozen=True)
class AxisDeclaration:
    name: str
    #: The values the shipped product carries.
    values: tuple[Any, ...]
    #: "manifest" -- projected from `CHUNK_ARMS` and checked against it;
    #: "runtime" -- a host-side choice that leaves no symbol behind, so the
    #: census cannot discover it and an undeclared one would be invisible.
    source: str
    basis: str

    def __post_init__(self):
        assert self.basis, f"axis {self.name!r} declares no basis"
        assert self.source in ("manifest", "runtime")


AXIS_DECLARATIONS: tuple[AxisDeclaration, ...] = (
    AxisDeclaration(
        "D", tuple(sorted({D for (D, _B) in BUILT_TOPOLOGIES})), "manifest",
        basis="the number of routed levels. A different `D` is a different "
              "topology -- a different function of different inputs -- so it is "
              "swept rather than compared, and every built value must be "
              "reached by some family or the census is claiming a shape nobody "
              "ran."),
    AxisDeclaration(
        "B", tuple(sorted({B for (_D, B) in BUILT_TOPOLOGIES})), "manifest",
        basis="the uniform level width. The built manifest is uniform-width, so "
              "one number fixes the whole topology beside `D`; it also fixes "
              "`N = B ** D`, which is what the owner blocking and the page "
              "granule are stated in."),
    AxisDeclaration(
        "paging", (False, True), "runtime",
        basis="the state's backing. A page table changes the ADDRESS of an "
              "atom's rows and nothing else, so a residency decision that moved "
              "a number would be a defect. It leaves no symbol in the binary, "
              "so it is declared here or it is invisible."),
)

AXIS_BY_NAME: Mapping[str, AxisDeclaration] = {a.name: a for a in AXIS_DECLARATIONS}


def instantiation_of(config: Mapping[str, Any]) -> dict[str, Any]:
    """A family config's coordinates on the axes above."""
    widths = tuple(config["widths"])
    return {"D": len(widths), "B": widths[0], "paging": bool(config["paging"])}


#: The clauses `rola.engine.rules.envelope.arm_envelope` refuses on, each paired with the
#: substring its production message carries. The mirror below answers with a
#: clause NAME rather than with prose, so the two can be compared without one
#: file having to restate the other's wording -- a comparison on wording would
#: fail on a punctuation edit and pass on a swapped clause, which is exactly
#: backwards.
REFUSAL_CLAUSES: Mapping[str, str] = {
    "d_v": "d_v=",
    "decay": "decay",
    "widths": "widths=",
    "topology": "topology (",
}


def kernel_refusal(config: Mapping[str, Any]) -> str | None:
    """The clause on which the chunk arm refuses this config, or None if it runs.

    A pure-CPU MIRROR of `rola.engine.rules.envelope.arm_envelope`, clause for clause and in
    the same order, so `families.py` can stay device-free while still knowing
    which arm each family executes on. A mirror can drift, so it is not trusted:
    a CUDA row in `test_families.py` asserts this clause is the one production
    actually refuses on, for every declared family -- the same "a declaration
    cannot lie" discipline the regime cells are already held to.
    """
    widths = tuple(config["widths"])
    if config["d_v"] != BUILT_DV:
        return "d_v"
    if config["decay"]:
        return "decay"
    if len(set(widths)) != 1:
        return "widths"
    if (len(widths), widths[0]) not in BUILT_TOPOLOGIES:
        return "topology"
    return None
