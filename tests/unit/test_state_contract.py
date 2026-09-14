"""THE STATE OWNS ITS FORMAT: the descriptor's field list, pinned without a kernel.

`rola/_state.py::StateFormat` is what every kernel derives its addressing from and what a
later call is checked against (`docs/internals/state.md#format`). This file pins that
contract at the Python object -- no CUDA extension, no kernel body, no GPU required -- so
the pipelined body inherits a descriptor whose shape cannot silently drift underneath it.

Symbols, each at first use: `D` routing levels; `B_l` level `l`'s digit width (the
descriptor's `B` tuple); `DV` the value width; `cols = DV + 1` the page's logical width
(the mass column); an **atom** is 16 consecutive canonical leaves, the page granule.

**THE ONE ADMISSION LAW.** K50: "prefill and decode share ONE admission law" -- there is no
decode-only shape, tolerance or topology. `RoLAState._kernel_entry` (prefill) and
`RoLAState._decode_entry` (decode) both build the call's presented format through the SAME
free function, `_format_of`, and check it against the bound one through the SAME method,
`StateFormat.check` -- never a second comparison written at either seam. That is asserted
here structurally (each entry point's own source names both calls), which is what makes a
future edit that reintroduces a decode-only shape check a self-evident diff instead of a
silent divergence.
"""

from __future__ import annotations

import inspect

import pytest

from rola._state import (
    CANONICAL_ORDER,
    MIN_LEVEL_WIDTH,
    PAGE_RECTANGLE_BITS,
    SPLIT_PLANE_DTYPE,
    RoLAState,
    StateFormat,
    _format_of,
)

#: A LAWFUL DESCRIPTOR, D = 2, at the flagship's own widths -- the fixture every test below
#: mutates one field of at a time.
_FIELDS = dict(D=2, B=(256, 256), order=CANONICAL_ORDER, page_bits=PAGE_RECTANGLE_BITS,
               DV=64, dtype=SPLIT_PLANE_DTYPE, ids=8 * (65536 >> PAGE_RECTANGLE_BITS))


def _format(**over):
    return StateFormat(**{**_FIELDS, **over})


# -- THE FIELD LIST, AS ASSERTED -----------------------------------------------------------

def test_the_descriptor_is_exactly_these_seven_fields():
    """The kernel-facing surface: D, per-level B, order, page_bits, DV, dtype, ids.

    A field added or removed here is a contract change, not a refactor -- `check()`
    iterates a hand-named tuple of them (below), so an eighth field silently unchecked is
    exactly the drift this test exists to catch.
    """
    assert [f.name for f in StateFormat.__dataclass_fields__.values()] == [
        "D", "B", "order", "page_bits", "DV", "dtype", "ids"]


def test_the_descriptor_is_frozen():
    """A bound state's format never mutates in place -- rebinding is a fresh object."""
    fmt = _format()
    with pytest.raises((AttributeError, TypeError)):
        fmt.D = 3  # type: ignore[misc]


def test_derived_properties_read_off_the_fields_and_nothing_else():
    fmt = _format(D=2, B=(256, 256), DV=64)
    assert fmt.N == 256 * 256, "N is the mixed-radix product, never configured directly"
    assert fmt.cols == 65, "cols = DV + 1, the mass column"
    assert fmt.atoms_per_bh == fmt.N >> PAGE_RECTANGLE_BITS
    assert fmt.page_bytes == (1 << PAGE_RECTANGLE_BITS) * fmt.cols * 4, (
        "[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo] -- one page, in bytes")


# -- THE FIELDS' OWN LAWS, EACH A REFUSAL --------------------------------------------------

def test_b_must_agree_with_d_in_length():
    with pytest.raises(ValueError, match="D=2 levels and 3 widths"):
        _format(D=2, B=(256, 256, 256))


@pytest.mark.parametrize("width", [15, 24, 17, 0, -16])
def test_every_level_width_is_a_power_of_two_at_or_above_the_floor(width):
    """the restored floor: B_l >= 16 (= K_max), power of two -- no narrower level exists."""
    with pytest.raises(ValueError, match=f"level 0 is {width}"):
        _format(D=2, B=(width, 256))


def test_min_level_width_constant_is_the_floor_every_field_test_assumes():
    assert MIN_LEVEL_WIDTH == 16


def test_the_only_leaf_order_is_the_canonical_one():
    with pytest.raises(ValueError, match="only leaf order this build addresses"):
        _format(order="mixed-radix-lsb-first")


def test_page_bits_is_the_page_rectangle_and_nothing_else():
    with pytest.raises(ValueError, match="trailing .* canonical bits"):
        _format(page_bits=PAGE_RECTANGLE_BITS + 1)


def test_dv_is_even_for_the_split_planes_paired_access():
    with pytest.raises(ValueError, match="DV is even"):
        _format(D=1, B=(65536,), DV=63, ids=8 * (65536 >> PAGE_RECTANGLE_BITS))


def test_n_is_a_whole_number_of_pages_by_construction_not_by_padding():
    """RAGGED N IS REFUSED, NEVER PADDED: under the width floor every lawful `B`
    already makes `N` a multiple of the page, so the refusal is a standing invariant of
    the OTHER fields' laws, not a fourth, independent thing to satisfy."""
    fmt = _format(D=3, B=(16, 16, 16), ids=8 * ((16 ** 3) >> PAGE_RECTANGLE_BITS))
    assert fmt.N % (1 << PAGE_RECTANGLE_BITS) == 0


def test_dtype_names_the_split_bf16_planes():
    """R17 (BF16 EVERYTHING): the logical element is fp32, stored as hi||lo bf16 planes --
    `dtype` records the LAYOUT, not a narrowing of the logical word (`state.md#format`)."""
    assert _FIELDS["dtype"] == "fp32 as split bf16 planes" == SPLIT_PLANE_DTYPE


# -- THE ONE ADMISSION LAW, PREFILL AND DECODE IDENTICAL -----------------------------------

def test_check_accepts_an_identical_presented_format():
    bound = _format()
    bound.check(_format())  # no raise: same fields, different object


@pytest.mark.parametrize("field,over", [
    ("D", dict(D=3, B=(256, 256, 256), ids=8 * (256 ** 3 >> PAGE_RECTANGLE_BITS))),
    ("B", dict(B=(64, 1024))),
    ("DV", dict(DV=128)),
    ("dtype", dict(dtype="fp32")),
])
def test_check_refuses_a_call_that_disagrees_naming_the_field(field, over):
    bound = _format()
    with pytest.raises(ValueError, match=f"this state was bound with {field}="):
        bound.check(_format(**over))


def test_kernel_entry_and_decode_entry_both_call_the_one_admission_law():
    """STRUCTURAL PIN: `_format_of` builds the CALL's presented descriptor and
    `StateFormat.check` is the ONE comparison -- both entry points name both, so a future
    decode-only shape/tolerance path (K50: refused by name) shows up as a source diff here
    rather than as a silent divergence between the two seams."""
    prefill_src = inspect.getsource(RoLAState._kernel_entry)
    decode_src = inspect.getsource(RoLAState._decode_entry)
    for src, name in ((prefill_src, "_kernel_entry"), (decode_src, "_decode_entry")):
        assert "_format_of(" in src, f"{name} does not build its presented format via _format_of"
        assert ".check(" in src or "self._format = presented" in src, (
            f"{name} does not route through StateFormat.check (or bind it, on first call)")
    # decode NEVER binds -- a step continues, so its only legal path is the check.
    assert "self._format = presented" not in decode_src
    assert "self._format.check(" in decode_src


def test_format_of_sources_dv_from_the_value_stream_not_the_routing():
    """`cols` needs `d_v`, which the routing does not carry (`state.md#binding`)."""
    assert "d_v" in inspect.signature(_format_of).parameters


# -- K53: DECODE'S OWN OP ENTRY ASKS THE SAME LAW, NOT ONLY A BOUND STATE ------------------

def test_decode_op_entry_reuses_state_format_not_a_second_copy():
    """STRUCTURAL PIN, decode's side of `test_kernel_entry_and_decode_entry_both_call_the_one_admission_law`.

    That test pins `RoLAState._decode_entry` (a BOUND state's own check); this pins
    `rola.ops.decode`'s OP ENTRY, which a call reaches directly with no `RoLAState` in
    play at all (every oracle/integration fixture in `tests/oracle/test_decode_*.py` and
    `tests/integration/test_decode_*.py` calls `derive_decode_geometry` this way). Before
    K53 that path asked no floor at all -- `StateFormat` is imported here rather than a
    hand-rolled power-of-two check, so a future rewrite that duplicates the rule instead
    of asking the one class shows up as a source diff here.
    """
    from rola.ops import decode as decode_ops

    src = inspect.getsource(decode_ops._admit)
    assert "StateFormat(" in src, "_admit does not route through StateFormat"
    geometry_src = inspect.getsource(decode_ops.derive_decode_geometry)
    assert "_admit(" in geometry_src, "derive_decode_geometry does not call _admit"


def test_decode_op_entry_refuses_an_unlawful_topology_by_name():
    """K53: decode's D=4 arm's OLD topology, `(8, 8, 8, 8)`, is unlawful under the floor
    every prefill call already refuses at -- decode's own entry now refuses it by the
    SAME name (`StateFormat`'s), not a decode-shaped message of its own."""
    from rola.ops import decode as decode_ops

    with pytest.raises(ValueError, match="level 0 is 8"):
        decode_ops._admit((8, 8, 8, 8))


@pytest.mark.parametrize("widths", [(64, 64), (16, 16, 16), (16, 16, 16, 16)])
def test_decode_op_entry_accepts_every_shipped_decode_topology(widths):
    """The three topologies `test_decode_vs_oracle.py` carries post-K53 -- none of them
    is refused by the law that replaced the missing floor."""
    from rola.ops import decode as decode_ops

    decode_ops._admit(widths)  # no raise
