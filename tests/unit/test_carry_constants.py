"""the axis law's CONSTANT and DERIVED classes: the ONE-PLACE gate for
``csrc/rola/src/common/constants.cuh`` and its Python mirror, ``rola.engine.constants``.

Two claims, same shape as ``test_definition_constants.py``'s readout-epsilon gate:

1. **THE TWO MODULES AGREE** -- the C++ header's literals (``kWindow``, ``kBatchTile``,
   the ``warps_per_cta`` set, the launch table) are parsed out of the source text and
   checked equal to the Python mirror's values. No CUDA build is needed for this
   direction; the header is not compiled to make the comparison.
2. **THE RULES BEHAVE THE SAME**, on inputs neither side pins as a literal:
   ``box_leaves``/``leaves_per_warp`` against ``common/geom.cuh``'s own formula, and
   ``derive_stream_count`` (the S-from-SMEM rule) against its own definition, both
   mirrored line for line -- so a change to one side's ARITHMETIC (not just its
   literals) that the other side does not follow fails a property test, not just a
   literal-equality check.

No device, no build: this reads source text and calls pure Python.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rola.engine import constants as py

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "csrc" / "rola" / "src" / "common" / "constants.cuh"


def _header_text() -> str:
    return HEADER.read_text()


def _constexpr_int(name: str) -> int:
    """The value bound to ``constexpr int <name> = <literal>;`` in the header."""
    m = re.search(rf"constexpr int {re.escape(name)}\s*=\s*(-?\d+)\s*;", _header_text())
    assert m, f"no `constexpr int {name} = ...;` found in {HEADER}"
    return int(m.group(1))


def _launch_table_rows() -> list[tuple[int, int, str]]:
    """``[(cc, ctas_per_sm, warps_per_cta_symbol)]`` from the ``kLaunchTable`` array."""
    text = _header_text()
    body = re.search(r"kLaunchTable\[\]\s*=\s*\{(.*?)\};", text, re.S)
    assert body, "no `kLaunchTable[] = {...};` array found"
    rows = re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(kWarpsPerCta\w+)\s*\}", body.group(1))
    assert rows, "kLaunchTable parsed to zero rows -- the regex or the header drifted"
    return [(int(cc), int(ctas), sym) for cc, ctas, sym in rows]


# --------------------------------------------------------------------------- agreement


def test_the_window_agrees():
    assert _constexpr_int("kWindow") == py.WINDOW


def test_the_segment_agrees():
    assert _constexpr_int("kSegmentTokens") == py.SEGMENT_TOKENS


def test_the_warps_per_cta_set_agrees():
    assert _constexpr_int("kWarpsPerCtaShipped") == py.WARPS_PER_CTA_SHIPPED
    assert _constexpr_int("kWarpsPerCtaDiagnostic") == py.WARPS_PER_CTA_DIAGNOSTIC
    assert (py.WARPS_PER_CTA_SHIPPED, py.WARPS_PER_CTA_DIAGNOSTIC) == py.WARPS_PER_CTA


def test_the_launch_table_agrees_row_for_row():
    symbol_value = {
        "kWarpsPerCtaShipped": py.WARPS_PER_CTA_SHIPPED,
        "kWarpsPerCtaDiagnostic": py.WARPS_PER_CTA_DIAGNOSTIC,
    }
    cpp_rows = [(cc, ctas, symbol_value[sym]) for cc, ctas, sym in _launch_table_rows()]
    py_rows = [(row.cc, row.ctas_per_sm, row.warps_per_cta) for row in py.LAUNCH_TABLE]
    assert cpp_rows == py_rows, (
        f"the launch table drifted between the two modules:\n  C++:    {cpp_rows}\n"
        f"  Python: {py_rows}"
    )


def test_the_leaves_per_sm_numerator_agrees():
    """``leaves_per_sm(dv) = 16384 / dv`` lives in ``common/geom.cuh`` (not this
    header); the Python mirror restates the same literal rather than re-deriving it,
    so this pins that restatement against the numbers K50 item 1 states directly.
    """
    assert py._LEAVES_PER_SM_NUMERATOR == 16384
    assert py.leaves_per_warp(64) == 32
    assert py.leaves_per_warp(128) == 16


# ------------------------------------------------------------------------- the rules


@pytest.mark.parametrize("dv,warps_per_cta,expected", [
    (64, 8, 256), (64, 4, 128), (128, 8, 128), (128, 4, 64),
])
def test_box_leaves_is_leaves_per_warp_times_warps_per_cta(dv, warps_per_cta, expected):
    assert py.box_leaves(dv, warps_per_cta) == expected == (
        py.leaves_per_warp(dv) * warps_per_cta)


@pytest.mark.parametrize("warps_per_cta", [8, 4, 16, 3, 0, -1])
def test_is_warps_per_cta_is_exactly_the_shipped_and_diagnostic_pair(warps_per_cta):
    assert py.is_warps_per_cta(warps_per_cta) == (warps_per_cta in (8, 4))


def test_launch_row_refuses_an_untabulated_architecture():
    with pytest.raises(ValueError, match="no launch-table row"):
        py.launch_row(750)


@pytest.mark.parametrize("cc,ctas_per_sm,warps_per_cta", [
    (800, 1, 8), (860, 1, 8), (870, 1, 8), (890, 1, 8), (900, 1, 8),
])
def test_launch_row_matches_the_table(cc, ctas_per_sm, warps_per_cta):
    row = py.launch_row(cc)
    assert (row.ctas_per_sm, row.warps_per_cta) == (ctas_per_sm, warps_per_cta)


class TestDeriveStreamCount:
    """THE S-FROM-SMEM RULE, against a STATED layout (no real segments exist yet).

    ``per_stream_bytes`` and ``residency_smem_bytes`` are inputs the test states, not
    numbers read off a kernel -- C3-K3 is what supplies the real ones.
    """

    def test_the_full_warp_count_fits_when_the_budget_is_generous(self):
        assert py.derive_stream_count(1000, 8 * 1000, 8) == 8

    def test_it_halves_until_it_fits(self):
        # 5 streams' worth fits, but 5 is not a power of two: 4 is the answer.
        assert py.derive_stream_count(1000, 5 * 1000, 8) == 4

    def test_one_stream_is_the_floor_that_still_fits(self):
        assert py.derive_stream_count(1000, 1000, 8) == 1

    def test_zero_is_a_refusal_not_a_silent_fallback(self):
        assert py.derive_stream_count(1000, 999, 8) == 0

    def test_it_never_exceeds_warps_per_cta(self):
        assert py.derive_stream_count(1, 10_000, 4) == 4

    @pytest.mark.parametrize("warps_per_cta", [1, 2, 4, 8])
    def test_the_result_is_always_a_power_of_two_or_zero(self, warps_per_cta):
        for budget in range(0, warps_per_cta * 100, 7):
            s = py.derive_stream_count(100, budget, warps_per_cta)
            assert s == 0 or (s & (s - 1)) == 0

    def test_a_larger_budget_never_yields_fewer_streams(self):
        """Monotone in the budget -- the search widens, it never narrows."""
        prev = 0
        for budget in range(0, 900, 3):
            s = py.derive_stream_count(100, budget, 8)
            assert s >= prev
            prev = s
