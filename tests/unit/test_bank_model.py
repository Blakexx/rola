# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The bank model's wavefront counts on patterns whose cost the hardware fixes (KERNEL_STANDARDS section 30)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import bank_model  # noqa: E402


def test_consecutive_words_are_one_wavefront():
    assert bank_model.wavefronts([4 * lane for lane in range(32)], 4) == 1


def test_a_shared_word_is_a_broadcast():
    assert bank_model.wavefronts([64] * 32, 4) == 1


def test_a_stride_of_two_words_is_two_way():
    assert bank_model.wavefronts([8 * lane for lane in range(32)], 4) == 2


def test_a_row_pitch_of_32_words_is_32_way():
    assert bank_model.wavefronts([128 * lane for lane in range(32)], 4) == 32


def test_wide_accesses_serve_in_phases():
    assert bank_model.wavefronts([8 * lane for lane in range(32)], 8) == 2
    assert bank_model.wavefronts([16 * lane for lane in range(32)], 16) == 4
    assert bank_model.wavefronts([0] * 32, 16) == 4


def test_inactive_lanes_cost_nothing():
    assert bank_model.wavefronts([None] * 31 + [4], 4) == 1


def test_ldmatrix_rows_on_distinct_bank_groups_are_one_wavefront_a_matrix():
    rows = [64 * r + 16 * ((r >> 1) & 3) for r in range(8)]
    assert bank_model.ldmatrix([rows, [a ^ 16 for a in rows]]) == 2


def test_ldmatrix_rows_on_one_bank_group_conflict():
    assert bank_model.ldmatrix([[128 * r for r in range(8)]]) == 8


def test_misaligned_and_odd_widths_are_refused():
    with pytest.raises(ValueError):
        bank_model.wavefronts([2], 4)
    with pytest.raises(ValueError):
        bank_model.wavefronts([0], 2)
