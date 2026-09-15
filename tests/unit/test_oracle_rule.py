# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ORACLE RULE ITSELF, on planted numbers rather than a kernel: a slot passes within every clause of its output kind
and, where the kind is multilinear, within its envelope; the failure names the term that bound; a derived absolute term
the caller states moves every clause's atol; and every declared output kind carries clauses with a measurement behind
them, none of them tighter than the budget the kind itself declares. No CUDA: nothing here launches anything.
"""
from __future__ import annotations

import pytest
import torch

from tests.oracle.fixtures import allowances, assert_slots_close
from tests.oracle.tolerances import CARRY_NUM, OUTPUTS, Output

ONE_CLAUSE = Output("one clause", clauses=((1e-2, 1e-3),))
TWO_CLAUSES = Output("two clauses", clauses=((0.0, 1e-1), (1.0, 1e-3)))
WITH_ENVELOPE = Output("with envelope", clauses=((1e-2, 1e-3),), rtol=1e-2)


def test_a_clause_is_a_hinge_in_the_allowance_and_the_smallest_clause_binds():
    reference = torch.tensor([0.0, 1e-3, 1.0, 100.0], dtype=torch.float64)
    allowance, binding = allowances(reference, ONE_CLAUSE)
    #: max(rtol * |oracle|, atol): the atol below s = 0.1, the rtol above it
    assert allowance.tolist() == pytest.approx([1e-3, 1e-3, 1e-2, 1.0])
    assert binding.tolist() == [0, 0, 0, 0]
    allowance, binding = allowances(reference, TWO_CLAUSES)
    #: the absolute cap (clause 0) binds on the big slots, the relative clause on the small ones
    assert allowance.tolist() == pytest.approx([1e-3, 1e-3, 1e-1, 1e-1])
    assert binding.tolist() == [1, 1, 0, 0]


def test_the_envelope_binds_where_it_is_tighter_and_a_derived_atol_moves_every_clause():
    reference = torch.tensor([1.0], dtype=torch.float64)
    envelope = torch.tensor([0.5], dtype=torch.float64)
    allowance, binding = allowances(reference, WITH_ENVELOPE, envelope)
    assert (float(allowance), int(binding)) == (5e-3, 1)
    allowance, _ = allowances(reference, ONE_CLAUSE, derived_atol=0.25)
    assert float(allowance) == pytest.approx(0.25 + 1e-3)


def test_a_slot_outside_its_allowance_fails_and_the_message_names_what_bound_it():
    reference = torch.tensor([1.0, 1.0])
    assert_slots_close(reference + torch.tensor([0.0, 9e-3]), reference, output=ONE_CLAUSE, what="inside")
    with pytest.raises(AssertionError, match=r"clause \(0.01, 0.001\)"):
        assert_slots_close(reference + torch.tensor([0.0, 2e-2]), reference, output=ONE_CLAUSE, what="past its clause")
    with pytest.raises(AssertionError, match="the envelope"):
        assert_slots_close(reference + torch.tensor([0.0, 9e-3]), reference, output=WITH_ENVELOPE,
                           envelope=torch.tensor([1.0, 0.5]), what="past its envelope")


def test_a_slot_whose_allowance_is_zero_must_be_exact_and_a_non_finite_slot_fails():
    zero = Output("exact above a tenth", clauses=((1e-1, 0.0),))
    reference = torch.tensor([1.0])
    assert_slots_close(reference * 1.05, reference, output=zero, what="inside the tenth")
    with pytest.raises(AssertionError):
        assert_slots_close(reference * 1.5, reference, output=zero, what="past the tenth")
    with pytest.raises(AssertionError):
        assert_slots_close(torch.tensor([float("nan")]), reference, output=CARRY_NUM,
                           envelope=torch.tensor([1.0]), what="not finite")


def test_every_output_kind_declares_measured_clauses_no_tighter_than_its_own_budget():
    assert len({output.name for output in OUTPUTS}) == len(OUTPUTS)
    for output in OUTPUTS:
        if not output.measured:
            assert output.clauses == (), f"{output.name} is marked unmeasured, so it declares no clause"
            assert output.rtol is not None, f"{output.name} has neither clauses nor an envelope, so it grades nothing"
            continue
        assert output.clauses, f"{output.name} declares no clause"
        for rtol, atol in output.clauses:
            assert atol >= 0.0 and rtol >= 0.0, f"{output.name}: a clause is a pair of non-negative numbers"
            #: a clause below the kind's own declared relative budget would fail a draw that reached the budget
            assert rtol == 0.0 or output.rtol is None or rtol >= output.rtol, (
                f"{output.name}: clause ({rtol}, {atol}) is tighter than its declared budget {output.rtol}")
