# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer, WITHOUT a device: every band-2 fold, fed values and asserted on.

A rule is a pure host fold from facts to a decision, and it reads a fact's shape,
dtype, device and PRESENCE but never its contents. That is what makes this battery
possible at all: it feeds plain tuples, ints and a fake manifest, calls no kernel and
touches no device, and it must pass under `CUDA_VISIBLE_DEVICES=""`. Before P75 the
run table was reachable only through a launch and the arm only through a built
extension (docs/internals/engine/rules.md).

The run-table rows are the DERIVATION, not a spot check: every one of the eight
liveness bytes is asserted against every row of `engine/rules.md#run-table`, and the row
lines themselves are matched against that document so the doc and the rule cannot
drift apart.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest
import torch

import rola.engine.rules as rules_pkg
from rola.engine.rules import (
    admission,
    arm,
    arm_envelope,
    decode_mask,
    envelope,
    mask,
    paging_preference,
    run_table,
    training_envelope,
)
from rola.engine.runner import Node
from rola.engine.types import Arm, CallClass, Mask

ROOT = Path(__file__).resolve().parents[2]

#: A manifest as `rola.engine.facts.manifest.chunk_arms` publishes it: `(C, BC, D, B)` rows.
#: Fake, and deliberately so -- the rule's input is the census, so a test that read
#: the real one would only prove the binary agrees with itself.
_MANIFEST = ((32, 64, 2, 16), (32, 128, 2, 64), (32, 64, 2, 64), (32, 64, 3, 8),
             (16, 128, 2, 64))


# ---------------------------------------------------------------------------
# The layer itself
# ---------------------------------------------------------------------------

def _rule_modules():
    return [importlib.import_module(f"rola.engine.rules.{info.name}")
            for info in pkgutil.iter_modules(rules_pkg.__path__)]


def test_every_function_under_rules_is_a_declared_rule():
    """THE MARKER IS THE INVENTORY. `rola/engine/rules/` holds band-2 folds and nothing
    else, so an ordinary helper landing here -- one that could launch, allocate or
    read a tensor's contents unnoticed -- fails by being unmarked."""
    unmarked = []
    for module in _rule_modules():
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            if obj.__module__ != module.__name__:
                continue
            if not isinstance(getattr(obj, "node", None), Node):
                unmarked.append(f"{module.__name__}.{name}")
    assert not unmarked, (
        f"{unmarked} live under rola/engine/rules/ without the `rule` marker; a rule "
        "declares its facts and its band, and an unmarked function declares nothing")


def test_a_rule_declares_the_host_band_and_no_memo():
    """Band 2 is pure host work on the DAG's parallel front: no stream, no join, no
    device-to-host read, and -- this stage -- no memo anywhere new."""
    for module in _rule_modules():
        for name, obj in vars(module).items():
            declared = getattr(obj, "node", None)
            if not isinstance(declared, Node):
                continue
            where = f"{module.__name__}.{name}"
            assert declared.stream == "host", where
            assert not declared.waits and not declared.host_sync, where
            assert declared.pure_of == () and not declared.per_state, where
            assert declared.pure is obj, where


def test_a_rule_whose_parameters_are_not_its_facts_is_refused():
    """The adapter from the fact bag to the fold is a LOOKUP, so the parameter names
    are the fact names -- checked at declaration rather than at the walk."""
    from rola.engine.runner import rule

    with pytest.raises(ValueError, match="parameter names are the FACT names"):
        @rule(gives=("arm",), needs=("widths", "manifest"))
        def wrong(widths, census):
            return None


def test_a_rule_runs_as_a_node_over_a_fact_bag():
    """Rules are NODES, not a phase: the same fold a caller invokes with values is
    what the walk runs, and `.node` is the one it walks."""
    from rola.engine.runner import run

    facts = run((mask.node, arm.node), {"call_class": CallClass(False, False, False, False),
                                        "widths": (64, 64), "manifest": _MANIFEST})
    assert facts["mask"] is Mask.M1
    assert facts["arm"] == Arm(32, 128)


# ---------------------------------------------------------------------------
# MaskRule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state_in", [False, True])
@pytest.mark.parametrize("state_out", [False, True])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("grad", [False, True])
def test_the_mask_is_m1_exactly_when_no_state_bit_is_engaged(state_in, state_out, paged, grad):
    """GRAD IS NOT A MASK: it is a retention decision, and the mask must not move
    with it -- a grad-bearing stateless call runs the subgraph exactly."""
    engaged = state_in or state_out or paged
    assert mask(CallClass(state_in, state_out, paged, grad)) is (Mask.M2 if engaged
                                                                 else Mask.M1)


@pytest.mark.parametrize("paged", [False, True])
def test_the_decode_mask_is_the_backing_and_nothing_else(paged):
    """A GROWTH STEP IS NOT A MASK. M4 is walked twice around an admission, so the
    only thing that selects a different node set at T = 1 is dense vs paged."""
    assert decode_mask(paged) is (Mask.M4 if paged else Mask.M3)


def test_the_four_masks_are_the_four_call_classes():
    """The set is CLOSED at four: prefill and decode, each over its two backings."""
    assert [m.name for m in Mask] == ["M1", "M2", "M3", "M4"]


# ---------------------------------------------------------------------------
# ArmRule
# ---------------------------------------------------------------------------

def test_the_arm_prefers_the_pin_and_falls_back_to_the_largest_built_bc():
    assert arm((64, 64), _MANIFEST) == Arm(32, 128)
    assert arm((16, 16), _MANIFEST) == Arm(32, 64)


def test_the_arm_reads_only_rows_of_its_own_chunk_depth_and_width():
    """`(C=16, BC=128, D=2, B=64)` is in the manifest and must not be reachable: the
    pinned `C` is part of the plan, so a row at another chunk length is another
    operator's arm."""
    assert arm((64, 64), ((16, 128, 2, 64),)) is None
    assert arm((4, 4, 4), _MANIFEST) is None
    assert arm((64, 64, 64), _MANIFEST) is None


def test_the_arm_takes_no_width_threshold():
    """BRANCH ON STRUCTURE, NEVER ON A THRESHOLD: the rule's inputs are the topology
    and the census, and `d_v` is not among them -- there is nothing to compare a
    dimension against."""
    assert tuple(inspect.signature(arm).parameters) == ("widths", "manifest")


# ---------------------------------------------------------------------------
# EnvelopeRule
# ---------------------------------------------------------------------------

def test_the_arm_envelope_names_each_clause_in_order():
    good = dict(widths=(64, 64), d_v=64, decay=None,
                arm=Arm(32, 128))
    assert arm_envelope(**good) is None
    assert arm_envelope(**{**good, "d_v": 48}) == "d_v=48; the chunk matrix is built at d_v=64"
    assert "decay is not implemented in the operator" in arm_envelope(**{**good, "decay": object()})
    assert arm_envelope(**{**good, "widths": (64, 16)}) == (
        "widths=(64, 16); the built manifest is uniform-width")
    assert arm_envelope(**{**good, "arm": None}) == (
        "topology (D=2, B=64) has no built arm in the manifest")


def test_the_arm_envelope_refuses_the_widest_clause_first():
    """A call outside several clauses is named by the FIRST one, so the sentence a
    caller reads is stable rather than a race between checks."""
    assert arm_envelope(widths=(64, 16), d_v=48, decay=object(),
                        arm=None).startswith("d_v=48")


def test_the_operand_envelope_names_the_activation_before_the_device():
    assert envelope((), torch.device("cuda")) is None
    assert envelope((), torch.device("cpu")) == (
        "the operands are on cpu; the arm is a CUDA kernel")
    assert envelope(("entmax(1.7)",), torch.device("cpu")) == (
        "unratified activations ('entmax(1.7)',); no kernel was measured for them "
        "(docs/ratification.md)")


def test_training_is_stateless_only():
    class _State:
        pass

    assert training_envelope(False, None) is None
    assert training_envelope(True, None) is None
    assert training_envelope(False, _State()) is None
    assert "trains STATELESS ONLY" in training_envelope(True, _State())


# ---------------------------------------------------------------------------
# PagingPreferenceRule
# ---------------------------------------------------------------------------

def test_paging_is_the_default_and_the_expert_may_only_turn_it_off():
    from rola.expert import PlanOverrides

    assert paging_preference(None) is True
    assert paging_preference(PlanOverrides()) is True
    assert paging_preference(PlanOverrides(paging=False)) is False


# ---------------------------------------------------------------------------
# AdmissionRule (the pure half)
# ---------------------------------------------------------------------------

def test_a_run_that_fits_the_remainder_stays_in_the_open_extent():
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=3) == 5


def test_a_run_that_does_not_fit_opens_the_next_extent():
    """STRUCTURAL, not a threshold: the count is compared against the remainder, so a
    large admission lands contiguous rather than straddling."""
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=4) == 8
    assert admission(allocated=8, capacity=64, extent_atoms=8, count=8) == 8


def test_an_empty_run_moves_nothing():
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=0) == 5


def test_admission_is_all_or_nothing_at_the_ceiling():
    #: THE ALIGNMENT IS INSIDE THE CEILING CHECK: this run does not fit the open
    #: extent's remainder, so it starts at 64 and there is nothing left -- a rule
    #: that checked the demand against `capacity - allocated` would grant it.
    with pytest.raises(RuntimeError, match="5 new atoms demanded, 0 free"):
        admission(allocated=60, capacity=64, extent_atoms=8, count=5)
    with pytest.raises(RuntimeError, match="grants ALL or refuses"):
        admission(allocated=0, capacity=64, extent_atoms=8, count=65)
    #: and a run that DOES fit the remainder is served from it, at the same cursor.
    assert admission(allocated=60, capacity=64, extent_atoms=8, count=4) == 60


# ---------------------------------------------------------------------------
# RunTableRule -- the derivation, over every bit value and every documented row
# ---------------------------------------------------------------------------

_PRESENT = object()

#: `docs/internals/engine/rules.md#run-table`, transcribed: the row's line as it must
#: appear in that document, the three PRESENCE facts it is keyed by, and its rule.
_DOC_ROWS = (
    ("| stateless `y`-only | absent | absent | absent | `R && W` |",
     [(None, None, None)],
     lambda R, W, S: R and W),
    ("| state-in `y`-only, paged | present | absent | present | `R && (W \\| S)` |",
     [(_PRESENT, None, _PRESENT)],
     lambda R, W, S: R and (W or S)),
    ("| state-in `y`-only, dense | present | absent | absent | `R` |",
     [(_PRESENT, None, None)],
     lambda R, W, S: R),
    ("| continued state-out, paged | present | present | present | `(R && (W \\| S)) \\| W` |",
     [(_PRESENT, _PRESENT, _PRESENT)],
     lambda R, W, S: (R and (W or S)) or W),
    ("| fresh state-out, paged | absent | present | present | `W` |",
     [(None, _PRESENT, _PRESENT)],
     lambda R, W, S: W),
    ("| continued state-out, dense | present | present | absent | `R \\| W` |",
     [(_PRESENT, _PRESENT, None)],
     lambda R, W, S: R or W),
    ("| fresh state-out, dense | absent | present | absent | `W` |",
     [(None, _PRESENT, None)],
     lambda R, W, S: W),
)


def test_the_documented_rows_are_the_rows_the_document_carries():
    """The seven-row prose became an executable fixture, and this is the tie that
    keeps it one: a doc edit that changes a rule fails here."""
    doc = (ROOT / "docs" / "internals" / "engine" / "rules.md").read_text()
    missing = [row for row, _triples, _rule in _DOC_ROWS if row not in doc]
    assert not missing, f"rules.md#run-table no longer carries {missing}"


@pytest.mark.parametrize("row,triples,predicate", _DOC_ROWS,
                         ids=[r.split("|")[1].strip() for r, _t, _p in _DOC_ROWS])
def test_the_run_table_derives_every_bit_value_of_every_row(row, triples, predicate):
    """VACUITY GUARD: all eight liveness bytes, every row -- the derivation, never a
    spot check of the rows someone remembered."""
    for state_in, state_out, page_table in triples:
        table = run_table(state_in, state_out, page_table)
        for v in range(8):
            want = predicate(bool(v & 1), bool(v & 2), bool(v & 4))
            assert bool(table >> v & 1) is bool(want), (
                f"{row}: byte {v:#05b} derives {bool(table >> v & 1)}, the rule says "
                f"{bool(want)}")


def test_every_argument_triple_is_covered_by_a_row_or_refused_upstream():
    """Eight triples, seven rows: the one triple no row carries is a page table with no
    state, which the binding refuses before the derivation is reached -- and which
    derives the stateless row regardless, so nothing depends on the refusal."""
    covered = {t for _row, triples, _rule in _DOC_ROWS for t in triples}
    every = {(s_in, s_out, tbl)
             for s_in in (None, _PRESENT)
             for s_out in (None, _PRESENT)
             for tbl in (None, _PRESENT)}
    assert every - covered == {(None, None, _PRESENT)}
    assert run_table(None, None, _PRESENT) == run_table(None, None, None)


def test_the_run_table_reads_presence_and_never_a_tensor():
    """THE HARD INVARIANT, on the one rule whose inputs are tensors: it is handed
    planes whose contents would raise if read, and it derives the same table."""

    class _Unreadable:
        def __bool__(self):
            raise AssertionError("a rule read a fact's CONTENTS")

    assert run_table(_Unreadable(), _Unreadable(), _Unreadable()) == run_table(
        _PRESENT, _PRESENT, _PRESENT)


@pytest.fixture(scope="module", autouse=True)
def _extension_calls_at_start():
    from rola.ops._ext import extension

    info = extension.cache_info()
    return info.hits + info.misses


def test_no_rule_touched_the_extension(_extension_calls_at_start):
    """DEVICE-FREE IS THE CLAIM, and this is it measured rather than asserted: the
    battery must not have resolved the compiled extension at all. A DELTA from this
    module's start, because a shared pytest process may already have loaded the
    extension for another file's tests."""
    from rola.ops._ext import extension

    info = extension.cache_info()
    assert info.hits + info.misses == _extension_calls_at_start, (
        f"the rule battery called into the extension ({info})")
