# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The DAG framework's structure and its two caches, WITHOUT a device.

Memoization is a node attribute rather than a per-site convention, and the structural
refusals fire at DAG declaration rather than at launch. Both are host-only properties
and this battery holds them that way: no CUDA, no extension, no kernel. That is a new
capability -- the conventions these replace were only reachable through a launch.
"""
from __future__ import annotations

import gc

import pytest
import torch

from rola.engine.runner import Node, node, run, validate_dag
from rola.engine.types import Mask


def _counting(**kwargs):
    """A node whose body records every call, so a cache HIT is observable."""
    calls: list[tuple] = []

    @node(**kwargs)
    def probe(f):
        calls.append(tuple(f[name] for name in kwargs.get("needs", ())))
        return len(calls)

    return probe, calls


class _Sequence:
    """A weak-referenceable stand-in for the state object a sequence is keyed by."""


def test_a_pure_of_node_runs_once_per_key():
    probe, calls = _counting(gives=("geometry",), needs=("widths",), pure_of=("widths",))
    dag = (probe,)

    first = run(dag, {"widths": (64, 64)})["geometry"]
    again = run(dag, {"widths": (64, 64)})["geometry"]
    other = run(dag, {"widths": (16, 16)})["geometry"]

    assert first == again, "an identical key rebuilt the fact"
    assert other != first, "a different key returned the cached fact"
    assert len(calls) == 2, f"the body ran {len(calls)} times over two distinct keys"


def test_a_per_state_node_never_shares_between_two_live_sequences():
    probe, calls = _counting(gives=("scratch",), needs=("state", "widths"),
                             pure_of=("widths",), per_state=True)
    dag = (probe,)
    one, two = _Sequence(), _Sequence()

    a = run(dag, {"state": one, "widths": (64, 64)})["scratch"]
    a_again = run(dag, {"state": one, "widths": (64, 64)})["scratch"]
    b = run(dag, {"state": two, "widths": (64, 64)})["scratch"]

    assert a == a_again, "one sequence rebuilt its own scratch"
    assert b != a, "two concurrent sequences shared one buffer"
    assert len(calls) == 2


def test_a_dropped_sequence_drops_its_scratch():
    """The weakness is required, not incidental: a live table keyed on dead sequences
    is a leak of exactly the device allocations paging exists to bound."""
    @node(gives=("scratch",), needs=("state",), per_state=True)
    def probe(f):
        #: deliberately not closing over the state: a probe that recorded its own
        #: argument would pin the sequence and the table would look strong.
        return object()

    dag = (probe,)

    sequence = _Sequence()
    run(dag, {"state": sequence})
    assert len(probe.memo) == 1
    del sequence
    gc.collect()
    assert len(probe.memo) == 0, "the scratch outlived the sequence that owned it"


#: THE FACT BAG each shipped `per_state` node is driven with by the discipline gate
#: below. A node arriving with no bag fails there BY NAME, which is what makes the
#: discipline a gate rather than a habit.
def _decode_scratch_facts():
    from rola.ops.decode import derive_decode_geometry
    from tests.oracle.oracle_fixtures import _topology

    return {"geometry": derive_decode_geometry(_topology((16, 16)), d_v=64, decay=False),
            "BH": 2, "device": "cpu"}


PER_STATE_FACTS = {"scratch": _decode_scratch_facts}


def _shipped_per_state_nodes() -> list[Node]:
    from rola.engine.dags.decode_dag import DECODE_STEP
    from rola.engine.dags.prefill_dag import PREFILL_DAG

    seen: dict[int, Node] = {}
    for dag in (PREFILL_DAG, DECODE_STEP):
        seen.update({id(step): step for step in dag if step.per_state})
    return list(seen.values())


class _Poison:
    """A sequence whose every attribute read is a failure."""

    def __getattr__(self, item):
        raise AssertionError(f"a per_state node read the sequence: state.{item}")


def test_a_per_state_node_reads_the_state_as_a_KEY_and_never_as_an_INPUT():
    """THE DISCIPLINE: a `per_state` cache holds REBUILDABLE TRANSIENTS and nothing else.

    Two halves, and together they are the claim. STRUCTURALLY, every need but `state`
    is a declared memo key, so the cached object is a function of facts a rebuild
    has -- a need that is not a key is an input the memo cannot see change.
    FUNCTIONALLY, the body runs with the state replaced by an object that raises on
    any attribute read, so nothing about the sequence can enter what is cached, and
    two objects built for one geometry are interchangeable BY CONSTRUCTION rather than
    by inspection.

    The CROSS-STEP half -- that a reused object carries nothing from one step to the
    next -- is not assertable without a device and is gated where a real step runs:
    `DecodeScratch`'s counters are self-resetting and
    `tests/integration/test_decode_graph_step.py` replays a captured step across an
    admission against eager. This file is device-free by charter, and running one pair
    of steps here would sample one geometry where the poison probe covers every field.
    """
    nodes = _shipped_per_state_nodes()
    assert nodes, "no shipped node is per_state; this gate lost its subject"

    for step in nodes:
        assert set(step.needs) - {"state"} == set(step.pure_of), (
            f"{step.name} reads "
            f"{sorted(set(step.needs) - {'state'} - set(step.pure_of))} without "
            "declaring it a memo key: a per_state cache may hold only what a rebuild "
            "from the declared keys would produce")
        assert step.name in PER_STATE_FACTS, (
            f"{step.name} is a new per_state node with no bag in PER_STATE_FACTS; "
            "declare one so the transients-only discipline is proven for it too")

        facts = PER_STATE_FACTS[step.name]()
        one = step({**facts, "state": _Poison()})
        two = step({**facts, "state": _Poison()})

        assert type(one) is type(two)
        for name in one.__slots__:
            a, b = getattr(one, name), getattr(two, name)
            assert type(a) is type(b), f"{step.name}.{name}"
            if isinstance(a, torch.Tensor):
                assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device), (
                    f"{step.name}.{name} is not a function of the declared keys")
                #: A ZERO-ELEMENT buffer owns no bytes, so "shared" is not a property
                #: it can have -- torch gives every empty tensor the null pointer.
                #: Asserting distinctness there would be asserting an allocator detail.
                assert a.numel() == 0 or a.data_ptr() != b.data_ptr(), (
                    f"{step.name}.{name} is SHARED between two sequences' objects")


def test_a_memo_key_that_is_not_an_input_is_refused():
    with pytest.raises(ValueError, match="does not read"):
        node(gives=("geometry",), needs=("widths",), pure_of=("d_v",))(lambda f: None)


def test_a_per_state_node_must_read_the_state():
    with pytest.raises(ValueError, match="must read `state`"):
        node(gives=("scratch",), needs=("widths",), per_state=True)(lambda f: None)


def test_a_need_that_arrives_later_is_a_declaration_error():
    late, _ = _counting(gives=("arm",))
    early, _ = _counting(gives=("plan",), needs=("arm",))
    with pytest.raises(ValueError, match="reads"):
        validate_dag((early, late), seeds=())


def test_a_dag_declares_exactly_one_join_and_at_most_one_host_read():
    first, _ = _counting(gives=("a",), waits=("residency",))
    second, _ = _counting(gives=("b",), waits=("residency",))
    with pytest.raises(ValueError, match="exactly one Join"):
        validate_dag((first, second), seeds=())

    reader, _ = _counting(gives=("a",), host_sync=True)
    other, _ = _counting(gives=("b",), host_sync=True)
    with pytest.raises(ValueError, match="one device-to-host read"):
        validate_dag((reader, other), seeds=())


def test_a_masked_out_node_leaves_its_facts_null():
    """The runtime-null OUTPUT skip: intra-mask variation is a null plan field, never
    a mask split, so a skipped band's facts must be PRESENT and null rather than
    absent -- a downstream node reads them either way."""

    skipped, calls = _counting(gives=("arena",), masks=(Mask.M2,))
    facts = run((skipped,), {"mask": Mask.M1})
    assert facts["arena"] is None and not calls


def test_a_supplied_fact_short_circuits_its_producer():
    probe, calls = _counting(gives=("read_plane",))
    facts = run((probe,), {"read_plane": "supplied"})
    assert facts["read_plane"] == "supplied" and not calls


# ---------------------------------------------------------------------------
# the decode family's two structural declarations
# ---------------------------------------------------------------------------

def test_a_capturable_node_may_not_read_the_device_on_the_host():
    """The two attributes are one claim, so declaring both is refused at declaration."""
    with pytest.raises(ValueError, match="cannot be capturable"):
        node(gives=("grow",), host_sync=True, capturable=True)(lambda f: True)


def test_the_decode_step_is_declared_capturable_end_to_end():
    """THE CAPTURABILITY CLAIM, structurally: no host read, and every node says so.

    `tests/integration/test_decode_graph_step.py` proves a real capture replays; this
    proves the claim is a property of the DECLARATION, so a node added to the step
    without the attribute fails here rather than at someone's first `graph.replay()`.
    """
    from rola.engine.dags.decode_dag import DECODE_STEP

    assert all(step.capturable for step in DECODE_STEP)
    assert not any(step.host_sync for step in DECODE_STEP)


def test_a_reference_only_node_is_refused_by_every_shipped_dag():
    """§H.7's hand-written warning, made mechanical.

    `write_atom_bitmap` is the gate's independent second derivation of the write-atom
    set. Splicing it into a shipped DAG would make the gate compare the probe against
    itself, and `validate_dag` is where that is refused.
    """
    from rola.engine.dags.decode_dag import DECODE_STEP, write_atom_bitmap
    from rola.engine.dags.prefill_dag import PREFILL_DAG

    assert write_atom_bitmap.reference_only
    for dag in (PREFILL_DAG, DECODE_STEP):
        assert not any(step.reference_only for step in dag)
    with pytest.raises(ValueError, match="second derivation"):
        validate_dag((write_atom_bitmap,), ("routes", "geometry"))


def test_the_decode_dag_ends_at_the_step_and_its_verdict_is_read_outside_it():
    """The step is the LAST node, and no node reads the host. The verdict is the step's own output, read by
    `growth_pending` at the caller's synchronization -- after the launch that publishes it, and only where there is a
    page table to publish about -- so the whole tuple is the captured one."""
    from rola.engine.dags.decode_dag import DECODE_STEP

    assert all(isinstance(node, Node) for node in DECODE_STEP)
    assert DECODE_STEP[-1].name == "step", "the step is the last thing the decode tuple does"
    assert not any(node.host_sync for node in DECODE_STEP), "a host read cannot be captured"
