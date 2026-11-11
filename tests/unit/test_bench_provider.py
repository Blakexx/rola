"""rola's runner (measure/provider.py): it reads central cells, names an arm by its subject, its call count
and a layer arm's construction, offers the arms this binary carries at the cell's shape, and refuses a cell by name --
an uncarried carry arm, an iteration build, a cell of another kind. No GPU: nothing here builds an arm, and the binary's
arm tables are planted (the carry's as the arms the cells' own carves select: this line declares none)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "measure", ROOT / "tools"):
    sys.path.insert(0, str(path))

from measure import hardware, provider, subjects  # noqa: E402
from measure.cells import by_name  # noqa: E402
from measure.cells.layer import CONSTRUCTIONS  # noqa: E402
from measure.subjects import SUBJECTS  # noqa: E402

CARRY_CELLS = ("flagship-alt-k4", "flat-small-alt-k16")
DECODE_CELL = "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"


@pytest.fixture
def binary(monkeypatch):
    """A binary carrying the carry arms `CARRY_CELLS`' carves select and every decode arm (the intra reads its widths
    by rule)."""
    import rola.ops.carry as carry
    import rola.ops.decode as decode

    tables = SimpleNamespace(carry=[carry.arm_key(subjects.cell_carve(by_name(name))) for name in CARRY_CELLS],
                             decode=[(dv, d, decay) for dv in (32, 64) for d in (1, 2, 3, 4) for decay in (False, True)])
    monkeypatch.setattr(carry, "arms", lambda: tables.carry)
    monkeypatch.setattr(decode, "arms", lambda: tuple(tables.decode))
    #: the window an sm_86 card's build runs (hardware.json), planted as the tables are: no device answers it here
    monkeypatch.setattr(subjects, "_shared_window", lambda: hardware.load()["arch"]["sm_86"]["build"]["window"])
    return tables


def test_an_arm_is_named_by_its_subject_its_call_count_and_its_construction():
    assert provider.arm_name("facts_pass") == "facts_pass"
    assert provider.arm_name("prefill", calls=4) == "prefill@calls=4"
    assert provider.arm_name("decode_step", construction="chunk-decode-w16") == "decode_step@layer=chunk-decode-w16"


def test_a_cells_arms_are_the_subjects_it_takes(binary):
    names = set(provider.arms(by_name("flagship-alt-k4")))
    assert {"prefill", "facts_pass"} <= names
    assert all(name.split("@")[0] in SUBJECTS for name in names)
    #: a call shorter than a window is a call: the prefill is offered
    assert "prefill" in provider.arms(by_name("flat-small-alt-k16"))


def test_a_layer_cells_arms_are_its_declared_constructions(binary):
    names = set(provider.arms(by_name(DECODE_CELL)))
    declared = sorted(c.name for c in CONSTRUCTIONS.values() if DECODE_CELL in c.cells)
    assert declared == ["chunk-decode-w16"]
    assert {"entmax_solve@layer=chunk-decode-w16", "decode_step@layer=chunk-decode-w16"} <= names
    assert all(name.endswith("@layer=chunk-decode-w16") for name in names)


def test_an_arm_is_offered_only_where_the_binary_carries_its_kernel(binary, monkeypatch):
    import rola.ops.intra as intra

    monkeypatch.setattr(intra, "reads", lambda widths: False)
    names = set(provider.arms(by_name("flagship-alt-k4")))
    assert "facts_pass" in names and "prefill" not in names
    assert "decode_step@layer=chunk-decode-w16" in provider.arms(by_name(DECODE_CELL))
    binary.decode = []
    assert not any(name.startswith("decode_step") for name in provider.arms(by_name(DECODE_CELL)))


def test_a_cell_is_refused_by_name(binary, monkeypatch):
    import gen_shards

    assert "prefill" not in provider.arms(by_name("deep3-dense"))
    monkeypatch.setattr(gen_shards, "CARRY_ARMS", tuple(binary.carry))
    with pytest.raises(TypeError, match="rola's runner takes a central carry cell or layer cell"):
        provider.arms(by_name("qkv-L1024-dv64"))
    binary.carry = []
    with pytest.raises(RuntimeError, match="iteration build"):
        provider.arms(by_name("flagship-alt-k4"))
