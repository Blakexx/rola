"""rola's runner (benchmarks/bench/provider.py): it reads central cells, names an arm by its subject, the dials it reads
and a layer arm's construction, offers the arms this binary carries at the cell's shape, and refuses a cell by name --
an uncarried carry arm, an iteration build, a cell of another kind. No GPU: nothing here builds an arm, and the binary's
arm tables are planted."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "benchmarks", ROOT / "tools"):
    sys.path.insert(0, str(path))

import gen_shards  # noqa: E402

from bench import provider  # noqa: E402
from bench.subjects import SUBJECTS  # noqa: E402
from benchmarks.cells import by_name  # noqa: E402
from benchmarks.cells.layer import CONSTRUCTIONS  # noqa: E402

SHIPPED = [tuple(row) for row in gen_shards.CARRY_ARMS]
DECODE_CELL = "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"


@pytest.fixture
def binary(monkeypatch):
    """A binary carrying the shipped carry arms, the intra arm at depth 2 and the carry's window, and every decode arm."""
    import rola.ops.carry as carry
    import rola.ops.decode as decode
    import rola.ops.intra as intra

    tables = SimpleNamespace(carry=list(SHIPPED), intra=[(2, 256, carry.WINDOW, 0)],
                             decode=[(dv, d, decay) for dv in (32, 64) for d in (1, 2, 3, 4) for decay in (False, True)])
    monkeypatch.setattr(carry, "arms", lambda: tables.carry)
    monkeypatch.setattr(intra, "arms", lambda: tables.intra)
    monkeypatch.setattr(decode, "arms", lambda: tuple(tables.decode))
    return tables


def test_an_arm_is_named_by_its_subject_its_non_default_dials_and_its_construction():
    assert provider.arm_name("carry_forward") == "carry_forward"
    assert provider.arm_name("prefill_op", calls=4) == "prefill_op@calls=4"
    assert provider.arm_name("carry_forward", schedule="identity") == "carry_forward@schedule=identity"
    assert provider.arm_name("decode_step", construction="chunk-decode-w16") == "decode_step@layer=chunk-decode-w16"


def test_a_cells_arms_are_the_subjects_it_takes_under_the_dials_each_reads(binary):
    names = set(provider.arms(by_name("flagship-alt-k4")))
    assert {"carry_forward", "carry_forward@schedule=identity", "liveness_pass", "intra_forward", "prefill_op"} <= names
    assert not any("@schedule=" in name for name in names if not name.startswith("carry_forward"))
    assert all("schedule" in SUBJECTS[name.split("@")[0]].dials for name in names if "@schedule=" in name)
    #: a partial window has no intra grid: the combined op is not offered
    assert not any(name.startswith(("prefill_op", "intra_forward")) for name in provider.arms(by_name("flat-small-alt-k16")))


def test_a_layer_cells_arms_are_its_declared_constructions(binary):
    names = set(provider.arms(by_name(DECODE_CELL)))
    declared = sorted(c.name for c in CONSTRUCTIONS.values() if DECODE_CELL in c.cells)
    assert declared == ["chunk-decode-w16"]
    assert {"entmax_solve@layer=chunk-decode-w16", "decode_step@layer=chunk-decode-w16"} <= names
    assert all(name.endswith("@layer=chunk-decode-w16") for name in names)


def test_an_arm_is_offered_only_where_the_binary_carries_its_kernel(binary):
    binary.intra = []
    names = set(provider.arms(by_name("flagship-alt-k4")))
    assert "carry_forward" in names and not any(name.startswith(("prefill_op", "intra_forward")) for name in names)
    assert "decode_step@layer=chunk-decode-w16" in provider.arms(by_name(DECODE_CELL))
    binary.decode = []
    assert not any(name.startswith("decode_step") for name in provider.arms(by_name(DECODE_CELL)))


def test_a_cell_is_refused_by_name(binary):
    with pytest.raises(LookupError, match=r"deep3-dense: this binary carries no carry arm \(3, 64, 8\)"):
        provider.arms(by_name("deep3-dense"))
    with pytest.raises(TypeError, match="rola's runner takes a central carry cell or layer cell"):
        provider.arms(by_name("qkv-L1024-dv64"))
    binary.carry = []
    with pytest.raises(RuntimeError, match="iteration build"):
        provider.arms(by_name("flagship-alt-k4"))
