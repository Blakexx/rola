"""rola's runner (benchmarks/bench/provider.py) and the comparison (tools/compare.py): a cell is its data provider and
parameters; the runner names an arm by its subject and the dials it reads, offers the arms this binary carries at the
cell's shape, and refuses a cell by name -- an uncarried carry arm, an iteration build, data of another library; the
comparison's point is a registered one narrowed by name, or one of rola cells. No GPU: nothing here builds an arm, and
the binary's arm tables are planted."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "benchmarks", ROOT / "tools"):
    sys.path.insert(0, str(path))

import compare  # noqa: E402
import gen_shards  # noqa: E402
from rola_devtools.cells import build  # noqa: E402

from bench import provider  # noqa: E402
from bench.subjects import SUBJECTS  # noqa: E402
from benchmarks.cells import by_name  # noqa: E402
from benchmarks.cells.layer import by_name as layer_by_name  # noqa: E402
from benchmarks.cells.registry import registry  # noqa: E402

SHIPPED = [tuple(gen_shards.CARRY_ARMS[i]) for i in gen_shards.CARRY_SHIPPED_ARMS]


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


def data(name):
    return build(registry().cell(name))


def test_a_cell_is_its_data_provider_and_its_parameters():
    assert data("flat-small-alt-k16") == by_name("flat-small-alt-k16")
    layer = next(name for name, c in registry().cells.items() if c["data"].endswith(":layer_cell"))
    assert data(layer) == layer_by_name(layer)


def test_an_arm_is_named_by_its_subject_and_its_non_default_dials():
    assert provider.arm_name("carry_forward") == "carry_forward"
    assert provider.arm_name("prefill_op", calls=4, state="continuation") == "prefill_op@calls=4@state=continuation"
    assert provider.arm_name("carry_forward", schedule="identity") == "carry_forward@schedule=identity"


def test_a_cells_arms_are_the_subjects_it_takes_under_the_dials_each_reads(binary):
    names = set(provider.arms(data("flagship-alt-k4")))
    assert {"carry_forward", "carry_forward@schedule=identity", "liveness_pass", "intra_forward",
            "prefill_op@state=paged"} <= names
    assert not any("@schedule=" in name for name in names if not name.startswith("carry_forward"))
    assert all("state" in SUBJECTS[name.split("@")[0]].dials for name in names if "@state=" in name)
    #: a partial window has no intra grid: the combined op is not offered
    assert not any(name.startswith(("prefill_op", "intra_forward")) for name in provider.arms(data("flat-small-alt-k16")))


def test_an_arm_is_offered_only_where_the_binary_carries_its_kernel(binary):
    binary.intra = []
    names = set(provider.arms(data("flagship-alt-k4")))
    assert "carry_forward" in names and not any(name.startswith(("prefill_op", "intra_forward")) for name in names)
    decode_cell = next(name for name, c in registry().cells.items()
                       if c["data"].endswith(":layer_cell") and c["params"]["decode_steps"] > 0)
    assert "decode_step" in provider.arms(data(decode_cell))
    binary.decode = []
    assert "decode_step" not in provider.arms(data(decode_cell))


def test_a_cell_is_refused_by_name(binary):
    with pytest.raises(LookupError, match=r"deep3-dense: this binary carries no carry arm \(3, 64, 8\)"):
        provider.arms(data("deep3-dense"))
    with pytest.raises(TypeError, match="rola's runner takes a carry cell"):
        provider.arms({"tokens": 256})
    binary.carry = []
    with pytest.raises(RuntimeError, match="iteration build"):
        provider.arms(data("flagship-alt-k4"))


def test_the_comparisons_arms_parse_as_their_flags_say():
    arm = compare.rola_arm("label:tip,arm:carry_forward@schedule=identity,worktree:/w/tip,venv:/w/venv-tip")
    assert arm == {"label": "tip", "arm": "carry_forward@schedule=identity", "worktree": "/w/tip", "venv": "/w/venv-tip"}
    foreign = compare.foreign_arm("label:attention,provider:pkg.attention:arms,arm:flash,python:/p/python,cwd:/c,"
                                  "runner:attn")
    assert (foreign["provider"], foreign["runner"]) == ("pkg.attention:arms", "attn")
    with pytest.raises(argparse.ArgumentTypeError, match="lacks"):
        compare.foreign_arm("label:attention,arm:flash")


def test_the_comparisons_point_is_registered_and_narrowed_by_name_or_is_rola_cells(tmp_path):
    extra = tmp_path / "registry.json"
    extra.write_text(json.dumps({"schema": 1, "data": "pkg.cells:qkv",
                                 "cells": [{"name": "attn-L1024-dv64", "tokens": 1024, "dv": 64}],
                                 "points": [{"name": "L1024", "equal": ["tokens", "dv"],
                                             "runners": {"rola": ["flagship-alt-k4", "flagship-dense"],
                                                         "attention": ["attn-L1024-dv64"]}}]}))

    def args(**kw):
        return SimpleNamespace(**{"point": None, "cells": None, "registry": [str(extra)], **kw})

    point = compare.point_of(args(point="L1024", cells="flagship-dense,attn-L1024-dv64"))
    assert {r: [c["name"] for c in cells] for r, cells in point["runners"].items()} == {
        "rola": ["flagship-dense"], "attention": ["attn-L1024-dv64"]}
    assert [c["name"] for c in compare.point_of(args(cells="flat-small-alt-k16"))["runners"]["rola"]] == [
        "flat-small-alt-k16"]
    with pytest.raises(SystemExit, match="not cells of point L1024"):
        compare.point_of(args(point="L1024", cells="flat-small-alt-k16"))
    with pytest.raises(SystemExit, match="name a --point"):
        compare.point_of(args())

