# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""`benchmarks/registry.py`: rola's registrations with the measurement service load as units of the kinds they claim,
name tools and data this checkout carries, depend on the build, and accept a cell exactly where this binary runs it.
No GPU and no built binary: the arm tables are planted and nothing is set up."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "benchmarks", ROOT / "tools"):
    sys.path.insert(0, str(path))

import gen_shards  # noqa: E402
from rola_devtools.measure.worker import Registry  # noqa: E402

from benchmarks import registry as rola_registry  # noqa: E402
from benchmarks.cells import by_name  # noqa: E402

REF = "benchmarks.registry:registry"


@pytest.fixture
def binary(monkeypatch):
    import rola.ops.carry as carry
    import rola.ops.decode as decode
    import rola.ops.intra as intra

    tables = SimpleNamespace(carry=[tuple(row) for row in gen_shards.CARRY_ARMS], intra=[(2, 256, carry.WINDOW, 0)],
                             decode=[(dv, d, False) for dv in (32, 64) for d in (1, 2, 3, 4)])
    monkeypatch.setattr(carry, "arms", lambda: tables.carry)
    monkeypatch.setattr(intra, "arms", lambda: tables.intra)
    monkeypatch.setattr(decode, "arms", lambda: tuple(tables.decode))
    return tables


def test_every_registration_is_a_unit_of_its_kind_and_every_unit_but_the_clock_depends_on_the_build():
    reg = Registry(REF)
    kinds = {name: unit.kind for name, unit in reg.units.items()}
    assert (kinds["build"], kinds["clock"], kinds["carry.phases"], kinds["carry_forward"]) == (
        "build", "clock", "instrument", "arm")
    assert "decode_step@layer=chunk-decode-w16" in kinds
    for name, registration in reg.registrations.items():
        if name not in ("build", "clock"):
            assert registration.deps == ("build",), name
        if kinds[name] != "clock":
            assert reg.units[name].location.startswith("rola/"), name


def test_every_tool_and_its_data_are_this_checkouts():
    for unit in Registry(REF).units.values():
        if isinstance(unit, rola_registry.Tool):
            assert (ROOT / unit.tool).is_file(), unit.tool
            assert all((ROOT / d).exists() for d in unit.data if not d.startswith("build/")), unit.data


def test_a_unit_accepts_a_cell_where_this_binary_runs_it(binary):
    units = Registry(REF).units
    flagship, deep, decode = by_name("flagship-alt-k4"), by_name("deep3-dense"), by_name("layer-B2-T128-H2-h128-dv64-fp32-s4-dec8")
    assert units["carry.phases"].accepts(flagship) is None
    assert "carries no carry arm (3, 64, 8)" in units["carry.phases"].accepts(deep)
    assert units["carry.phases"].accepts(decode).endswith("not a carry cell")
    assert units["carry_forward"].accepts(flagship) is None
    assert units["carry_forward"].accepts(decode) is not None
    assert units["decode_step@layer=chunk-decode-w16"].accepts(decode) is None
    assert units["decode_step@layer=chunk-decode-w16"].accepts(flagship) is not None
    assert units["carry.sass"].accepts(None) is None


def test_no_registration_names_a_machine_path():
    blob = json.dumps([registration.params for registration in Registry(REF).registrations.values()])
    assert "/home/" not in blob and "/Users/" not in blob
