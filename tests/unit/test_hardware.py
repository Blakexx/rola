# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HARDWARE CONFIGURATION (`measure/hardware.py` over `csrc/rola/src/common/hardware.json`): the header is what the
file generates and every source holds; a tampered rate or a stale header is caught; a product's override wins over its
architecture's default; and the Python cost is `common/cost.cuh`'s on its parity cases."""
from __future__ import annotations

import json
import re

from measure import hardware


def test_the_configuration_checks_clean():
    assert hardware.check() == []


def test_a_measured_rate_off_its_record_is_a_finding():
    cfg = hardware.load()
    cfg["product"]["rtx3080ti"]["rates"]["mma"]["hmma_f16_f32"]["cycles"] = 30.0
    assert any("hmma_f16_f32" in f for f in hardware.check(cfg))


def test_a_stale_header_is_a_finding():
    assert any("hardware.cuh" in f for f in hardware.check(header_text=hardware.HEADER.read_text() + "\n"))


def test_a_product_override_wins_over_its_architecture():
    assert hardware.resolve("sm_86")["mma.hmma_f16_f32"].cycles == 16
    assert abs(hardware.resolve(product="rtx3080ti")["mma.hmma_f16_f32"].cycles - 32.13) < 0.01
    assert hardware.product_of("NVIDIA GeForce RTX 3080 Ti") == "rtx3080ti"
    assert json.loads(hardware.CONFIG.read_text()) == hardware.load()


def test_the_python_cost_is_the_headers_on_its_parity_cases():
    text = (hardware.COMMON / "cost.cuh").read_text()
    cases = re.findall(r"static_assert\(cost\(Quanta\{\{([\d, ]+)\}, (\d+), (\d+), (\d+)\},\s*"
                       r"Load\{\{([\d, ]+)\}, (\d+), (\d+), (\d+)\}\)\s*==\s*(\d+)\);", text)
    assert len(cases) == 2
    q = hardware.quanta(hardware.resolve(product="rtx3080ti"))
    for mma_q, wave_q, atom_q, atom_wave_q, mma, waves, atoms, atom_waves, want in cases:
        assert q == hardware.Quanta(tuple(int(x) for x in mma_q.split(",")), int(wave_q), int(atom_q),
                                    int(atom_wave_q))
        assert hardware.cost(q, [int(x) for x in mma.split(",")], int(waves), int(atoms), int(atom_waves)) == int(want)


def test_every_architecture_names_its_family_and_each_built_one_its_window():
    cfg = hardware.load()
    assert {a: r["build"]["family"] for a, r in cfg["arch"].items()} == {
        "sm_80": "ampere", "sm_86": "ampere", "sm_87": "ampere", "sm_89": "ampere", "sm_90": "hopper"}
    cfg["arch"]["sm_86"]["build"].pop("window")
    assert any("sm_86 is built but its build chooses no window" in f for f in hardware.check(cfg))


def test_products_equal_in_build_traits_share_a_variant_and_an_override_is_the_next():
    cfg = hardware.load()
    assert {p: hardware.variant_of(p, cfg) for p in cfg["product"]} == {"a100": 0, "rtx3080ti": 0}
    cfg["product"]["rtx3080ti"]["build"] = {"window": 1024}
    cfg["product"]["other86"] = {"arch": "sm_86", "device": "NVIDIA other86", "build": {"window": 1024}}
    assert hardware.variant_of("rtx3080ti", cfg) == hardware.variant_of("other86", cfg) == 1
    assert [(a, v) for a, v, _ in hardware.builds(cfg) if a == "sm_86"] == [("sm_86", 0), ("sm_86", 1)]
    assert "if (b.cc == 860 && b.variant == 1)" in hardware.header(cfg)


def test_a_product_build_override_outside_the_traits_is_a_finding():
    cfg = hardware.load()
    cfg["product"]["rtx3080ti"]["build"] = {"family": "hopper"}
    assert any("overrides a trait outside window" in f for f in hardware.check(cfg))


def test_a_product_build_override_is_refused_until_a_variant_is_gated():
    cfg = hardware.load()
    cfg["product"]["rtx3080ti"]["build"] = {"window": 1024}
    assert any("a build variant is not gated yet" in f for f in hardware.check(cfg))
