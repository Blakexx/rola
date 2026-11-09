"""The coverage gate (`tools/lint/coverage.py`): a kernel no declaration names, and a pointer to nothing, are findings.

See docs/internals/tools/coverage.md.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "lint"))

import coverage  # noqa: E402
import kernels  # noqa: E402


def test_an_undeclared_kernel_is_a_finding(monkeypatch):
    real = kernels.kernels()
    ghost = replace(real[0], name="ghost_kernel")
    monkeypatch.setattr(kernels, "kernels", lambda src=kernels.SRC: [*real, ghost])
    assert any("`ghost_kernel`" in f and "is not declared" in f for f in coverage.findings())


def test_a_test_node_that_is_gone_does_not_resolve():
    here = "tests/unit/test_coverage_gate.py"
    assert coverage.test_node_exists(f"{here}::test_a_test_node_that_is_gone_does_not_resolve[param]")
    assert not coverage.test_node_exists(f"{here}::test_that_was_renamed")
    assert not coverage.test_node_exists("tests/unit/no_such_file.py")


def test_every_sanitizer_row_the_declaration_names_launches_a_real_test():
    rows = coverage.sanitizer_rows()
    assert rows["decode"][1] and coverage.test_node_exists(rows["decode"][0])


def test_an_owed_obligation_is_backlog_and_not_a_finding():
    assert not any("owes" in f for f in coverage.findings())
    assert any("budget" in line for line in coverage.owed())
