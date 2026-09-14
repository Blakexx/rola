"""The `entmax` (DeepSPIN, PyPI) version pin -- checkable without a build.

Mirrors `test_vendored_pin.py`'s shape for the CUTLASS pin, adapted for a pip-installed
reference rather than a vendored subtree: `rola/routing/entmax/pin.py::gate()` refuses
an installed `entmax` whose version or on-disk bytes drift from
`rola/routing/entmax/PIN.json`, and this file checks the three claims the pin makes:

1. the recorded digest describes the package that is actually installed;
2. a version mismatch is refused, named on both sides;
3. a digest mismatch under the SAME version string is refused too -- the reason both
   checks exist rather than the version alone (an editable or re-published install can
   carry one version string over different bytes).

`gate()` already ran once, successfully, when this test file imported
`rola.routing.entmax.reference` (directly or transitively) -- so a version drift on
THIS box would already have failed collection before reaching these tests. That is the
gate doing its job, not a reason to skip checking it; the tests below drive `gate()`
and `pin.py`'s helpers directly, with a planted wrong version, so the refusal itself is
exercised rather than assumed from the fact that collection succeeded.

Restored 2026-08-29 (S5c ITEM 0): this file was folded into
`tools/lint/lint_standards.py::check_entmax_pin` by K46, but it EXECUTES real code
(`pin.gate()`, monkeypatched refusal shapes) rather than reading files -- a check that
executes the code under test is a test, not a lint (docs/testing.md), and a lint that
imports `rola` crashes under pre-commit's isolated hook environment, which has no
`rola` installed. `check_entmax_pin` is removed from lint_standards.py; this is its
restoration, verbatim.
"""
from __future__ import annotations

import pytest

from rola.routing.entmax import pin


def test_pin_records_every_field_the_gate_relies_on():
    recorded = pin.load_pin()
    for key in ("name", "upstream", "version", "package_sha256", "file_count"):
        assert recorded.get(key), f"PIN.json is missing `{key}`"
    assert recorded["distribution"] == "PyPI"


def test_installed_entmax_matches_its_pin():
    """The green path: what `gate()` runs on every import of `reference.py` today."""
    recorded = pin.load_pin()
    root = pin.installed_package_root()
    here, count = pin.package_sha256(root)
    assert count == recorded["file_count"], (
        f"installed entmax file count moved: {count} on disk, "
        f"{recorded['file_count']} pinned")
    assert here == recorded["package_sha256"], (
        "the installed entmax package does not match its pin -- gate() would refuse "
        "this install")
    # gate() must therefore not raise, which is the claim these two asserts predict.
    pin.gate()


def test_a_version_drift_is_refused_and_both_versions_are_named(monkeypatch):
    ratified = pin.load_pin()["version"]
    monkeypatch.setattr(pin.metadata, "version", lambda name: "9.9.9")
    with pytest.raises(RuntimeError, match=r"installed entmax==9\.9\.9") as excinfo:
        pin.gate()
    assert ratified in str(excinfo.value), (
        "the refusal names the drifted version but not the ratified one -- a reader "
        "would have to open PIN.json to know what to roll back to")


def test_a_digest_drift_under_the_same_version_is_refused(monkeypatch, tmp_path):
    """The check the version string alone cannot make: same `version`, different
    bytes on disk (an editable or re-published install moved under the pin)."""
    fake_root = tmp_path / "entmax"
    fake_root.mkdir()
    (fake_root / "__init__.py").write_text("# not the ratified entmax\n")
    monkeypatch.setattr(pin, "installed_package_root", lambda: fake_root)
    with pytest.raises(RuntimeError, match="does not match its pinned digest"):
        pin.gate()


def test_a_missing_pin_file_is_not_silently_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(pin, "PIN_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        pin.gate()
