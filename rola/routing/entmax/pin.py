"""The version pin for the `entmax` (DeepSPIN, PyPI) producer reference.

CLOSED-WORLD RULE, entmax's shape of it. `csrc/third_party/cutlass/PIN.json` pins a
VENDORED codegen input with a tag, a commit and a tree digest, and refuses a build
whose vendored subtree does not match (`setup.py::_gate_vendored`,
`tools/lint/lint_standards.py::check_vendored_pin`). `entmax` is the same class of input by a
different route -- not vendored, because a pure-python test dependency has nothing to
vendor, but pip-installed and unpinned in `pyproject.toml` before this module existed.

WHY THE PIN IS THE ONLY GUARD. `tools/oracle_dual_run.py` runs the OLD side as a real
prior commit of THIS repository in a worktree, and the NEW side as the tip -- but both
sides import the SAME installed `entmax`. A dependency bump that changes `entmax15`'s
tie-breaking or `entmax_bisect`'s semantics re-anchors every Tier 1 routing gate and
the dual run stays green throughout, because there is no OLD `entmax` for the NEW one
to be diffed against. This module -- checked at `reference.py`'s import site, not just
in a test -- is the only thing in this tree that can refuse that drift.

`PIN.json` records a VERSION (like CUTLASS's tag) and a PACKAGE DIGEST (like CUTLASS's
`tree_sha256`), because a version string alone permits a re-published or editable
install of the same nominal release to drift silently under it; the digest is over the
files actually on disk, the way CUTLASS's is.
"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

PIN_PATH = Path(__file__).resolve().parent / "PIN.json"


def load_pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def installed_package_root() -> Path:
    """Where pip put `entmax`, not a path this repository owns."""
    import entmax as _entmax_pkg
    return Path(_entmax_pkg.__file__).resolve().parent


def package_sha256(root: Path) -> tuple[str, int]:
    """The recipe `PIN.json` states, re-implemented from the RECIPE so that a digest
    checked by the same code that wrote it checks something."""
    digest = hashlib.sha256()
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def gate() -> None:
    """Refuse to proceed if the installed `entmax` is not the ratified one.

    Two independent checks, because either alone is beatable: a version string does
    not prove the files it names are the files on disk (an editable or re-published
    install can carry the same version string over different bytes), and a bare
    digest with no recorded version gives a reader nothing to look up. Called at
    `reference.py`'s import site -- the crown-jewel routing reference is unusable
    without a live `entmax` import, so this is the one place that import cannot be
    skipped.
    """
    pin = load_pin()
    installed_version = metadata.version("entmax")
    if installed_version != pin["version"]:
        raise RuntimeError(
            f"CLOSED-WORLD: installed entmax=={installed_version}, ratified pin "
            f"names {pin['version']} ({PIN_PATH}). The routing reference's measured "
            "figures (the 1.421e-14 dual-run agreement, PRODUCER_ATOL) describe the "
            "ratified version only. The dual-run protocol imports the same install on "
            "both sides and cannot see this drift itself -- re-measure against the "
            "new version and re-pin (update PIN.json's version and package_sha256) "
            "before using it as a reference.")
    root = installed_package_root()
    here, count = package_sha256(root)
    if count != pin["file_count"] or here != pin["package_sha256"]:
        raise RuntimeError(
            f"CLOSED-WORLD: installed entmax=={installed_version} at {root} does not "
            f"match its pinned digest ({PIN_PATH}) -- {count} files on disk vs "
            f"{pin['file_count']} pinned, digest {here} vs {pin['package_sha256']}. "
            "Same version string, different bytes: an editable or re-published "
            "install moved under the pin. Re-measure and re-pin before trusting it as "
            "a reference.")
