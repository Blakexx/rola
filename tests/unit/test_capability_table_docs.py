"""The README's capability table is the activation registry's own output: the block between its generated markers is
`rola.routing.activations.capability_table()` byte for byte, placed by tools/supported.py. A test, not a lint: it runs
the registry it reads (docs/testing.md); the README's other generated block is the standards lint's
(`check_supported_table`). No CUDA, no GPU."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import supported  # noqa: E402

MARKER = "rola.routing.activations.capability_table"


def _block(text: str) -> str:
    found = supported.block(text, MARKER)
    assert found is not None, f"README.md has no generated block for {MARKER}"
    return found


def test_the_readme_capability_table_is_the_registry_s():
    assert _block(supported.README.read_text()) == supported.BLOCKS[MARKER]()


def test_a_hand_edit_is_caught():
    end = supported.markers(MARKER)[1]
    edited = supported.README.read_text().replace(end, f"x\n{end}")
    assert _block(edited) != supported.BLOCKS[MARKER]()
