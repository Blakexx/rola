"""A measurement row names the imported `rola` by its path inside the measured checkout, refuses a `rola` imported from
outside it, and carries a worker's message without this machine's paths (tools/probe_cells.py). No GPU."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import probe_cells  # noqa: E402


def test_the_imported_rola_is_named_inside_its_checkout(tmp_path):
    checkout = tmp_path / "wt"
    (checkout / "rola").mkdir(parents=True)
    assert probe_cells.rola_file_in(str(checkout), str(checkout / "rola" / "__init__.py")) == ("rola/__init__.py", None)
    assert probe_cells.rola_file_in(str(checkout), "rola/__init__.py") == ("rola/__init__.py", None)
    assert probe_cells.rola_file_in(str(checkout), None) == (None, None)


def test_a_rola_imported_from_outside_the_checkout_is_a_refusal(tmp_path):
    (tmp_path / "wt").mkdir()
    path, refusal = probe_cells.rola_file_in(str(tmp_path / "wt"), str(tmp_path / "other" / "rola" / "__init__.py"))
    assert path is None and "outside the measured checkout wt" in refusal


def test_a_worker_message_carries_no_machine_path(tmp_path):
    from rola_results import machine_paths

    checkout = tmp_path / "wt"
    checkout.mkdir()
    message = f'File "{checkout.resolve()}/tools/probe_cells.py", line 9\nFile "{Path.home()}/venv/torch.py"'
    portable = probe_cells.portable_error(message, str(checkout))
    assert portable.startswith('File "tools/probe_cells.py"') and machine_paths(portable) == []
