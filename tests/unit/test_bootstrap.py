"""`tools/bootstrap.py`'s one restated rule -- the dev config's directory and the rola-devtools checkout its workspace
section names -- agrees with the reader's (`rola_devtools.config`): the graph's front door finds the checkout the dev
config names, before the reader is importable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import bootstrap  # noqa: E402
import dev_config  # noqa: E402
from rola_devtools import config as reader  # noqa: E402


def test_the_pointer_is_the_readers():
    assert bootstrap.POINTER == reader.POINTER


@pytest.mark.parametrize("named", [False, True])
def test_the_devtools_checkout_is_the_dev_configs(tmp_path, monkeypatch, named):
    root = tmp_path / "config"
    root.mkdir()
    if named:
        (root / "workspace.json").write_text(json.dumps({"devtools": str(tmp_path / "elsewhere")}))
    monkeypatch.setenv(reader.POINTER, str(root))
    dev_config.reload()
    try:
        assert bootstrap.devtools() == Path(dev_config.get("workspace.devtools"))
    finally:
        monkeypatch.delenv(reader.POINTER)
        dev_config.reload()
