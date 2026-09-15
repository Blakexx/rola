"""`rola` loads its kernels from the binary plugin at exactly its own version, and refuses otherwise by naming the extra.

No GPU: the plugin's modules are planted, so what is tested is the loader's decision (`rola.ops._ext._load`), never the
binary.
"""
from __future__ import annotations

import types

import pytest

from rola import __version__
from rola.ops import _ext


def _plant(monkeypatch, record_version=None, library_error=None):
    """`importlib.import_module` as a machine with this plugin: no record when `record_version` is None."""
    def import_module(name):
        if name == f"{_ext.PLUGIN}._build_config":
            if record_version is None:
                raise ImportError(f"No module named {_ext.PLUGIN!r}")
            return types.SimpleNamespace(BUILD_CONFIG={"version": record_version})
        if name == f"{_ext.PLUGIN}._C":
            if library_error:
                raise ImportError(library_error)
            return types.SimpleNamespace(loaded=True)
        raise AssertionError(f"the loader imported {name}")
    monkeypatch.setattr(_ext.importlib, "import_module", import_module)


def test_a_plugin_at_this_version_loads(monkeypatch):
    _plant(monkeypatch, record_version=__version__)
    module, why = _ext._load()
    assert module.loaded and why is None


def test_no_plugin_is_named(monkeypatch):
    _plant(monkeypatch)
    module, why = _ext._load()
    assert module is None and f"no `{_ext.PLUGIN}` binary plugin is installed" in why


def test_another_version_never_loads_its_library(monkeypatch):
    _plant(monkeypatch, record_version="0.0.0", library_error="the library must not be reached")
    module, why = _ext._load()
    assert module is None and "0.0.0" in why and __version__ in why


def test_the_refusal_names_the_extra_at_this_version():
    message = _ext._MISSING.format(why="no plugin", extra=_ext.EXTRA, version=__version__)
    assert f'pip install "rola[{_ext.EXTRA}]=={__version__}"' in message
    with pytest.raises(ImportError):
        raise ImportError(message)
