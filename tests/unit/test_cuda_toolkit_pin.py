"""The pinned CUDA toolkit (tools/cuda_toolkit_pin.json) is the graph's: rules_cuda's toolkit reads the manifest the pin
names, by the same digest and version (MODULE.bazel's `cuda.redist_json`), every component the graph compiles with is
one of the pin's, and its compiler is a declared toolchain. So the nvcc and ptxas outside the graph's rules are the
graph's own. No CUDA, no GPU."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import toolchains  # noqa: E402

PIN = json.loads((ROOT / "tools" / "cuda_toolkit_pin.json").read_text())


def _redist_json() -> str:
    """MODULE.bazel's `cuda.redist_json(...)` call."""
    return re.search(r"^cuda\.redist_json\((.*?)^\)", (ROOT / "MODULE.bazel").read_text(), re.S | re.M).group(1)


def test_the_graphs_toolkit_reads_the_pins_manifest():
    call = _redist_json()
    assert re.search(r'sha256 = "([0-9a-f]{64})"', call).group(1) == PIN["manifest_sha256"]
    assert re.search(r'version = "([^"]+)"', call).group(1) == PIN["version"]
    assert PIN["manifest_url"].endswith(f"/redistrib_{PIN['version']}.json")


def test_every_component_the_graph_compiles_with_is_pinned():
    #: rules_cuda names a component without the manifest's `cuda_` or `lib` prefix
    short = re.findall(r'"([a-z_]+)"', re.search(r"components = \[(.*?)\]", _redist_json(), re.S).group(1))
    missing = [c for c in short if f"cuda_{c}" not in PIN["components"] and f"lib{c}" not in PIN["components"]]
    assert short and not missing, missing


def test_the_pinned_compiler_is_a_declared_toolchain():
    assert any(PIN["version_string"] in toolchain.ptxas for toolchain in toolchains.records().values())
