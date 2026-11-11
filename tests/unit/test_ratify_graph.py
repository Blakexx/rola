"""Ratify on the graph's build (tools/ratify.py's `--graph`): the census record's paths resolve against the runfiles
and its tools become this process's; an iteration build is told from the shipped one by the record's own rule; the
census reads each unit's report, and an instantiation two units emit is refused, as the compile path refuses it. No
CUDA, no GPU: the reports are text, the object scan stands in."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ratify  # noqa: E402
import sass  # noqa: E402

REPORT = """\
ptxas info    : Compiling entry function '_Z1kPi' for 'sm_86'
ptxas info    : Function properties for _Z1kPi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 360 bytes cmem[0]
"""

SHIPPED = {"archs": "80,86", "carry_arms": "", "decode_arms": ""}


@pytest.fixture()
def graph(tmp_path, monkeypatch):
    """A census record of two units under a runfiles root; restores this module's graph state after."""
    monkeypatch.setattr(ratify, "_GRAPH", {})
    monkeypatch.setattr(ratify, "cuda_bin", ratify.cuda_bin)
    monkeypatch.setattr(sass, "_GRAPH", {})
    monkeypatch.setattr(ratify, "toolchain", lambda name=None: type("T", (), {"archs": lambda self: ["80", "86"]})())
    runfiles = tmp_path / "runfiles" / "_main"
    (runfiles / "objs").mkdir(parents=True)
    for unit in ("a", "b"):
        (runfiles / "objs" / f"{unit}.ptxas.txt").write_text(REPORT if unit == "a" else "")
        (runfiles / "objs" / f"{unit}.pic.o").write_bytes(b"")

    def write(parameters=SHIPPED, both_report=False):
        if both_report:
            (runfiles / "objs" / "b.ptxas.txt").write_text(REPORT)
        census = runfiles / "census.json"
        census.write_text(json.dumps({
            "units": [{"source": f"csrc/rola/src/{u}.cu", "report": f"objs/{u}.ptxas.txt", "object": f"objs/{u}.pic.o"}
                      for u in ("a", "b")],
            "tools": {"ptxas": "../cuda+/bin/ptxas", "cuobjdump": "../cuda_tools+/bin/cuobjdump",
                      "nvdisasm": "../cuda_tools+/bin/nvdisasm", "mold": "../mold+/bin/mold"},
            "derivation": {"harness": "tools/derivation/derivation_harness.cu", "binary": "tools/derivation_harness"},
            "parameters": parameters}))
        ratify.use_graph(census, runfiles)
        return runfiles

    return write


def test_the_census_paths_resolve_against_the_runfiles_and_its_tools_are_this_process_s(graph):
    runfiles = graph()
    assert ratify.cuda_bin("ptxas") == str(runfiles.parent / "cuda+" / "bin" / "ptxas")
    assert sass.tool("cuobjdump") == str(runfiles.parent / "cuda_tools+" / "bin" / "cuobjdump")
    assert ratify._GRAPH["units"][0]["report"] == runfiles / "objs" / "a.ptxas.txt"
    assert ratify._GRAPH["harness_binary"] == runfiles / "tools" / "derivation_harness"


def test_the_shipped_parameters_are_not_an_iteration_build(graph):
    graph()
    assert ratify.graph_parameters() == (("80", "86"), False)


@pytest.mark.parametrize("change", [{"archs": "86"}, {"decode_arms": "0"}])
def test_any_other_parameter_is_an_iteration_build(graph, change):
    graph({**SHIPPED, **change})
    assert ratify.graph_parameters()[1] is True


def test_the_census_reads_each_unit_s_report(graph, monkeypatch):
    graph()
    monkeypatch.setattr(ratify, "_scan_object", lambda obj, names: {n: {"body_sha256": "x", "hot_locals": [],
                                                                       "local_ops": 0} for n in names})
    entries, unhonored, scan = ratify.measure_built()
    assert sorted(entries) == ["sm_86:_Z1kPi"]
    assert entries["sm_86:_Z1kPi"]["regs"] == 8
    assert unhonored == [] and scan["sm_86:_Z1kPi"]["body_sha256"] == "x"


def test_an_instantiation_two_units_emit_is_refused(graph, monkeypatch):
    graph(both_report=True)
    monkeypatch.setattr(ratify, "_scan_object", lambda obj, names: {})
    with pytest.raises(SystemExit, match="compiled by more than one translation unit"):
        ratify.measure_built()
