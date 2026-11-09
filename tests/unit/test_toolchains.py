"""The toolchain records (tools/toolchains.py): the configured assembler selects its record, and everything a record
cannot vouch for is refused -- an undeclared assembler, a malformed record, two records for one assembler, a torch of
another CUDA major, and a build directory another toolkit wrote. No CUDA, no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import toolchains  # noqa: E402

A = "ptxas: NVIDIA (R) Ptx optimizing assembler Cuda compilation tools, release 13.0, V13.0.48 Build x_0"
B = "ptxas: NVIDIA (R) Ptx optimizing assembler Cuda compilation tools, release 12.4, V12.4.99 Build y_0"


@pytest.fixture()
def declared(tmp_path, monkeypatch):
    def make(**records) -> Path:
        for name, record in records.items():
            (tmp_path / "toolchains").mkdir(exist_ok=True)
            (tmp_path / "toolchains" / f"{name}.json").write_text(json.dumps(record))
        monkeypatch.setattr(toolchains, "RECORDS", tmp_path / "toolchains")
        monkeypatch.setattr(toolchains, "MANIFESTS", tmp_path / "manifests")
        toolchains.records.cache_clear()
        return tmp_path

    yield make
    toolchains.records.cache_clear()


def _record(ptxas, major=13):
    return {"ptxas": ptxas, "readers": {"cuobjdump": f"cuobjdump {major}", "nvdisasm": f"nvdisasm {major}"},
            "sanitizer": f"compute-sanitizer {major}",
            "cuda_major": major, "torch_index": f"https://download.pytorch.org/whl/cu{major}0"}


def test_the_assembler_selects_its_record_whitespace_normalized(declared):
    root = declared(cu13=_record(A), cu124=_record(B, 12))
    (root / "manifests" / "cu13").mkdir(parents=True)
    (root / "manifests" / "cu13" / "sm_86.json").write_text("{}")
    toolchain = toolchains.for_ptxas(A.replace(" ", "\n  "))
    assert toolchain.name == "cu13"
    assert toolchain.manifest_path("80") == root / "manifests" / "cu13" / "sm_80.json"
    assert toolchain.archs() == ["86"]


def test_an_undeclared_assembler_is_refused_naming_the_declared_ones(declared):
    declared(cu13=_record(A))
    with pytest.raises(toolchains.ToolchainError, match="cu13"):
        toolchains.for_ptxas(B)


def test_two_records_for_one_assembler_are_refused(declared):
    declared(first=_record(A), second=_record(A))
    with pytest.raises(toolchains.ToolchainError, match="same assembler"):
        toolchains.records()


@pytest.mark.parametrize("record", [{"ptxas": A, "cuda_major": 13}, {**_record(A), "extra": 1},
                                    {**_record(A), "cuda_major": "13"},
                                    {**_record(A), "readers": {"cuobjdump": "cuobjdump 13"}}])
def test_a_malformed_record_is_refused(declared, record):
    declared(bad=record)
    with pytest.raises(toolchains.ToolchainError):
        toolchains.records()


def test_torch_of_another_cuda_major_is_refused_naming_the_index(declared):
    declared(cu13=_record(A))
    toolchain = toolchains.named("cu13")
    toolchains.torch_pairing(toolchain, "13.2")
    for torch_cuda in ("12.4", None):
        with pytest.raises(toolchains.ToolchainError, match="whl/cu130"):
            toolchains.torch_pairing(toolchain, torch_cuda)
