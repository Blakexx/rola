"""THE TOOLCHAINS: each a named record of the exact assembler a fatbin is built and ratified under.

`tools/toolchains/<name>.json` records one toolchain; its ratification is `tools/manifests/<name>/sm_XX.json`. The build
takes no toolchain parameter: the configured toolkit's `ptxas --version` selects the record that names it, and a
toolkit no record names refuses. Docs: docs/internals/tools/toolchains.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "tools" / "toolchains"
MANIFESTS = ROOT / "tools" / "manifests"

_FIELDS = {"ptxas": str, "cuda_major": int, "torch_index": str}


class ToolchainError(RuntimeError):
    """A toolkit no record names, a malformed record, two records for one assembler, or a mismatched torch."""


@dataclass(frozen=True)
class Toolchain:
    name: str
    ptxas: str
    cuda_major: int
    torch_index: str

    @property
    def manifest_dir(self) -> Path:
        return MANIFESTS / self.name

    def manifest_path(self, arch: str) -> Path:
        return self.manifest_dir / f"sm_{arch}.json"

    def archs(self) -> list[str]:
        return sorted(p.stem.removeprefix("sm_") for p in self.manifest_dir.glob("sm_*.json"))


def normalize(version: str) -> str:
    return " ".join(version.split())


@cache
def records() -> dict[str, Toolchain]:
    out: dict[str, Toolchain] = {}
    for path in sorted(RECORDS.glob("*.json")):
        blob = json.loads(path.read_text())
        if set(blob) != set(_FIELDS):
            raise ToolchainError(f"{path}: fields must be exactly {sorted(_FIELDS)}, got {sorted(blob)}")
        for field, kind in _FIELDS.items():
            if type(blob[field]) is not kind:
                raise ToolchainError(f"{path}: `{field}` must be {kind.__name__}, got {blob[field]!r}")
        toolchain = Toolchain(name=path.stem, ptxas=normalize(blob["ptxas"]), cuda_major=blob["cuda_major"],
                              torch_index=blob["torch_index"])
        twin = next((t for t in out.values() if t.ptxas == toolchain.ptxas), None)
        if twin is not None:
            raise ToolchainError(f"{path}: names the same assembler as {twin.name}; one assembler is one toolchain")
        out[toolchain.name] = toolchain
    if not out:
        raise ToolchainError(f"no toolchain records in {RECORDS}")
    return out


def named(name: str) -> Toolchain:
    if name not in records():
        raise ToolchainError(f"no toolchain named {name!r}; declared: {sorted(records())}")
    return records()[name]


def for_ptxas(version: str) -> Toolchain:
    version = normalize(version)
    for toolchain in records().values():
        if toolchain.ptxas == version:
            return toolchain
    declared = "\n".join(f"  {t.name}: {t.ptxas}" for t in records().values())
    raise ToolchainError(
        f"CLOSED-WORLD RULE 1: no toolchain record names this assembler:\n  {version}\ndeclared:\n{declared}\n"
        "Point toolchain.cuda_home at a declared toolkit, or bring up the toolchain (docs/bringup.md).")


def torch_pairing(toolchain: Toolchain, torch_cuda: str | None) -> None:
    if torch_cuda is None or int(torch_cuda.split(".", 1)[0]) != toolchain.cuda_major:
        raise ToolchainError(
            f"toolchain {toolchain.name} is CUDA {toolchain.cuda_major}, but torch was built for CUDA {torch_cuda}; "
            f"install torch from {toolchain.torch_index}")
