"""THE TOOLCHAINS: each a named record of the exact assembler a fatbin is built and ratified under.

`tools/toolchains/<name>.json` records one toolchain; its ratification is `tools/manifests/<name>/sm_XX.json`. The build
takes no toolchain parameter: the pinned toolkit's `ptxas --version` (tools/cuda_toolkit_pin.json) selects the record
that names it, and a toolkit no record names refuses. Docs: docs/internals/tools/toolchains.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
RECORDS = ROOT / "tools" / "toolchains"
MANIFESTS = ROOT / "tools" / "manifests"

_FIELDS = {"ptxas": str, "readers": dict, "sanitizer": str, "cuda_major": int, "torch_index": str}
#: the disassemblers the binary gates read with, each pinned by its whole `--version` text as the assembler is
READERS = ("cuobjdump", "nvdisasm")


class ToolchainError(RuntimeError):
    """A toolkit no record names, a malformed record, two records for one assembler, or a mismatched torch."""


@dataclass(frozen=True)
class Toolchain:
    name: str
    ptxas: str
    readers: tuple[tuple[str, str], ...]
    #: compute-sanitizer's whole `--version` text: the sanitizer lane reads a binary with the toolkit that built it
    sanitizer: str
    cuda_major: int
    torch_index: str

    @property
    def manifest_dir(self) -> Path:
        return MANIFESTS / self.name

    def manifest_path(self, arch: str) -> Path:
        return self.manifest_dir / f"sm_{arch}.json"

    def archs(self) -> list[str]:
        return sorted(p.stem.removeprefix("sm_") for p in self.manifest_dir.glob("sm_*.json"))


def built_extension(root: Path = ROOT) -> Path:
    """The extension a checkout's in-place build produced: `rola_<toolchain>/_C.*so`, inside the binary plugin the build
    ships as (docs/build.md#wheels). Refuses a checkout with none, or with more than one toolchain's."""
    found = sorted(root.glob("rola_*/_C.*so"))
    if len(found) != 1:
        raise FileNotFoundError(f"{root} holds {len(found)} built extensions (rola_<toolchain>/_C.*so): "
                                f"{[str(p.relative_to(root)) for p in found]}")
    return found[0]


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
        if set(blob["readers"]) != set(READERS) or not all(type(v) is str for v in blob["readers"].values()):
            raise ToolchainError(f"{path}: `readers` must name {list(READERS)}, each its `--version` text")
        toolchain = Toolchain(name=path.stem, ptxas=normalize(blob["ptxas"]),
                              readers=tuple(sorted((k, normalize(v)) for k, v in blob["readers"].items())),
                              sanitizer=normalize(blob["sanitizer"]),
                              cuda_major=blob["cuda_major"],
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
        "Install the pinned toolkit (bazel run //:init), or bring up the toolchain (docs/bringup.md).")


def version_text(binary: str) -> str:
    """A toolkit binary's whole `--version` output, the text a record pins (normalize it to compare)."""
    import subprocess

    return subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout


def check_readers(toolchain: Toolchain, readers: dict[str, str]) -> None:
    """Each disassembler a gate reads with (name -> its `--version` text) is the record's byte for byte, every one the
    record names: a binary is read with the toolkit it was ratified under."""
    pinned = dict(toolchain.readers)
    if set(readers) != set(pinned):
        raise ToolchainError(f"CLOSED-WORLD RULE 1: the readers checked are {sorted(readers)}, the record names "
                             f"{sorted(pinned)}")
    for name, text in readers.items():
        if normalize(text) != pinned[name]:
            raise ToolchainError(f"CLOSED-WORLD RULE 1: {name} is not toolchain {toolchain.name}'s.\n  record: "
                                 f"{pinned[name]}\n  here  : {normalize(text)}")


def rule1(toolchain: Toolchain, ptxas: str, manifest: dict, torch_cuda: str | None, readers: dict[str, str]) -> str:
    """CLOSED-WORLD RULE 1: `ptxas` (a `--version` text) is the toolchain record's and the ratified manifests' assembler
    byte for byte; the torch the extension builds against is the toolchain's CUDA major; and each of `readers` (the
    disassemblers the binary gates read with: name -> `--version` text) is the record's byte for byte, so the gates read
    a binary with the toolkit it was ratified under. The normalized assembler string, on success."""
    here, pinned = normalize(ptxas), normalize(manifest["toolchain"]["ptxas"])
    if here != toolchain.ptxas or here != pinned:
        raise ToolchainError(f"CLOSED-WORLD RULE 1: toolchain {toolchain.name}'s manifests were ratified under a different "
                             f"assembler.\n  record  : {toolchain.ptxas}\n  manifest: {pinned}\n  here    : {here}\n"
                             "Re-ratify (`bazel run //:ratify -- --write`) and record the new numbers -- do not "
                             "hand-edit the manifest.")
    check_readers(toolchain, readers)
    torch_pairing(toolchain, torch_cuda)
    return here


def torch_pairing(toolchain: Toolchain, torch_cuda: str | None) -> None:
    if torch_cuda is None or int(torch_cuda.split(".", 1)[0]) != toolchain.cuda_major:
        raise ToolchainError(
            f"toolchain {toolchain.name} is CUDA {toolchain.cuda_major}, but torch was built for CUDA {torch_cuda}; "
            f"install torch from {toolchain.torch_index}")
