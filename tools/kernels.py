#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE KERNEL LIST: every device translation unit the build compiles and every `__global__` kernel the source
defines, DERIVED FROM THE TREE -- never hand-listed. The build, the ratification and every gate take their targets
from here, so a new kernel is built, ratified and gated the day it lands, and no gate can narrow back to one family.

    bazel run //tools:kernels                      # the list: family, kernel, file:line
    bazel run //tools:kernels -- --units           # the translation units the build compiles
    bazel run //tools:kernels -- --check-manifest # the ratified manifests name exactly the source's kernels

A UNIT is a `.cu` under `csrc/rola/src`: the hand-written ones, plus the per-arm carry units `gen_shards` selects
under `instantiations/`. A KERNEL is a `__global__` function definition in a `.cu`/`.cuh` there, read from the source
with comments and string literals blanked; its FAMILY is its directory under `src/`.
Docs: docs/internals/tools/kernels.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
SRC = ROOT / "csrc" / "rola" / "src"
MANIFESTS = ROOT / "tools" / "manifests"
#: the generated per-arm units: the build compiles the ones the arm selection names, not the directory
GENERATED = "instantiations"

_BLANK = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)
_GLOBAL = re.compile(
    r"__global__\s+(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?void\s+(\w+)\s*\(")
_TEMPLATE_TAIL = re.compile(r"template\s*<[^;{}]*>\s*$")


@dataclass(frozen=True)
class Kernel:
    name: str
    family: str
    path: Path
    line: int
    template: bool

    def where(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}"


def blank(text: str) -> str:
    """The text with comments and string literals replaced by spaces, newlines kept, so offsets and lines hold."""
    return _BLANK.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def parse(text: str) -> list[tuple[str, int, bool]]:
    """`[(name, line, template), ...]` for every `__global__` definition in one file's text."""
    code = blank(text)
    out = []
    for m in _GLOBAL.finditer(code):
        head = code[:m.start()].rstrip()
        cut = max(head.rfind(";"), head.rfind("}"), head.rfind("{"))
        out.append((m.group(1), code.count("\n", 0, m.start()) + 1, bool(_TEMPLATE_TAIL.search(head[cut + 1:]))))
    return out


def family_of(path: Path, src: Path = SRC) -> str:
    return path.relative_to(src).parts[0]


def kernels(src: Path = SRC) -> list[Kernel]:
    """Every `__global__` definition under `src`, in path order."""
    found = []
    for path in sorted(p for p in src.rglob("*") if p.suffix in (".cu", ".cuh")):
        for name, line, template in parse(path.read_text()):
            found.append(Kernel(name, family_of(path, src), path, line, template))
    return found


def hand_units(src: Path = SRC) -> list[Path]:
    """The hand-written translation units: every `.cu` under `src` outside the generated per-arm directory."""
    return sorted(p for p in src.rglob("*.cu") if GENERATED not in p.relative_to(src).parts)


def runtime_units(src: Path = SRC) -> list[Path]:
    """The RUNTIME units: the hand units under `common/` that define no kernel -- the card's facts, the geometry and the
    selection that every kernel's host side calls. The library links them, and so does an instrument's library
    (bazel/defs.bzl's `rola_instrument`), whose host side calls them too."""
    with_kernels = {k.path for k in kernels(src)}
    return [p for p in hand_units(src) if p.relative_to(src).parts[0] == "common" and p not in with_kernels]


def arm_units(arms=None) -> list[Path]:
    """The per-arm carry units for `arms` (every declared arm by default), from `gen_shards`' one enumeration."""
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_shards

    files = gen_shards.carry_sources()
    want = range(len(gen_shards.CARRY_ARMS)) if arms is None else arms
    return [SRC / GENERATED / name for name in sorted({files[i] for i in want})]


def units(arms=None) -> list[Path]:
    """Every translation unit the build compiles under csrc/rola/src."""
    return hand_units() + arm_units(arms)


_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def closure(unit: Path, src: Path = SRC) -> set[Path]:
    """The files under `src` a unit compiles: itself and every quoted include it reaches, resolved against the
    including file's directory and then `src` (the build's include root)."""
    seen, todo = set(), [unit]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _INCLUDE.findall(path.read_text()):
            for base in (path.parent, src):
                hit = (base / name).resolve()
                if hit.is_file() and hit.is_relative_to(src):
                    todo.append(hit)
                    break
    return seen


def units_touching(paths) -> list[Path]:
    """The units whose compiled files include any of `paths`: what an edit to them makes a unit-level gate re-read."""
    changed = {Path(p).resolve() for p in paths}
    return [u for u in units() if closure(u) & changed]


def units_of(kernel: Kernel) -> list[Path]:
    """The translation units that compile a kernel's definition."""
    return [u for u in units() if kernel.path in closure(u)]


def demangled_names(mangled: list[str]) -> list[str]:
    """The unqualified function name of each mangled symbol."""
    if not mangled:
        return []
    text = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True, check=True).stdout
    return [function_name(sig) for sig in text.splitlines()]


def function_name(signature: str) -> str:
    """`void ns::name<1, 2>(Params)` -> `name`: the identifier before the depth-0 argument list, template args cut."""
    depth, cut = 0, len(signature)
    for i, ch in enumerate(signature):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    head = signature[:cut]
    if head.endswith(">"):
        depth = 0
        for i in range(len(head) - 1, -1, -1):
            depth += {">": 1, "<": -1}.get(head[i], 0)
            if depth == 0:
                head = head[:i]
                break
    return re.split(r"::|\s", head.strip())[-1]


def manifest_kernels(arch: str) -> set[str]:
    """The kernel names every toolchain's ratified manifest for `arch` records."""
    mangled = []
    for path in sorted(MANIFESTS.glob(f"*/sm_{arch}.json")):
        mangled += [key.split(":", 1)[1] for key in json.loads(path.read_text()).get("entries", {})]
    return set(demangled_names(mangled))


def check_manifest(arch: str) -> list[str]:
    """Findings: a source kernel no manifest records, and a recorded kernel the source no longer defines."""
    source = {k.name: k for k in kernels()}
    recorded = manifest_kernels(arch)
    out = [f"{k.where()}: `{name}` ({k.family}) is not in any sm_{arch} manifest -- unratified"
           for name, k in sorted(source.items()) if name not in recorded]
    out += [f"tools/manifests/*/sm_{arch}.json: `{name}` is recorded but no source defines it -- stale"
            for name in sorted(recorded - set(source))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", action="store_true", help="print the translation units the build compiles")
    ap.add_argument("--touching", nargs="*", default=None, metavar="FILE",
                    help="with --units: only the units whose compiled files include one of these")
    ap.add_argument("--check-manifest", action="store_true", help="compare the source's kernels to the manifests")
    ap.add_argument("--arch", default="86")
    a = ap.parse_args()
    if a.units:
        for path in (units() if a.touching is None else units_touching(a.touching)):
            print(path.relative_to(ROOT))
        return 0
    if a.check_manifest:
        findings = check_manifest(a.arch)
        for f in findings:
            print(f"  {f}")
        print(f"kernels: {len(kernels())} source kernel(s), {len(findings)} finding(s) against sm_{a.arch}")
        return 1 if findings else 0
    for k in kernels():
        print(f"{k.family:8s} {k.name:28s} {'template' if k.template else '        '} {k.where()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
