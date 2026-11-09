# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BUILD RECORD: `rola_cu13/_build_config.py`, which answers "which measured evidence backs this exact .so?" from
an installed wheel with no checkout -- and rule 3, which assembles the ratification it names.

Written by the graph's record action (`bazel/record.py`), standard library only so the graph runs it in its sandbox.
Docs: docs/build-graph.md#gates.
"""
from __future__ import annotations

import hashlib
import json
import pprint
import re
from pathlib import Path


def version(init: Path) -> str:
    """The package's `__version__`, from its `__init__.py`. No setuptools_scm: a tag must not be able to disagree with
    the shipped `__version__`."""
    match = re.search(r'^__version__\s*=\s*([\'"])([^\'"]+)\1', init.read_text(), re.M)
    if not match:
        raise RuntimeError(f"no __version__ in {init}")
    return match.group(2)


def ratified_manifest(archs: list[str], manifest_dir: Path) -> dict:
    """RULE 3: every arch in the fatbin has a ratified manifest, and they agree. Returns the MERGED record -- the union
    of exactly the archs being built, which is what the record's `manifest_sha256` then describes."""
    toolchain = None
    shards = None
    entries: dict = {}
    for a in archs:
        path = manifest_dir / f"sm_{a}.json"
        if not path.exists():
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: sm_{a} is in the build's architectures but there is no ratified manifest at "
                f"{path}. Ratify it first:\n    python tools/ratify.py --arch {a} --write\nand record why the numbers "
                f"are what they are (docs/bringup.md). Building an architecture nobody measured is the failure this "
                f"rule exists to prevent.")
        blob = json.loads(path.read_text())
        if not blob["entries"]:
            raise RuntimeError(f"CLOSED-WORLD RULE 3: {path} exists but ratifies NOTHING. File existence is the cheap "
                               "form of the question, not a weaker one.")
        if toolchain is None:
            toolchain, shards = blob["toolchain"], blob["shards"]
        elif blob["toolchain"] != toolchain:
            raise RuntimeError(f"CLOSED-WORLD RULE 1: {path.name} pins a different assembler than the other manifests. "
                               "The per-arch split is a FILING decision; one fatbin may not be gated against two "
                               "ratifications.")
        elif blob["shards"] != shards:
            raise RuntimeError(f"CLOSED-WORLD RULE 3: {path.name} records a different instantiation membership than "
                               "the other manifests. The per-arch split is a FILING decision; one fatbin may not be "
                               "gated against two matrices.")
        entries.update(blob["entries"])
        print(f"rule 3 OK  sm_{a}: {len(blob['entries'])} ratified instantiations ({path.name})")
    return {"toolchain": toolchain, "shards": shards, "entries": entries}


def vendored_tree_sha256(root: Path) -> tuple[str, int]:
    """A digest of a DIRECTORY, defined so it cannot be argued with: sha256 over the sorted relative POSIX paths, each
    followed by a NUL byte and the sha256 of that file's bytes; `PIN.json` is excluded because it carries the answer.
    Paths participate as well as contents, so a rename -- which changes what an `#include` resolves to -- moves it."""
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "PIN.json")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def vendored_pin(root: Path) -> dict:
    """RULE 5: the vendored tree at `root` is the one its `PIN.json` records -- a vendored header is code ptxas
    assembles, so a drift is a codegen change, not a packaging one. Returns the pin."""
    pin_path = root / "PIN.json"
    if not pin_path.exists():
        raise RuntimeError(f"CLOSED-WORLD RULE 5: {pin_path} is missing. A vendored library with no pin is an "
                           "unmeasured codegen input; restore the pin or remove the vendored tree.")
    pin = json.loads(pin_path.read_text())
    here, count = vendored_tree_sha256(root)
    if here != pin["tree_sha256"]:
        raise RuntimeError(f"CLOSED-WORLD RULE 5: the vendored tree {root.name} does not match its pin.\n  PIN.json: "
                           f"{pin['tree_sha256']} ({pin['file_count']} files)\n  on disk : {here} ({count} files)\n"
                           "Re-vendor the pinned tag, or move the pin DELIBERATELY and re-ratify "
                           "(`bazel run //:ratify -- --write`).")
    print(f"rule 5 OK  {root.name} {pin['tag']} ({pin['commit'][:12]}): {count} files, sha256 {here[:16]}...")
    return pin


def manifest_digest(manifest: dict) -> str:
    """Canonical JSON of `{toolchain, entries, shards}` with sorted keys -- not the bytes of any file -- so the digest
    is a property of the RATIFICATION, not of its filing. The shard block is membership only (`{name: {members_sha256,
    count}}`), so it records what was ratified without how it was filed."""
    blob = json.dumps({"toolchain": manifest["toolchain"], "entries": manifest["entries"],
                       "shards": manifest["shards"]}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def config(*, version: str, archs: list[str], iteration: bool, toolchain: str, ptxas: str, torch: str,
           cxx11_abi: bool, manifest: dict, cutlass_pin: dict) -> dict:
    """The record's fields. `iteration`: an arm, part or architecture set other than the shipped one, never a wheel.
    `shards` is the ratified membership `{shard: members_sha256}` the dispatch table's rows are joined against at
    runtime."""
    return {
        "version": version,
        "archs": [f"sm_{a}" for a in archs],
        "iteration": iteration,
        "toolchain": toolchain,
        "ptxas": ptxas,
        "torch": torch,
        "cxx11_abi": cxx11_abi,
        "manifest_sha256": manifest_digest(manifest),
        "manifest_entries": {f"sm_{a}": sum(1 for k in manifest["entries"] if k.startswith(f"sm_{a}:")) for a in archs},
        "shards": {name: block["members_sha256"] for name, block in sorted(manifest["shards"].items())},
        "ptx_jit_fallback": False,
        "vendored": {"cutlass": {"tag": cutlass_pin["tag"], "commit": cutlass_pin["commit"],
                                 "tree_sha256": cutlass_pin["tree_sha256"]}},
    }


def render(record: dict) -> str:
    """The module's text. `pprint`, not `json.dumps`: it is Python, and JSON spells the booleans `true`/`false`."""
    return ('"""GENERATED by the build -- do not edit.\n\n'
            "Answers 'which measured evidence backs this exact .so?' from an installed\n"
            "wheel, with no repository checkout: `manifest_sha256` is the hash of the\n"
            "ratification this binary was gated against.\n"
            '"""\n\n'
            f"BUILD_CONFIG = {pprint.pformat(record, indent=4, sort_dicts=False, width=88)}\n\n\n"
            "def show():\n"
            '    """A copy, so a caller cannot mutate the record."""\n'
            "    return dict(BUILD_CONFIG)\n")
