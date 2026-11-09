# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRAPH'S CUDA COMPILE ACTION (`bazel/cuda.bzl`'s `cuda_unit`): nvcc with a dependency file; the headers it did not
read become the action's unused inputs, so Bazel drops them from the unit's cache key and an edit to one recompiles
nothing; ptxas's report is kept as an output. Docs: docs/build-graph.md#compile.

    cuda_compile.py --source S.cu --candidates C.txt --unused U.txt --report R.txt -- NVCC ARGS...
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: a Make prerequisite: a run of characters that are not unescaped whitespace
_WORD = re.compile(r"(?:\\.|[^\s\\])+")


def prerequisites(depfile: str) -> list[str]:
    """Every prerequisite the dependency file's rules name, unescaped (`\\ ` a space, `$$` a dollar)."""
    found = []
    for rule in depfile.replace("\\\n", " ").splitlines():
        _target, colon, words = rule.partition(": ")
        if colon:
            found += [re.sub(r"\\(.)", r"\1", word).replace("$$", "$") for word in _WORD.findall(words)]
    return found


def unused(candidates: list[str], read: list[str]) -> list[str]:
    """The candidates the compile did not read, compared as files (each side's real path): the sandbox links every
    input, and nvcc may name a header by a path that resolves through those links. A directory is never listed: what
    under it was read is not a path comparison, so it stays in the key."""
    seen = {os.path.realpath(path) for path in read}
    return [path for path in candidates if not os.path.isdir(path) and os.path.realpath(path) not in seen]


def ptxas_info(line: str, inside: bool) -> bool:
    """Whether a line of nvcc's diagnostics belongs to ptxas's `-v` report: its `ptxas info` lines and the indented
    lines under them. Its warnings and every other diagnostic do not."""
    return line.startswith("ptxas info") or (inside and line[:1].isspace())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="the unit's source, which its dependency file must name")
    ap.add_argument("--candidates", type=Path, required=True, help="the unit's prunable inputs: its declared headers")
    ap.add_argument("--unused", type=Path, required=True, help="the action's unused_inputs_list")
    ap.add_argument("--report", type=Path, required=True, help="nvcc's whole diagnostics, ptxas's report among them")
    ap.add_argument("nvcc", nargs=argparse.REMAINDER, help="-- then nvcc and its arguments")
    a = ap.parse_args()
    command = a.nvcc[1:] if a.nvcc[:1] == ["--"] else a.nvcc
    with tempfile.TemporaryDirectory(prefix="rola_nvcc_") as scratch:
        depfile = Path(scratch) / "unit.d"
        done = subprocess.run([*command, "-MD", "-MF", str(depfile)], stderr=subprocess.PIPE, text=True)
        a.report.write_text(done.stderr)
        inside = False
        for line in done.stderr.splitlines(keepends=True):
            inside = ptxas_info(line, inside)
            if done.returncode or not inside:
                sys.stderr.write(line)
        if done.returncode:
            return done.returncode
        read = prerequisites(depfile.read_text())
    if os.path.realpath(a.source) not in {os.path.realpath(path) for path in read}:
        raise SystemExit(f"cuda_compile: nvcc's dependency file does not name its source {a.source}")
    a.unused.write_text("".join(path + "\n" for path in unused(a.candidates.read_text().splitlines(), read)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
