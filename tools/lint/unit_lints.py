#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE UNIT LINTS: clang-tidy's two profiles and the host warnings, each read from one unit's own compile action in
the build graph (`bazel/defs.bzl`'s `unit_lints` aspect), and the editor's compile database from the same commands.

    unit_lints.py command --argv A.json --arch 80 --cuda-root DIR --out C.json   # the unit's clang command
    unit_lints.py tidy --command C.json --clang-tidy BIN --config F --package csrc/rola --out V.json
    unit_lints.py warnings --argv A.json --arch 80 --package csrc/rola --out V.json   # the host compile, warnings on
    unit_lints.py verdicts [--expect PATTERN...] V.json...             # red on any finding, or a fixture's mismatch
    unit_lints.py database --checkout DIR --out F C.json...            # compile_commands.json, from the run's execroot

A unit's compile is the graph's own (its argv, the action's inputs), so the lint reads exactly what the build compiled,
for one architecture a configuration. Standard library only: the graph's base interpreter runs it.
Docs: docs/internals/tools/lint.md#unit-lints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: the compile's arguments clang reads as nvcc and gcc do -- the include roots, the macros, the forced includes, the
#: language standard and the optimization level (it sets `__OPTIMIZE__`); the rest are a code generator's or a driver's
_WITH_VALUE = ("-I", "-isystem", "-iquote", "-include", "-D", "-U")
_JOINED = ("-I", "-D", "-U", "-std=", "-O")
#: the path-valued ones, which the editor's database makes absolute
_PATH_VALUED = ("-I", "-isystem", "-iquote", "-include")
#: the architecture an nvcc `-gencode` builds (`arch=compute_80,code=sm_80`)
_GENCODE = re.compile(r"code=sm_(\d+)")
#: a diagnostic: `path:line:col: severity: message`
_DIAGNOSTIC = re.compile(r"^(?P<path>[^:\s][^:]*):\d+:\d+: (?P<severity>warning|error|fatal error): ")
#: an action's absolute path, up to the execution root it ran in
_SANDBOX = re.compile(r"/\S*?/execroot/_main/")
#: the host warnings' flags: they select the diagnostics of the same compile, and are never a build's
HOST_WARNING_FLAGS = ("-Wall", "-Wextra", "-Wunused")
#: each profile's clang-tidy arguments past its config: the device profile's dead-store analysis stays inside a function
#: (`ipa=none`), and its findings in its own package's headers count (`{package}`); the host profile reads its main
#: file alone (`.clang-tidy`'s `HeaderFilterRegex`)
PROFILES = {
    "device": ("--header-filter={package}/.*", "--extra-arg=-Xclang", "--extra-arg=-analyzer-config",
               "--extra-arg=-Xclang", "--extra-arg=ipa=none"),
    "host": (),
}


def clang_command(argv: list[str], arch: str, cuda_root: str) -> dict:
    """The unit's compile as clang's: its source, its kept arguments, and -- for a CUDA unit -- clang's CUDA front end
    on the host side for `arch`, the configuration's one architecture, which the compile must build alone."""
    source, archs, kept = None, [], []
    i = 1
    while i < len(argv):
        word = argv[i]
        if word == "-c":
            source = argv[i + 1]
            i += 2
            continue
        if word == "-gencode":
            archs += _GENCODE.findall(argv[i + 1])
            i += 2
            continue
        if word in _WITH_VALUE:
            kept += [word, argv[i + 1]]
            i += 2
            continue
        if word.startswith(_JOINED):
            kept.append(word)
        i += 1
    if source is None:
        raise SystemExit(f"unit_lints: the compile names no source (`-c`): {' '.join(argv[:4])} ...")
    if argv[0].endswith("nvcc"):
        if set(archs) != {arch}:
            raise SystemExit(f"unit_lints: {source} compiles for {sorted(set(archs)) or 'no'} architecture(s); the lint "
                             f"reads one architecture's compile, sm_{arch}")
        flags = ["-x", "cuda", "--cuda-host-only", f"--cuda-path={cuda_root}", f"--cuda-gpu-arch=sm_{arch}",
                 "-nocudalib", *kept]
        return {"file": source, "profile": "device", "arch": arch, "flags": flags}
    return {"file": source, "profile": "host", "arch": arch, "flags": ["-x", "c++", *kept]}


def _verdict(lint: str, command: dict, package: str, started: float, output: str, returncode: int) -> dict:
    """The diagnostics that count: a warning or error in the unit's own package (a vendored or a toolkit header's is not
    its to fix), and any compiler error -- a unit the compiler cannot parse is a unit no lint read."""
    findings, errors = [], 0
    #: the action's own directory is the sandbox's: a path is reported from the execution root
    output = _SANDBOX.sub("", output)
    for line in output.splitlines():
        m = _DIAGNOSTIC.match(line)
        if m and (m["path"].startswith(package + "/") or m["severity"] != "warning"):
            findings.append(line)
            errors += m["severity"] != "warning"
    if returncode and not errors:
        #: a failure no located diagnostic explains (a driver's refusal): the run is red, never clean by absence
        last = [line for line in output.splitlines() if line.strip()][-3:]
        findings.append(f"exit {returncode}, no located error: " + " | ".join(last))
    return {"lint": lint, "file": command["file"], "arch": command["arch"], "seconds": round(time.monotonic() - started),
            "returncode": returncode, "findings": findings}


def tidy(command: dict, clang_tidy: str, config: str, package: str) -> dict:
    started = time.monotonic()
    profile = command["profile"]
    arguments = [word.format(package=package) for word in PROFILES[profile]]
    run = subprocess.run([clang_tidy, command["file"], f"--config-file={config}", *arguments, "--", *command["flags"],
                          "-ferror-limit=0"], capture_output=True, text=True)
    return _verdict(f"clang-tidy {profile}", command, package, started, run.stdout + run.stderr, run.returncode)


def warnings(argv: list[str], arch: str, package: str) -> dict:
    """The host compile itself, front end only (`-fsyntax-only`: no object), with the warning flags last."""
    started = time.monotonic()
    out, source, i = [argv[0]], None, 1
    while i < len(argv):
        word = argv[i]
        if word in ("-o", "-MF"):
            i += 2
            continue
        if word == "-c":
            source = argv[i + 1]
        elif word not in ("-MD", "-MMD", "-MP"):
            out.append(word)
        i += 1
    run = subprocess.run([*out, "-fsyntax-only", *HOST_WARNING_FLAGS], capture_output=True, text=True)
    return _verdict("host warnings", {"file": source, "arch": arch}, package, started, run.stdout + run.stderr,
                    run.returncode)


def _exactly(findings: list[str], expect: list[str]) -> bool:
    """Each pattern matches its own finding, and no finding is left over."""
    left = list(findings)
    for pattern in expect:
        hit = next((f for f in left if re.search(pattern, f)), None)
        if hit is None:
            return False
        left.remove(hit)
    return not left


def verdicts(paths: list[str], expect: list[str]) -> int:
    """A unit's verdicts are red on any finding. A fixture's (`expect`) are red unless each holds exactly the expected
    findings: a lint fires on its must-fire case and on nothing else."""
    red = 0
    for path in sorted(paths):
        v = json.loads(Path(path).read_text())
        held = _exactly(v["findings"], expect) if expect else not v["findings"]
        print(f"== {v['lint']}: {v['file']} (sm_{v['arch']}), {v['seconds']} s, {len(v['findings'])} finding(s)"
              + ("" if held or not expect else ", NOT the fixture's expected findings") + " ==")
        for line in v["findings"]:
            print(f"  {line}")
        red += not held
    if not paths:
        print("unit lints: no verdict -- a lint with no unit does not pass")
        return 1
    print(f"unit lints: {red} of {len(paths)} verdict(s) red" + (" (a fixture's)" if expect else ""))
    return 1 if red else 0


def database(commands: list[str], checkout: str) -> list[dict]:
    """Each unit's clang command, every path absolute: the tree's under the checkout, the build's outputs under the
    execution root this run reads (`bazel-out/`), the repositories under the output base (`external/`)."""
    cwd = os.getcwd()
    execroot = cwd[:cwd.index("/bazel-out/")]
    output_base = execroot[:execroot.rindex("/execroot/")]

    def absolute(path: str) -> str:
        if path.startswith("bazel-out/"):
            return f"{execroot}/{path}"
        if path.startswith("external/"):
            return f"{output_base}/{path}"
        return os.path.normpath(f"{checkout}/{path}")

    entries = []
    for path in commands:
        c = json.loads(Path(path).read_text())
        flags, out = c["flags"], []
        for i, word in enumerate(flags):
            if i and flags[i - 1] in _PATH_VALUED:
                out.append(absolute(word))
            elif word.startswith("-I") and len(word) > 2:
                out.append("-I" + absolute(word[2:]))
            elif word.startswith("--cuda-path="):
                out.append("--cuda-path=" + absolute(word.split("=", 1)[1]))
            else:
                out.append(word)
        entries.append({"directory": checkout, "file": absolute(c["file"]), "arguments": ["clang++", *out, absolute(c["file"])]})
    return sorted(entries, key=lambda e: e["file"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("command")
    c.add_argument("--argv", type=Path, required=True)
    c.add_argument("--arch", required=True, help="the configuration's architecture (`80`)")
    c.add_argument("--cuda-root", required=True)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tidy")
    t.add_argument("--command", type=Path, required=True)
    t.add_argument("--clang-tidy", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--package", required=True, help="the unit's package: its findings are the ones that count")
    t.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("warnings")
    w.add_argument("--argv", type=Path, required=True)
    w.add_argument("--arch", required=True)
    w.add_argument("--package", required=True)
    w.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verdicts")
    v.add_argument("--expect", action="append", default=[], help="a fixture's expected finding (a pattern)")
    v.add_argument("paths", nargs="*")
    d = sub.add_parser("database")
    d.add_argument("--checkout", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("commands", nargs="+")

    a = ap.parse_args()
    if a.mode == "command":
        a.out.write_text(json.dumps(clang_command(json.loads(a.argv.read_text()), a.arch, a.cuda_root), indent=1) + "\n")
    elif a.mode == "tidy":
        a.out.write_text(json.dumps(tidy(json.loads(a.command.read_text()), a.clang_tidy, a.config, a.package), indent=1)
                         + "\n")
    elif a.mode == "warnings":
        a.out.write_text(json.dumps(warnings(json.loads(a.argv.read_text()), a.arch, a.package), indent=1) + "\n")
    elif a.mode == "verdicts":
        return verdicts(a.paths, a.expect)
    else:
        entries = database(a.commands, a.checkout)
        a.out.write_text(json.dumps(entries, indent=1) + "\n")
        print(f"compile database: {len(entries)} unit(s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
