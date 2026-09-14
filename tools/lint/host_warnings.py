#!/usr/bin/env python3
"""HOST warnings dry-compile (REPORT-ONLY).

Recompiles every HOST TU (`.cpp` entries in `compile_commands.json`, the same
scope `tools/lint/run_clang_tidy.sh` names -- currently exactly one,
`csrc/rola/rola_api.cpp`) with `-Wall -Wextra -Wunused`, using the flags an
already-built tree's compile database recorded, but stopping after the
FRONT END (`-fsyntax-only`) so this is a dry compile: no object file, no link,
nothing else in the tree needs to be built first.

WHY THESE FLAGS LIVE HERE AND NOT IN `tools/build_flags.py`. That file is the
ONE shared codegen flag list `setup.py` (the actual build) and `tools/ratify.py`
(the register/spill census) both compile against -- its own docstring states
the failure mode of a second, drifted copy. `-Wall -Wextra -Wunused` are not
codegen flags (they select which diagnostics the SAME compile emits) and this
check is a LINT, never part of what ships; adding them to the shared list would
either (a) start emitting these warnings in every real build (noise on a build
log nobody asked to gate) or (b) require build_flags.py to grow a report-only
mode, which is scope this stage does not own (build_flags.py is shared
build-machinery another track may be editing concurrently). They are declared
here, once, and read by nothing else -- if `build_flags.py` ever grows a
documented "lint flags" export, this module should switch to importing it
rather than re-declare.

Every finding is reported for a human to rule on; this script always exits 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CC_JSON = ROOT / "compile_commands.json"

#: LINT-ONLY diagnostic flags (see module docstring for why these are not in
#: `tools/build_flags.py`). `-Wunused` is a superset umbrella pycodestyle-style
#: alias GCC/Clang both accept; `-Wall -Wextra` are the standard broad sets.
HOST_WARNING_FLAGS = ("-Wall", "-Wextra", "-Wunused")


def host_entries(cc_json: Path) -> list[dict]:
    db = json.loads(cc_json.read_text())
    return [e for e in db if e["file"].endswith(".cpp")]


def dry_compile(entry: dict) -> tuple[int, str]:
    """`-fsyntax-only` recompile of one host TU's own command line, with the
    lint's warning flags appended (appended, not prepended, so a repeated flag
    the original command already carried loses to nothing -- last one wins on
    both gcc and clang)."""
    import shlex

    toks = shlex.split(entry["command"])
    # Drop `-o <obj>` / `-c` -- `-fsyntax-only` replaces both; drop `-MMD -MF x -MP`
    # dependency-file generation, irrelevant to a syntax-only pass.
    skip_with_arg = {"-o", "-MF"}
    skip_bare = {"-c", "-MMD", "-MP"}
    out = [toks[0]]
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in skip_with_arg:
            i += 2
            continue
        if t in skip_bare:
            i += 1
            continue
        out.append(t)
        i += 1
    out += ["-fsyntax-only", *HOST_WARNING_FLAGS]
    proc = subprocess.run(out, cwd=entry["directory"], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def self_test() -> int:
    """Excluded from the commit gate (the fixtures convention): the fixture is
    self-contained C++, no project headers, no compdb needed."""
    fixture = ROOT / "tools" / "lint" / "fixtures" / "host_warnings_unused_param.cpp"
    proc = subprocess.run(["c++", str(fixture), "-fsyntax-only", *HOST_WARNING_FLAGS],
                           capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if "unused parameter" not in out or "unused_param" not in out:
        print("SELF-TEST FAILED: expected finding on fixture's 'bad' did not fire", file=sys.stderr)
        return 1
    if "'a'" in out or "'b'" in out:
        print("SELF-TEST FAILED: fixture's 'ok' fired unexpectedly", file=sys.stderr)
        return 1
    print("host_warnings.py --test-fixtures: PASS")
    return 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()

    if not CC_JSON.is_file():
        print("host_warnings: SKIPPED -- no compile_commands.json "
              "(run tools/lint/compdb.sh against a built tree first)")
        return 0

    entries = host_entries(CC_JSON)
    if not entries:
        print("host_warnings: no .cpp (host) TU in the compile database")
        return 0

    #: ONLY this codebase's own lines are ON-TARGET (`csrc/rola/`); torch/pybind11's
    #: own headers are reached transitively (`torch/extension.h`) and produce
    #: warnings this project does not own and cannot fix -- reported as a
    #: separate count, same "noise, not silenced" shape as the device profile.
    total_on_target = 0
    total_header = 0
    for e in entries:
        rc, out = dry_compile(e)
        lines = [l for l in out.splitlines() if ": warning:" in l]
        on_target = [l for l in lines if "/csrc/rola/" in l]
        header = [l for l in lines if "/csrc/rola/" not in l]
        total_on_target += len(on_target)
        total_header += len(header)
        rel = Path(e["file"]).name
        print(f"== host_warnings: {rel} ({len(on_target)} on-target, "
              f"{len(header)} header, compiler rc={rc}) ==")
        for l in on_target:
            print(f"  {l}")

    print(f"host_warnings: {total_on_target} on-target warning(s) in csrc/rola/ across "
          f"{len(entries)} host TU(s); {total_header} in vendored headers "
          f"(torch/pybind11, not this codebase's -- excluded from the on-target count)")

    #: GATING on the ON-TARGET count only. A vendored header's warning is not this
    #: codebase's to fix and never gates; a warning on our own line is a defect the
    #: compiler already found.
    return 1 if total_on_target else 0


if __name__ == "__main__":
    sys.exit(main())
