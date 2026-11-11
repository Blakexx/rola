#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The compute-sanitizer NAMED gate on the facts, decode and entmax oracle cells.

`docs/testing.md`'s "sanitizer lane" section records the shape (a SCHEDULED lane
over a named subset, never a PR gate -- the tools cost one to two orders of
magnitude -- `--error-exitcode=1`, a committed skip-list whose entries carry a
reason) and its one measurement so far: memcheck/initcheck/synccheck/racecheck all
clean on the TILED consumer's two registered probes. It also states the gap this
script closes: "no such measurement exists for the chunk arm yet."

This is that measurement, promoted from an ad-hoc invocation (recorded once, by
hand, in FINDINGS `## PROBE-K35` under "memcheck on paging") to a script anyone
can run and CI can schedule: `racecheck`/`synccheck`/`initcheck` (memcheck already
ran, ad hoc, on paging -- K39 card) against ONE named oracle cell per family
(facts, decode, entmax), each run as its own
`compute-sanitizer`-wrapped pytest process so a hang or an error in one tool
cannot block another, under the GPU lock every campaign shares.

Usage -- the lane is the build graph's, which passes `--sanitizer`, the toolkit's own compute-sanitizer by its path:
    bazel test //tests:sanitizer [--test_arg=--tool --test_arg=TOOL ...] [--test_arg=--family --test_arg=FAMILY ...]
        [--test_arg=--timeout --test_arg=SECONDS]

Invoked BARE: this script takes `gpu_lock()` itself, once, for the whole
run, exactly the way every
other GPU entry point does -- never wrap it in an external `flock` on the
lock path, which serializes against nothing (the lock is the files beside it;
`rola_devtools.locks.gpu`'s module docstring).

Exit code is 0 iff every (family, tool) pair is CLEAN or DECLARED. DECLARED is
the one adjudicated exception, not a skip: the pair still runs and its whole
report is still read, and it passes only if EVERY record in it is the one
declared idiom (see `ASSERTED`). A per-pair timeout (default
600s) turns a hang into a NAMED failure rather than a stuck lane holding the GPU
lock forever (KERNEL_STANDARDS §14: "a hang here blocks the whole lock queue for
every other agent").
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import toolchains
from rola_devtools.locks import host
from rola_devtools.locks.gpu import gpu_lock

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Family:
    """One family's sanitizer row: the cell, the kernel it launches, and whether that
    kernel is built on this line at all."""

    test: str
    kernel: str
    present: bool


#: one cell per family, chosen for being ALREADY a named, gated test -- this script
#: adds no new cell, it reruns an existing one under the sanitizer. `-k` narrows to
#: exactly one parametrization so each (family, tool) pair is one launch, not a
#: whole file's worth serialized under the heaviest instrumentation this repo has.
#:
#: THE FAMILIES, RE-KEYED TO WHAT THIS LINE CARRIES. Each row names one already-gated
#: cell -- this script adds no cell of its own, it reruns an existing one under the
#: sanitizer -- and states whether the family's kernel EXISTS.
#:
#: AN ABSENT FAMILY IS REPORTED, NEVER RUN AND NEVER CLEAN. A family with no body on this line: a
#: sanitizer pair over a cell that refuses at the extension boundary launches nothing,
#: and a tool that reports no hazards over no launch is the vacuous pass this script
#: exists to refuse (its own rule, learned when a CLEAN came back from a binary that
#: did not carry the cell's arm). The declaration is checked in BOTH directions: an
#: absent family whose cell starts passing fails the run, because the kernel it names
#: has arrived and the row owes a flip.
#:
#: `-k`-free ids: each row names ONE parametrization so a pair is one launch, not a
#: whole file serialized under the heaviest instrumentation this repo has.
_FAMILIES = {
    #: THE FACTS PASS is the one bit-exact device fact every consumer of the lists rests on: the vote's transposes, the
    #: union's ranks and the passes' bounds, the lists' walks and their pads, on both sides' lists at two levels of
    #: unequal width and a tail word.
    "facts": Family(
        test="tests/integration/test_facts_pass.py::"
             "test_the_lists_are_the_model_bit_for_bit[list_groups1-alternating-widths3-33]",
        kernel="rola::facts::rola_facts_forward", present=True),
    #: THE CARRY OVER A WHOLE CALL, absent: no carry kernel is built on this line, so the oracle's whole-call cell
    #: refuses by name, and the row flips when a schedule lands.
    "carry_call": Family(
        test="tests/oracle/test_carry_vs_oracle.py::"
             "test_the_inter_numerator_is_the_fp64_reference[flagship-cohort-k4-L4096]",
        kernel="rola::carry::carry_forward", present=False),
    #: THE DECODE STEP on the whole resident grid: the work queue's visits, claims and head switches.
    "decode": Family(
        test="tests/oracle/test_decode_vs_oracle.py::"
             "test_g1_the_state_is_byte_reproducible_against_itself[widths0-64]",
        kernel="rola::decode::rola_decode_forward", present=True),
    #: THE PRODUCER'S SOLVES: their own block-wide reductions, on the routing path.
    "entmax": Family(
        test="tests/oracle/test_entmax_cuda_gates.py::"
             "test_teeth_B_values_reject_beyond_tolerance[1.5]",
        kernel="rola::entmax::union_forward", present=True),
}

_FAMILY_TESTS = {name: row.test for name, row in _FAMILIES.items()}

#: THE TOOLS (docs/KERNEL_STANDARDS.md#sanitizer): memcheck, synccheck and initcheck by default.
_TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
#: RACECHECK IS DEFERRED (Blake, 2026-10-01): out of the default run for every kernel, runnable only by an explicit
#: `--tool racecheck`, until it is taken up again (docs/testing.md#the-sanitizer-lane).
_DEFAULT_TOOLS = ("memcheck", "synccheck", "initcheck")

#: entries here are (family, tool) pairs with a recorded reason, mirroring
#: `docs/testing.md`'s "committed suppressions file, and an explicit skip-list
#: whose entries carry their reason" -- empty by design (docs/testing.md: "An
#: empty skip-list is the standard the successor lane inherits").
SKIP: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class AssertedRaces:
    """ONE IDIOM'S REPORTS, DECLARED AND PINNED TO THE CODE THAT PRODUCES THEM.

    This is NOT a skip and NOT a suppression: the tool still runs, its whole report
    is still read, and the pair still FAILS on anything the declaration does not
    name. What the declaration buys is the ability to say "these exact reports, from
    these exact expressions, in this exact kernel, are the tool modelling less than
    the program does" -- with the proof of that claim attached and re-checkable.
    KERNEL_STANDARDS.md §18: a declared exception carrying its exit condition, never
    a blanket skip.

    The declaration LOCATES ITSELF IN THE SOURCE rather than pinning line numbers:
    `writes` and `reads` are the expressions whose lines racecheck may name, found in
    `source` at gate time. An unrelated edit above the combine shifts the lines and
    changes nothing; an edit TO THE COMBINE moves the expressions and re-opens the
    proof, which is the property that matters.
    """

    kernel: str
    source: str
    writes: tuple[str, ...]
    reads: tuple[str, ...]
    hazard_types: frozenset[str]
    proof: str
    exit_condition: str

    def lines(self, exprs: tuple[str, ...]) -> set[int]:
        text = (ROOT / self.source).read_text().splitlines()
        found: set[int] = set()
        for expr in exprs:
            hits = [i + 1 for i, line in enumerate(text) if expr in line]
            if len(hits) != 1:
                raise LookupError(
                    f"{self.source}: the declared expression {expr!r} was found "
                    f"{len(hits)} times, expected exactly 1 -- the combine moved, so "
                    f"the K47 proof no longer describes this code. Re-derive it "
                    f"(FINDINGS ## K47) before re-declaring the exception.")
            found.add(hits[0])
        return found


#: THE DECLARED EXCEPTIONS, empty. The one it held -- the decode step's warp arrival count,
#: a fence and an atomic in place of a barrier, which racecheck does not model -- died with
#: the arrival count: the decode visit folds its warps behind a barrier (DELETIONS.md,
#: 2026-09-29).
ASSERTED: dict[tuple[str, str], AssertedRaces] = {}

#: the kernel's demangled name contains spaces, so the name group is GREEDY and the
#: file:line is anchored on the LAST " in <path>:<line>" of the record line.
_RACE_WRITE = re.compile(r"Race reported between (\w+) access at (.+) in (\S+):(\d+)")
_RACE_OTHER = re.compile(r"and (\w+) access at (.+) in (\S+):(\d+)")
_HAZARD = re.compile(r"Potential (\w+) hazard")
_SUMMARY = re.compile(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)")


def _conforms(m: re.Match, kind: str, want: set[int], decl: AssertedRaces) -> bool:
    """One access line IS the declared idiom -- direction, kernel, file and line.

    The DIRECTION is declared too: the proof covers a deposit READ BY the last
    arriver, so a WAW or WAR record on the very same two lines is not it.
    """
    return (m.group(1) == kind
            and decl.kernel in m.group(2)
            and Path(m.group(3)).name == Path(decl.source).name
            and int(m.group(4)) in want)


def adjudicate(decl: AssertedRaces, out: str) -> tuple[bool, str]:
    """Read the WHOLE report and pass only if every record is the declared idiom."""
    m = _SUMMARY.search(out)
    if m and m.group(1) == "0":
        return True, ("DECLARED EXCEPTION IS NOW DEAD -- racecheck reported no hazards. "
                      f"Remove its ASSERTED entry. {decl.exit_condition}")
    want_w, want_r = decl.lines(decl.writes), decl.lines(decl.reads)
    types = set(_HAZARD.findall(out))
    if types - decl.hazard_types:
        return False, f"hazard types {sorted(types - decl.hazard_types)} are not declared (declared: {sorted(decl.hazard_types)})"

    records, offenders = 0, []
    pending = False
    for line in out.splitlines():
        mw = _RACE_WRITE.search(line)
        if mw:
            pending = True
            if not _conforms(mw, "Write", want_w, decl):
                offenders.append(line.strip())
            continue
        mo = _RACE_OTHER.search(line)
        if mo and pending:
            records += 1
            if not _conforms(mo, "Read", want_r, decl):
                offenders.append(line.strip())
    if offenders:
        head = "\n".join(f"    {o}" for o in offenders[:10])
        return False, (f"{len(offenders)} racecheck record(s) fall OUTSIDE the declared "
                       f"idiom -- these are NOT covered by any proof:\n{head}")
    if records == 0:
        return False, "racecheck failed but produced no parsable race records -- read the raw output"
    return True, (f"{records} race record(s), every one the DECLARED idiom "
                  f"(writes at {sorted(want_w)} -> reads at {sorted(want_r)} in "
                  f"{decl.source}, {sorted(decl.hazard_types)} only).")


def run_one(sanitizer: str, python: str, family: str, tool: str, timeout: int) -> tuple[bool, str]:
    row = _FAMILIES[family]
    cmd = [sanitizer, f"--tool={tool}", "--error-exitcode=1", python, "-m", "pytest", row.test, "-x", "-q"]
    #: ITS OWN SESSION, killed whole on a timeout: compute-sanitizer runs the target under a subreaper, so killing the
    #: direct child left the target running on the GPU under the next pair's lock.
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return False, f"TIMEOUT after {timeout}s -- a hang under {tool}, not a correctness verdict"
    out = stdout + stderr
    ok = proc.returncode == 0
    #: A SKIPPED CELL EXITS ZERO AND READS AS CLEAN (the carved family once reported
    #: CLEAN on a binary that did not carry its TEST arm, so the kernel never ran).
    #: The tool refuses the vacuous pass rather than reporting coverage it does not have.
    if ok and not re.search(r"\b[1-9]\d* passed", out):
        return False, ("VACUOUS: the wrapped pytest exited 0 without passing a test (a skip "
                       "or an empty selection). The kernel never ran, so this pair measures "
                       "nothing -- build the cell's arm (`bazel run //:library`, every arm) and re-run.\n"
                       + "\n".join(out.splitlines()[-20:]))
    #: THE DECLARED EXCEPTION IS ADJUDICATED, NOT ASSUMED. The tool ran, the whole
    #: report is read, and a pair with a declaration passes ONLY if every record in it
    #: is the declared idiom -- so a real new hazard in the same kernel still fails.
    decl = ASSERTED.get((family, tool))
    if decl is not None:
        passed, note = adjudicate(decl, out)
        return passed, ("DECLARED EXCEPTION -- " + note + "\n  proof: " + decl.proof
                        + "\n  exit:  " + decl.exit_condition) if passed else (
                        "DECLARED EXCEPTION DOES NOT COVER THIS RUN -- " + note)
    tail = "\n".join(stdout.splitlines()[-30:] + stderr.splitlines()[-30:])
    return ok, tail


def run_absent(python: str, family: str, tool: str, timeout: int) -> tuple[bool, str]:
    """An absent family's cell must still FAIL -- otherwise the kernel has arrived.

    The pair is not run under the sanitizer (there is nothing to instrument); what runs
    is the cell itself, and the declaration holds only while it is red. A pass here is a
    failure of THIS script: the row says `present=False` and the tree disagrees.
    """
    row = _FAMILIES[family]
    proc = subprocess.run([python, "-m", "pytest", row.test, "-x", "-q"],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    if re.search(r"\b[1-9]\d* passed", out):
        return False, (f"{row.kernel} is declared ABSENT but {row.test} PASSES -- the "
                       f"kernel is built, so this row owes `present=True` and the pair "
                       f"owes a real sanitizer run.\n" + "\n".join(out.splitlines()[-20:]))
    return True, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", action="append", choices=_TOOLS, default=None)
    ap.add_argument("--family", action="append", choices=tuple(_FAMILY_TESTS), default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--python", default=sys.executable)
    #: the toolkit's own, never whichever is first on PATH: the lane reads the library with the release that built it
    ap.add_argument("--sanitizer", required=True,
                    help="the compute-sanitizer binary; `bazel test //tests:sanitizer` passes the toolkit's own")
    args = ap.parse_args()
    sanitizer = Path(args.sanitizer)
    if not (sanitizer.is_absolute() and sanitizer.is_file() and os.access(sanitizer, os.X_OK)):
        ap.error(f"--sanitizer {args.sanitizer}: an absolute path to an executable, never a name PATH resolves -- the "
                 "lane passes the toolkit's own: bazel test //tests:sanitizer")
    #: the release the toolchain record pins, byte for byte (normalized), as rule 1 pins the assembler and the readers
    record = toolchains.named(toolchains.built_extension(ROOT).parent.name.removeprefix("rola_"))
    here = toolchains.normalize(toolchains.version_text(str(sanitizer)))
    if here != record.sanitizer:
        ap.error(f"--sanitizer {sanitizer} is not toolchain {record.name}'s compute-sanitizer:\n  record: "
                 f"{record.sanitizer}\n  here  : {here}")

    tools = args.tool or list(_DEFAULT_TOOLS)
    families = args.family or list(_FAMILY_TESTS)

    failures: list = []
    absent: list = []
    #: a sanitizer run is CORRECTNESS work on the GPU
    #: (`mode="shared"`) AND draws K=2 of the host-compute budget
    #: -- `compute-sanitizer` itself is CPU-heavy (instrumenting every
    #: memory access), on top of whatever the wrapped pytest process spends.
    with host.acquire(2, label="sanitize_oracle"), gpu_lock(mode="shared"):
        for family in families:
            for tool in tools:
                reason = SKIP.get((family, tool))
                if reason is not None:
                    print(f"SKIP {family}/{tool}: {reason}")
                    continue
                row = _FAMILIES[family]
                if not row.present:
                    ok, tail = run_absent(args.python, family, tool, args.timeout)
                    print(f"{'ABSENT' if ok else 'FAIL'} {family}/{tool}: "
                          f"{row.kernel} has no implementation on this line -- coverage "
                          f"owed by the stage that builds it")
                    if not ok:
                        print(tail)
                        failures.append((family, tool))
                    else:
                        absent.append((family, tool))
                    continue
                print(f"RUN  {family}/{tool} ({row.test}) ...", flush=True)
                ok, tail = run_one(args.sanitizer, args.python, family, tool, args.timeout)
                declared = (family, tool) in ASSERTED
                status = ("DECLARED" if declared else "CLEAN") if ok else "FAIL"
                print(f"{status} {family}/{tool}")
                if not ok or declared:
                    print(tail)
                if not ok:
                    failures.append((family, tool))

    if absent:
        print(f"\n{len(absent)} pair(s) ABSENT (the kernel is not built on this line, "
              f"so the coverage is owed, not held): {absent}")
    if failures:
        print(f"\n{len(failures)} sanitizer failure(s): {failures}")
        return 1
    live = len(tools) * len([f for f in families if _FAMILIES[f].present]) - len(failures)
    print(f"\n{live} LIVE (family, tool) pair(s) CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
