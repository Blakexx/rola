"""SPILL-MANIFEST GATE for the extension's launch bounds.

**WHAT THIS CONVERTS, AND WHY.** A measured spill profile accepted as an optimum is
a judgment recorded in a document, and a judgment that lives nowhere else means
**nothing fails if a later codegen change grows it.** This file makes it an ASSERTED
INVARIANT, on the same doctrine every other gate in this tree runs on -- a property
the batch believes is a property the batch checks. The profile is a RATCHET per
instantiation, not a spill ban: the gate holds each entry AT its ratified bytes.

The gate measures the shipped translation units together: the chunk consumer's
instantiation shards, the DECODE GEMV and the two UNION-ENTMAX kernels, so every
`__launch_bounds__` in the extension is read off `ptxas -v` instead of asserted in a
comment.

SIX OUTCOMES, and the difference between them is deliberate:

* **FAIL -- toolchain drift.** The `ptxas` this gate invokes reports a version string
  different from the one the manifest was ratified under. **CLOSED-WORLD CODEGEN POLICY
  RULE 1 (user directive, 2026-08-01.)** A spill/register manifest is a property of the
  ASSEMBLER as much as of the source: every number in this file was produced by one
  `ptxas`, and a different one may legitimately reach different allocations. Checking a
  new assembler's output against an old assembler's ratification is wrong in BOTH
  directions -- it can fail a regression nobody introduced, and it can pass one the new
  allocator did introduce while the stale numbers happened to be looser. So the version
  is RECORDED in the manifest and CHECKED, and a mismatch is a hard failure demanding
  re-ratification (`--write`), never a warning: a warning is read once and thereafter
  ignored, which is the same as not checking.
* **FAIL -- spill growth.** Any instantiation whose spill bytes EXCEED its manifest
  entry. Spills are the thing being bounded; growth is a regression whatever caused it.
* **FAIL -- an unhonored bound.** Any `Value of threads per SM ... is out of range`
  warning. That entry compiled with NO launch bound, so `ptxas` was free to spend
  registers as it liked. It is a hard failure even at zero spills.
* **FAIL -- source drift (session 9).** The tree's `csrc/rola/src` digest differs
  from the one the manifest was ratified against. The manifest's numbers -- and the
  register-budget derivation they justify -- describe a kernel that no longer
  exists; re-ratify with `--write`. This is the structural fix for the
  stale-constant trap: a measured constant's justification must not outlive the
  working set it was measured on.
* **FAIL -- hot-loop growth (session 9).** Any instantiation executing MORE
  `LDL`/`STL` inside a loop (backward-`BRA` interval in the SASS) than its ratified
  `hot_locals` count. In-loop spill is a memory round-trip per iteration and is
  disqualifying whatever the byte count; the count is a RATCHET rather than a ban so
  that structural dynamically-indexed local-array traffic no budget removes is held
  at its measured level instead of forbidden.
* **REPORT -- register drift.** A changed `Used N registers` with no spill growth is
  printed, not failed. The register count is an ALLOCATOR OUTCOME that legitimately
  moves when the kernel changes; what must not move is the spilling.

THE CENSUS (`--census [--json PATH]`) is the measurement mode the register rung is
derived from: a compile of every built arm at the SHIPPED derived bound, reported
per structural key and per SHARD. It never touches a manifest. Measuring a
CANDIDATE bound is a scratch-tree edit of the bound itself
(`docs/measurement.md`), not a flag here -- a forced bound must never be
ratifiable, and the cheapest way to guarantee that is for the tool not to have
one.

**RE-RATIFICATION IS A MEASUREMENT, NOT AN EDIT.** The manifest is regenerated only by
`--write`, which re-runs this same compile and records what it reports. Hand-editing an
entry to make the gate pass would be asserting a spill nobody measured, which is
exactly the failure this file replaces. When a kernel change legitimately moves the
numbers: re-run `--write`, and put the new numbers in the findings document with the
reason they moved.

Usage (no GPU required -- this is a COMPILE-time gate, so it does not take the lock):

    bazel test //:ratified                     # the verdict on the shipped build
    bazel run //:ratify -- --write             # the manifests, re-recorded from the build (a milestone's)
    bazel run //:ratify -- --census            # every instantiation's registers and spills, for a scope a flag sets

`--shard` is a MEASUREMENT scope and never a ratification scope: a ratchet refusal is
investigated one shard's slice at a time instead of by re-deriving every entry, but
`--write` refuses it, because a manifest written from a subset of the translation
units would silently drop every instantiation it did not compile.

`--self-test` requires the gate to FAIL on the things it must reject: a deliberately
tightened manifest, a perturbed `ptxas` version, an arch with no committed manifest, a
per-arch split that does not rejoin to one digest, per-arch files that disagree about
shard membership, a shard block that is decorative or that is a property of the
filing, a perturbed source digest, and in-loop local-access growth past a ratified
count (plus the hot-loop parser proving it counts an in-loop `LDL` and clears a cold
one on synthetic SASS). That is what makes a green run mean something: a gate that
cannot fail is not a gate (`rewrite.md`'s vacuity discipline, applied to itself).

MANIFEST FORMAT: `tools/manifests/<toolchain>/sm_XX.json`, ONE FILE PER ARCH, each
`{"toolchain": {ptxas, defines, nvcc_flags, sass_hash_version, mold}, "source":
{csrc_sha256, date, image_digest}, "shards": {...},
"entries": {<mangled name>:
{...}}}`. The metadata blocks are siblings of the entries rather than reserved keys
inside them, so an instantiation can never be confused for metadata by a name that
happens to collide. `shards` is `{shard name: {members_sha256, count}}` -- MEMBERSHIP
ONLY, no paths and no ordering -- and it enters the digest, which is what makes "only
shards X and Y were re-measured" a provable statement rather than a claim. Every arch
file must agree on it: the per-arch split is a filing decision and may not become a
split MATRIX.

**WHY PER ARCH.** *"Does arch X have a ratified manifest?"* becomes a file-existence
question, which is what the build's rule-3 gate (the graph's record action) asks before
anything is linked; a new architecture's bring-up is then one reviewable file in a PR instead of a
1024-line diff; and an `--arch 89` run cannot restate measurements it did not take. The
SPLIT IS FILING ONLY, and that is checked rather than asserted: the digest
`_build_config.py` records is taken over the merged canonical record, so N files hash to
exactly what one file hashed to (`--self-test`, last arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402
import derivation  # noqa: E402
import dev_config  # noqa: E402
import gen_shards  # noqa: E402
import kernels  # noqa: E402
import mold_toolchain  # noqa: E402
import sass  # noqa: E402
import toolchains  # noqa: E402
from build_record import manifest_digest  # noqa: E402
from csrc_stamp import INC as CSRC_STAMP_INC  # noqa: E402
from csrc_stamp import digest as _csrc_digest  # noqa: E402
from csrc_stamp import header as _csrc_header  # noqa: E402
from rola_devtools.locks import host  # noqa: E402

os.environ["TORCH_NO_COMPILER_WRAPPER"] = "1"

#: ONE toolchain location for nvcc and ptxas, the build's own (`dev_config.cuda_bin`): the version this gate reads is
#: from the assembler that produced the numbers it parses.
cuda_bin = dev_config.cuda_bin

#: THE GRAPH'S BUILD (`--graph`, bazel/defs.bzl's `ratify_census`): the units its compile rule compiled -- each one's ptxas report and
#: object stand in for this gate's recompile -- its assembler and disassemblers, its derivation harness and the build's
#: parameters. -- see docs/internals/tools/ratify.md#graph
_GRAPH: dict = {}


def _graph_bin(name: str) -> str:
    """The graph's toolkit binary `name`."""
    return str(_GRAPH["tools"][name])


def use_graph(census: Path, runfiles: Path) -> None:
    """Read the graph's census record, whose paths are relative to `runfiles`: this process's assembler and
    disassemblers become the graph's, and its census reads the graph's compiles."""
    global cuda_bin
    blob = json.loads(census.read_text())

    def where(short: str) -> Path:
        return Path(os.path.normpath(runfiles / short))

    _GRAPH.update(tools={name: where(path) for name, path in blob["tools"].items()},
                  units=[{"source": Path(u["source"]), "report": where(u["report"]), "object": where(u["object"])}
                         for u in blob["units"]],
                  harness=where(blob["derivation"]["harness"]), harness_binary=where(blob["derivation"]["binary"]),
                  parameters=blob["parameters"])
    cuda_bin = _graph_bin
    sass.use_graph(cuobjdump=_graph_bin("cuobjdump"), nvdisasm=_graph_bin("nvdisasm"))


def graph_parameters() -> tuple[tuple[str, ...], bool]:
    """The graph build's architectures, and whether it is an iteration build -- `gen_shards.is_iteration` on the
    build's parameters, as its record decides it (bazel/record.py)."""
    p = _GRAPH["parameters"]
    archs = tuple(x for x in p["archs"].split(",") if x)
    carry = gen_shards.family_arms(gen_shards.CARRY, p["carry_arms"], "--//:carry_arms")
    decode = gen_shards.family_arms(gen_shards.DECODE, p["decode_arms"], "--//:decode_arms")
    parts, trivial = gen_shards.carry_parts_spec(p["carry_parts"], "--//:carry_parts")
    return archs, gen_shards.is_iteration(carry, decode, parts, trivial, archs, toolchain().archs())


MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
SRC = Path(__file__).resolve().parents[1] / "csrc" / "rola" / "src"


def _compile_dir() -> Path:
    """The compile path's working directory, in the dev config's scratch, its per-unit records beside it."""
    return sass.scratch("ratify")


#: THE TRANSLATION UNITS THIS GATE COMPILES ARE THE BUILD'S: `tools/kernels.py`'s one list, so every kernel the
#: binary ships is measured and no second enumeration can drift from the build's (it did: three kernels went
#: unratified behind a hand-listed tuple). The kernels live in NAMED namespaces, so their mangled names do not depend
#: on which translation unit expands them. -- see docs/internals/tools/ratify.md
def tu_paths(shards: tuple[str, ...] | None = None, arms=None) -> list[Path]:
    """The translation units to compile. `shards` scopes to a subset by stem.

    Scoping is a MEASUREMENT scope, never a ratification scope: `--write` still
    refuses to file a manifest that does not carry the whole matrix, because a
    partial manifest would silently drop every arm it did not compile.
    """
    paths = kernels.units(arms)
    if shards is not None:
        by_stem = {p.stem: p for p in paths}
        unknown = sorted(set(shards) - set(by_stem))
        if unknown:
            raise SystemExit(f"no such shard: {', '.join(unknown)}\n"
                             f"    known: {', '.join(sorted(by_stem))}")
        return [by_stem[s] for s in sorted(shards)]
    return paths


#: The vendored CuTe/CUTLASS headers, in the SAME order the graph passes them
#: (repository headers first). This gate's whole premise is that it compiles the
#: shipped translation units under the shipped kernel's flags; an include path the
#: extension has and this gate does not is that premise failing silently.
CUTLASS_INC = Path(__file__).resolve().parents[1] / "csrc" / "third_party" / "cutlass" / "include"

#: The compile path's `-gencode` targets by default, overridable per run with `--arch`. BOTH are measured by default, because
#: both are in the shipped fat binary and the two allocators reach different
#: answers: sm_80 admits more CTAs by shared memory, so its bound is a different
#: number and its spill profile is a different measurement.
DEFAULT_ARCHES = ("80", "86")


def toolchain(name: str | None = None) -> toolchains.Toolchain:
    """The named toolchain, or the one whose assembler the configured toolkit's is."""
    return toolchains.named(name) if name else toolchains.for_ptxas(ptxas_version())


def manifest_path(arch: str, name: str | None = None) -> Path:
    """ONE FILE PER ARCH, per toolchain.

    *"Does arch X have a ratified manifest?"* is then a FILE-EXISTENCE question --
    which is what the build's rule-3 gate (the graph's record action) asks first,
    and what makes a new-architecture bring-up a single reviewable file in a PR
    rather than a 1024-line diff nobody can read. The union of the files is the
    ratification; `load_manifest` is the only thing that forms it.
    """
    return toolchain(name).manifest_path(arch)


def load_manifest(arches, name: str | None = None) -> dict:
    """The per-arch files, read as ONE ratification record.

    Every file must pin the SAME assembler. A manifest whose halves were ratified
    under different `ptxas` builds is not one ratification, and silently merging
    them would produce a record that describes no binary that was ever built.
    """
    toolchain = None
    shards = None
    entries: dict[str, dict] = {}
    sources: dict[str, dict | None] = {}
    for arch in arches:
        path = manifest_path(arch, name)
        if not path.exists():
            raise SystemExit(
                f"no ratified manifest for sm_{arch} at {path}; run\n"
                f"    python tools/ratify.py --arch {arch} --write\n"
                f"and put the numbers in the findings document")
        blob = json.loads(path.read_text())
        sources[arch] = blob.get("source")
        #: THE SHARD MEMBERSHIP is a property of the ratification, not of one arch:
        #: the same source files were compiled for every target. Two files that
        #: disagree about which instantiations exist are not one ratification, and
        #: merging them would produce a record describing no binary ever built --
        #: the same failure the split-toolchain check above refuses.
        if shards is None:
            shards = blob.get("shards")
        elif blob.get("shards") != shards:
            raise SystemExit(
                f"{path.name} records a different shard membership than the other "
                f"manifests. The per-arch SPLIT is a filing decision; it may not "
                f"become a split MATRIX. Re-ratify every arch off one tree.")
        if toolchain is None:
            toolchain = blob["toolchain"]
        elif blob["toolchain"] != toolchain:
            raise SystemExit(
                f"{path.name} pins a different assembler than the other manifests:\n"
                f"    {blob['toolchain']['ptxas']}\n"
                f"    {toolchain['ptxas']}\n"
                "The per-arch SPLIT is a filing decision; it may not become a split "
                "TOOLCHAIN. Re-ratify every arch under one assembler.")
        for key, value in blob["entries"].items():
            if key in entries:
                raise SystemExit(f"{key} is ratified in two files")
            entries[key] = value
    return {"toolchain": toolchain, "entries": entries, "shards": shards,
            "sources": sources}


ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+a?)'")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
REGS = re.compile(r"Used (\d+) registers")
OUT_OF_RANGE = re.compile(r"Value of threads per SM for entry (\S+) is out of range")


#: The FACTS kernels' mangled template arguments, in DECLARATION order.  The
#: union-table pass is templated on the PLAN TYPE, so its key
#: carries the plan's `(D, B, s_0..s_3)` -- the SPANS, which is what the pass is
#: actually parameterized by -- rather than the `BC` a lex arm is named by;
#: `atom_bitmap_kernel<int D, int B>` and `block_bitmap_kernel<int D, int B, int BC>`
#: are lex-only reductions and keep their integer keys.  Decoded in ONE place: the
#: manifest key is what the assembler saw, and every consumer of it -- the census's
#: per-key grouping, the hot-loop gate's reporting, the fatbin gate's ownership
def shard_field(entries: dict) -> dict[str, str]:
    """`{entry key -> the kernel it instantiates}`, the census's per-row shard column.

    The shard is read off the symbol the assembler saw (`tools/kernels.py`'s demangling), so it cannot disagree
    with the binary and every family answers in the same schema.
    """
    keys = sorted(entries)
    return dict(zip(keys, kernels.demangled_names([k.split(":", 1)[1] for k in keys])))


#: The calibration macros a shipped binary compiles under (coverage-closure audit
#: item 10). EMPTY since the endgame executor: the ring
#: (`ROLA_C_RING_DEPTH`) and the write-panel window (`ROLA_C_WT_WINDOW`) died
#: with the aligned walk -- the kernel has no compile-time calibration macros
#: left. The mechanism stays so the NEXT macro is a one-line addition, and a
#: manifest still records the (empty) block, keeping "what did this compile
#: under" a checkable statement rather than an unverifiable claim.
#: Each entry is `(macro, the header that defines its default)`, RELATIVE TO `SRC`.
#: The header is named PER MACRO, because a calibration macro is a property of the
#: kernel that reads it: naming one header for the whole record makes the record
#: depend on a file it may have no macro in, and a file the record does not need is a
#: file the record must not read. An empty tuple reads nothing.
RATIFIED_DEFINES: tuple[tuple[str, str], ...] = ()


def shipped_defines() -> dict[str, int]:
    """The `#define` defaults of :data:`RATIFIED_DEFINES` in the digested source."""
    out = {}
    for name, rel in RATIFIED_DEFINES:
        path = SRC / rel
        if not path.exists():
            raise SystemExit(f"{rel} is ratified as {name}'s source but does not "
                             f"exist; the provenance record cannot be produced")
        m = re.search(rf"^#define {name} (\d+)$", path.read_text(), re.MULTILINE)
        if m is None:
            raise SystemExit(f"{rel} no longer defines a default for {name}; the "
                             f"provenance record cannot be produced")
        out[name] = int(m.group(1))
    return out


#: Flags that say HOW THE COMPILE IS RUN rather than WHAT CODE IT PRODUCES, and are
#: therefore excluded from the ratified list. `-gencode` names the targets, which the
#: per-arch FILING already records -- keeping it would make one arch's manifest
#: disagree with another's on a block every arch file must share. `--threads` is a
#: parallelism knob (`host.nvcc_threads` in the dev config), so including it would
#: make a build that spends more cores on itself read as a codegen change.
_NON_CODEGEN_FLAG_PREFIXES = ("-gencode", "--threads=")


def ratified_flags() -> list[str]:
    """The codegen flags a manifest is measured under, ARCH- AND JOB-INDEPENDENT.

    Everything `tools/build_flags.py` emits except :data:`_NON_CODEGEN_FLAG_PREFIXES`:
    the optimization level, the language standard, `-lineinfo`, the torch defines,
    `--ptxas-options=-v` and the fatbin compression. Each of those either changes what
    `ptxas` assembles or changes the report this gate parses.
    """
    #: any thread count: `--threads` is not a codegen flag and leaves the list
    flags = build_flags.torch_defines() + build_flags.nvcc_flags((), nvcc_threads=1)
    return [f for f in flags if not f.startswith(_NON_CODEGEN_FLAG_PREFIXES)]


def check_flags(manifest: dict) -> bool:
    """The recorded flag list and the tree's flag list must be the same.

    THE GAP THIS CLOSES, recorded when `-Xfatbin -compress-all` landed: the manifest
    pinned the assembler and the calibration macros but not the flags, so a
    codegen-affecting flag change left no fingerprint in anything a gate reads. It was
    caught then only because both compilers import one flag list -- a property of the
    code, not a gate. A missing block is a failure for the same reason a missing
    `defines` block is: a manifest that cannot say what it compiled under cannot
    certify what shipped.
    """
    want = ratified_flags()
    got = manifest["toolchain"].get("nvcc_flags")
    if got == want:
        print(f"flags OK  {len(want)} codegen flags")
        return True
    missing = [f for f in want if f not in (got or [])]
    extra = [f for f in (got or []) if f not in want]
    if got is not None and not missing and not extra:
        #: SAME FLAGS, DIFFERENT ORDER. Order is compared deliberately -- `nvcc` reads
        #: its flag list left to right and a later flag can override an earlier one, so
        #: a reordering is not provably a no-op -- but a refusal whose two difference
        #: lists are both empty says nothing, and a reader would take it for a bug in
        #: the gate rather than a difference in the manifest.
        print(f"\nFAIL  FLAG DRIFT (ORDER). The manifest records the same {len(want)} "
              f"flags in a different order.\n  tree:     {want}\n  manifest: {got}\n"
              f"Order is compared because `nvcc` reads the list left to right and a "
              f"later flag can override an earlier one; re-ratify with --write.")
        return False
    print(f"\nFAIL  FLAG DRIFT. The manifest was measured under a different flag list "
          f"than the tree compiles with.\n  in the tree, not the manifest: {missing}\n"
          f"  in the manifest, not the tree: {extra}\n"
          f"The ratified register and spill numbers describe a compile nobody runs; "
          f"re-ratify with --write.")
    return False


def check_defines(manifest: dict) -> bool:
    """The recorded calibration macros and the tree's defaults must be the same.

    A missing block is a failure too: a manifest ratified before audit item 10
    landed cannot answer what the binary compiled, which is the gap the record
    exists to close -- `--write` on the current tree produces it.
    """
    want = shipped_defines()
    got = manifest["toolchain"].get("defines")
    if got == want:
        print(f"defines OK  {want}")
        return True
    print(f"\nFAIL  DEFINE DRIFT. The manifest records calibration macros {got} but "
          f"the tree's shipped defaults are {want}. The ratified numbers were "
          f"measured under macros the tree no longer compiles (or were never "
          f"recorded at all); re-ratify with --write.")
    return False


def csrc_digest() -> str:
    """A digest of the kernel sources the manifest's numbers were measured against (`tools/csrc_stamp.py`'s).

    THE FRESHNESS TRIGGER. A register budget is a MEASURED constant; its justification EXPIRES when `csrc/` moves, so
    a ratification whose sources have materially changed is a hard failure demanding `--write`, a fresh measurement.
    """
    return _csrc_digest(SRC)


# ---------------------------------------------------------------------------
# THE DERIVATION, RATIFIED
#
# The addressing block is HOST code: one function, `derive_carry_geom`, run once per
# launch, whose output every kernel of the family then addresses through. Its numbers
# are as much a measured property of this tree as a register count is -- an off-by-one
# in a span or a shift is a wrong answer nothing else in this gate would see, because
# the SASS it produces is somebody else's kernel. So it is ratified the same way the
# codegen is, and with the same two independent statements:
#
#   * A CODE HASH over the sources the derivation is computed from. That is the
#     wholesale trigger: any edit there invalidates the record at once.
#   * GOLDEN OUTPUTS -- the derived block, byte for byte, on a DECLARED set of named
#     cases, plus the REFUSAL each declared-illegal case produces. That is the
#     per-case answer the hash can only ask globally, and it reads the same two ways
#     the `sass_sha256` ratchet does: with the hash matching, no golden may move; with
#     the hash moved, the gate names exactly which cases moved and which did not.
#
# The refusals are ratified beside the outputs deliberately. R13 makes the kernel's
# strict shape half a contract whose other half is what it REFUSES, and an unstated
# refusal is the half that rots -- a derivation that quietly starts admitting a
# non-power-of-two width would pass a gate that only checked the shapes it admits.
#
# The harness compiles the shipped header directly and links no extension, so this
# check cannot be answered by a stale `.so`: there is no binary in the loop to be
# stale. That is deliberate -- an extension-trap-proof check of a host function.
DERIVATION_CASES_JSON = MANIFEST_DIR / "derivation_cases.json"
DERIVATION_JSON = MANIFEST_DIR / "derivation.json"


def load_derivation_cases() -> dict:
    """The declared cases, read and refused field by field."""
    if not DERIVATION_CASES_JSON.exists():
        raise SystemExit(f"no declared derivation cases at {DERIVATION_CASES_JSON}; "
                         f"the derivation cannot be ratified against nothing")
    blob = json.loads(DERIVATION_CASES_JSON.read_text())
    if blob.get("schema") != 1:
        raise SystemExit(f"{DERIVATION_CASES_JSON.name} declares schema "
                         f"{blob.get('schema')!r}; this gate reads schema 1")
    if not blob.get("sources"):
        raise SystemExit(f"{DERIVATION_CASES_JSON.name} names no `sources`; a golden "
                         f"output with no code hash beside it cannot say what it is "
                         f"the output OF")
    for rel in blob["sources"]:
        if not (SRC / rel).exists():
            raise SystemExit(f"{DERIVATION_CASES_JSON.name} names the derivation "
                             f"source {rel}, which does not exist under {SRC}")
    names = [c["name"] for c in blob["geometry"]] + [c["name"] for c in blob["sub_boxes"]]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"{DERIVATION_CASES_JSON.name}: duplicate case name(s) "
                         f"{dupes}; a golden output is keyed by name, so two cases "
                         f"sharing one would file one measurement over the other")
    if not names:
        raise SystemExit(f"{DERIVATION_CASES_JSON.name} declares no case; a record "
                         f"that measures nothing would pass by existence alone")
    return blob


def derivation_sources(cases: dict) -> dict[str, str]:
    """`{declared source -> sha256}`, the derivation's code hash, per file.

    PER FILE rather than one digest over the set, so a drift report can name which
    header moved. `csrc_digest()` is the wholesale trigger for the CODEGEN; this is
    the narrow one for the derivation, and it must not be widened to all of `csrc/`:
    a change to a kernel body is not a change to the derivation, and a record that
    expires on every unrelated edit is a record people re-write without reading.
    """
    return {rel: hashlib.sha256((SRC / rel).read_bytes()).hexdigest()
            for rel in sorted(cases["sources"])}


def cases_digest(cases: dict) -> str:
    """A digest of the DECLARED CASES themselves.

    Without it a green run would be ambiguous: goldens can also stop failing because
    somebody deleted the case that failed.
    """
    payload = {"geometry": cases["geometry"], "sub_boxes": cases["sub_boxes"],
               "sources": cases["sources"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _golden_from_output(text: str) -> dict[str, str]:
    """The harness's stdout, parsed into `{case name -> golden}`.

    An `ok` line's byte dump is hashed HERE rather than in the harness so that one
    hash function serves the whole gate; a `refused` line keeps its words, because
    the words are the ratified thing.
    """
    golden: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts[1] == "refused":
            golden[parts[0]] = f"refused: {parts[2]}"
        elif parts[1] == "ok":
            golden[parts[0]] = (f"ok: {parts[2]} bytes, sha256 "
                                f"{hashlib.sha256(parts[3].encode()).hexdigest()}")
        else:
            raise SystemExit(f"the derivation harness printed an unreadable line: "
                             f"{line!r}")
    return golden


#: ONE COMPILE PER HARNESS TEXT, keyed by the text itself. The gate calls the
#: derivation measurement more than once in a run (the check, and every self-test
#: mutant), and the harness is a pure function of the declared cases -- so a second
#: compile could only ever produce the same answer more slowly, or a different one,
#: and a different one would mean the measurement is not reproducible, which is a
#: property this record needs anyway.
_DERIVATION_MEASURED: dict[str, dict[str, str]] = {}


def measure_derivation(cases: dict) -> dict[str, str]:
    """Compile and run the harness. Returns its goldens.

    No GPU and no extension: `nvcc` is used because the header carries
    `__host__ __device__` on the functions the device also calls, and defining those
    away for a host compiler would measure a source the tree does not have.
    """
    import tempfile

    text = derivation.harness(cases)
    memo = _DERIVATION_MEASURED.get(text)
    if memo is not None:
        return memo
    if _GRAPH:
        #: the graph compiled and linked the harness (tools/BUILD.bazel's derivation_harness) from this same rendering
        if _GRAPH["harness"].read_text() != text:
            raise SystemExit(f"the graph's derivation harness {_GRAPH['harness']} is not this tree's rendering of "
                             f"{DERIVATION_CASES_JSON.name}: rebuild")
        run = subprocess.run([str(_GRAPH["harness_binary"])], capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit(f"the derivation harness exited {run.returncode}:\n{run.stderr}")
        return _derivation_golden(cases, run.stdout, text)
    with tempfile.TemporaryDirectory(prefix="rola_derivation_") as tmp:
        src = Path(tmp) / "derivation_harness.cu"
        exe = Path(tmp) / "derivation_harness"
        src.write_text(text)
        #: THE HARNESS NAMES AN ARCHITECTURE. The addressing block's state capacity is a
        #: capability-row quantity, so `common/geom.cuh` reaches `common/arch_caps.cuh`,
        #: whose tabulation refuses `nvcc`'s DEFAULT device architecture by name. The
        #: derivation is invariant across the tabulated set -- every row carries the same
        #: register file -- so naming the smallest tabulated one measures the shipped
        #: derivation rather than pinning it to a part.
        cmd = [cuda_bin("nvcc"), "-std=c++20", "-O0", "-arch=sm_80", f"-I{SRC}",
               str(src), "-o", str(exe)]
        with host.acquire(1, label="ratify's derivation harness"):
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"the derivation harness does not compile against the tree's "
                f"{', '.join(cases['sources'])}:\n{proc.stderr}\n"
                f"    {shlex.join(cmd)}")
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit(f"the derivation harness exited "
                             f"{run.returncode}:\n{run.stderr}")
    return _derivation_golden(cases, run.stdout, text)


def _derivation_golden(cases: dict, stdout: str, text: str) -> dict[str, str]:
    """The harness's output as goldens, every declared case among them."""
    golden = _golden_from_output(stdout)
    declared = [c["name"] for c in cases["geometry"]] + \
               [c["name"] for c in cases["sub_boxes"]]
    missing = [n for n in declared if n not in golden]
    if missing:
        raise SystemExit(f"the derivation harness printed no line for {missing}; a "
                         f"record that silently drops a case proves nothing about it")
    _DERIVATION_MEASURED[text] = golden
    return golden


def derivation_record(cases: dict) -> dict:
    """The whole ratified record, measured now."""
    golden = measure_derivation(cases)
    return {"schema": 1,
            "sources": derivation_sources(cases),
            "cases_sha256": cases_digest(cases),
            "date": time.strftime("%Y-%m-%d"),
            "golden": golden}


def write_derivation() -> int:
    """Re-measure and file `tools/manifests/derivation.json`."""
    cases = load_derivation_cases()
    record = derivation_record(cases)
    with host.file_lock("ratify_derivation"):
        DERIVATION_JSON.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    n_ok = sum(1 for v in record["golden"].values() if v.startswith("ok:"))
    n_ref = len(record["golden"]) - n_ok
    print(f"wrote {DERIVATION_JSON} ({n_ok} derived blocks, {n_ref} ratified "
          f"refusals) over {len(record['sources'])} source file(s)")
    return 0


def check_derivation(record: dict | None = None) -> bool:
    """The derivation gate: the code hash, the case set, and every golden."""
    cases = load_derivation_cases()
    if record is None:
        if not DERIVATION_JSON.exists():
            print(f"\nFAIL  THE DERIVATION IS NOT RATIFIED. There is no "
                  f"{DERIVATION_JSON.name}: the addressing block's outputs are "
                  f"measured by nothing. Ratify them:\n"
                  f"    bazel run //:ratify -- --write-derivation")
            return False
        record = json.loads(DERIVATION_JSON.read_text())
    if record.get("schema") != 1:
        print(f"\nFAIL  {DERIVATION_JSON.name} records schema "
              f"{record.get('schema')!r}; this gate reads schema 1")
        return False

    want_sources = derivation_sources(cases)
    fresh = record.get("sources") == want_sources
    want_cases = cases_digest(cases)
    same_cases = record.get("cases_sha256") == want_cases
    golden = measure_derivation(cases)
    ratified = record.get("golden", {})

    moved = sorted(n for n in golden if n in ratified and golden[n] != ratified[n])
    added = sorted(n for n in golden if n not in ratified)
    dropped = sorted(n for n in ratified if n not in golden)

    if not same_cases:
        print(f"\nFAIL  DERIVATION CASE DRIFT. The declared cases hash "
              f"{want_cases[:16]}...; the record was measured on "
              f"{str(record.get('cases_sha256'))[:16]}.... A golden that stops "
              f"failing because its case was edited or deleted proves nothing; "
              f"re-ratify with --write-derivation.")
    if not fresh:
        print(f"\nFAIL  DERIVATION SOURCE DRIFT. The record was measured against "
              f"{record.get('sources')}, the tree hashes to {want_sources}. The "
              f"derivation's outputs must be re-measured "
              f"(--write-derivation).")
        unchanged = [n for n in golden if golden.get(n) == ratified.get(n)]
        print(f"      {len(unchanged)} of {len(golden)} golden outputs are "
              f"unchanged and are still described by the record; "
              f"{len(moved)} moved: {moved}")
    if (moved or added or dropped) and fresh and same_cases:
        print("\nFAIL  DERIVATION OUTPUT DRIFT with the sources and the cases "
              "UNCHANGED. The addressing block derives different values from the "
              "same code on the same inputs, which is a toolchain or a "
              "platform difference, not an edit -- investigate before re-writing.")
    for name in moved:
        print(f"      {name}\n        ratified: {ratified[name]}\n"
              f"        this run: {golden[name]}")
    for name in added:
        print(f"      {name}: measured now, ratified by nothing")
    for name in dropped:
        print(f"      {name}: ratified, but the harness produced no such case")

    if fresh and same_cases and not (moved or added or dropped):
        n_ref = sum(1 for v in golden.values() if v.startswith("refused:"))
        print(f"derivation OK  {len(golden)} golden outputs ({n_ref} ratified "
              f"refusals) over {len(want_sources)} source file(s)")
        return True
    return False


# ---------------------------------------------------------------------------
# THE DEVICE STAMP'S CSRC KEY: `tools/csrc_stamp.py`, which the build's stamp action runs too.
def csrc_stamp_header(digest: str | None = None) -> str:
    """The generated header the stamp sites include."""
    return _csrc_header(digest if digest is not None else csrc_digest())


def write_csrc_stamp(digest: str | None = None) -> Path:
    """Write `build/generated/csrc_stamp.inc` for this tree on every call."""
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = gen_shards.GENERATED_DIR / CSRC_STAMP_INC
    with host.file_lock("csrc_stamp_header"):
        path.write_text(csrc_stamp_header(digest))
    return path


def _nvcc_command(arches, source: Path, obj_path: str) -> list[str]:
    """The compile path's nvcc command for `source`: the graph's compile of it (docs/build-graph.md#compile) -- the
    codegen flags `tools/build_flags.py` holds for both, torch's extension defines, the include roots in the graph's
    order (the repository's, CUTLASS's, the generated headers, then torch's and Python's) -- with a dependency file for
    the per-unit record. The `-gencode` targets are THIS RUN's (`--arch`, the toolchain's ratified architectures by
    default), never a build's: a flag naming what was compiled, inherited, would narrow the ratification to a single-arch
    build's subset (docs/build.md#arm-subset). `ratified_flags()` records the codegen flags in the manifest and
    `check_flags()` compares them."""
    #: torch.utils.cpp_extension resolves its toolkit when first imported: a ratification compiles against the pinned one
    os.environ["CUDA_HOME"] = dev_config.cuda_home()
    import torch.utils.cpp_extension as ext

    cmd = [cuda_bin("nvcc"), "--generate-dependencies-with-compile", "--dependency-output", f"{obj_path}.d",
           "-c", str(source), f"-I{SRC}", f"-I{CUTLASS_INC}", f"-I{gen_shards.GENERATED_DIR}"]
    for inc in ext.include_paths(device_type="cuda"):
        cmd += ["-I", inc]
    cmd += ["-I", sysconfig.get_paths()["include"]]
    cmd += build_flags.torch_defines()
    cmd += build_flags.nvcc_flags(arches, nvcc_threads=dev_config.get("host.nvcc_threads"))
    #: the graph's units are position-independent objects compiled with torch's extension defines: host-side, inert
    #: in the codegen this gate reads, passed so the compile is the graph's
    cmd += ["-Xcompiler", "-fPIC", *build_flags.extension_defines()]
    return cmd + ["-o", obj_path]


def ptxas_version() -> str:
    """`ptxas --version`, whitespace-normalized to one line.

    The WHOLE output, not a parsed release number: the build string
    (`cuda_12.4.r12.4/compiler.33961263_0`) distinguishes two shipments of the same
    nominal release, and it is exactly those that can move an allocation while the
    release number says nothing changed. Normalizing whitespace makes the comparison
    insensitive to line endings and nothing else.
    """
    ptxas = cuda_bin("ptxas")
    proc = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"`{ptxas} --version` FAILED (rc={proc.returncode}); the gate "
                         f"cannot ratify a manifest against an assembler it cannot ask")
    return " ".join(proc.stdout.split())


def check_toolchain(got: str, want: str) -> bool:
    """Rule 1: the recorded assembler and the invoked assembler must be the same one."""
    if got == want:
        print(f"toolchain OK  ptxas: {got}")
        return True
    print(f"\nFAIL  TOOLCHAIN DRIFT. This manifest's spill and register numbers were "
          f"ratified under a different assembler:\n"
          f"        manifest  {want}\n"
          f"        invoked   {got}\n"
          f"      A launch-bound manifest is a property of the assembler as much as of "
          f"the source, so these numbers say nothing about this `ptxas`. RE-RATIFICATION "
          f"IS REQUIRED: re-run with --write and record the new numbers, and the version "
          f"they were taken under, in the findings document.")
    return False


def check_freshness(manifest: dict, digest: str | None = None) -> bool:
    """THE FRESHNESS TRIGGER (session 9, R1). The manifest's numbers -- and the
    register-budget derivation they justify -- are dated against a source tree;
    when `csrc/rola/src` moves materially, the ratification is a description of a
    kernel that no longer exists and must be RE-MEASURED (`--write`), never
    trusted. This is the structural fix for the stale-constant trap that blocked
    K5: a register budget's justification cannot silently outlive the working
    set it was measured on, because the gate that every build runs re-checks the
    tree the measurement was taken from."""
    digest = digest if digest is not None else csrc_digest()
    ok = True
    for arch, src in sorted(manifest["sources"].items()):
        if src is None:
            print(f"\nFAIL  sm_{arch}'s manifest predates the source-freshness "
                  f"trigger (no `source` block). Re-ratify: "
                  f"bazel run //:ratify -- --write")
            ok = False
        elif src["csrc_sha256"] != digest:
            print(f"\nFAIL  SOURCE DRIFT. sm_{arch}'s manifest was ratified against "
                  f"csrc {src['csrc_sha256'][:16]}... on {src.get('date', '?')}; "
                  f"the tree is now {digest[:16]}... . The register/spill numbers "
                  f"and the budget derivation they justify describe a kernel that "
                  f"no longer exists. Re-ratify with --write and record the new "
                  f"numbers in the findings document.")
            ok = False
    if ok:
        print(f"source freshness OK  csrc sha256 {digest[:16]}... "
              f"({len(manifest['sources'])} arch manifest(s))")
    return ok


def _parse_report(stderr: str) -> tuple[dict, list[str]]:
    entries: dict[str, dict] = {}
    unhonored = sorted(set(OUT_OF_RANGE.findall(stderr)))
    key = None
    for line in stderr.splitlines():
        m = ENTRY.search(line)
        if m:
            #: The MANGLED name is the key. It is exact, stable, and encodes every
            #: template argument, so an entry cannot be silently re-pointed at a
            #: different instantiation by a rename or a reordering.
            key = f"sm_{m.group(2)}:{m.group(1)}"
            entries[key] = dict(regs=0, spill_stores=0, spill_loads=0)
            continue
        if key is None:
            continue
        m = SPILL.search(line)
        if m:
            entries[key]["spill_stores"] = int(m.group(1))
            entries[key]["spill_loads"] = int(m.group(2))
        m = REGS.search(line)
        if m:
            entries[key]["regs"] = int(m.group(1))
    return entries, unhonored


#: WHERE A MEASURED TRANSLATION UNIT IS REMEMBERED: beside the compile path's working directory, in the scratch.
def _cache_dir() -> Path:
    return _compile_dir() / "units"


_FILE_SHA: dict[str, str] = {}


def _file_sha(path: str) -> str:
    """Content hash of one dependency, memoised: the arm TUs share nearly every
    header, so the closure is hashed once per FILE and not once per unit."""
    if path not in _FILE_SHA:
        try:
            _FILE_SHA[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            _FILE_SHA[path] = "ABSENT"
    return _FILE_SHA[path]


def _depfile_inputs(depfile: Path) -> list[str]:
    """Every file `nvcc` reported this translation unit reading, from the depfile the
    compile already writes (`--generate-dependencies-with-compile`)."""
    text = depfile.read_text().replace("\\\n", " ")
    _, _, rhs = text.partition(":")
    return sorted({tok for tok in rhs.split() if tok and not tok.endswith(":")})


def _tu_fingerprint(cmd: list[str], deps: list[str]) -> str:
    """WHAT MAKES A MEASUREMENT REUSABLE: the exact argv (less the object and depfile
    paths, which are a scratch directory's business) and the CONTENT of every input
    the compiler reported reading.  A header edit moves it; a new `#include` moves it
    too, because adding one edits a file already in the closure."""
    h = hashlib.sha256()
    h.update(f"v1|{SASS_HASH_VERSION}|".encode())
    skip = False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok in ("-o", "--dependency-output"):
            skip = True
            continue
        h.update(tok.encode())
        h.update(b"\0")
    for dep in deps:
        h.update(dep.encode())
        h.update(_file_sha(dep).encode())
    return h.hexdigest()


def _cache_read(source: Path, cmd: list[str]) -> dict | None:
    """The stored measurement for this unit, or `None` if it does not describe it.

    The stored DEPENDENCY LIST is what the fingerprint is recomputed over -- the
    record has to say what it depended on, or "unchanged" is a claim about nothing.
    """
    rec_path = _cache_dir() / f"{source.stem}.json"
    if not rec_path.exists():
        return None
    try:
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError):
        return None
    if rec.get("fingerprint") != _tu_fingerprint(cmd, rec.get("deps", [])):
        return None
    return rec


def _cache_write(source: Path, cmd: list[str], deps: list[str], entries: dict,
                 unhonored: list[str], scan: dict) -> None:
    _cache_dir().mkdir(parents=True, exist_ok=True)
    (_cache_dir() / f"{source.stem}.json").write_text(json.dumps(
        {"fingerprint": _tu_fingerprint(cmd, deps), "deps": deps, "entries": entries,
         "unhonored": unhonored, "scan": scan}, indent=1, sort_keys=True) + "\n")


def _scan_object(obj: str, wanted: set) -> dict:
    with sass.sass_lines(obj) as lines:
        return sass_scan(lines, wanted)


def measure_units(arches, objdir: str, sources: list[Path], arms=None) -> tuple[dict, list[str], dict]:
    """COMPILE AND DISASSEMBLE EACH TRANSLATION UNIT, OR REUSE ITS RECORD.

    The gate used to recompile the whole matrix on every run, which is what made it a
    40-minute act and therefore a milestone act (KERNEL_STANDARDS section 16). With one
    arm per translation unit the matrix is 30-odd independent units, and a unit whose
    input closure has not moved has nothing left to measure: its registers, its spills
    and its SASS body are the same numbers by the same argument the source digest makes
    globally. So the unit is the cache grain, keyed on the argv and the CONTENT of every
    input the compiler itself reported reading -- never on a timestamp.

    The reuse is not a weakening of the gate: what is replayed is a MEASUREMENT of an
    identical compile, and the thing that decides identity is the same content hash the
    freshness trigger uses, applied per unit instead of per tree.
    """
    #: WRITE THIS RUN'S SELECTION HEADER, ON EVERY RUN, BEFORE COMPILING ANYTHING, and it is not optional: docs/build.md records the exact trap
    #: this prevents -- "an arm-subset build wrote its -D into those very flags, and
    #: a later --write ... inherited the SUBSET and certified a manifest for arms
    #: nobody built." The old fix was argv-substitution; the header gets the same
    #: unconditional-overwrite treatment for the same reason: `arms=None` here means
    #: the shipped set (`kernels.arm_units`'s own default), never "whatever the last
    #: build left behind".
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    want_arms = sorted(set(range(len(gen_shards.CARRY_ARMS)) if arms is None else arms))
    #: ONE SHARED OUTPUT FILE: two concurrent ratify/build
    #: invocations racing this write must not interleave -- one must finish
    #: writing the header before the other starts compiling against it.
    with host.file_lock("carry_selection_header"):
        (gen_shards.GENERATED_DIR / gen_shards.CARRY_SELECTION_INC).write_text(
            gen_shards.carry_selection_header(want_arms))
        (gen_shards.GENERATED_DIR / gen_shards.CARRY_PARTS_INC).write_text(
            gen_shards.carry_parts_header(gen_shards.CARRY_PARTS, ()))
        (gen_shards.GENERATED_DIR / gen_shards.DECODE.header).write_text(
            gen_shards.decode_selection_header(range(len(gen_shards.DECODE_ARMS))))
        gen_shards.write_builds(arches)
    #: AND THE CSRC STAMP, for the same reason and from the same function the graph's
    #: stamp header runs: the stamp sites include it, so a gate that did not write it would
    #: measure a tree nobody builds.
    write_csrc_stamp()

    t0 = time.perf_counter()
    compile_dir = _compile_dir()

    def one(source: Path):
        obj = str(Path(objdir) / f"{source.stem}.o")
        cmd = _nvcc_command(arches, source, obj)
        rec = _cache_read(source, cmd)
        if rec is not None:
            return source, rec, True
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(compile_dir))
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"{source.name} FAILED to compile (rc={proc.returncode})")
        got, bad = _parse_report(proc.stderr)
        got_scan = _scan_object(obj, set(got))
        dep = Path(f"{obj}.d")
        deps = _depfile_inputs(dep) if dep.exists() else []
        rec = {"entries": got, "unhonored": bad, "scan": got_scan, "deps": deps}
        if deps:
            _cache_write(source, cmd, deps, got, bad, got_scan)
        return source, rec, False

    #: ONE MACHINE, ONE COMPILE BUDGET (KERNEL_STANDARDS §14 addendum 2): the compiles run under the host budget's
    #: slots, one job a `host.nvcc_threads` of them, as every build does through the graph's front door
    with host.acquire(host.budget_slots(), label="ratify's compile path") as held:
        jobs = max(1, held // dev_config.get("host.nvcc_threads"))
        with ThreadPoolExecutor(max_workers=min(jobs, len(sources))) as pool:
            measured = list(pool.map(one, sources))
    hits = [source.stem for source, _rec, hit in measured if hit]
    wall = time.perf_counter() - t0
    print(f"{len(sources)} translation unit(s) measured in {wall:.1f} s at "
          f"{min(jobs, len(sources))} jobs over {len(arches)} targets "
          f"({', '.join('sm_' + a for a in arches)}); {len(hits)} reused an unchanged "
          f"unit's record, {len(sources) - len(hits)} recompiled")
    return _merge_units([(source, rec) for source, rec, _hit in measured])


def measure_built() -> tuple[dict, list[str], dict]:
    """THE GRAPH'S CENSUS: each unit its build compiled, read from that compile -- ptxas's report and the object it
    wrote -- with no compile of its own."""
    units = _GRAPH["units"]
    t0 = time.perf_counter()

    def one(unit: dict):
        got, bad = _parse_report(unit["report"].read_text())
        return unit["source"], {"entries": got, "unhonored": bad, "scan": _scan_object(str(unit["object"]), set(got))}

    #: the disassemblers' reads hold the host budget too, a job a slot (KERNEL_STANDARDS §14 addendum 2): the
    #: machine's under `bazel run //:ratify`; under the sandboxed `//:ratified`, whose /tmp and dev config are its own,
    #: a pool of the sandbox's, the default slot count
    with host.acquire(host.budget_slots(), label="ratify's census") as jobs, ThreadPoolExecutor(
            max_workers=min(jobs, len(units))) as pool:
        measured = list(pool.map(one, units))
    print(f"{len(units)} translation unit(s) read from the graph's build in {time.perf_counter() - t0:.1f} s at "
          f"{min(jobs, len(units))} jobs")
    return _merge_units(measured)


def _merge_units(measured: list[tuple[Path, dict]]) -> tuple[dict, list[str], dict]:
    """One census from each unit's record: its entries, its unhonored bounds, its SASS scan."""
    entries: dict[str, dict] = {}
    unhonored: set[str] = set()
    scan: dict[str, dict] = {}
    owner: dict[str, str] = {}
    for source, rec in measured:
        for key, value in rec["entries"].items():
            if key in entries:
                #: TWO TRANSLATION UNITS EMITTING ONE INSTANTIATION is the exact state the split exists to prevent, and
                #: merging their reports would hide it behind numbers that look fine.
                raise SystemExit(
                    f"{key} was compiled by more than one translation unit "
                    f"({source.name} is the second, after {owner[key]}). Nothing is "
                    f"sharded; run `bazel run //tools:gen_shards -- --check`")
            entries[key] = value
            owner[key] = source.name
        unhonored |= set(rec["unhonored"])
        scan.update(rec["scan"])
    scan = {name: scan.get(name, {"body_sha256": None, "hot_locals": [], "local_ops": 0})
            for name in entries}
    return entries, sorted(unhonored), scan


def check(entries: dict, unhonored: list[str], want: dict) -> bool:
    ok = True
    if unhonored:
        ok = False
        print(f"\nFAIL  {len(unhonored)} instantiation(s) compiled with the launch "
              f"bound DISCARDED ('out of range'). `ptxas` was unconstrained there:")
        for name in unhonored[:8]:
            print(f"        {name}")
        if len(unhonored) > 8:
            print(f"        ... and {len(unhonored) - 8} more")

    missing = sorted(set(want) - set(entries))
    added = sorted(set(entries) - set(want))
    if missing or added:
        ok = False
        print(f"\nFAIL  the instantiation set moved: {len(missing)} manifest entries "
              f"absent from the build, {len(added)} new entries not in the manifest. "
              f"Re-ratify with --write once the matrix change is intended.")
        for name in (missing[:4] + added[:4]):
            print(f"        {name}")

    grew, drifted = [], []
    for name, got in sorted(entries.items()):
        exp = want.get(name)
        if exp is None:
            continue
        if (got["spill_stores"] > exp["spill_stores"]
                or got["spill_loads"] > exp["spill_loads"]):
            grew.append((name, exp, got))
        elif got["regs"] != exp["regs"]:
            drifted.append((name, exp, got))
    if grew:
        ok = False
        print(f"\nFAIL  {len(grew)} instantiation(s) SPILL MORE than the manifest:")
        for name, exp, got in grew:
            print(f"        {name}\n"
                  f"          manifest {exp['spill_stores']} st / {exp['spill_loads']} ld"
                  f"   realized {got['spill_stores']} st / {got['spill_loads']} ld")
    if drifted:
        print(f"\nREPORT  {len(drifted)} instantiation(s) changed register count with "
              f"NO spill growth (an allocator outcome, not a regression):")
        for name, exp, got in drifted[:10]:
            print(f"        {exp['regs']} -> {got['regs']}  {name}")
        if len(drifted) > 10:
            print(f"        ... and {len(drifted) - 10} more")

    spillers = [(n, e) for n, e in sorted(entries.items()) if e["spill_stores"]
                or e["spill_loads"]]
    print(f"\n{len(entries)} instantiations measured, {len(unhonored)} with an unhonored "
          f"bound, {len(spillers)} that spill at all:")
    for name, e in spillers:
        print(f"    {e['spill_stores']:4d} st / {e['spill_loads']:4d} ld  "
              f"{e['regs']:3d} regs  {name}")
    print(f"\n{'PASS' if ok else 'FAIL'}  launch-bound spill manifest")
    return ok


#: --------------------------------------------------------------------------
#: THE HOT-LOOP LOCAL-ACCESS GATE (session 9, R1 enforcement).
#:
#: A spill's COST is where it lands, not how much of it there is: the `base[D]`
#: prologue array spilling once per CTA is free, the same bytes inside the
#: 2,000-3,000-instruction tile loop are a memory round-trip per iteration. The
#: register-rung scoping mechanized that as "no
#: `LDL`/`STL` at loop depth >= 1", and this gate is that test as a BUILD FAILURE.
#:
#: **IT IS A RATCHET, NOT AN ABSOLUTE, and the difference was MEASURED before the
#: gate landed (session 9).** The decay arms execute dynamically-indexed LOCAL
#: ARRAYS (`LDL.64 R28, [R26]` -- computed addresses, `ptxas` spill bytes ZERO)
#: inside their loops; that is the "local ARRAYS, not pressure" class,
#: structural to the arm's indexing and unfixable by any register budget (three
#: keys still carry it at 255 registers). An absolute in-loop ban would therefore
#: fail every baseline forever on traffic that is not the defect. At SASS level
#: array traffic and pressure spill are not separable by address form (both reach
#: `[R1+imm]`), so the gate pins the MEASURED per-instantiation in-loop count in
#: the manifest and fails on GROWTH: a budget change that pushes pressure spill
#: into a loop adds in-loop `LDL`/`STL` and is refused; the structural array
#: traffic is ratified as what it is. (the own key changes are held to the
#: stricter reading -- no increase at all on a changed key -- by the same number.)
#:
#: Loop membership is computed from the SASS itself: an instruction is "in a loop"
#: iff it lies inside the interval of a BACKWARD branch (a `BRA` whose target
#: precedes it in the function). That is the same reading the scoping session used.

#: the section headers and the body lines, off the shared grammar (`tools/sass.py`, the one reader of the disassemblers)
_SASS_FUNC = sass.FUNCTION
_SASS_ARCH = sass.ARCH
_SASS_INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
_SASS_LABEL_DEF = re.compile(r"^\s*(\.L_[\w.]+):")
_SASS_LOCAL = re.compile(r"\b(LDL|STL)\b")
_SASS_BRA = re.compile(r"\bBRA\b.*?(?:`?\((\.L_[\w.]+)\)|0x([0-9a-f]+))")

#: THE BODY-HASH LINE RULES, and they are deliberately WIDER than `_SASS_INST`'s.
#: The hot-loop parser wants an instruction it can reason about, so it requires a
#: terminating `;`; the body hash wants EVERY line that carries codegen, which
#: includes the continuation line holding an instruction's second encoding word. Two
#: instructions can print identically and encode differently -- a predicate, a
#: scheduling control field, a reuse flag -- so the encoding words are in the hash
#: and must be. Not in it: `Function :` headers, `.headerflags`, lineinfo, and the
#: padding that aligns the encoding column, which is a property of the LISTING and
#: moves with the longest instruction in it.
#: THE SCANNER'S OWN VERSION. What reaches the digest is a property of
#: :func:`sass_scan`, not of the binary: widening the capture once moved every digest
#: in an unchanged binary, which reads exactly like a codegen change and is not one.
#: Every artifact carrying one of these hashes is stamped with it -- the maps
#: `tools/sass_bodies.py` writes and the per-instantiation `sass_sha256` in every
#: manifest -- and each refuses a comparison across two versions. BUMP IT in the same
#: commit that changes which lines `sass_scan` feeds the digest.
#: 1 = instruction text only (the pre-2026-08-05 capture, which dropped the encoding
#: words); 2 = every offset-marked and encoding line, whitespace-normalised.
SASS_HASH_VERSION = 2

_SASS_BODY_INST = re.compile(r"/\*[0-9a-f]{4,}\*/")
_SASS_BODY_ENC = sass.ENCODING


def _hot_locals_in_function(items: list) -> list:
    """`items` = [(pos, kind, payload)] for one function, in order; kinds are
    'label' (payload: name), 'inst' (payload: (addr, text)). Returns the hot
    local accesses: [(addr, opcode_text)] that sit inside a backward-BRA interval."""
    labels: dict[str, int] = {}
    addr_pos: dict[int, int] = {}
    for pos, kind, payload in items:
        if kind == "label":
            labels[payload] = pos
        else:
            addr_pos.setdefault(payload[0], pos)
    intervals = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        m = _SASS_BRA.search(text)
        if not m:
            continue
        target = labels.get(m.group(1)) if m.group(1) else \
            addr_pos.get(int(m.group(2), 16))
        if target is not None and target < pos:
            intervals.append((target, pos))
    if not intervals:
        return []
    hot = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        if _SASS_LOCAL.search(text) and any(a <= pos <= b for a, b in intervals):
            hot.append((addr, text.strip()))
    return hot


def sass_scan(sass_lines, wanted: set | None = None) -> dict:
    """ONE pass over `cuobjdump -sass`, answering every question asked of the SASS.

    `{key -> {"body_sha256", "hot_locals": [(addr, inst)], "local_ops": int}}`:

    * `body_sha256` is the assembled body's identity, and is what "the same code"
      means in this repository -- names are the weaker check twice over, since a
      mangling can move while the code is identical (a translation-unit restructure
      does exactly that to anonymous-namespace kernels) and the code can move while
      every name is identical.
    * `hot_locals` is the in-loop local traffic the ratchet bounds.
    * `local_ops` is EVERY `LDL`/`STL`, in a loop or not -- a seconds-cheap presence
      count that catches local traffic the byte-level spill report does not (a
      dynamically indexed local array is not a spill) and that the in-loop count does
      not see outside a backward-branch interval.

    `wanted` filters; `None` scans every entry point, which is what a whole-binary
    comparison needs. A name appearing twice within one arch with DIFFERENT bodies
    is refused rather than resolved last-writer-wins: it means the kernel is
    duplicated across objects -- the one-instantiation-per-TU rule broken -- and
    hashing one of the two copies would hide it behind a green comparison.
    """
    report: dict[str, dict] = {}
    arch, key, items, pos, digest, locals_seen = None, None, [], 0, None, 0

    def flush():
        if key is None or (wanted is not None and key not in wanted):
            return
        entry = {"body_sha256": digest.hexdigest(),
                 "hot_locals": _hot_locals_in_function(items),
                 "local_ops": locals_seen}
        if key in report and report[key]["body_sha256"] != entry["body_sha256"]:
            raise SystemExit(f"{key} is disassembled twice with DIFFERENT bodies; "
                             f"the same kernel is emitted by two objects")
        report[key] = entry

    for line in sass_lines:
        m = _SASS_ARCH.match(line)
        if m:
            flush()
            arch, key, items = f"sm_{m.group(1)}", None, []
            continue
        m = _SASS_FUNC.match(line)
        if m:
            flush()
            key, items, pos = f"{arch}:{m.group(1)}", [], 0
            digest, locals_seen = hashlib.sha256(), 0
            continue
        if key is None:
            continue
        if _SASS_BODY_INST.search(line) or _SASS_BODY_ENC.match(line):
            digest.update(" ".join(line.split()).encode())
            digest.update(b"\n")
        m = _SASS_LABEL_DEF.match(line)
        if m:
            items.append((pos, "label", m.group(1)))
            pos += 1
            continue
        m = _SASS_INST.search(line)
        if m:
            items.append((pos, "inst", (int(m.group(1), 16), m.group(2))))
            pos += 1
            if _SASS_LOCAL.search(m.group(2)):
                locals_seen += 1
    flush()
    return report


def check_hot_loop(hot: dict[str, int], want: dict) -> bool:
    """The gate proper: any instantiation whose in-loop local-access count GREW
    past its ratified count is a build failure (the ratchet)."""
    grew = [(n, want[n].get("hot_locals", 0), c) for n, c in sorted(hot.items())
            if n in want and c > want[n].get("hot_locals", 0)]
    if grew:
        print(f"\nFAIL  HOT-LOOP GROWTH: {len(grew)} instantiation(s) execute MORE "
              f"local accesses INSIDE a loop than were ratified. In-loop spill is "
              f"a memory round-trip per iteration and is disqualifying whatever "
              f"the byte count; lower the pressure or raise that key's budget:")
        for name, exp, got in grew[:10]:
            print(f"        ratified {exp} -> realized {got}  {name}")
        if len(grew) > 10:
            print(f"        ... and {len(grew) - 10} more")
        return False
    total = sum(hot.values())
    print(f"hot-loop gate: in-loop local accesses within ratified counts on all "
          f"{len(hot)} entries (tree total {total}); PASS")
    return True


def check_local_presence(local_ops: dict[str, int], want: dict) -> bool:
    """The CHEAP RATCHET: `LDL`/`STL` PRESENCE, in a loop or not.

    A complement to the two expensive gates rather than a replacement for either.
    `ptxas`'s spill bytes miss local traffic that is not a spill -- a dynamically
    indexed local array is the case that matters here -- and the in-loop count only
    sees what sits inside a backward-branch interval. Counting every local access is
    a single integer per instantiation, read off the disassembly the other gates
    already produced, and it moves whenever either of them would.

    A ratchet, not a ban: the decay arms carry structural local traffic that no
    register budget removes, so each entry is held AT its ratified count.
    """
    missing = sorted(n for n in local_ops if n in want and "local_ops" not in want[n])
    if missing:
        print(f"\nFAIL  {len(missing)} manifest entr(ies) predate the local-presence "
              f"ratchet (no `local_ops`). An unarmed ratchet is not a looser gate, "
              f"it is no gate; re-ratify with --write, e.g.\n"
              f"        {missing[0]}")
        return False
    grew = [(n, want[n]["local_ops"], c) for n, c in sorted(local_ops.items())
            if n in want and c > want[n]["local_ops"]]
    if grew:
        print(f"\nFAIL  LOCAL-MEMORY GROWTH: {len(grew)} instantiation(s) execute "
              f"MORE local accesses than were ratified. Local traffic is a memory "
              f"round-trip the register file was supposed to absorb:")
        for name, exp, got in grew[:10]:
            print(f"        ratified {exp} -> realized {got}  {name}")
        if len(grew) > 10:
            print(f"        ... and {len(grew) - 10} more")
        return False
    print(f"local-presence gate: LDL/STL counts within ratified values on all "
          f"{len(local_ops)} entries (tree total {sum(local_ops.values())}); PASS")
    return True


def check_sass_bodies(bodies: dict[str, str], want: dict, fresh: bool,
                      stamped_version) -> bool:
    """THE INCREMENTAL RATCHET: per-instantiation codegen identity.

    The source digest is a WHOLESALE trigger -- any edit under `csrc/rola/src` moves
    it and invalidates every one of the ratified entries at once -- and that is
    correct as a trigger and useless as a diagnosis. The body hash is the per-entry
    answer to the question the digest can only ask globally: which instantiations'
    CODE actually moved.

    So the check reads two ways, and both are hard:

    * `fresh` (the digest matches): every ratified body must still hash the same. The
      digest says the inputs did not move; this says the OUTPUT did not either, which
      is an independent statement and the one the spill numbers depend on.
    * not `fresh` (the digest moved): the numbers must be re-measured, but the entries
      whose bodies are unchanged are provably still described by their ratified
      numbers. Naming the moved ones turns "re-ratify" into a scoped, checkable act
      instead of a 1,216-entry re-derivation.
    """
    if stamped_version != SASS_HASH_VERSION:
        print(f"\nFAIL  SASS HASH VERSION. The ratified bodies were hashed by scanner "
              f"version {stamped_version!r}; this tool is version "
              f"{SASS_HASH_VERSION}. Two versions hash DIFFERENT text, so their "
              f"digests disagree on every entry while describing identical code -- "
              f"the same false verdict the path trap produces. Re-ratify with --write.")
        return False
    missing = sorted(n for n in bodies if n in want and "sass_sha256" not in want[n])
    if missing:
        print(f"\nFAIL  {len(missing)} manifest entr(ies) predate the SASS-body "
              f"ratchet (no `sass_sha256`), so codegen identity is unratified and "
              f"unfalsifiable. Re-ratify with --write, e.g.\n        {missing[0]}")
        return False
    moved = sorted(n for n, h in bodies.items()
                   if n in want and h != want[n]["sass_sha256"])
    if not moved:
        if fresh:
            print(f"SASS-body gate: all {len(bodies)} ratified bodies unchanged; PASS")
            return True
        print("\nFAIL  the source digest moved but NO ratified body did. The "
              "manifest still describes this codegen exactly; re-ratify with "
              "--write to record the new source digest.")
        return False
    print(f"\nFAIL  SASS BODY DRIFT: {len(moved)} of {len(bodies)} ratified "
          f"instantiation(s) assemble to different code than was ratified"
          + ("" if fresh else " (the source digest moved too, so this is the SCOPE "
                             "of that move)") + ":")
    for name in moved[:10]:
        print(f"        {name}")
    if len(moved) > 10:
        print(f"        ... and {len(moved) - 10} more")
    if fresh:
        print("      The source digest did NOT move, so this is codegen changing "
              "under a source tree that did not: a toolchain or environment "
              "difference the digest cannot see.")
    return False


#: --------------------------------------------------------------------------
#: THE CENSUS (session 9, R0) -- the permanent instrument the register rung is
#: derived from. `--census` compiles every built arm at the shipped derived bound
#: and prints the per-key register/spill table. It never touches a manifest:
#: ratification is the recording of the SHIPPED bound, and conflating it with a
#: measurement is how a stale budget constant survived three kernel rungs.

def _regs_ctas(regs: int) -> int:
    """Register-file-side CTAs/SM at 256 threads after sm_86's 8-register
    allocation granularity. The REGISTER term only -- the driver's own occupancy
    query stays the sole residency authority."""
    alloc = 256 * ((regs + 7) // 8) * 8
    return min(6, 65536 // alloc) if alloc else 6


def census_report(entries: dict, unhonored: list[str],
                  hot: dict[str, int] | None = None) -> None:
    hot = hot or {}
    shard = shard_field(entries)
    by_key: dict[tuple, list] = {}
    for name, e in sorted(entries.items()):
        by_key.setdefault((name.split(":", 1)[0], shard[name]), []).append((e, hot.get(name, 0)))
    print(f"\n{'arch':6} {'shard':24} {'n':>4} {'regs':>9} {'ctas(reg)':>9} "
          f"{'st':>6} {'ld':>6} {'#spill':>6} {'worst st':>8} {'hot':>5}")
    for k, es in sorted(by_key.items()):
        regs = sorted(e["regs"] for e, _ in es)
        ctas = sorted(_regs_ctas(r) for r in regs)
        st = sum(e["spill_stores"] for e, _ in es)
        ld = sum(e["spill_loads"] for e, _ in es)
        nsp = sum(1 for e, _ in es if e["spill_stores"] or e["spill_loads"])
        worst = max(e["spill_stores"] for e, _ in es)
        nhot = sum(h for _, h in es)
        rg = f"{regs[0]}-{regs[-1]}" if regs[0] != regs[-1] else f"{regs[0]}"
        cg = f"{ctas[0]}-{ctas[-1]}" if ctas[0] != ctas[-1] else f"{ctas[0]}"
        print(f"{k[0]:6} {k[1]:24} {len(es):>4} {rg:>9} {cg:>9} {st:>6} "
              f"{ld:>6} {nsp:>6} {worst:>8} {nhot:>5}")
    total_st = sum(e["spill_stores"] for e in entries.values())
    total_ld = sum(e["spill_loads"] for e in entries.values())
    zero = sum(1 for e in entries.values()
               if not e["spill_stores"] and not e["spill_loads"])
    print(f"\nTOTAL {len(entries)} entries: {total_st} spill st / {total_ld} "
          f"spill ld, {zero} zero-spill, register sum "
          f"{sum(e['regs'] for e in entries.values())}, "
          f"{sum(hot.values())} in-loop local accesses, "
          f"{len(unhonored)} unhonored bound(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="RE-RATIFY: record what this compile reports as the manifest")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the gate can fail, by re-checking against a "
                         "deliberately tightened manifest")
    ap.add_argument("--arch", action="append", metavar="XX",
                    help="a compute capability to measure and ratify, repeatable "
                         f"(default: {' '.join(DEFAULT_ARCHES)}). One manifest file "
                         "per arch, so bringing up a new one touches no other arch's "
                         "measurements.")
    ap.add_argument("--write-derivation", action="store_true",
                    help="RE-RATIFY THE DERIVATION ONLY: re-measure the addressing "
                         "block's golden outputs and file "
                         "tools/manifests/derivation.json. Its own mode because it "
                         "compiles one host harness and no kernel: the derivation is "
                         "arch-independent, so it is neither measured per arch nor "
                         "worth a whole codegen ratification to refresh.")
    ap.add_argument("--census", action="store_true",
                    help="FORCING CENSUS: compile the whole matrix, print the "
                         "per-key register/spill table, touch no manifest")
    ap.add_argument("--json", metavar="PATH",
                    help="census only: dump the raw per-instantiation record for "
                         "arm-vs-arm diffing")
    ap.add_argument("--shard", action="append", metavar="NAME",
                    help="MEASUREMENT SCOPE: compile only these instantiation shards "
                         "(repeatable). A ratchet refusal is investigated one shard "
                         "at a time; --write still refuses a partial matrix.")
    ap.add_argument("--arm", action="append", type=int, metavar="N",
                    help="MEASUREMENT SCOPE: measure only these carry arms by index "
                         "(repeatable). Same rule as --shard: --write refuses it, "
                         "because a manifest written from a subset of the arms would "
                         "silently drop every one it did not compile.")
    ap.add_argument("--graph", type=Path, metavar="CENSUS",
                    help="READ THE GRAPH'S BUILD instead of compiling: its census record (bazel/defs.bzl's ratify_census) names each "
                         "unit's ptxas report and object, the graph's toolkit and derivation harness, and the build's "
                         "parameters, which are the scope (no --arch, --shard or --arm)")
    ap.add_argument("--runfiles", type=Path, metavar="DIR",
                    help="--graph: the directory the census record's paths are relative to")
    a = ap.parse_args()

    if a.graph:
        if a.arch or a.shard or a.arm:
            raise SystemExit("--graph measures the build's own scope, its parameters (--//:archs, --//:carry_arms, "
                             "...): --arch, --shard and --arm scope a compile of this gate's")
        if a.runfiles is None:
            raise SystemExit("--graph needs --runfiles, the directory its census record's paths are relative to")
        use_graph(a.graph, a.runfiles)
        graph_archs, iteration = graph_parameters()
        #: an iteration build is never ratified (docs/build-graph.md#compile); its census is a measurement scope
        if iteration and not a.census:
            raise SystemExit("this is an iteration build (its parameters are not the shipped set): it is never "
                             "ratified. Build //:library with its default parameters, or take --census")

    if a.write_derivation:
        if a.write or a.census or a.self_test:
            raise SystemExit("--write-derivation is its own mode")
        return write_derivation()

    arches = graph_archs if a.graph else tuple(a.arch) if a.arch else DEFAULT_ARCHES
    if (a.shard or a.arm) and a.write:
        raise SystemExit("--shard/--arm are MEASUREMENT scopes, not ratification "
                         "scopes: a manifest written from a subset of the translation "
                         "units would silently drop every instantiation it did not "
                         "compile")
    arms = sorted(set(a.arm)) if a.arm else None
    if arms is not None:
        unknown = [i for i in arms if not 0 <= i < len(gen_shards.CARRY_ARMS)]
        if unknown:
            raise SystemExit(f"no such carry arm: {unknown}; the list has "
                             f"{len(gen_shards.CARRY_ARMS)} rows (0.."
                             f"{len(gen_shards.CARRY_ARMS) - 1})")
    sources = [] if a.graph else tu_paths(tuple(a.shard) if a.shard else None, arms)

    #: the linker of the measured library, the graph's mold; the compile path links nothing
    mold_info = ({"version": mold_toolchain.mold_version(_graph_bin("mold")),
                  "sha256": mold_toolchain.binary_sha256(_graph_bin("mold"))} if a.graph else None)
    version = ptxas_version()
    #: every SASS body below is read with the record's disassemblers, or it is not the ratification's: checked before
    #: the first read, so `--write` and `--census` record nothing an unchecked reader produced
    toolchains.check_readers(toolchain(), {tool: toolchains.version_text(sass.tool(tool)) for tool in toolchains.READERS})
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rola_ratify_") as tmp:
        entries, unhonored, scan = measure_built() if a.graph else measure_units(arches, tmp, sources, arms)
        hot = {name: len(s["hot_locals"]) for name, s in scan.items()}
        bodies = {name: s["body_sha256"] for name, s in scan.items()}
        local_ops = {name: s["local_ops"] for name, s in scan.items()}
        for name, s in scan.items():
            entries[name]["hot_locals"] = len(s["hot_locals"])
            entries[name]["local_ops"] = s["local_ops"]
            entries[name]["sass_sha256"] = s["body_sha256"]
        if a.census:
            census_report(entries, unhonored, hot)
            if a.json:
                Path(a.json).write_text(json.dumps(
                    dict(ptxas=version,
                         csrc_sha256=csrc_digest(), entries=entries,
                         shards=shard_field(entries)),
                    indent=1, sort_keys=True) + "\n")
                print(f"census written to {a.json}")
            return 0
    if a.write:
        #: THE MANIFEST IS ONE SHARED ARTIFACT PER ARCH:
        #: two concurrent `--write` ratifications of the same arch would
        #: interleave their `path.write_text` calls otherwise.
        with host.file_lock("ratify_manifest"):
            #: WRITTEN PER ARCH, from the one compile. Splitting here rather than at
            #: read time is what keeps `sm_86.json` untouched by an `--arch 80` run:
            #: a bring-up must not restate measurements it did not take.
            for arch in arches:
                prefix = f"sm_{arch}:"
                mine = {k: v for k, v in sorted(entries.items()) if k.startswith(prefix)}
                if not mine:
                    raise SystemExit(
                        f"the compile reported NO entries for sm_{arch}; refusing to "
                        f"write an empty ratification. A manifest that exists but "
                        f"measures nothing would pass rule 3 by file existence alone.")
                path = manifest_path(arch)
                path.parent.mkdir(parents=True, exist_ok=True)
                #: The SOURCE DIGEST is recorded beside the toolchain: a manifest is a
                #: measurement OF a source tree UNDER an assembler, and both halves of
                #: that sentence must be checkable (the freshness trigger, session 9).
                path.write_text(json.dumps(
                    {"toolchain": {"ptxas": version, "defines": shipped_defines(),
                                    #: THE CODEGEN FLAG LIST, arch-independent part.
                                    #: A ratification is a measurement OF sources UNDER
                                    #: an assembler AND its flags; without this the
                                    #: third of those was unrecorded and a
                                    #: codegen-affecting flag could land without moving
                                    #: anything a gate reads.
                                    "nvcc_flags": ratified_flags(),
                                    #: The scanner version behind every entry's
                                    #: `sass_sha256`. Same argument as the flag list:
                                    #: a record that cannot say what produced its
                                    #: hashes cannot certify them.
                                    "sass_hash_version": SASS_HASH_VERSION,
                                    #: THE LINKER of the measured library, for
                                    #: PROVENANCE: a linker does not touch codegen,
                                    #: so its version is recorded, never compared
                                    #: against a SASS body. `None` on the compile
                                    #: path, which links nothing.
                                    "mold": {"version": mold_info["version"],
                                             "sha256": mold_info["sha256"]}
                                            if mold_info else None},
                     "source": {"csrc_sha256": csrc_digest(),
                                "date": time.strftime("%Y-%m-%d"),
                                #: where it was measured: the devbox image's digest, empty on
                                #: a host; recorded, never hashed -- see
                                #: docs/setup.md#reproducibility-as-a-statement
                                "image_digest": dev_config.devbox_image()},
                     #: WHICH INSTANTIATIONS THIS RATIFICATION COVERS, by membership
                     #: hash rather than by file list. It enters the digest, so a
                     #: re-partition is a visible re-ratification and a scoped
                     #: re-measurement is a provable statement.
                     "shards": gen_shards.shard_manifest_block(),
                     "entries": mine}, indent=1) + "\n")
                print(f"wrote {path} ({len(mine)} instantiations) under ptxas: {version}")
            print(f"{len(unhonored)} unhonored bounds across {len(arches)} arch(es)")
        #: OUTSIDE the manifest lock and outside the per-arch loop: the derivation is
        #: host code, measured once and filed once, and a re-ratification that
        #: refreshed the codegen numbers while leaving the derivation's goldens at
        #: their old values would file a record of two different trees.
        write_derivation()
        return 0
    manifest = load_manifest(arches)
    want = manifest["entries"]
    print(f"manifest sha256 {manifest_digest(manifest)}  "
          f"({len(want)} entries over {', '.join('sm_' + x for x in arches)})")
    #: THE TOOLCHAIN CHECK RUNS FIRST AND ITS RESULT IS ANDed IN, rather than
    #: short-circuiting: on a version bump the entry-level report is still the most
    #: useful thing to print (it shows WHAT the new assembler did), and printing it
    #: costs nothing once the compile has already happened. What it must not do is
    #: turn a green entry comparison into a green gate.
    ok = check_toolchain(version, manifest["toolchain"]["ptxas"])
    ok = check_defines(manifest) and ok
    ok = check_flags(manifest) and ok
    #: The freshness verdict is REUSED rather than recomputed: the body gate reads
    #: differently on each side of it, and two independent computations of "is the
    #: source digest current" could disagree.
    fresh = check_freshness(manifest)
    ok = fresh and ok
    ok = check_derivation() and ok
    ok = check(entries, unhonored, want) and ok
    ok = check_hot_loop(hot, want) and ok
    ok = check_local_presence(local_ops, want) and ok
    ok = check_sass_bodies(bodies, want, fresh,
                           manifest["toolchain"].get("sass_hash_version")) and ok

    if a.self_test:
        #: NON-VACUITY. Tighten one entry that actually spills (or, if nothing
        #: spills, any entry) and require the gate to FAIL on it. A gate whose green
        #: run is unfalsifiable says nothing, and this batch has already recorded two
        #: gates that failed themselves as vacuous before they were fixed.
        target = next((n for n, e in sorted(want.items())
                       if e["spill_stores"] or e["spill_loads"]), None)
        target = target or sorted(want)[0]
        tightened = json.loads(json.dumps(want))
        tightened[target]["spill_stores"] = -1
        tightened[target]["spill_loads"] = -1
        print(f"\n--- SELF-TEST: the same result against a manifest tightened at\n"
              f"    {target}\n    (this run MUST fail) ---")
        if check(entries, unhonored, tightened):
            print("\nSELF-TEST FAILED: the gate PASSED a manifest it must reject")
            return 1
        print("\nSELF-TEST PASSED: the gate rejects a spill it was not ratified for")

        #: NON-VACUITY OF RULE 1, the same way. The version string is perturbed rather
        #: than a second `ptxas` installed -- the property under test is the COMPARISON,
        #: and a comparison that cannot fail on a string it was not given is exactly the
        #: vacuous gate this discipline exists to catch.
        perturbed = manifest["toolchain"]["ptxas"] + " (perturbed by --self-test)"
        print("\n--- SELF-TEST: the real ptxas version against a perturbed manifest\n"
              "    entry (this run MUST fail) ---")
        if check_toolchain(version, perturbed):
            print("\nSELF-TEST FAILED: the gate ACCEPTED a toolchain it was not "
                  "ratified under")
            return 1
        print("\nSELF-TEST PASSED: the gate rejects an assembler it was not ratified "
              "under")

        #: NON-VACUITY OF THE PER-ARCH SPLIT. Rule 3 is now a FILE-EXISTENCE
        #: question, and a question whose answer is always yes is not a gate: an arch
        #: nobody ratified must RAISE, not merge into an empty entry set that then
        #: passes every comparison below it vacuously.
        print("\n--- SELF-TEST: an arch with no committed manifest\n"
              "    (this lookup MUST fail) ---")
        try:
            load_manifest(("999",))
        except SystemExit as exc:
            print(f"SELF-TEST PASSED: {str(exc).splitlines()[0]}")
        else:
            print("\nSELF-TEST FAILED: an UNRATIFIED arch loaded as though it were "
                  "ratified; rule 3 would pass on a manifest that does not exist")
            return 1

        #: NON-VACUITY OF THE SPLIT'S CLAIM TO BE NON-SEMANTIC. The digest is over
        #: the merged canonical record, so two files must hash to what one file hashed
        #: to. If the filing could move the digest, `_build_config.py`'s
        #: `manifest_sha256` would stop identifying the ratification and start
        #: identifying a directory listing.
        halves = [load_manifest((x,)) for x in arches]
        rejoined = {"toolchain": halves[0]["toolchain"], "shards": halves[0]["shards"],
                    "entries": {k: v for h in halves for k, v in h["entries"].items()}}
        if manifest_digest(rejoined) != manifest_digest(manifest):
            print("\nSELF-TEST FAILED: the per-arch files do not rejoin to the same "
                  "record; the split is not filing-only")
            return 1
        print(f"SELF-TEST PASSED: {len(arches)} per-arch files rejoin to one digest "
              f"{manifest_digest(rejoined)[:16]}...")

        #: ... AND THE SHARD BLOCK DOES NOT MAKE THE DIGEST A FILING PROPERTY. It is
        #: membership only, so re-filing the same measurements -- reordering the
        #: block's keys, or splitting the arches differently -- must move nothing. A
        #: block that carried paths or an ordering would fail exactly here, which is
        #: why it carries neither.
        refiled = {"toolchain": manifest["toolchain"],
                   "shards": dict(reversed(list(manifest["shards"].items()))),
                   "entries": manifest["entries"]}
        if manifest_digest(refiled) != manifest_digest(manifest):
            print("\nSELF-TEST FAILED: re-filing the shard block moved the manifest "
                  "digest; the block is recording HOW the ratification is filed, not "
                  "WHAT it covers")
            return 1
        #: ... and it must still be LOAD-BEARING: a changed membership must move it.
        moved = {"toolchain": manifest["toolchain"], "entries": manifest["entries"],
                 "shards": {k: dict(v, count=v["count"] + 1)
                            for k, v in manifest["shards"].items()}}
        if manifest_digest(moved) == manifest_digest(manifest):
            print("\nSELF-TEST FAILED: a changed shard membership did not move the "
                  "manifest digest; the block is decorative")
            return 1
        print("SELF-TEST PASSED: the shard block is membership-only (re-filing is "
              "digest-neutral) and load-bearing (a membership change is not)")

        #: NON-VACUITY OF THE CROSS-ARCH AGREEMENT. Two arch files that disagree
        #: about which instantiations exist are not one ratification.
        print("\n--- SELF-TEST: per-arch manifests disagreeing on shard membership\n"
              "    (this load MUST fail) ---")
        real = manifest_path(arches[-1])
        saved = real.read_text()
        blob = json.loads(saved)
        blob["shards"] = dict(blob.get("shards") or {}, __self_test__={"count": 1})
        try:
            real.write_text(json.dumps(blob, indent=1) + "\n")
            load_manifest(arches)
        except SystemExit as exc:
            print(f"SELF-TEST PASSED: {str(exc).splitlines()[0]}")
        else:
            print("\nSELF-TEST FAILED: two manifests with different shard "
                  "memberships loaded as one ratification")
            return 1
        finally:
            real.write_text(saved)

        #: NON-VACUITY OF THE FRESHNESS TRIGGER. A perturbed source digest must
        #: fail: a freshness gate that passes a tree it was not measured on is the
        #: stale-constant trap re-armed.
        print("\n--- SELF-TEST: the manifest against a perturbed source digest\n"
              "    (this run MUST fail) ---")
        if check_freshness(manifest, digest="0" * 64):
            print("\nSELF-TEST FAILED: the gate ACCEPTED a source tree the "
                  "manifest was not measured on")
            return 1
        print("SELF-TEST PASSED: the gate rejects a moved source tree")

        #: NON-VACUITY OF THE HOT-LOOP GATE, against synthetic SASS: one LDL
        #: inside a backward-branch interval MUST be flagged, and the same LDL
        #: outside every loop MUST NOT be.
        hot_sass = [
            "arch = sm_86",
            "                Function : k_hot",
            "        /*0000*/ MOV R1, c[0x0][0x28] ;",
            ".L_x_0:",
            "        /*0010*/ LDL R2, [R1] ;",
            "        /*0020*/ @P0 BRA `(.L_x_0) ;",
            "                Function : k_cold",
            "        /*0000*/ LDL R2, [R1] ;",
            ".L_x_1:",
            "        /*0010*/ IADD3 R3, R3, 1, RZ ;",
            "        /*0020*/ @P0 BRA `(.L_x_1) ;",
        ]
        rep = sass_scan(iter(hot_sass), {"sm_86:k_hot", "sm_86:k_cold"})
        print("\n--- SELF-TEST: the hot-loop parser on synthetic SASS ---")
        counts = {k: len(v["hot_locals"]) for k, v in rep.items()}
        if counts != {"sm_86:k_hot": 1, "sm_86:k_cold": 0}:
            print(f"\nSELF-TEST FAILED: hot-loop parser reported {counts}; it "
                  f"must count exactly the in-loop LDL (k_hot=1, k_cold=0)")
            return 1
        print("SELF-TEST PASSED: the parser counts the in-loop LDL and clears the "
              "cold one")
        #: ... and the RATCHET must fail on growth past a ratified count.
        print("\n--- SELF-TEST: hot-loop growth against a ratified count of 0\n"
              "    (this run MUST fail) ---")
        if check_hot_loop({"sm_86:k_hot": 1}, {"sm_86:k_hot": {"hot_locals": 0}}):
            print("\nSELF-TEST FAILED: the ratchet PASSED an in-loop count it was "
                  "not ratified for")
            return 1
        print("SELF-TEST PASSED: the ratchet rejects in-loop growth")

        #: NON-VACUITY OF THE LOCAL-PRESENCE RATCHET, on the same synthetic SASS.
        #: `k_cold`'s `LDL` sits outside every loop, so the hot-loop gate clears it
        #: and this one must NOT -- which is the whole reason the cheap count is
        #: kept beside the expensive one.
        print("\n--- SELF-TEST: local-presence growth on an out-of-loop LDL\n"
              "    (this run MUST fail) ---")
        assert rep["sm_86:k_cold"]["local_ops"] == 1
        if check_local_presence({"sm_86:k_cold": 1},
                                {"sm_86:k_cold": {"local_ops": 0}}):
            print("\nSELF-TEST FAILED: the presence ratchet PASSED local traffic "
                  "it was not ratified for")
            return 1
        print("SELF-TEST PASSED: the presence ratchet catches a local access the "
              "in-loop gate is right to ignore")

        #: NON-VACUITY OF THE BODY HASH, both directions. Two functions whose
        #: instruction text differs must hash differently, or the comparison is
        #: decorative; the SAME text laid out with different column padding must hash
        #: identically, or every rebuild reads as a codegen change and the gate gets
        #: turned off.
        print("\n--- SELF-TEST: the SASS body hash on synthetic SASS ---")
        moved_sass = [
            "arch = sm_86",
            "                Function : k_a",
            "        /*0000*/ IADD3 R3, R3, 1, RZ ;",
            "                                     /* 0x000fe200078e0203 */",
            "                Function : k_b",
            "        /*0000*/ IADD3 R3, R3, 2, RZ ;",
            "                                     /* 0x000fe200078e0203 */",
            "                Function : k_a_repadded",
            "   /*0000*/     IADD3   R3,  R3, 1, RZ ;",
            "     /* 0x000fe200078e0203 */",
        ]
        got = sass_scan(iter(moved_sass))
        a, b = got["sm_86:k_a"]["body_sha256"], got["sm_86:k_b"]["body_sha256"]
        repad = got["sm_86:k_a_repadded"]["body_sha256"]
        if a == b:
            print("\nSELF-TEST FAILED: two different instruction texts hashed the "
                  "same; the body hash cannot detect a codegen change")
            return 1
        if a != repad:
            print("\nSELF-TEST FAILED: re-padding the listing moved the body hash; "
                  "the hash is over the LISTING's column alignment, not the code")
            return 1
        print("SELF-TEST PASSED: the body hash separates changed code and ignores "
              "listing padding")

        #: ... and the GATE built on it must fail on a perturbed ratified hash.
        print("\n--- SELF-TEST: a ratified body hash that no longer matches\n"
              "    (this run MUST fail) ---")
        if check_sass_bodies({"sm_86:k_a": a},
                             {"sm_86:k_a": {"sass_sha256": "0" * 64}}, True,
                             SASS_HASH_VERSION):
            print("\nSELF-TEST FAILED: the gate ACCEPTED an instantiation whose "
                  "assembled body is not the one that was ratified")
            return 1
        print("SELF-TEST PASSED: the gate rejects codegen it did not ratify")

        #: ... and an UNARMED entry is a failure, not a pass. A ratchet with no
        #: recorded value is the vacuous gate this discipline exists to catch.
        print("\n--- SELF-TEST: a manifest entry with no ratified body hash\n"
              "    (this run MUST fail) ---")
        if check_sass_bodies({"sm_86:k_a": a}, {"sm_86:k_a": {}}, True,
                             SASS_HASH_VERSION):
            print("\nSELF-TEST FAILED: an unratified body hash passed as though it "
                  "had been checked")
            return 1
        print("SELF-TEST PASSED: an unarmed body ratchet fails rather than passes")

        #: ... and a body hash produced by a DIFFERENT scanner version is refused
        #: whatever it says. This is the map-stamp argument applied to the
        #: manifest: two versions hash different text, so an unstamped or
        #: differently-stamped ratification disagrees on every entry while
        #: describing identical code.
        print("\n--- SELF-TEST: ratified bodies hashed by another scanner version\n"
              "    (both of these MUST fail) ---")
        for stamped in (SASS_HASH_VERSION - 1, None):
            if check_sass_bodies({"sm_86:k_a": a},
                                 {"sm_86:k_a": {"sass_sha256": a}}, True, stamped):
                print(f"\nSELF-TEST FAILED: bodies stamped {stamped!r} were compared "
                      f"against version {SASS_HASH_VERSION} digests")
                return 1
        print("SELF-TEST PASSED: the gate rejects hashes it did not produce")

        #: THE DERIVATION RECORD, mutated one field at a time. The mutants are on the
        #: RECORD rather than on the sources, so the harness is measured once and each
        #: arm isolates exactly one of the record's claims -- and a mutant that only
        #: moved a hash would not prove the golden comparison runs, which is why one
        #: arm moves a golden with the hashes left intact.
        print("\n--- SELF-TEST: the derivation record, mutated (all of these MUST "
              "fail) ---")
        base = json.loads(DERIVATION_JSON.read_text())
        a_case = sorted(base["golden"])[0]
        a_source = sorted(base["sources"])[0]
        mutants = {
            "a moved golden output": lambda b: b["golden"].__setitem__(
                a_case, "ok: 4 bytes, sha256 " + "0" * 64),
            "a perturbed source hash": lambda b: b["sources"].__setitem__(
                a_source, "0" * 64),
            "a perturbed case digest": lambda b: b.__setitem__(
                "cases_sha256", "0" * 64),
            "a dropped golden": lambda b: b["golden"].pop(a_case),
            "a golden for no declared case": lambda b: b["golden"].__setitem__(
                "__self_test__", "ok: 4 bytes, sha256 " + "0" * 64),
            "an unreadable schema": lambda b: b.__setitem__("schema", 99),
        }
        for name, mutate in mutants.items():
            blob = json.loads(json.dumps(base))
            mutate(blob)
            if check_derivation(blob):
                print(f"\nSELF-TEST FAILED: the derivation gate ACCEPTED {name}")
                return 1
        print(f"SELF-TEST PASSED: the derivation gate rejects all "
              f"{len(mutants)} record mutants")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
