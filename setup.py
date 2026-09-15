"""THE BUILD -- ahead-of-time, arch-gated, and refused when the evidence is missing.

**WHAT THIS REPLACES.** In the fork the extension was built by
``torch.utils.cpp_extension.load(...)`` on the first call into the kernel: a JIT
compile at runtime, in the user's process, under whatever toolchain the user's
machine happened to have. The closed-world codegen policy forbids that (rule 2), and
the ratification manifest -- a per-instantiation register/spill measurement taken
under one named ``ptxas`` -- is meaningless against a binary somebody else's compiler
produced. So the build moved here, and the gates moved with it.

**THE FOUR RULES, and where each one lives in this file.**

===== ==================================================================== ==========
rule   what it forbids                                                      here
===== ==================================================================== ==========
R1     a build under an assembler the manifest was not ratified under       ``_gate_toolchain``
R2     ``code=compute_XX`` (PTX) anywhere in the fatbin                     ``gencodes`` + its assertion
R3     an architecture in the fatbin with no ratified manifest              ``_gate_ratified``
R4     a *launch* on a device whose arch was never measured                 ``csrc/rola/src/chunk/arch_caps.cuh`` (C++)
R5     a vendored codegen INPUT that does not match its recorded pin        ``_gate_vendored``
===== ==================================================================== ==========

**R5 AND WHY A HEADER-ONLY LIBRARY IS A CODEGEN INPUT, NOT A PACKAGING DETAIL.**
``csrc/third_party/cutlass`` vendors CUTLASS/CuTe headers. Header-only or not, every
line of it that the kernel instantiates is code ``ptxas`` assembles, so a silent edit
to a vendored header moves the register allocation the manifest ratified exactly as an
edit to ``chunk/chunk_kernel.cuh`` would. ``_gate_vendored`` therefore hashes the whole
vendored subtree and compares it against ``PIN.json``'s recorded digest BEFORE the
compile, in the same pass as rules 1 and 3, and the pin (upstream, tag, commit, digest)
is stamped into ``rola_cu13/_build_config.py`` so an installed wheel can answer "which
CUTLASS is inside this ``.so``?" with no checkout. The vendored subtree is
byte-identical to the upstream tag, which is what makes the digest checkable against
something other than itself.

R1 and R3 run BEFORE ``build_ext`` -- they are cheap, and failing after a multi-hour
compile helps nobody. The manifest gate runs AFTER, because the pre-build check
proves the *inputs* were ratified and only the post-build run proves the *shipped
codegen* still matches what was measured. A source install therefore re-runs the
ratification gates by construction; it is not an optional ``make`` target.

**FLAG FIDELITY (extraction W1).** Every codegen-affecting flag here is the flag the
fork's JIT build passed, and the list is not a redesign. ``bindings.py`` passed
``-O3 -std=c++17 -lineinfo --threads=2`` plus torch's own ``COMMON_NVCC_FLAGS``
(``-D__CUDA_NO_HALF_OPERATORS__``, ``-D__CUDA_NO_HALF_CONVERSIONS__``,
``-D__CUDA_NO_BFLOAT16_CONVERSIONS__``, ``-D__CUDA_NO_HALF2_OPERATORS__``,
``--expt-relaxed-constexpr``) and ``-D_GLIBCXX_USE_CXX11_ABI=<torch's>``; ``CUDAExtension``
adds exactly that same set, so it is reproduced rather than restated. TWO deliberate
deviations, both non-codegen-affecting, both stated so a reader does not have to
diff:

* ``--ptxas-options=-v`` is ADDED. It changes what the assembler PRINTS, not what it
  emits, and ``tools/ratify.py`` parses exactly that output.
* the design's ``-U__CUDA_NO_HALF_OPERATORS__`` / ``-U__CUDA_NO_HALF_CONVERSIONS__``
  / ``-U__CUDA_NO_BFLOAT16_CONVERSIONS__`` / ``--expt-extended-lambda`` are NOT
  adopted. Those would UNDO torch's own defines and enable a language feature the
  measured build did not have. The manifest was ratified without them; adopting them
  would be a codegen change smuggled in as a packaging change.

``--use_fast_math`` is likewise refused: Tier-1's tolerances are derived as fp32
accumulation tolerances against an fp64 oracle, and fast-math silently changes the
numeric contract those tolerances were computed under.
"""
from __future__ import annotations

import hashlib
import json
import os
import pprint
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc" / "rola"
CUTLASS_DIR = ROOT / "csrc" / "third_party" / "cutlass"

#: `tools/` is not an installed package -- it is the repository's instruments -- but
#: the build and the ratification gate must not maintain two copies of the codegen
#: flag list or two copies of the instantiation matrix. Both are imported from there.
sys.path.insert(0, str(ROOT / "tools"))
# These follow the sys.path insert above and cannot be hoisted into the import block
# at the top of the file.
# isort: off
import dev_config  # noqa: E402

#: torch.utils.cpp_extension resolves its toolkit when it is imported, so the dev config's is handed to it first.
if dev_config.get("toolchain.cuda_home"):
    os.environ["CUDA_HOME"] = dev_config.get("toolchain.cuda_home")
#: torch >= 2.10 prefixes any `ccache`/`sccache` on PATH to every compiler it runs; the only cache is the pinned one.
os.environ["TORCH_NO_COMPILER_WRAPPER"] = "1"
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402
import build_flags  # noqa: E402
import build_lock  # noqa: E402
import gen_shards  # noqa: E402
import mold_toolchain  # noqa: E402
import ratify  # noqa: E402
import sccache_toolchain  # noqa: E402
import toolchains  # noqa: E402


# isort: on

#: THE RATIFIED SET. Overridable only downward in practice: adding an arch here
#: without a manifest fails the build at ``_gate_ratified`` (rule 3), which is the
#: point -- the supported matrix IS the ratified set, not a wish list.
def _declared_archs() -> list[str]:
    """The architectures this build compiles, from the environment or the manifests.

    ``ROLA_CUDA_ARCHS`` names them explicitly. Unset, they are read from the ratified
    manifests of this build's toolchain -- the same files rule 3 then gates each
    arch against, so the fatbin's arch list and the ratification's have ONE source.
    An unset variable used to default to a hard-coded pair, which is a second
    declaration of the supported matrix and the reason a build asked for one arch's
    worth of slots and compiled two arches' worth of ``cicc``.
    """
    env = os.getenv("ROLA_CUDA_ARCHS")
    if env is not None:
        return [a for a in env.split(";") if a]
    named = TOOLCHAIN.archs()
    if not named:
        raise RuntimeError(
            f"ROLA_CUDA_ARCHS is unset and toolchain {TOOLCHAIN.name} ratifies no "
            f"architecture ({TOOLCHAIN.manifest_dir}), so this build has no declared arch "
            "list. Name the arches (ROLA_CUDA_ARCHS=86) or ratify one first "
            "(tools/ratify.py --arch 86 --write)."
        )
    return named


#: THE TOOLCHAIN: the declared record naming the configured toolkit's assembler (tools/toolchains.py). A Python-only
#: install compiles nothing and resolves none.
TOOLCHAIN = None if os.getenv("ROLA_NO_EXTENSION") == "1" else toolchains.for_ptxas(ratify.ptxas_version())
ROLA_CUDA_ARCHS = [] if TOOLCHAIN is None else _declared_archs()

#: THE MILESTONE/CI SWITCH (K43, KERNEL_STANDARDS §16). Manifests go stale between
#: milestones BY DESIGN -- ratification is a per-milestone act, not a per-stage one --
#: so the POST-BUILD gate that compares the freshly built binary against the
#: checked-in ``tools/manifests/<toolchain>/sm_XX.json`` (``_post_build_manifest_check``'s
#: ``ratify.py`` re-run) REPORTS a mismatch instead of failing the install by default.
#: ``ROLA_STRICT_MANIFEST=1`` restores the pre-K43 hard failure: the milestone/CI gate
#: build that immediately precedes ``tools/ratify.py --write`` sets it (docs/build.md),
#: because THAT build must fail loudly if the tree it is about to ratify is not
#: actually shippable. This never weakens ``tools/ratify.py`` itself -- ``--write``/
#: ``--check`` invoked directly are untouched, as are every PRE-build gate below
#: (``_gate_ratified``/``_gate_toolchain``/``_gate_vendored``/``_gate_shards``: those
#: are prerequisites, not staleness) -- only whether THIS INSTALL treats a
#: stale-manifest finding as fatal.
ROLA_STRICT_MANIFEST = os.getenv("ROLA_STRICT_MANIFEST") == "1"


#: THE ARM SUBSET IS A FILE SUBSET (KERNEL_STANDARDS section 7, K36). Instantiation
#: cost is the dominant build resource, so an arm the build does not want is an arm
#: whose TRANSLATION UNIT is not in the build at all -- not a macro that still costs a
#: parse. ``ROLA_CARRY_ARMS=11`` (or ``11,12``, or ``all``) names rows of the carry
#: family's arm list by index (`tools/gen_shards.py`'s ``CARRY_ARMS``); each selected
#: row contributes its generated ``carry_arm_<i>.cu``, and the SELECTION -- which of
#: those rows the dispatch's ``CARRY_ARMS_X`` expands to -- is a GENERATED HEADER
#: (K36b, ``_write_carry_selection``), not a compiler flag: it used to be a
#: ``-DROLA_CARRY_BUILD_<i>`` that ``CUDAExtension`` applied to EVERY source's argv,
#: which is why an arm-set change forced ninja to rebuild every translation unit even
#: though only the per-arm TUs and the carry dispatch TU read those macros.
#:
#: THE DEFAULT IS THE SHIPPED ROWS. A row tagged TEST in the arm list is a conformance
#: cell -- the fp64 oracle materializes ``[B,T,H,N]``, so ``N = 4096`` is where a whole
#: cell is checkable, and the ``W = 64`` arms exist so that it is -- and it is built
#: for the battery (``ROLA_CARRY_ARMS=all ROLA_CUDA_ARCHS=86``) and never shipped. A
#: binary is SELF-IDENTIFYING either way: ``rola.ops.carry.arms()`` lists exactly what
#: was built and every entry point refuses an arm the binary does not carry.
#:
#: THE LIST IS EMPTY ON THIS LINE (C0, card C): the carry family's kernels are deleted,
#: so every scope resolves to no arm and no per-arm TU joins the build. The mechanism
#: stays because it is the BUILD's contract; G4 refills the list on the final key.
#: docs/build.md.
def _carry_arms() -> list[int]:
    spec = os.getenv("ROLA_CARRY_ARMS", "").replace(",", " ").split()
    if not spec:
        return list(gen_shards.CARRY_SHIPPED_ARMS)
    n = len(gen_shards.CARRY_ARMS)
    out: list[int] = []
    for tok in spec:
        if tok == "all":
            out += list(range(n))
            continue
        try:
            i = int(tok)
        except ValueError:
            raise SystemExit(f"ROLA_CARRY_ARMS: {tok!r} is not an arm index or 'all'")
        if not 0 <= i < n:
            raise SystemExit(f"ROLA_CARRY_ARMS: no such carry arm {i}; the list has "
                             f"{n} rows (0..{n - 1})")
        out.append(i)
    return sorted(set(out))


def _write_carry_selection() -> None:
    """Write `build/generated/carry_selection.inc` for THIS build's arm scope (K36b).

    Runs BEFORE the compile, next to `_gate_shards()` -- cheap, pre-compile,
    source-of-truth-for-what-gets-built. Only the per-arm TUs and the carry
    dispatch TU ever include the generated header, so
    writing it here (rather than passing `-D` on `nvcc_flags`) is what keeps an
    arm-set change from perturbing every other translation unit's argv.
    """
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    text = gen_shards.carry_selection_header(_carry_arms())
    (gen_shards.GENERATED_DIR / gen_shards.CARRY_SELECTION_INC).write_text(text)


#: THE COVERAGE SWEEP'S PROBE, run in its own interpreter against the freshly built
#: extension LOADED BY PATH.  By path, and not by `import rola`, because the fact
#: wanted here is a property of the artifact this build just produced: importing the
#: package would resolve through whatever is installed in the running environment,
#: which on a developer box is routinely another worktree's build.
_ARM_SWEEP_PROBE = """
import importlib.util, json, sys
from pathlib import Path
import torch  # noqa: F401 -- the extension links against it and must load first
name = Path(sys.argv[1]).name.split(".")[0]  # the module's init symbol is PyInit_<name>
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # loading the library registers torch.ops.rola
print(json.dumps([list(row) for row in torch.ops.rola.carry_arms()]))
"""


def _built_extension(build_lib: str) -> Path | None:
    """The `_C` extension this build produced, found under its own output tree."""
    found = sorted(Path(build_lib).glob(f"{PLUGIN}/_C.*so"))
    return found[0] if found else None


def _arm_coverage_sweep(build_lib: str, want: list[int]) -> int:
    """R9's fourth layer, host side: every built arm answered for by the binary.

    The binary is asked what it carries and the answer is required to be exactly the
    rows this build selected from the declaration. That is what makes `arm_switch`'s
    exhaustiveness a checked property of the shipped artifact rather than of the
    generator alone: the generator, the selection header and the binary are three
    statements of one set, and this is where the third one is read.
    """
    so = _built_extension(build_lib)
    if so is None:
        raise RuntimeError(
            f"no {PLUGIN}/_C.*so under {build_lib} after the build; the arm coverage "
            f"sweep has no artifact to ask, and a sweep that silently asks nothing "
            f"is the vacuous gate this project keeps re-learning about")
    #: ABSOLUTE: a wheel's `build_lib` is relative to this checkout, and the probe runs outside it (so `import rola`
    #: cannot resolve the source tree instead of the build).
    proc = subprocess.run([sys.executable, "-c", _ARM_SWEEP_PROBE, str(so.resolve())],
                          capture_output=True, text=True, cwd=str(ROOT.parent))
    if proc.returncode != 0:
        raise RuntimeError(
            f"the arm coverage sweep could not load {so}:\n{proc.stderr}")
    got = sorted(tuple(row) for row in json.loads(proc.stdout))
    expect = sorted(tuple(gen_shards.CARRY_ARMS[i]) for i in want)
    if got != expect:
        raise RuntimeError(
            f"{so.name} reports the carry arms {[list(r) for r in got]}, this build "
            f"selected {[list(r) for r in expect]}. A binary that cannot name what it "
            f"carries cannot refuse what it does not.")
    return len(got)


def _carry_parts() -> tuple[str, ...]:
    """The carry kernel's REAL parts for this build (the composer's switches): every part
    unless `ROLA_CARRY_PARTS` names a subset -- `all`, or a comma list of `gen_shards.CARRY_PARTS`
    names, or `none`. A subset stubs the rest and makes the build an ITERATION build."""
    spec = os.getenv("ROLA_CARRY_PARTS", "all").strip()
    if spec in ("", "all"):
        return gen_shards.CARRY_PARTS
    if spec == "none":
        return ()
    names = tuple(x for x in spec.replace(" ", "").split(",") if x)
    bad = sorted(set(names) - set(gen_shards.CARRY_PARTS))
    if bad:
        raise SystemExit(f"ROLA_CARRY_PARTS: no such part(s) {bad}; the parts are {gen_shards.CARRY_PARTS}")
    return names


def _write_carry_parts() -> None:
    """Write `build/generated/carry_parts.inc` on EVERY build, so a stubbed mask never
    outlives the build that asked for it."""
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (gen_shards.GENERATED_DIR / gen_shards.CARRY_PARTS_INC).write_text(
        gen_shards.carry_parts_header(_carry_parts()))


def _write_csrc_stamp() -> None:
    """Write `build/generated/csrc_stamp.inc` for THIS tree (the device stamp's key).

    Beside `_write_carry_selection()`, pre-compile, and from `tools/ratify.py`'s own
    function so the stamp this build compiles is the stamp that gate assembles. It is
    a generated HEADER rather than a `-D` for the same reason the arm selection is:
    the define would reach every translation unit's argv, so every `csrc/` edit would
    rebuild the whole extension instead of the stamp sites.
    """
    ratify.write_csrc_stamp()


def _carry_sources() -> list[str]:
    """The per-arm translation units this build compiles, TAKEN FROM THE GENERATOR.

    Two rows that are ONE kernel share a file (the generator's `carry_kernel_key`), so
    this is a set and not a list comprehension over the indices.
    """
    files = gen_shards.carry_sources()
    return sorted({f"csrc/rola/src/instantiations/{files[i]}" for i in _carry_arms()})


def _is_iteration_build() -> bool:
    """A build whose arm set is not the SHIPPED set is an ITERATION build.

    The post-build gate proves SHIPPABILITY -- that the fatbin carries every ratified
    instantiation. A binary built with a different arm set carries a different set BY
    CONSTRUCTION, so that gate would fail for the one reason that is not a defect. It
    is skipped here and nowhere else, behind a banner, and the binary stays
    self-identifying.

    NOTE the asymmetry, and that it is deliberate: a build carrying MORE than the
    shipped rows (the battery's `all`) is an iteration build too. The manifest ratifies
    the shipped set; a binary carrying arms it does not ratify is not the thing that
    ships, whichever direction it differs in.
    """
    return (set(_carry_arms()) != set(gen_shards.CARRY_SHIPPED_ARMS) or _decode_arm_subset_active()
            or set(_carry_parts()) != set(gen_shards.CARRY_PARTS))


#: THE ITERATION ARM SUBSET FOR DECODE (KERNEL_STANDARDS section 7, D1b). Same
#: mechanism as ``_carry_arms``, one family over: ``ROLA_DECODE_ARMS=5`` (or
#: ``5,13``) names rows of the decode family's declared arm list by index
#: (``csrc/rola/src/decode/decode.cu``'s ``DECODE_ARM_<i>``, the ``d_v x D x decay``
#: matrix in that order). The DEFAULT IS EVERY DECLARED ARM, so a plain build -- and
#: every gate build -- is the closed-world one. A subset binary is SELF-IDENTIFYING:
#: ``rola.ops.decode.arms()`` lists exactly what was built and the launch refuses an
#: arm the binary does not carry. Iteration builds pair it with
#: ``ROLA_CUDA_ARCHS=86``; docs/build.md.
def _decode_arm_subset_active() -> bool:
    return bool(os.getenv("ROLA_DECODE_ARMS", "").strip())


def _decode_arm_subset() -> list[str]:
    spec = os.getenv("ROLA_DECODE_ARMS", "").replace(",", " ").split()
    if not spec:
        return []
    rows = " ".join(f"DECODE_ARM_{int(i)}(X_)" for i in spec)
    return [f"-DROLA_DECODE_ARMS(X_)={rows}"]


#: THE EXTENSION LIVES IN THE TOOLCHAIN'S BINARY PLUGIN, ``rola_cu13`` (wheel naming D,
#: docs/build.md#wheels): ``rola`` is pure Python, and the binary, its build record and its
#: manifests ship as ``rola-cu13``. A dotted name makes setuptools place the binary inside
#: the plugin's package in an in-place build and in a wheel alike, so a checkout imports it
#: by the path an install does, and the repo root stays clean.
#:
#: This is a PACKAGING name: the registration TU's ``PyInit__C`` is the one place csrc
#: spells its last component, and the operators it registers live in ``torch.ops.rola``;
#: the C++ namespaces (``rola``, ``rola::paging``) and every kernel symbol are untouched.
#: The manifests key on MANGLED KERNEL NAMES, and the module-init symbol they do not
#: contain is the only thing this name can move.
PLUGIN, EXTRA = "rola_cu13", "cu13"
if TOOLCHAIN is not None and TOOLCHAIN.name != EXTRA:
    raise RuntimeError(f"toolchain {TOOLCHAIN.name} has no binary plugin; this tree ships {PLUGIN} alone")
MODULE_NAME = f"{PLUGIN}._C"

#: THE STABLE ABI TARGET (`tools/build_flags.py`'s `TORCH_MIN`, docs/build.md#stable-abi): every compile targets it, and
#: `CUDAExtension(py_limited_api=True)` builds the module object against Python's limited API, defining the flag itself.
STABLE_DEFINES = list(build_flags.STABLE_DEFINES)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
def read_version() -> str:
    """From ``rola/__init__.py``. No setuptools_scm: the tag must not be able to
    disagree with the shipped ``__version__``, which is also stamped next to the
    manifest hashes in ``_build_config.py``."""
    text = (ROOT / "rola" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*([\'"])([^\'"]+)\1', text, re.M)
    if not match:
        raise RuntimeError("no __version__ in rola/__init__.py")
    return match.group(2)


# ---------------------------------------------------------------------------
# RULE 2 -- gencode
# ---------------------------------------------------------------------------
def assert_no_ptx(flags: list[str]) -> None:
    """Not decoration: this is what makes rule 2 survive a future editor who adds a
    'just for the next architecture' PTX line. The build fails rather than silently
    shipping unmeasured codegen."""
    bad = [f for f in flags if "code=compute_" in f]
    if bad:
        raise RuntimeError(
            "CLOSED-WORLD RULE 2: PTX JIT fallback must never be emitted. Offending "
            f"gencode flag(s): {bad}. A `code=compute_XX` entry lets the driver "
            "compile this kernel for an architecture nobody measured; ratify the "
            "architecture instead (docs/bringup.md)."
        )


# ---------------------------------------------------------------------------
# RULES 1 and 3 -- the pre-build gates
# ---------------------------------------------------------------------------
cuda_bin = dev_config.cuda_bin


def ptxas_version() -> str:
    """The WHOLE ``ptxas --version`` output, whitespace-normalized -- the same
    normalization ``tools/ratify.py`` records, so the two strings are comparable
    byte for byte. The build string distinguishes two shipments of one nominal
    release, and it is exactly those that can move an allocation."""
    proc = subprocess.run([cuda_bin("ptxas"), "--version"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot ask `{cuda_bin('ptxas')} --version`; a closed-world "
                           f"build cannot proceed without identifying its assembler")
    return " ".join(proc.stdout.split())


def manifest_path(arch: str) -> Path:
    """ONE FILE PER ARCH (§3.7), so rule 3 is a FILE-EXISTENCE question.

    That matters because rule 3 runs BEFORE the compile: the cheapest possible check
    is the one that must never be tempting to skip. It also makes a new
    architecture's bring-up one reviewable file in a PR rather than a diff inside a
    thousand-line document nobody reads to the end.
    """
    return TOOLCHAIN.manifest_path(arch)


def _gate_ratified(archs: list[str]) -> dict:
    """RULE 3: every arch in the fatbin has a ratified manifest, and they agree.

    Returns the MERGED record -- the union of exactly the archs being built, which is
    what ``manifest_sha256`` must then describe. A wheel built for one arch is gated
    against one arch's measurements, and its recorded digest says so.
    """
    toolchain = None
    shards = None
    entries: dict = {}
    for a in archs:
        path = manifest_path(a)
        if not path.exists():
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: sm_{a} is in ROLA_CUDA_ARCHS but there is no "
                f"ratified manifest at {path}. Ratify it first:\n"
                f"    python tools/ratify.py --arch {a} --write\n"
                f"and record why the numbers are what they are (docs/bringup.md). "
                f"Building an architecture nobody measured is the failure this rule "
                f"exists to prevent."
            )
        blob = json.loads(path.read_text())
        if not blob["entries"]:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: {path} exists but ratifies NOTHING. File "
                f"existence is the cheap form of the question, not a weaker one."
            )
        if toolchain is None:
            toolchain, shards = blob["toolchain"], blob["shards"]
        elif blob["toolchain"] != toolchain:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 1: {path.name} pins a different assembler than the "
                f"other manifests. The per-arch split is a FILING decision; one fatbin "
                f"may not be gated against two ratifications."
            )
        elif blob["shards"] != shards:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: {path.name} records a different instantiation "
                f"membership than the other manifests. The per-arch split is a FILING "
                f"decision; one fatbin may not be gated against two matrices."
            )
        entries.update(blob["entries"])
        print(f"rule 3 OK  sm_{a}: {len(blob['entries'])} ratified instantiations "
              f"({path.name})")
    return {"toolchain": toolchain, "shards": shards, "entries": entries}


def vendored_tree_sha256(root: Path) -> tuple[str, int]:
    """A digest of a DIRECTORY, defined so it cannot be argued with.

    sha256 over the sorted relative POSIX paths, each followed by a NUL byte and by
    the sha256 DIGEST of that file's bytes; ``PIN.json`` itself is excluded because it
    carries the answer. Paths participate as well as contents, so a RENAME -- which
    changes which header an ``#include`` resolves to, and therefore what is compiled --
    moves the digest exactly as a content edit does.
    """
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "PIN.json")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def _gate_vendored() -> dict:
    """RULE 5: the vendored codegen inputs are the ones that were pinned.

    Runs BEFORE the compile, with rules 1 and 3, for the same reason they do: it is
    cheap, and a vendored header that drifted is not something to discover from a
    manifest diff after a multi-hour build.
    """
    pin_path = CUTLASS_DIR / "PIN.json"
    if not pin_path.exists():
        raise RuntimeError(
            f"CLOSED-WORLD RULE 5: {pin_path} is missing. A vendored library with no "
            f"pin is an unmeasured codegen input; restore the pin or remove the "
            f"vendored tree."
        )
    pin = json.loads(pin_path.read_text())
    here, count = vendored_tree_sha256(CUTLASS_DIR)
    if here != pin["tree_sha256"]:
        raise RuntimeError(
            "CLOSED-WORLD RULE 5: the vendored CUTLASS subtree does not match its "
            f"pin.\n  PIN.json: {pin['tree_sha256']} ({pin['file_count']} files)\n"
            f"  on disk : {here} ({count} files)\n"
            "A vendored header is code `ptxas` assembles, so this is a codegen change "
            "and not a packaging one. Re-vendor the pinned tag, or move the pin "
            "DELIBERATELY and re-ratify (`python tools/ratify.py --write`)."
        )
    print(f"rule 5 OK  cutlass {pin['tag']} ({pin['commit'][:12]}): {count} files, "
          f"sha256 {here[:16]}...")
    return pin


def _gate_toolchain(manifest: dict) -> str:
    import torch

    pinned = toolchains.normalize(manifest["toolchain"]["ptxas"])
    here = ptxas_version()
    if pinned != here or here != TOOLCHAIN.ptxas:
        raise RuntimeError(
            f"CLOSED-WORLD RULE 1: toolchain {TOOLCHAIN.name}'s manifests were ratified under a "
            f"different assembler.\n  manifest: {pinned}\n  this host: {here}\n"
            f"Re-ratify (`python tools/ratify.py --write`) and record the new numbers "
            "-- do not hand-edit the manifest."
        )
    toolchains.torch_pairing(TOOLCHAIN, torch.version.cuda)
    print(f"rule 1 OK  toolchain {TOOLCHAIN.name}, torch CUDA {torch.version.cuda}, ptxas: {here}")
    return here


def _gate_sccache() -> dict | None:
    """Wire ``tools/sccache_nvcc.py`` in as ``PYTORCH_NVCC``, if the pinned cache is
    available (closed-world rule 1's extension, ``tools/sccache_toolchain.py``).

    Runs AFTER ``_gate_toolchain``, and points the wrapper at exactly the ``nvcc``
    that gate's ``ptxas --version`` check already resolved and pinned -- the wrapper
    itself never re-resolves it (see that module's docstring for why letting it do
    so would reopen the hole rule 1 exists to close).
    """
    info = sccache_toolchain.gate()
    if info is None:
        return None
    real_nvcc = cuda_bin("nvcc")
    wrapper, env = sccache_toolchain.wrap(real_nvcc, info["path"])
    os.environ.update(env)
    os.environ["PYTORCH_NVCC"] = wrapper
    print(f"sccache: wired as PYTORCH_NVCC ({wrapper} -> {real_nvcc})")
    return info


# ---------------------------------------------------------------------------
# _build_config.py -- the self-describing binary
# ---------------------------------------------------------------------------
def write_build_config(archs: list[str], ptxas: str, manifest: dict, pin: dict,
                        sccache: dict | None) -> Path:
    import torch

    config = {
        "version": read_version(),
        "archs": [f"sm_{a}" for a in archs],
        #: A binary whose arm or part set is not the shipped one: never a wheel (`tools/wheels.py`).
        "iteration": _is_iteration_build(),
        "toolchain": TOOLCHAIN.name,
        "ptxas": ptxas,
        "torch": torch.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        #: TAKEN FROM `tools/ratify.py`, NEVER RESTATED. This digest identifies the
        #: ratification a binary was gated against, and the tier-2 census recomputes
        #: it independently and requires a match -- so a third spelling of "what goes
        #: into the hash" is a spelling that can disagree with both.
        "manifest_sha256": ratify.manifest_digest(manifest),
        "manifest_entries": {f"sm_{a}": sum(1 for k in manifest["entries"]
                                            if k.startswith(f"sm_{a}:"))
                             for a in archs},
        #: THE RATIFIED SHARD MEMBERSHIP, `{shard: members_sha256}`, taken from the
        #: manifest rather than recomputed. It is what the dispatch table's rows are
        #: joined against at runtime: a row whose shard membership is not the one that
        #: was MEASURED describes an arm nobody has register or spill numbers for.
        #: An installed wheel carries no `tools/manifests/`, so without this the
        #: membership could only be read from a repository checkout -- which is the
        #: half of the closed world a user does not have.
        "shards": {name: block["members_sha256"]
                   for name, block in sorted(manifest["shards"].items())},
        "ptx_jit_fallback": False,
        #: THE VENDORED CODEGEN INPUTS, one entry per pinned library. Recorded next to
        #: `ptxas` because it answers the same question about the same binary: what
        #: assembled this, and from what sources.
        "vendored": {
            "cutlass": {"tag": pin["tag"], "commit": pin["commit"],
                        "tree_sha256": pin["tree_sha256"]},
        },
        #: THE COMPILER CACHE, recorded per the input-closure rule (closed-world
        #: rule 1's extension, `tools/sccache_toolchain.py`): any cache config that
        #: CAN affect output is part of the fingerprint, even though this one is
        #: measured transparent (docs/build.md#sccache) -- `None` when the build ran
        #: uncached, never omitted, so "was this binary's compile cached at all" is
        #: answerable from the wheel with no checkout.
        "sccache": sccache,
    }
    body = (
        '"""GENERATED by setup.py -- do not edit.\n\n'
        "Answers 'which measured evidence backs this exact .so?' from an installed\n"
        "wheel, with no repository checkout: `manifest_sha256` is the hash of the\n"
        "ratification this binary was gated against.\n"
        '"""\n\n'
        #: `pprint`, not `json.dumps`: this is a PYTHON module, and JSON spells the
        #: booleans `true`/`false`, which do not parse here.
        f"BUILD_CONFIG = {pprint.pformat(config, indent=4, sort_dicts=False, width=88)}\n\n\n"
        "def show():\n"
        '    """A copy, so a caller cannot mutate the record."""\n'
        "    return dict(BUILD_CONFIG)\n"
    )
    record = ROOT / PLUGIN / "_build_config.py"
    record.write_text(body)
    print(f"wrote {PLUGIN}/_build_config.py (manifest sha256 "
          f"{config['manifest_sha256'][:16]}...)")
    return record


# ---------------------------------------------------------------------------
# the extension
# ---------------------------------------------------------------------------
#: THE TRANSLATION UNITS. Every one is hand-written on this line: C0 deleted the two
#: families that had GENERATED instantiation units (the carry family's per-arm TUs and
#: the stats passes' shard), so nothing is appended to this list but the per-arm carry
#: units the arm table names -- and it names none until G4.
SOURCES = [
    "csrc/rola/rola_api.cpp",
    # TRACK D (thunder §6b): the T=1 decode path, one TU.
    #: The carry family's dispatch: the kernel boundary's refusals, the addressing
    #: block's one derivation and the arm resolution. Holds no arm body -- those are
    #: the generated per-arm units, which is what makes an arm subset a file subset.
    "csrc/rola/src/carry/carry.cu",
    "csrc/rola/src/decode/decode.cu",
    "csrc/rola/src/entmax/entmax.cu",
    "csrc/rola/src/entmax/factor.cu",
    "csrc/rola/src/paging/vmm_owner.cu",
    #: G -- the addressing block's host seam (derivation + refusals). Its own TU so
    #: that the foundation lands without recompiling the dispatch; it holds no kernel.
    "csrc/rola/src/common/geom.cu",
    #: The closed-world arch rule's runtime half (setup.py's rule R4): its own TU,
    #: beside arch_caps.cuh's compile-time half, so every launching family links one
    #: definition rather than each carrying its own copy.
    "csrc/rola/src/common/arch_runtime.cu",
    "csrc/rola/src/facts/liveness.cu",
    #: The extension's freshness fact: one kernel returning the csrc content hash it
    #: compiled against. Its own TU so that a source edit rebuilds this one file
    #: rather than every translation unit that would otherwise carry the constant.
    "csrc/rola/src/common/build_stamp.cu",
    #: The SM's effective clock under load, read off the device: the harness records it
    #: with every measurement, because the driver's reported clock is not it on this host.
    "csrc/rola/src/common/sm_clock.cu",
    # K31 R2: the standalone deterministic intra kernel, one TU.
    "csrc/rola/src/intra/intra.cu",
]


def _gate_shards() -> None:
    """The checked-in shards are the ones the arm list produces.

    Runs BEFORE the compile, with the other cheap gates. Generated sources are
    committed so that what is reviewed is what compiles (closed-world); that promise
    is only worth anything if a stale generated file fails the build instead of
    silently shipping an arm list nobody wrote.
    """
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_shards.py"), "--check"],
                          cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            "the checked-in instantiation shards do not match the arm list "
            "in tools/gen_shards.py. Regenerate them (`python tools/gen_shards.py "
            "--write`) and review the diff -- a generated file that disagrees with "
            "its generator is an arm list nobody wrote."
        )


def _ninja_jobs(held_slots: int) -> int:
    """Ninja's job count, so the concurrent ``cicc`` count is the slots held.

    A slot in ``tools/host_budget.py`` is one ``cicc``, and one ``nvcc`` runs one
    ``cicc`` PER ARCHITECTURE, up to ``--threads``: the two parallelism axes multiply,
    so the job count is the budget divided by that per-process factor. Setting ninja's
    ``-j`` to the slot count directly oversubscribes the budget by exactly the number
    of architectures compiled.
    """
    per_process = min(dev_config.get("host.nvcc_threads"), max(1, len(ROLA_CUDA_ARCHS)))
    return max(1, held_slots // per_process)


def _link_flags() -> list[str]:
    ldflags = ["-lcuda"]
    #: WSL puts the driver stub outside the toolkit's lib dir. A build knob, so it
    #: belongs to the build; the runtime (`rola/ops/_ext.py`) has none of this.
    wsl = Path("/usr/lib/wsl/lib")
    if (wsl / "libcuda.so").exists():
        ldflags.insert(0, f"-L{wsl}")
    #: mold is THE linker, unconditionally (Blake ruling, 2026-08-28): raises if
    #: the pin (tools/mold_pin.json) is not satisfied -- no fallback to
    #: ld.bfd/ld.gold. See tools/mold_toolchain.py's module docstring.
    ldflags += mold_toolchain.link_flags()
    return ldflags


#: THE PART HARNESS'S DRIVER (`benchmarks/unit/bench_carry_parts.py`), built AHEAD OF TIME
#: beside the arm under `ROLA_BUILD_PARTS=1` -- the same pass, the same lock, the same cache
#: -- where a torch JIT `load()` of the same three sources ran for minutes and was killed
#: by the host's memory watchdog (KERNEL_STANDARDS §22: a gating instrument that cannot
#: run is a defect). Never shipped: an iteration-build instrument, `-lineinfo` for the
#: ledgers. docs/build.md#parts
PARTS_MODULE_NAME = f"{PLUGIN}._C_parts"
PARTS_SOURCES = ["benchmarks/unit/carry_parts/carry_parts.cu",
                 "csrc/rola/src/common/arch_runtime.cu", "csrc/rola/src/common/geom.cu"]


def build_parts_extension():
    nvcc_threads = dev_config.get("host.nvcc_threads")
    nvcc_flags = build_flags.nvcc_flags(ROLA_CUDA_ARCHS, nvcc_threads) \
        + list(build_flags.DEPFILE_FLAGS) + ["-lineinfo"]
    assert_no_ptx(nvcc_flags)
    return CUDAExtension(
        name=PARTS_MODULE_NAME,
        sources=PARTS_SOURCES,
        include_dirs=[str(CSRC / "src"), str(CUTLASS_DIR / "include"), str(ROOT / "build" / "generated")],
        extra_compile_args={"cxx": ["-O3", "-std=c++20", *STABLE_DEFINES], "nvcc": nvcc_flags + STABLE_DEFINES},
        extra_link_args=_link_flags(),
        py_limited_api=True,
    )


def desired_jobs() -> int:
    """The ninja jobs a build asks the host budget for: `host.build_jobs`, else derived from `host.nvcc_threads`."""
    return dev_config.get("host.build_jobs") or build_flags.default_max_jobs(dev_config.get("host.nvcc_threads"))


def build_extension():
    nvcc_threads = dev_config.get("host.nvcc_threads")

    nvcc_flags = build_flags.nvcc_flags(ROLA_CUDA_ARCHS, nvcc_threads) \
        + list(build_flags.DEPFILE_FLAGS) + _decode_arm_subset()
    assert_no_ptx(nvcc_flags)

    ldflags = _link_flags()

    #: The generated selection header's directory (K36b). A FIXED path, added to
    #: every build regardless of arm selection -- it does not itself vary per-arm-set
    #: and so does not perturb other TUs' argv/cache-key any more than the CUTLASS
    #: include dir already does. Must exist before `CUDAExtension(...)` is
    #: constructed: nvcc/gcc get handed `-I<dir>` at configure time even on a
    #: completely fresh checkout, before `_write_carry_selection` ever runs.
    generated_dir = ROOT / "build" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    return CUDAExtension(
        name=MODULE_NAME,
        sources=SOURCES + _carry_sources(),
        #: The vendored CuTe/CUTLASS headers come LAST, so a repository header always
        #: wins a name collision and the vendored tree can never shadow ours.
        include_dirs=[str(CSRC / "src"), str(CUTLASS_DIR / "include"), str(generated_dir)],
        extra_compile_args={"cxx": ["-O3", "-std=c++20", *STABLE_DEFINES], "nvcc": nvcc_flags + STABLE_DEFINES},
        extra_link_args=ldflags,
        py_limited_api=True,
    )


class RoLABuildExtension(BuildExtension):
    """``BuildExtension`` plus the closed-world gates, before and after."""

    def finalize_options(self):
        """Pin ``build_temp`` to a FIXED, repo-relative directory, overriding
        whatever ``pip``'s ``build_editable`` hook chose.

        ``pip install -e . --no-build-isolation`` runs the PEP 660 editable-build
        hook in a FRESH ``tempfile.mkdtemp()`` directory every single invocation
        -- verified: two consecutive installs of an UNCHANGED tree produced
        ``/tmp/tmpXXXXXXXX.build-temp`` with different random suffixes each time.
        Ninja's build graph (and therefore its OWN incremental-rebuild skip, and
        sccache's cache key -- both measured to include the invoking ``nvcc``
        process's CWD, ``docs/build.md#sccache``) lives inside that directory, so
        a random CWD every build defeats BOTH: ninja treats an unchanged source
        as new work because there is no ``build.ninja`` from last time to compare
        against, and sccache never hits across builds because the two ``nvcc``
        invocations of the identical TU ran from two different working
        directories. Pinning ``build_temp`` here fixes the object-file layer for
        the same reason multiple worktree builds already need SOURCE-content-only
        mangled names (P19, ``docs/build.md``'s path-trap section): a build
        artifact's identity must be a function of the tree's CONTENT, not of
        which ephemeral directory pip happened to allocate this time.
        """
        super().finalize_options()
        self.build_temp = str(ROOT / "build" / "temp")

    def run(self):
        manifest = _gate_ratified(ROLA_CUDA_ARCHS)
        ptxas = _gate_toolchain(manifest)
        pin = _gate_vendored()
        _gate_shards()
        _write_carry_selection()
        _write_carry_parts()
        _write_csrc_stamp()
        sccache = _gate_sccache()
        #: THE MACHINE-WIDE CICC BUDGET (KERNEL_STANDARDS §14 addendum 2, Blake
        #: ruling 2026-08-28-night): TOOL-ENFORCED here so no brief, no
        #: worktree and no agent can forget it -- see tools/build_lock.py's
        #: module docstring for why this cannot stay a documentation-only
        #: convention. A gate build (the shipped arm set) takes the WHOLE
        #: budget; an iteration build takes whatever share is free right now
        #: and MAX_JOBS is clamped to that share, never to what was asked.
        #: `_is_iteration_build` covers BOTH families (ROLA_CARRY_ARMS,
        #: ROLA_DECODE_ARMS) -- one mechanism, not two (§18).
        with build_lock.acquire(gate=not _is_iteration_build(),
                                 desired=desired_jobs()) as held_slots:
            os.environ["MAX_JOBS"] = str(_ninja_jobs(held_slots))
            super().run()
        self._post_build_arm_table_check()
        self._post_build_manifest_check()
        record = write_build_config(ROLA_CUDA_ARCHS, ptxas, manifest, pin, sccache)
        #: A wheel's build tree took the package's files before this record was written; it ships this build's.
        if not self.inplace:
            shutil.copy(record, Path(self.build_lib) / PLUGIN / record.name)

    def _iteration_banner(self) -> bool:
        if not _is_iteration_build():
            return False
        print("=" * 78)
        if set(_carry_arms()) != set(gen_shards.CARRY_SHIPPED_ARMS):
            print("ITERATION BUILD -- NOT SHIPPABLE.  This build carries carry arms "
                  f"{_carry_arms()}, which is not the SHIPPED set "
                  f"{list(gen_shards.CARRY_SHIPPED_ARMS)}, so the shippability gate "
                  "(the manifest) is skipped. Build with "
                  "ROLA_CARRY_ARMS unset to get a gateable binary.")
        if set(_carry_parts()) != set(gen_shards.CARRY_PARTS):
            print("ITERATION BUILD -- NOT SHIPPABLE.  ROLA_CARRY_PARTS stubs "
                  f"{sorted(set(gen_shards.CARRY_PARTS) - set(_carry_parts()))} of the carry kernel "
                  "(a composer measurement build: the stubbed parts keep their HMMAs and drop "
                  "their other work). Build with ROLA_CARRY_PARTS unset to get the kernel.")
        if _decode_arm_subset_active():
            print("ITERATION BUILD -- NOT SHIPPABLE.  ROLA_DECODE_ARMS is set, so this "
                  "fatbin carries a SUBSET of the decode arms, so the shippability "
                  "gate (the manifest) is skipped. "
                  "rola.ops.decode.arms() lists what it does carry. Build with "
                  "ROLA_DECODE_ARMS unset to get a gateable binary.")
        print("=" * 78)
        return True

    def _post_build_arm_table_check(self):
        """The built arm table against the DECLARED SHIPPED SET.

        Three statements, asserted after the compile rather than before it, because
        what is being checked is what this build PRODUCED:

        1. the generator's table still matches `tools/manifests/shipped_set.json`,
           re-read from disk now. The table was built from the file at import; a
           build runs for long enough that "the declaration says what it said" is a
           second statement rather than a restatement of the first;
        2. the selection header this build compiled against names exactly the arms
           this build selected -- the binary's arm set is the declaration's, scoped,
           and not whatever a previous run left in `build/generated/`;
        3. the coverage sweep: every arm the binary carries is exercised through
           `arm_switch`, which is R9's fourth layer on the host side.

        A FAILURE HERE IS FATAL WHATEVER `ROLA_STRICT_MANIFEST` SAYS. That switch
        is about manifest STALENESS -- measurements that go out of date between
        milestones by design -- and none of these three is a measurement. A binary
        whose arm set disagrees with the declaration is not stale, it is wrong.
        """
        bad = gen_shards.declaration_matches_table()
        if bad:
            raise RuntimeError(
                "the carry arm table does not match the declared shipped set at "
                f"{gen_shards.SHIPPED_SET_JSON}:\n" + "\n".join(bad))

        want = _carry_arms()
        header = gen_shards.GENERATED_DIR / gen_shards.CARRY_SELECTION_INC
        if not header.exists():
            raise RuntimeError(
                f"{header} does not exist after the build; the selection header is "
                f"what the per-arm translation units and the arm set compile "
                f"against, so a build without one compiled against something else")
        text = header.read_text()
        named = sorted(int(m) for m in re.findall(
            r"^#define ROLA_CARRY_BUILD_(\d+) 1$", text, re.MULTILINE))
        if named != want:
            raise RuntimeError(
                f"the selection header names carry arms {named}, this build selected "
                f"{want}. The header is what the translation units compiled against, "
                f"so the binary carries the header's set and not the build's.")
        count = re.search(r"^#define ROLA_CARRY_ARM_COUNT (\d+)$", text, re.MULTILINE)
        declared = re.search(r"^#define ROLA_CARRY_ARM_DECLARED (\d+)$", text,
                             re.MULTILINE)
        if count is None or declared is None:
            raise RuntimeError(
                f"{header} carries no ROLA_CARRY_ARM_COUNT/ROLA_CARRY_ARM_DECLARED; "
                f"`arm_switch` has no set to dispatch over and no way to tell an "
                f"unbuilt arm from an undeclared one")
        if int(count.group(1)) != len(want) or \
                int(declared.group(1)) != len(gen_shards.CARRY_ARMS):
            raise RuntimeError(
                f"{header} declares {count.group(1)} built / "
                f"{declared.group(1)} declared arms; this build selected "
                f"{len(want)} of {len(gen_shards.CARRY_ARMS)} declared")
        members = sorted(tuple(int(x) for x in m) for m in re.findall(
            r"^  F\((\d+), (\d+), (\d+), (\d+)\)", text, re.MULTILINE))
        if members != sorted((i,) + tuple(gen_shards.CARRY_ARMS[i]) for i in want):
            raise RuntimeError(
                f"the arm set names {members}, which is not this build's rows of the "
                f"declaration. The set and the selection are two enumerations of one "
                f"thing and must not be able to disagree.")

        swept = _arm_coverage_sweep(self.build_lib, want)
        print(f"arm table OK  {len(want)} built of {len(gen_shards.CARRY_ARMS)} "
              f"declared, keyed {list(gen_shards.ARM_KEY)}; {swept} arm(s) swept")

    def _post_build_manifest_check(self):
        """The pre-build gates prove the INPUTS were ratified. This proves the
        codegen still matches what was measured: `tools/ratify.py` re-compiles the
        SHIPPED translation units -- the eight instantiation shards plus
        `decode.cu` and `entmax.cu`, under the same flag list this build uses
        (`tools/build_flags.py`) -- and fails on any spill growth or unhonored
        launch bound.

        EVERY one of `ratify.py`'s checks (entry set, SASS-body hashes, the
        hot-loop/local-presence ratchets) is a comparison against the COMMITTED
        `tools/manifests/<toolchain>/sm_XX.json`, and that manifest is allowed to be stale
        between milestones by design (KERNEL_STANDARDS §16) -- unlike the fatbin
        gate above, there is no sub-check here that is independent of the manifest,
        so the K43 split is a single blanket one: FAIL is fatal only under
        `ROLA_STRICT_MANIFEST=1` (set by the milestone/CI gate build that precedes
        `ratify.py --write`, docs/build.md); otherwise it is a REPORT and the
        install proceeds. This never touches `ratify.py` itself -- `--write`/
        `--check` invoked directly still behave exactly as before.
        """
        if _is_iteration_build():
            return
        if os.getenv("ROLA_SKIP_POST_BUILD_RATIFY") == "1":
            print("post-build ratification SKIPPED (ROLA_SKIP_POST_BUILD_RATIFY=1)")
            return
        print("post-build: re-running the ratification gate ...")
        #: EXACTLY the archs being built, passed explicitly. The gate has its own
        #: default arch list, and a default that silently differed from
        #: ``ROLA_CUDA_ARCHS`` would ratify a fatbin nobody produced.
        cmd = [sys.executable, str(ROOT / "tools" / "ratify.py")]
        for a in ROLA_CUDA_ARCHS:
            cmd += ["--arch", a]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            message = (
                "post-build ratification FAILED (see the entry/body counts printed "
                "above): the built sources no longer match the committed manifest. "
                "Re-ratify (`python tools/ratify.py --write`) only once the change is "
                "understood and the new numbers are recorded."
            )
            if ROLA_STRICT_MANIFEST:
                raise RuntimeError(message + " The binary is NOT shippable.")
            print("\n" + "=" * 78)
            print("WARNING -- post-build ratification FAILED (counts printed above "
                  "this banner). Install proceeding because ROLA_STRICT_MANIFEST is "
                  "not set: manifests go stale between milestones by design "
                  "(KERNEL_STANDARDS §16). This binary is NOT certified shippable -- "
                  "re-run with ROLA_STRICT_MANIFEST=1 (the milestone/CI gate build "
                  "always does) before trusting it, or `python tools/ratify.py "
                  "--write` if the drift is understood and intentional.")
            print("=" * 78 + "\n")


#: THE EXTRAS, here and not in pyproject.toml because `cu13` names this version: `pip install "rola[cu13]"` installs the
#: binary plugin built at exactly this `rola` (docs/build.md#wheels), and `rola.ops._ext` refuses any other.
EXTRAS = {
    EXTRA: [f"{PLUGIN.replace('_', '-')}=={read_version()}"],
    #: `pytest-benchmark` was REMOVED, measured: it drove the latency gate's repetition, and its round schedule -- not
    #: the kernel -- set the number the gate read once the committed spread narrowed to a device-event IQR.
    #: `rola_results` holds the record now (docs/measurement.md). THE ATTENTION REFERENCE every reported carry number
    #: sits beside is rola-bench's arm (torch's own flash backend, docs/measurement.md), so it needs no package here.
    "bench": ["matplotlib", "pandas"],
    #: `entmax` (DeepSPIN) is the fp64 REFERENCE the routing gates are anchored to -- the paper authors' own
    #: implementation, pure python with torch as its only dependency. A TEST dependency and never a runtime one:
    #: `rola/routing/entmax/production.py` is RoLA's own hand-CUDA producer and imports nothing from it, which
    #: `tests/integration/test_production_levels.py` asserts rather than assumes.
    #:
    #: PINNED to `==1.3`: `rola/routing/entmax/PIN.json` records the version AND a digest of the installed package, and
    #: `rola/routing/entmax/reference.py` refuses to import (`pin.gate()`) an install that drifted from either. This `==`
    #: is the coarse half of that guard -- the resolver refuses the bump before pip finishes -- and the pin is the fine
    #: half, because the resolver cannot see an editable or re-published install that keeps the version string. See
    #: `PIN.json`'s `why_this_is_the_only_guard`: the dual-run protocol imports the same installed package on both its
    #: sides and cannot see this drift itself.
    "test": ["pytest", "pytest-xdist", "entmax==1.3"],
    "dev": ["pytest", "ruff==0.14.10", "pre-commit==4.6.2", "vulture==2.16", "clang-format==19.1.7",
            "ast-grep-cli==0.45.2"],
}


setup(
    name="rola",
    version=read_version(),
    packages=find_packages(include=["rola", "rola.*", PLUGIN]),
    #: entmax's pin record, read beside `rola/routing/entmax/pin.py`: without it an installed reference cannot gate itself.
    package_data={"rola.routing.entmax": ["PIN.json"]},
    extras_require=EXTRAS,
    #: ROLA_NO_EXTENSION=1 installs the Python package with NO extension build -- for a
    #: worktree that receives a built _C from the integration checkout at the same
    #: SHA (`python tools/dev.py worktree`); the device stamp still gates freshness.
    ext_modules=[] if os.getenv("ROLA_NO_EXTENSION") == "1" else (
        [build_extension()] + ([build_parts_extension()] if os.getenv("ROLA_BUILD_PARTS") == "1" else [])),
    cmdclass={"build_ext": RoLABuildExtension},
    zip_safe=False,
)
