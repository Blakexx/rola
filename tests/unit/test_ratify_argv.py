"""THE RATIFICATION'S COMPILE TARGETS ARE THIS RUN'S, NOT THE LAST BUILD'S.

``tools/ratify.py`` reconstructs its ``nvcc`` argv out of the build's own
``build.ninja`` -- deliberately, for sccache-cache-key fidelity, and that is the
only way to be right about flags nobody transcribed correctly three times running
(``_nvcc_command``'s docstring). But ``build.ninja`` also records the ``-gencode``
list of *whatever the last build compiled for*, and a ratification is a per-arch
measurement whose target list is its own argument.

Inheriting it had a failure mode that reads as a lie about the tree: after any
single-arch build (``ROLA_CUDA_ARCHS=86``), ``ratify.py`` compiled sm_86 only and
then refused with *"the compile reported NO entries for sm_80"* -- a source-drift
message for an environment fact. That is the extension-trap family, and the same
answer applies: the tool must not be able to inherit the thing it is measuring.

This test is the regression. It drives ``_nvcc_command`` against a synthetic
ninja template so it needs no build, no CUDA toolchain and no GPU -- a CPU-lane
test, like every other consumer of this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import build_flags  # noqa: E402  (the tools/ path is inserted above)
import ratify  # noqa: E402

#: A ninja template whose recorded build targeted sm_86 ALONE, which is exactly
#: the state any `ROLA_CUDA_ARCHS=86` build leaves behind.
_ONE_ARCH_TEMPLATE = (
    "/usr/local/cuda/bin/nvcc",
    "-I/somewhere/csrc -c",
    "-O3 -std=c++17 --ptxas-options=-v "
    "-gencode=arch=compute_86,code=sm_86 -DTORCH_EXTENSION_NAME=_C",
)


@pytest.fixture()
def ninja(monkeypatch):
    monkeypatch.setattr(ratify, "_ninja_cuda_template", lambda: _ONE_ARCH_TEMPLATE)


def _gencodes(argv: list[str]) -> list[str]:
    return [a for a in argv if a.startswith(("-gencode", "--generate-code"))]


def test_gencode_is_the_runs_arch_list_not_the_last_builds(ninja):
    argv = ratify._nvcc_command(("80", "86"), Path("x.cu"), "/tmp/x.o")
    assert _gencodes(argv) == build_flags.gencodes(("80", "86"))


def test_a_single_arch_run_compiles_that_arch_only(ninja):
    argv = ratify._nvcc_command(("80",), Path("x.cu"), "/tmp/x.o")
    assert _gencodes(argv) == build_flags.gencodes(("80",))


def test_the_inherited_flags_survive_apart_from_the_targets(ninja):
    """The cache-key fidelity the ninja reuse exists for is not weakened: every
    non-target flag the build recorded is still passed, verbatim."""
    argv = ratify._nvcc_command(("80", "86"), Path("x.cu"), "/tmp/x.o")
    for flag in ("-O3", "-std=c++17", "--ptxas-options=-v", "-DTORCH_EXTENSION_NAME=_C"):
        assert flag in argv


@pytest.mark.parametrize(
    "flags, kept",
    [
        (["-gencode=arch=compute_86,code=sm_86", "-O3"], ["-O3"]),
        (["-gencode", "arch=compute_86,code=sm_86", "-O3"], ["-O3"]),
        (["--generate-code=arch=compute_80,code=sm_80", "-O3"], ["-O3"]),
        (["--generate-code", "arch=compute_80,code=sm_80", "-O3"], ["-O3"]),
    ],
)
def test_both_of_nvccs_gencode_spellings_are_stripped(flags, kept):
    assert ratify._without_gencode(flags) == kept


def test_the_forcing_census_is_gone():
    """`--force-ctas` and its `ROLA_FORCE_CTAS` define were deleted (P65 S2,
    `docs/internals/DELETIONS.md`) once nothing in `csrc/` read the macro. A
    tool that can force a launch bound is a tool that can ratify one; the
    measurement lives in `docs/measurement.md` as a scratch-tree edit."""
    assert "force_ctas" not in ratify.__dict__
    src = (ROOT / "tools" / "ratify.py").read_text()
    assert "ROLA_FORCE_CTAS" not in src and "--force-ctas" not in src
    for path in (ROOT / "csrc").rglob("*"):
        if path.is_file() and path.suffix in (".cu", ".cuh", ".cpp", ".h"):
            assert "ROLA_FORCE_CTAS" not in path.read_text(errors="ignore"), path
