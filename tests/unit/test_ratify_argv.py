"""THE RATIFICATION'S COMPILE TARGETS ARE THIS RUN'S, NOT A BUILD'S.

The compile path (``tools/ratify.py``'s ``_nvcc_command``, an architecture's first manifest) compiles each unit for the
architectures the run names. A ratification is a per-arch measurement whose target list is its own argument: when the
command inherited a build's ``-gencode`` list, a single-arch build narrowed the next ratification to that architecture,
which then refused with *"the compile reported NO entries for sm_80"* -- a source-drift message for an environment fact
(the extension-trap family: the tool must not be able to inherit the thing it is measuring).

It drives ``_nvcc_command`` alone, so it needs no build, no CUDA toolchain and no GPU: a CPU-lane test.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import build_flags  # noqa: E402  (the tools/ path is inserted above)
import ratify  # noqa: E402


def _gencodes(argv: list[str]) -> list[str]:
    return [a for a in argv if a.startswith(("-gencode", "--generate-code"))]


def test_gencode_is_the_runs_arch_list():
    argv = ratify._nvcc_command(("80", "86"), Path("x.cu"), "/tmp/x.o")
    assert _gencodes(argv) == build_flags.gencodes(("80", "86"))


def test_a_single_arch_run_compiles_that_arch_only():
    argv = ratify._nvcc_command(("80",), Path("x.cu"), "/tmp/x.o")
    assert _gencodes(argv) == build_flags.gencodes(("80",))


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
