"""The HOST dispatch primitive, exercised against a synthetic arm set.

`csrc/rola/src/common/arm_switch.cuh` dispatches over the arm set the generated
selection header carries; this file never depends on which arms are actually shipped
(today's shipped set is one arm, `(2, 64, 8)`, per `tools/manifests/shipped_set.json`).
This file compiles a probe against the SHIPPED HEADER with
a synthetic three-arm selection header in front of it -- rendered by the one renderer
`setup.py` and `tools/ratify.py` use, so what is tested is what a build would emit --
and checks every layer of the contract:

* the two counts are the generated ones, and the set names its members;
* a BUILT arm dispatches, and its body reads its own compile-time constants;
* an arm the DECLARATION declares but this build did not carry is refused, and the
  refusal says so with both counts;
* an arm outside the declaration is refused with the FIELD that misses named -- one
  case per field, because a refusal that names the wrong field is worse than one that
  names none;
* the EMPTY set refuses everything, exercised here as a synthetic case (the shipped
  binary's own set carries one arm, not zero).

The compile runs under the host budget (`rola_devtools.locks.host`): a compile is a compile.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "csrc" / "rola" / "src"
PROBE = pathlib.Path(__file__).resolve().parent / "fixtures" / "arm_switch_probe.cpp"
BUILD_LOCK = [sys.executable, "-m", "rola_devtools.locks.host", "--"]

#: THE SYNTHETIC TABLE, and the only mirror in this file: three rows on the real key,
#: of which a build carries the first and the last. The middle row is what makes
#: "declared but not built" a distinct case from "not declared".
TABLE = ((1, 64, 8), (2, 64, 8), (2, 128, 4))
BUILT = [0, 2]


def _selection_header(arms):
    sys.path.insert(0, str(REPO / "tools"))
    import gen_shards

    saved = gen_shards.CARRY_ARMS
    gen_shards.CARRY_ARMS = TABLE
    try:
        return gen_shards.carry_selection_header(arms)
    finally:
        gen_shards.CARRY_ARMS = saved


def _build(tmp_path, arms):
    import torch
    from torch.utils import cpp_extension

    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no host C++ compiler; the probe cannot be built")
    generated = tmp_path / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "carry_selection.inc").write_text(_selection_header(arms))
    out = tmp_path / "probe"
    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    #: THE ABI THE INSTALLED TORCH WAS BUILT WITH. `TORCH_CHECK` resolves into
    #: `libc10`, whose `torchCheckFail` takes a `std::string`, so a probe compiled
    #: under the other `std::string` ABI links against a symbol that is not there.
    abi = int(bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    cmd = [*BUILD_LOCK, cxx, "-std=c++17", "-O0",
           f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
           f"-I{generated}", f"-I{SRC}"]
    cmd += [f"-I{p}" for p in cpp_extension.include_paths()]
    cmd += [str(PROBE), "-o", str(out), f"-L{torch_lib}", "-lc10",
            f"-Wl,-rpath,{torch_lib}"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, f"the probe did not compile:\n{done.stderr}"
    run = subprocess.run([str(out)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    return {line.split("\t")[0]: line.split("\t")[1:]
            for line in run.stdout.splitlines() if line}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("arm_switch_built"), BUILT)


@pytest.fixture(scope="module")
def empty(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("arm_switch_empty"), [])


def test_the_counts_and_the_members_are_the_generated_ones(built):
    assert built["count"] == ["2", "3"]
    assert built["members"] == ["(1, 64, 8), (2, 128, 4)"]


@pytest.mark.parametrize("label,want", [
    ("built_first", ["0", "1", "64", "8"]),
    ("built_last", ["2", "2", "128", "4"]),
])
def test_a_built_arm_dispatches_with_its_own_constants(built, label, want):
    """The body runs under ITS member: a body carrying another arm's constants is
    the failure this makes visible, and it is invisible to a switch that only
    reports whether it matched."""
    assert built[label][0] == "dispatched"
    assert built[label][1:] == want


def test_a_declared_arm_this_build_did_not_carry_is_refused_with_both_counts(built):
    assert built["declared_unbuilt"][0] == "refused"
    message = built["declared_unbuilt"][1]
    assert "carries no carry arm" in message
    assert "2 of 3 declared arms" in message


@pytest.mark.parametrize("label,field", [
    ("undeclared_depth", "D"),
    ("undeclared_width", "DV"),
    ("undeclared_warps", "warps_per_cta"),
])
def test_the_refusal_names_the_field_that_misses(built, label, field):
    assert built[label][0] == "refused"
    assert f"the field that misses is {field}." in built[label][1]


def test_an_empty_set_refuses_every_call_and_says_it_carries_none(empty):
    """The shipped binary's state: a family with no built arm refuses by name."""
    assert empty["count"] == ["0", "3"]
    assert empty["members"] == ["none"]
    for label in ("built_first", "built_last", "declared_unbuilt"):
        assert empty[label][0] == "refused"
        assert "0 of 3 declared arms" in empty[label][1]
        assert "none" in empty[label][1]
