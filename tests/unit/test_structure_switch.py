"""R9's ONE PRIMITIVE, exercised on a real device: `rola::uniform_switch`.

`csrc/rola/src/common/structure_switch.cuh` is a device primitive with no shipped caller
yet (`docs/internals/common/structure_switch.md` names the pipelined body as the one that
consumes it), so nothing in the extension can prove it works. This file compiles a toy
`__global__` against the SHIPPED HEADER and runs it, which is the only proof there is:

* **the generated set is covered** -- one member per index, each body carrying its own
  member's compile-time constant, so a body running under the wrong member is visible;
* **the branch is uniform** -- every lane of the warp takes the same member, and each
  lane's own output differs only by its lane id;
* **the trapping default fires** -- an index OUTSIDE the set reaches `__trap()`, which is
  a device fault, and this is why the toy is a SEPARATE PROCESS: a trap kills the CUDA
  context it runs in, so it cannot be provoked inside the pytest process.

The compile runs under the host budget (`tools/host_budget.py`) -- a census compile is a compile
(`rola-build`), and this one is two.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "csrc" / "rola" / "src"
PROBE = pathlib.Path(__file__).resolve().parent / "fixtures" / "structure_switch_probe.cu"
BUILD_LOCK = [sys.executable, str(REPO / "tools" / "host_budget.py"), "--"]

#: the toy's own set size and value law, mirrored from the fixture: a mirror of FIVE
#: LINES is the price of asserting the bodies ran under the right member at all.
NA = 5


def _nvcc() -> str:
    import dev_config

    return dev_config.cuda_bin("nvcc")


@pytest.fixture(scope="module")
def probe_binary(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("the switch's trap is a DEVICE fault; this test needs the device")
    nvcc = _nvcc()
    if not pathlib.Path(nvcc).exists():
        pytest.skip(f"no nvcc at {nvcc}: the toy kernel cannot be compiled")
    major, minor = torch.cuda.get_device_capability()
    out = tmp_path_factory.mktemp("structure_switch") / "probe"
    cmd = [*BUILD_LOCK, nvcc, "-std=c++17", f"-arch=sm_{major}{minor}",
           "--expt-relaxed-constexpr", f"-I{SRC}", str(PROBE), "-o", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"the toy kernel did not compile:\n{done.stderr}"
    return out


def test_every_member_of_the_generated_set_runs_its_own_body(probe_binary):
    """Layer 1 and layer 4's positive half: the switch reaches each member, exactly."""
    done = subprocess.run([str(probe_binary)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [tuple(int(x) for x in line.split()) for line in done.stdout.split("\n") if line]
    #: a lane's word IS its member's constant plus its own id, so a body that ran under
    #: the wrong member shows up as another member's power of four, and a NON-UNIFORM
    #: branch shows up as two members inside one warp.
    assert rows == [(a, lane, (1 << (2 * a)) + lane) for a in range(NA) for lane in range(32)]


def test_an_index_outside_the_set_reaches_the_trap(probe_binary):
    """Layer 4: the miss path is `__trap()`, never a fallthrough.

    A fallthrough would leave the output word at its sentinel and exit zero; a trap
    faults the context, which the toy reports as a launch error and a non-zero exit."""
    done = subprocess.run([str(probe_binary), "trap"], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode != 0, f"an out-of-set index did NOT trap: {done.stdout}"
    assert "LAUNCH_ERROR" in done.stdout, done.stdout + done.stderr
