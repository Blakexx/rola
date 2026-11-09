"""The life-range report's liveness (`tools/life_ranges.py`, docs/internals/tools/life_ranges.md#liveness): a value held
across a shuffle's slow-path call and one held across a loop's exit through a convergence barrier both count -- the two
shapes nvdisasm's own counts drop -- proven on a compiled probe and on a listing that has the exit.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import dev_config  # noqa: E402
import life_ranges  # noqa: E402

#: 32 values held across a loop whose trip count differs by lane, so its shuffle's slow path is a subroutine call
PROBE = r"""
__global__ void shuffle_probe(float* y, const int* n) {
  float keep[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) keep[k] = y[k * 256 + threadIdx.x];
  y[threadIdx.x] = 0.0f;
  float v = keep[0];
  for (int i = 0; i < n[threadIdx.x]; ++i) v += __shfl_xor_sync(0xffffffffu, v, 1);
#pragma unroll
  for (int k = 0; k < 32; ++k) v += keep[k] * (k + 1);
  y[threadIdx.x + 256] = v;
}
"""


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    nvcc = dev_config.cuda_bin("nvcc")
    if nvcc is None or not Path(nvcc).exists():
        pytest.skip("no nvcc in this environment's toolkit")
    root = tmp_path_factory.mktemp("life_ranges")
    (root / "probe.cu").write_text(PROBE)
    subprocess.run([nvcc, "-cubin", "-arch=sm_86", "-O3", "-o", str(root / "probe.cubin"), str(root / "probe.cu")],
                   check=True, capture_output=True)
    return root / "probe.cubin"


def test_values_held_across_a_slow_path_call_count_before_the_call(cubin):
    """nvdisasm's marks have the call write registers the probe holds (its own count drops 16 of them at the row
    before the call)."""
    rows = life_ranges.live_counts(cubin, "shuffle_probe")
    before = [rows[k - 1][2] for k, (_, text, _) in enumerate(rows) if text.startswith("CALL") and "__internal" in text]
    assert before, "the probe's shuffle has no slow-path call: the probe no longer has the shape it tests"
    assert min(before) >= 32, rows


#: THE LISTING, as nvdisasm prints life ranges: `(label, row, reads, writes)` -- R2 written, then a loop that leaves
#: through a call to a local label (marked, as nvdisasm marks every call, as writing R2..R9), then R2 read
LOOP = [(None, "IMAD.MOV.U32 R2, RZ, RZ, 0x1", (), (2,)),
        (".L_x_0", "IADD3 R3, R3, 0x1, RZ", (3,), (3,)),
        (None, "@!P0 CALL.REL.NOINC `(.L_x_1)", (), tuple(range(2, 10))),
        (None, "BRA `(.L_x_0)", (), ()),
        (".L_x_1", "STG.E [R4.64], R2", (2, 4, 5), ()),
        (None, "EXIT", (), ())]


def listing(rows) -> list[str]:
    head = "                    // |   # " + " ".join(str(r) for r in range(12)) + " |"
    at = [m.end() - 1 for m in re.finditer(r"\d+", head[head.index("#"):])]
    at = [head.index("#") + a - head.index("// |") - 4 for a in at]
    out = [head]
    for k, (label, text, reads, writes) in enumerate(rows):
        if label:
            out.append(f"{label}:")
        marks = [" "] * (max(at) + 2)
        for r in set(reads) | set(writes):
            marks[at[r]] = "x" if r in reads and r in writes else "v" if r in reads else "^"
        out.append(f"        /*{16 * k:04x}*/                   {text} ;   // |{0:4d}{''.join(marks)[4:]}")
    return out


def test_a_value_held_across_a_loop_exit_through_a_call_counts_in_the_loop():
    rows = {off: live for off, _, live in life_ranges.live_rows(listing(LOOP), "the loop")}
    #: in the loop: R2 (held to the store) and R3 (written each trip); R4 and R5 are never written, so never count
    assert rows[0x10] == 2 and rows[0x20] == 2 and rows[0x30] == 2, rows
    assert rows[0x40] == 1, rows
