"""The measurement preconditions of the chunk-arm harness, ENFORCED rather than documented.

Six gates run before a number exists, in this order: the lock, the memory fraction, the
extension's identity, the tree's identity, the settled clock, the tier. Each of the first
five is a REFUSAL -- a run that cannot establish the precondition raises rather than
measuring -- and each has a planted-violation row in
the retired harness gate (baseline = tag `baseline/pre-k31`). The reasoning behind each gate is
`docs/measurement.md`; what lives here is the enforcement.

**THE LOCKING STATEMENT (superseding the WRAPPED design `docs/testing.md` used
to require).** This harness takes `gpu_lock()` ITSELF now, the same primitive every
other GPU entry point uses (`tools/gpu_lock.py`, `tests/conftest.py`,
`tools/probe_cells.py`, `tools/sanitize_oracle.py`) -- it is invoked BARE:

    python benchmarks/unit/bench_liveness.py

This harness used to be invoked WRAPPED (an external `flock` on the GPU lock
path around the whole command) and `require_gpu_lock` REFUSED a bare invocation by
checking that an ancestor process
held the lock path open. That design was the deliberate OPPOSITE of `tools/probe_cells.py`'s
self-locking one, reasoned as "the rule has no exception left" -- but it meant every
brief had to restate "under flock" correctly by hand, which is exactly the kind of rule
a tool should enforce rather than a caller remember (KERNEL_STANDARDS §18: "tools
refuse, they do not choose"). `disciplined()` below now holds `gpu_lock()` for its
whole body.
"""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from bench.regression import provenance, settle_clocks

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from gpu_lock import gpu_lock  # noqa: E402 -- path insert must precede this import

#: The fraction the device-heavy test modules already pin, after two whole-machine VRAM
#: freezes. Applied before the first fixture allocates.
MEMORY_FRACTION = 0.65

#: `light` is the quick answer during a batch; `landing` collects enough rounds for the
#: paired test to be able to reach its alpha (`bench.stats.MIN_ROUNDS`).
TIERS = ("light", "landing")


def _extension_in_this_checkout() -> str:
    """REFUSE to measure a binary that belongs to a DIFFERENT checkout.

    The sha, the dirty flag and the csrc digest all describe THIS tree; none of them
    describes the `rola` Python actually imported. `rola` installs editable into a venv,
    so a bench run from a worktree can import the main tree's extension and write this
    tree's identity onto another tree's kernel. Both arms then measure the same binary and
    simply AGREE -- which is also what a true negative looks like.
    """
    import rola

    pkg = Path(rola.__file__).resolve()
    try:
        pkg.relative_to(ROOT)
    except ValueError:
        raise SystemExit(
            f"REFUSING TO MEASURE: `import rola` resolved to {pkg}, which is NOT inside "
            f"this checkout ({ROOT}). The row would stamp this tree's sha and csrc digest "
            f"onto another tree's binary. Put this checkout ahead of the editable "
            f"install: PYTHONPATH={ROOT}") from None
    #: §17: a committed record carries repo-relative locations, never an ad-hoc
    #: absolute checkout path -- the relative_to(ROOT) above already proves this is safe.
    return str(pkg.relative_to(ROOT))


def _device_side_stamp() -> dict:
    """The binary's OWN account of itself, read off the loaded extension.

    A path check cannot catch a stale `.so` at the right path -- the fourth recurrence of
    that trap ran a full green battery against a binary nobody had rebuilt. `BUILD_CONFIG`
    is generated into the extension at build time, so this is a device-side fact rather
    than a filesystem one.
    """
    from rola._build_config import BUILD_CONFIG

    return {"manifest_sha256": BUILD_CONFIG["manifest_sha256"], "version": BUILD_CONFIG["version"]}


def _csrc_sha256() -> str:
    sys.path.insert(0, str(ROOT / "tools"))
    import ratify

    return ratify.csrc_digest()


def _gpu_temp_c() -> float | None:
    if not torch.cuda.is_available():
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
         f"--id={torch.cuda.current_device()}"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return float(out.stdout.strip().splitlines()[0])


class Run:
    """One harness invocation: its identity, its environment, and its row factory.

    Constructing a `Run` IS the identity capture, so there is no path that produces a
    number without one. `settle_clocks` is called here and an unsettled box aborts:
    a number taken while the clock is ramping is a number about the clock.
    """

    def __init__(self, *, tier: str, cmd: str, settle=settle_clocks):
        if tier not in TIERS:
            raise SystemExit(f"tier must be one of {TIERS}, not {tier!r}")
        self.tier = tier
        self.cmd = cmd
        self.run_id = uuid.uuid4().hex[:16]
        self.started = time.time()
        self.extension = _extension_in_this_checkout()
        self.stamp = _device_side_stamp()
        self.sha, self.dirty = _git_state()
        self.csrc_sha256 = _csrc_sha256()
        self.temp_start = _gpu_temp_c()

        clocks = settle()
        if not clocks["settled"]:
            raise SystemExit(
                f"REFUSING TO MEASURE: the SM clock did not settle ({clocks}). A number "
                f"taken while the clock is ramping is a number about the clock. A genuine "
                f"timeout means the box is throttled or busy, which is a fact about the "
                f"measurement and not a condition to work around.")
        self.clocks = clocks
        self.env = dict(provenance(), settled=True, sm_mhz=clocks["sm_mhz"],
                        plateau_mhz=clocks["plateau_mhz"],
                        settle_elapsed_s=clocks["elapsed_s"],
                        extension=self.extension,
                        build_stamp=self.stamp,
                        gpu_temp_c_start=self.temp_start)


def _git_state():
    """`(sha, dirty)` of the tree being measured (`rola_results.checkout`: dirty is a tracked diff against HEAD)."""
    from rola_results import checkout

    here = checkout(Path(__file__).resolve().parents[2])
    return here["git_sha"], here["diff_sha256"] is not None


@contextmanager
def disciplined(*, tier: str, cmd: str):
    """The lock, memory fraction, stamped identity, settled clock -- in that order.

    `gpu_lock()` is held for this whole context: the memory fraction is applied
    under it and before any fixture is built, because it bounds this process's share of
    a device another process may still be releasing. EXCLUSIVE mode (LOCKS brief,
    2026-08-29, item 2): a bench number is MEASURED work, so it excludes every other
    GPU-touching tool -- including a correctness-tier pytest run that would otherwise
    share the device under `mode="shared"`.
    """
    with gpu_lock(mode="exclusive"):
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device; this harness measures a CUDA kernel")
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
        yield Run(tier=tier, cmd=cmd)
