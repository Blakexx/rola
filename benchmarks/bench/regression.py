"""The CI regression layer: the judgement over stored measurements.

**WHAT IS SHARED AND WHAT IS NOT.** The methodology stays this repository's own --
no library provides per-pair ratio IQRs, counter corroboration, or a committed
statistics manifest. Storage is `rola_results`: run this commit's latency per
``(cell, arm)``, keep the RAW samples with the instrument that took them, and
compare against a stored baseline. A benchmark runner was tried
in that role and removed on measurement -- its round schedule, not the kernel, set
the number the gate read.

:func:`settle_clocks` lives here for the same reason the threshold does: it is a
PRECONDITION of a comparable number, not a property of any one fixture. A run that
measured while the box was still ramping is not a run this layer can judge.

**THE THRESHOLD IS DERIVED FROM THE CELL'S OWN MEASURED IQR, NEVER INVENTED.** A
flat "fail at +5 %" is a number about nobody's hardware: on a cell whose IQR is 12 %
it fires constantly, and on a cell whose IQR is 0.4 % it lets a real 4 % regression
through. The derivation:

    for a roughly normal sample, IQR = 1.349 * sigma          (Q3 - Q1 = 1.349 sigma)
    a one-sided 3-sigma alarm is therefore  median + 3 * IQR / 1.349
                                          = median + 2.224 * IQR

so ``ALARM_SIGMA = 3.0`` is the only free choice, it is stated as a false-alarm
rate (~0.1 % one-sided per cell per run) rather than as a percentage of anything,
and the per-cell number falls out of that cell's own spread. A cell too noisy to
discriminate the effect it is asked about says so by having a wide threshold --
which is the honest outcome, and it is the rule for reporting too.

**PROVENANCE IS NORMALIZED (O-11).** GPU model, driver, CUDA runtime, torch,
assembler, and the ratification digest the binary was gated against. **No hostname,
no username, no path, no account identifier** -- these files are published, and a
fleet hostname is not a fact about the kernel.

**AND IT NAMES THE STOPWATCH.** Every arm records the instrument it was measured
with (:data:`CANONICAL_INSTRUMENT`), and :func:`check_instrument`
refuses a comparison across two of them. A threshold derived from one instrument's
spread says nothing about another instrument's median.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time

import torch

#: The one free choice. A one-sided 3-sigma alarm: ~0.1 % false alarm per cell per
#: run. Everything else about the threshold is that cell's measured spread.
ALARM_SIGMA = 3.0

#: Q3 - Q1 of a normal sample, in sigma. The constant that turns a measured IQR into
#: a sigma, which is what makes the threshold derived rather than chosen.
IQR_PER_SIGMA = 1.349


def threshold_ms(median_ms: float, iqr_ms: float) -> float:
    """The latency above which this cell is declared REGRESSED. Derived, see above."""
    return median_ms + ALARM_SIGMA * iqr_ms / IQR_PER_SIGMA


def provenance() -> dict:
    """Everything a reader needs to know WHERE a number came from, and nothing that
    identifies WHOSE machine it was."""
    from rola._build_config import BUILD_CONFIG

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "sm": f"sm_{props.major}{props.minor}",
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "ptxas": BUILD_CONFIG["ptxas"],
        "manifest_sha256": BUILD_CONFIG["manifest_sha256"],
        "rola": BUILD_CONFIG["version"],
    }


def iqr(xs) -> float:
    """The same order statistic every recorded arm reports, so a baseline's spread
    and a run's spread are the same quantity."""
    xs = sorted(xs)
    n = len(xs)
    return xs[int(0.75 * n)] - xs[int(0.25 * n)]


#: THE COMMITTED CHOICE. Device events, because the quantity every published claim
#: is about is the kernel's own duration, and because it is the instrument the
#: campaign's banked numbers were actually taken with -- that recipe lived only in a
#: workflow document, so the tree and the record measured different quantities. It
#: is one name in one place: changing it is a re-measurement of every committed
#: baseline, which is what it should be.
CANONICAL_INSTRUMENT = "cuda_events"


def same_instrument(a: str, b: str, what: str) -> None:
    """Refuse to compare two numbers taken with different stopwatches.

    Not a warning. The gap between the instruments is not noise -- it is the launch
    and sync cost one of them includes -- so a comparison across them reports that
    difference as though it were a property of the kernel. The same refusal covers
    ``ncu`` durations, which are taken under profiling conditions (replay, flushed
    caches, base clocks) that no wall-clock instrument reproduces.
    """
    if a != b:
        raise ValueError(
            f"{what}: refusing to compare a number taken with instrument {a!r} "
            f"against one taken with {b!r}. These are different quantities, not the "
            f"same quantity measured twice. Re-measure the older side with the "
            f"current instrument ({CANONICAL_INSTRUMENT!r}) and record it as a new "
            f"measurement.")


def check_instrument(baseline_arm: dict, got_source: str, key: str) -> None:
    """Refuse a gate comparison whose two sides were taken with different timers.

    A baseline entry written before the instrument was recorded has no `source`, and
    that is ALSO a refusal: the quantity it holds is unknown, which is exactly the
    state this check exists to end. Re-record it -- a baseline is a measurement.
    """
    was = baseline_arm.get("source")
    if was is None:
        raise ValueError(
            f"{key}'s committed baseline does not say which stopwatch took it, so "
            f"nothing can be compared against it. Re-record the baseline: "
            f"a baseline is a measurement, so re-take it with the current "
            f"instrument ({CANONICAL_INSTRUMENT!r}).")
    same_instrument(was, got_source, f"latency gate for {key}")


def classify(history, candidate, *, paired_diffs=None) -> dict:
    """THE FLAGGING RULE: three gates, and a regression needs all three.

    `history` is the rolling window of clean ledger rows on this fingerprint,
    oldest first; `candidate` is the row being judged. `paired_diffs` are the
    per-round differences when the candidate came from an interleaved A/B -- the
    significance gate is `insufficient_data` without them rather than silently
    unpaired, because an unpaired test on paired data throws away the design.

    1. EFFECT SIZE -- the candidate's median above :func:`threshold_ms`, derived
       from the window's own spread. Never a flat percentage.
    2. SIGNIFICANCE -- :func:`bench.stats.paired_verdict` at its stated alpha.
    3. DEBOUNCE -- :func:`bench.stats.classify_flags` over the window's own
       violations plus this one: a single unlucky run is not a regression on a box
       with logged power capping.

    The verdict is `regression` only when the effect-size gate fires AND the
    significance gate says the arms differ AND the debounce says it persisted.
    Anything else is reported as what it is, never as a pass.
    """
    from bench.stats import classify_flags, paired_verdict

    window = list(history)
    if len(window) < 3:
        return {"verdict": "insufficient_data", "n_history": len(window),
                "reason": "fewer than 3 clean rows on this fingerprint"}

    medians = [statistics.median(r["values"]) for r in window]
    base_median, base_iqr = statistics.median(medians), iqr(medians)
    limit = threshold_ms(base_median, base_iqr)
    got = statistics.median(candidate["values"])

    violations = [m > limit for m in medians] + [got > limit]
    debounce = classify_flags(violations)
    significance = (paired_verdict(paired_diffs) if paired_diffs is not None
                    else {"verdict": "insufficient_data", "p": None,
                          "reason": "no paired rounds recorded for this candidate"})

    verdict = "no_regression"
    if got > limit:
        if significance["verdict"] == "b_slower" and debounce == "regression":
            verdict = "regression"
        else:
            verdict = "flagged_not_confirmed"
    elif debounce == "suspicious":
        verdict = "suspicious"

    return {"verdict": verdict, "median_ms": got, "limit_ms": limit,
            "baseline_median_ms": base_median, "baseline_iqr_ms": base_iqr,
            "n_history": len(window), "debounce": debounce,
            "significance": significance}


# ---------------------------------------------------------------------------
# CLOCK-SETTLING PROTOCOL (item 0b)
# ---------------------------------------------------------------------------

def _nvidia_smi_clock_mhz(idx: int, field: str) -> float:
    """One ``nvidia-smi`` clock field, in MHz, for device ``idx``.

    Raises rather than returning a sentinel on failure: a settle routine that
    cannot see the clock has to say so (`settle_clocks` catches this and reports
    it), not proceed as though the box were settled.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
         f"--id={idx}"],
        capture_output=True, text=True, timeout=5, check=True)
    return float(out.stdout.strip().splitlines()[0])


def settle_clocks(
    *,
    #: Consecutive `clocks.sm` samples that must all agree before the box counts
    #: as settled. JUDGMENT, not measured: chosen as "a handful of samples", not
    #: swept against alternatives. `0a0b-instrument.md`'s own verification runs
    #: (5 back-to-back gate invocations, 2026-08-01) settled in 0.9-1.2s at this
    #: value every time, so it was never the binding constraint there -- a smaller
    #: value was not shown to be unsafe, this one just was not tightened further.
    hold_polls=4,
    #: MEASURED, on this box (`0a0b-instrument.md`, 2026-08-01): a sustained fp16
    #: matmul loop at 100% utilization holds 1710-1740 MHz continuously
    #: (power/thermal-limited, not still ramping) -- `extraction.md`'s independent
    #: "forcing 1740 MHz by hand" note agrees. This box idles at ~210-240 MHz
    #: before the burn starts; the actual gate runs this settles
    #: (`test_latency_regression.py -m bench`, 5 back-to-back invocations,
    #: 2026-08-01) latched at 1845-1860 MHz -- a different, slightly higher
    #: plateau than the isolated-burn measurement, not reconciled further here.
    #: 30 MHz brackets both plateaus' observed jitter without being so tight the
    #: poll never latches; not derived from a formal noise-floor measurement, so
    #: treat as judgment inside an empirically-set order of magnitude.
    plateau_tol_mhz=30.0,
    #: JUDGMENT: fast enough that `hold_polls` samples land in ~0.6s (matching the
    #: 0.9-1.2s settle times `0a0b-instrument.md` actually measured), slow enough
    #: not to hammer `nvidia-smi` as a subprocess launch every tick. Not swept.
    poll_interval_s=0.15,
    #: JUDGMENT, a generous ceiling: every measured settle in
    #: `0a0b-instrument.md`'s verification runs (5 back-to-back gate invocations,
    #: 2026-08-01) finished in 0.9-1.2s, roughly 10x under this bound. Sized to
    #: fail loudly on a genuinely stuck clock rather than to bound the common case.
    timeout_s=12.0,
    #: JUDGMENT: large enough that the `a @ b` burn loop keeps the SM saturated
    #: between `nvidia-smi` polls (the failure mode this exists to avoid is the
    #: clock relaxing between bursts -- see the "stop/query/restart" note below),
    #: not picked from a sweep of matmul sizes against settle time or power draw.
    matmul_n=4096,
) -> dict:
    """Ramp this box to a steady SM clock before ANY latency in this process is
    timed -- the fix for the gate defect `extraction.md` "A GATE DEFECT FOUND ON THE
    WAY" recorded rather than worked around.

    **The defect.** The box idles at ~210-240 MHz of a ~2100 MHz rated max. The
    baseline RECORDER (`bench_regression.py`'s 35 reps * 5 blocks * 7 processes)
    ramps the clock as an incidental side effect of doing that much work; the
    per-commit GATE (`test_latency_regression.py`, 11 reps, one process) does not do
    enough work to ramp it the same way. The two sides of the comparison were
    therefore starting from different clock states, and the two sub-millisecond
    wave-quantized `topo-*` cells (`waves_per_multiprocessor = 0.53`, 128 CTAs over
    80 SMs) are narrow enough for that gap alone to read as a regression.

    **The target is a PLATEAU, not a fraction of the rated max.** The rated max
    clock (`clocks.max.sm`) is a boost ceiling this card does not sustain under
    continuous compute -- measured here at 100 % utilization it settles at
    1710-1740 MHz, ~82 % of the 2100 MHz rating, and holds there (power/thermal
    limited, not still ramping); `extraction.md`'s own "forcing 1740 MHz by hand"
    note independently measured the same ceiling. A threshold expressed as a
    fraction of `clocks.max.sm` is therefore unreachable and times out on every
    call. The correct target is: the clock has STOPPED CHANGING, wherever it stops
    -- so this polls `clocks.sm` while a continuous background load runs, and calls
    it settled once the last `hold_polls` samples all fall within
    `plateau_tol_mhz` of each other.

    **The fix.** A background thread keeps issuing matmuls on the SAME device the
    caller is about to time -- continuously, not start/stop per poll, because a
    stop/query/restart cycle lets the clock relax between bursts and never
    reflects the steady state the timed reps will actually run under. The main
    thread polls `nvidia-smi` (a separate process; it does not contend with the
    GIL the burn thread holds while launching kernels) until the plateau
    criterion holds or `timeout_s` elapses.

    **Reported, never silently skipped.** A box without `nvidia-smi`, or one whose
    query fails, gets ``settled: False`` and a ``reason`` in the returned dict
    instead of the routine pretending it settled a clock it never measured. Callers
    print this dict so a flaky run says whether settling even ran, rather than
    leaving the clock as an unstated variable the way the pre-existing gate did.
    """
    result: dict = {"settled": False}
    if not torch.cuda.is_available():
        result["reason"] = "no cuda device"
        return result
    idx = torch.cuda.current_device()
    if shutil.which("nvidia-smi") is None:
        result["reason"] = "nvidia-smi not found"
        return result
    try:
        max_mhz = _nvidia_smi_clock_mhz(idx, "clocks.max.sm")
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
        result["reason"] = f"could not query clocks.max.sm: {exc}"
        return result

    a = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    b = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    stop = threading.Event()

    def _burn() -> None:
        while not stop.is_set():
            a @ b  # noqa: B018 -- async launch is the point; the result is unused

    burner = threading.Thread(target=_burn, daemon=True)
    burner.start()
    try:
        samples: list[float] = []
        t0 = time.monotonic()
        cur = 0.0
        while time.monotonic() - t0 < timeout_s:
            time.sleep(poll_interval_s)
            try:
                cur = _nvidia_smi_clock_mhz(idx, "clocks.sm")
            except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
                result["reason"] = f"could not query clocks.sm: {exc}"
                return result
            samples.append(cur)
            window = samples[-hold_polls:]
            if len(window) >= hold_polls and max(window) - min(window) <= plateau_tol_mhz:
                result.update(settled=True, sm_mhz=cur, max_mhz=max_mhz,
                              plateau_mhz=round(max(window) - min(window), 1),
                              elapsed_s=round(time.monotonic() - t0, 3))
                return result
        result.update(reason="timeout", sm_mhz=cur, max_mhz=max_mhz,
                      elapsed_s=round(time.monotonic() - t0, 3))
        return result
    finally:
        stop.set()
        burner.join(timeout=2.0)
        torch.cuda.synchronize()
