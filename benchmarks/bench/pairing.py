"""HOW it is measured: the interleaved paired-arm engine. **THIS IS THE METHODOLOGY.**

Every performance number the chunk arm publishes comes through here, so the rules are
implemented once rather than repeated in each driver's docstring where they drift.

**1. THE ARMS ALTERNATE REP BY REP.** Never arm-A-block then arm-B-block. A block layout
lets a drift step -- a clock change, a thermal step, another process arriving -- land
BETWEEN the arms, where it is indistinguishable from the effect being measured. Measured
on this box by the retired harness, the bias was real: sequential blocks read 3.27-3.30x
where interleaving read 3.33-3.35x, and interleaving cut the run-to-run spread by ~7x.
:meth:`PairedResult.diffs` REFUSES a sample order that is not strictly alternating, so
the design cannot be thrown away by a driver that collected its samples the other way.

**2. THE REPORTABLE STATISTIC IS THE MEDIAN OF THE PER-PAIR RATIOS**, each pair being two
measurements adjacent in time -- not the ratio of the medians, which discards the pairing
that was paid for. The SIGNIFICANCE test runs on the per-ROUND differences, which is the
same design read at block grain.

**3. ABSOLUTE TIMES AND IQRs ARE REPORTED ALONGSIDE, NEVER INSTEAD.** A ratio without its
spread is a claim without an error bar.

**4. A CELL WHOSE IQR CANNOT DISCRIMINATE THE EFFECT BEING CLAIMED IS REPORTED AS A RANGE
OR DROPPED, AND WHICH WAS DONE IS STATED.**

**5. ONE STOPWATCH, AND ITS NAME TRAVELS WITH EVERY NUMBER.** The timer is selected from
:data:`INSTRUMENTS` by name; `bench.regression.same_instrument` refuses a comparison
across two of them.

**6. WARMUP IS A PRECONDITION, NOT A COURTESY.** Fewer than :data:`WARMUP_LAUNCHES`
discarded launches per arm is a refusal: the first launches build the kernel's schedule
and warm the instruction cache, and a median that contains them is a median about module
load. There is no flag that lowers it below the floor.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from bench.regression import CANONICAL_INSTRUMENT, iqr


def _elapsed_cuda_events(call):
    """Device-side elapsed time between two events straddling the launch.

    Both syncs are taken: the first so the start event is not queued behind unrelated
    work, the last because `elapsed_time` is defined only once both events completed.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _elapsed_perf_counter(call):
    """Host wall time around a full device sync on both sides -- a strictly larger
    quantity than the event interval, and a different one."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


#: THE INSTRUMENT REGISTRY. A timer is selected BY NAME and the name is recorded beside
#: the number, so an instrument cannot enter the record without a name to record.
INSTRUMENTS = {
    "cuda_events": _elapsed_cuda_events,
    "perf_counter": _elapsed_perf_counter,
}

#: Launches discarded before any is recorded, per arm. The campaign recipe's count.
WARMUP_LAUNCHES = 10

#: Reps per block. The block median is the unit the reported spread is taken over, so it
#: is odd: an even count would interpolate two samples into a value neither was.
REPS_PER_ROUND = 11


@dataclass(frozen=True)
class Arm:
    """One thing to time: a name and a zero-argument callable that issues it."""

    name: str
    call: object


@dataclass
class PairedResult:
    """The samples of one interleaved A/B, with the ORDER they were taken in.

    `order` is the load-bearing field. It is what makes the pairing checkable by a reader
    of the result rather than a property of the loop that produced it.
    """

    a: Arm
    b: Arm
    order: list = field(default_factory=list)
    a_times: list = field(default_factory=list)
    b_times: list = field(default_factory=list)
    rounds: int = 0
    reps_per_round: int = 0
    warmup: int = 0
    source: str = CANONICAL_INSTRUMENT

    def _refuse_unpaired(self) -> None:
        if len(self.a_times) != len(self.b_times):
            raise ValueError(
                f"unpaired samples: {len(self.a_times)} of {self.a.name!r} against "
                f"{len(self.b_times)} of {self.b.name!r}. A paired statistic over "
                f"unequal arms is not a weaker answer, it is a different one.")
        expect = ["a", "b"] * len(self.a_times)
        if self.order != expect:
            raise ValueError(
                "REFUSING a paired statistic: these samples were not taken by "
                "alternating the arms rep by rep. A block layout lets a drift step land "
                "between the arms, where it is indistinguishable from the effect -- "
                "measured here as a 7x wider run-to-run spread. Re-run through "
                "`interleaved_ab`; do not re-order the samples afterwards.")

    def ratios(self) -> list:
        """The per-pair ratios `b/a`, each pair two measurements adjacent in time."""
        self._refuse_unpaired()
        return [y / x for x, y in zip(self.a_times, self.b_times)]

    def _blocks(self, times) -> list:
        n = self.reps_per_round
        return [statistics.median(times[i:i + n]) for i in range(0, len(times), n)]

    def diffs(self) -> list:
        """The per-ROUND differences `median(b) - median(a)` -- the quantity
        `bench.stats.paired_verdict` consumes. A drift step inside a round is common to
        both arms and cancels here, which is the reason the pairing was paid for."""
        self._refuse_unpaired()
        return [y - x for x, y in zip(self._blocks(self.a_times), self._blocks(self.b_times))]

    def summary(self) -> dict:
        ratios = self.ratios()
        blocks_a, blocks_b = self._blocks(self.a_times), self._blocks(self.b_times)
        return {"a": self.a.name, "b": self.b.name, "source": self.source,
                "a_median_ms": statistics.median(blocks_a), "a_iqr_ms": iqr(blocks_a),
                "b_median_ms": statistics.median(blocks_b), "b_iqr_ms": iqr(blocks_b),
                "paired_ratio": statistics.median(ratios), "paired_iqr": iqr(ratios),
                "rounds": self.rounds, "reps_per_round": self.reps_per_round,
                "warmup": self.warmup}


def _require_warmup(warmup: int) -> None:
    if warmup < WARMUP_LAUNCHES:
        raise ValueError(
            f"REFUSING to measure with {warmup} warmup launches; the floor is "
            f"{WARMUP_LAUNCHES}. The first launches build the kernel's schedule and warm "
            f"the instruction cache, so a median that contains them is a median about "
            f"module load, not about the kernel.")


def warm(arm: Arm, *, warmup: int = WARMUP_LAUNCHES) -> None:
    """Discard `warmup` launches of one arm. The floor is enforced here, once."""
    _require_warmup(warmup)
    for _ in range(warmup):
        arm.call()
    torch.cuda.synchronize()


def measure_arm(arm: Arm, *, rounds: int = 8, reps_per_round: int = REPS_PER_ROUND,
                warmup: int = WARMUP_LAUNCHES,
                instrument: str = CANONICAL_INSTRUMENT) -> dict:
    """ONE arm's absolute times: the BASELINE mode, and the only unpaired one.

    A committed baseline is a per-cell latency, not a ratio, so it is measured alone --
    and it is compared only against another baseline of the same cell taken with the same
    stopwatch, which is what `bench.regression.check_instrument` enforces. Samples are
    returned RAW and in run order; the median and IQR are over the BLOCK medians, so the
    spread the threshold is derived from is the spread of the quantity being compared.
    """
    _require_warmup(warmup)
    timer = INSTRUMENTS[instrument]
    warm(arm, warmup=warmup)
    times = []
    for _ in range(rounds * reps_per_round):
        t, _ = timer(arm.call)
        times.append(t)
    blocks = [statistics.median(times[i:i + reps_per_round])
              for i in range(0, len(times), reps_per_round)]
    return {"arm": arm.name, "source": instrument, "values": times, "blocks": blocks,
            "median_ms": statistics.median(blocks), "iqr_ms": iqr(blocks),
            "rounds": rounds, "reps_per_round": reps_per_round, "warmup": warmup}


def interleaved_ab(a: Arm, b: Arm, *, rounds: int = 8,
                   reps_per_round: int = REPS_PER_ROUND, warmup: int = WARMUP_LAUNCHES,
                   instrument: str = CANONICAL_INSTRUMENT) -> PairedResult:
    """Interleaved paired A/B: rep by rep, both arms, one stopwatch.

    Both arms are warmed before either is recorded, so neither pays the other's first
    launch. `rounds` is what the significance test's power is set by
    (`bench.stats.MIN_ROUNDS`); the driver's tier chooses it.
    """
    _require_warmup(warmup)
    timer = INSTRUMENTS[instrument]
    warm(a, warmup=warmup)
    warm(b, warmup=warmup)
    result = PairedResult(a=a, b=b, rounds=rounds, reps_per_round=reps_per_round,
                          warmup=warmup, source=instrument)
    for _ in range(rounds * reps_per_round):
        ta, _ = timer(a.call)
        result.a_times.append(ta)
        result.order.append("a")
        tb, _ = timer(b.call)
        result.b_times.append(tb)
        result.order.append("b")
    return result
