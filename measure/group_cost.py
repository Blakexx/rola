# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GROUP COST MODEL, STRUCTURAL AND CARVE-AWARE: per warp and window, what a sparse side's step form costs when the
warp's sparse digits step in aligned groups of g (a group's tokens the union of its digits' lists, every digit's MMAs
over that union; g = 1 per list, g = all merged), as `common/cost.cuh` prices it -- the larger of the MMA cycles on the
busiest scheduler and the shared-memory cycles on the SM. Every count comes from the carve's shape and the bytes a step
moves at a conflict-free layout, none from a kernel: a step of 16 tokens issues (dense digits / 16) x (channels / 8)
MMAs a digit; the fold loads the tokens' dense coefficients (f16), values (f16) and the group's sparse bytes, the
readout their dense read bytes (u8) and the group's sparse coefficients, and adds 16 tokens x channels of s16 into its
panel by atomics. The rates are a product's or an architecture's, from `common/hardware.json`. Each carve priced on
the probe cells' own lists, per head (every CTA of the head, a sparse slice repeated over the dense slices and channel
groups): per list, merged, and the best g a warp.

    python -m measure.group_cost
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from measure import hardware

WARPS, STEP, WINDOW = 8, 16, 512


@dataclass(frozen=True)
class Phase:
    """A phase's warp: `sparse` digits (lists), `dense` digits of the other level, `channels`; its MMA kind; whether it
    adds into a shared panel."""
    name: str
    sparse: int
    dense: int
    channels: int
    mma: str
    panel: bool

    @property
    def mmas(self) -> int:
        """MMAs a digit a step: m16n8k16 over the dense digits and the channels."""
        return self.dense // 16 * (self.channels // 8)

    @property
    def wavefronts(self) -> int:
        """Shared-memory load wavefronts a group step: the 16 tokens' operand bytes, a load at least one."""
        if self.panel:
            return math.ceil(STEP * self.dense / 128) + 1
        return math.ceil(STEP * self.dense * 2 / 128) + math.ceil(STEP * self.channels * 2 / 128) + 1

    @property
    def atomics(self) -> int:
        """The panel add's atomic instructions a group step (16 tokens x channels as s16 pairs, 128 bytes each)."""
        return STEP * self.channels * 2 // 128 if self.panel else 0


@dataclass(frozen=True)
class Carve:
    name: str
    fold: Phase
    readout: Phase


CARVES = (
    Carve("32 x 32", Phase("fold", 4, 32, 32, "hmma_f16_f32", False),
          Phase("readout", 4, 32, 32, "hmma_f16_f16", True)),
    Carve("16 x 64", Phase("fold", 8, 16, 32, "hmma_f16_f32", False),
          Phase("readout", 2, 64, 32, "hmma_f16_f16", True)),
)


def lists(name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """The cell's sparse sides' live masks `[tokens, digits]` -- level S's reads (the readout's lists) and level R's
    writes (the fold's) -- and its value width."""
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, _, _, statics = bundle_of(spec, realize(spec, H=1))
    level_r = facts_ops.list_level(statics[1])
    read = (routes.amp_read[1 - level_r][0, :, 0, :] != 0).cpu().numpy()
    write = (routes.amp_write[level_r][0, :, 0, :] != 0).cpu().numpy()
    del routes
    torch.cuda.empty_cache()
    return read, write, spec.dv


def groups(phase: Phase) -> list[int]:
    return [1 << i for i in range(phase.sparse.bit_length()) if (1 << i) <= phase.sparse]


def steps_by_group(phase: Phase, live: np.ndarray) -> dict[int, np.ndarray]:
    """For each g, `[windows, slices, warps]` group steps: each group's union of live tokens a window over 16, summed
    over the warp's groups."""
    tokens, digits = live.shape
    w = live[: tokens // WINDOW * WINDOW].reshape(tokens // WINDOW, WINDOW, digits // (WARPS * phase.sparse), WARPS,
                                                  phase.sparse)
    out = {}
    for g in groups(phase):
        union = w.reshape(*w.shape[:4], phase.sparse // g, g).any(axis=-1).sum(axis=1)
        out[g] = np.ceil(union / STEP).astype(np.int64).sum(axis=-1)
    return out


def price(phase: Phase, live: np.ndarray, rates: dict[str, hardware.Rate], dense_width: int,
          dv: int) -> dict[str, tuple[float, dict[int, float]]]:
    """Each strategy's cycles a head a window (every CTA) and its share of warp-windows at each g."""
    mma = rates[f"mma.{phase.mma}"].cycles
    atomic = max(rates["atomic"].cycles, rates["atomic_wavefront"].cycles)
    by_g = steps_by_group(phase, live)
    tensor = {g: s * g * phase.mmas * mma for g, s in by_g.items()}
    smem = {g: s * (phase.wavefronts * rates["wavefront"].cycles + phase.atomics * atomic) for g, s in by_g.items()}
    schedulers = int(rates["schedulers"].cycles)
    repeats = dense_width // phase.dense * (dv // phase.channels)
    windows = next(iter(by_g.values())).shape[0]

    def head(t: np.ndarray, s: np.ndarray) -> float:
        per_scheduler = t.reshape(*t.shape[:2], WARPS // schedulers, schedulers).sum(axis=2).max(axis=-1)
        return float(np.maximum(per_scheduler, s.sum(axis=-1)).sum() * repeats / windows)

    out = {f"g{g}": (head(tensor[g], smem[g]), {g: 1.0}) for g in by_g}
    own = np.stack([np.maximum(tensor[g], schedulers * smem[g]) for g in by_g])
    pick = own.argmin(axis=0)
    gs = list(by_g)
    chosen_t = np.choose(pick, [tensor[g] for g in gs])
    chosen_s = np.choose(pick, [smem[g] for g in gs])
    out["best"] = (head(chosen_t, chosen_s), {g: float((pick == k).mean()) for k, g in enumerate(gs)})
    return out


CELLS = ("flagship-alt-k4-L16384", "flagship-alt-k16-L16384", "flagship-cohort-k16-L16384", "flagship-cohort-k4",
         "nl16k-alt-k4")
RATES = (("rtx3080ti", None), (None, "sm_86"), (None, "sm_80"))


def main() -> None:
    from measure.cells import by_name

    for carve in CARVES:
        for phase in (carve.fold, carve.readout):
            print(f"{carve.name} {phase.name}: a warp {phase.sparse} x {phase.dense} x {phase.channels}, "
                  f"{phase.mmas} MMAs a digit-step, {phase.wavefronts} load wavefronts + {phase.atomics} atomics "
                  f"a group step, g in {groups(phase)}")
    for name in CELLS:
        read, write, dv = lists(name)
        widths = by_name(name).widths
        print(f"\n{name}: lists a digit a window, reads {read.mean() * WINDOW:.1f}, writes {write.mean() * WINDOW:.1f}")
        for product, arch in RATES:
            rates = hardware.resolve(arch, product)
            for phase_name in ("fold", "readout"):
                row = []
                for carve in CARVES:
                    phase = getattr(carve, phase_name)
                    live = read if phase.panel else write
                    result = price(phase, live, rates, widths[0], dv)
                    best, share = result["best"]
                    picks = " ".join(f"g{g} {s:.0%}" for g, s in share.items() if s >= 0.005)
                    row.append(f"{carve.name}: per list {result['g1'][0] / 1e3:7.1f}K  merged "
                               f"{result[f'g{phase.sparse}'][0] / 1e3:7.1f}K  best {best / 1e3:7.1f}K ({picks})")
                print(f"  {product or arch:10s} {phase_name:8s} " + "  |  ".join(row))


if __name__ == "__main__":
    main()
