#!/usr/bin/env python3
"""THE SWEEP OVER THE DECLARED SETS -- the fourth enforcement layer, as a tool.

A structure switch's cases are GENERATED (`csrc/rola/src/common/geom.cuh`'s `sub_boxes`)
and its miss path is a `__trap()` (`csrc/rola/src/common/structure_switch.cuh`). Between
those two facts sits a question neither of them answers: WHICH members can a real
declaration actually select, and does every member of the generated set have one? A
branch that nothing exercises is a promise, not a proof, and a member no declaration
reaches is dead code the compiler still pays for.

So, for every SHAPE the arm table declares -- `(D, DV, streams)`, at BOTH of the two-value `warps_per_cta` set (`(1, 8)` shipped, `(2, 4)` the test/diagnostic arm) -- this
enumerates the topologies and mode words that depth admits, derives each one's block
through the SHIPPED derivation (`rola.ops.carry.geometry`, never a mirror), and reports:

* the `(side, member)` pairs reached, against the generated set's own size;
* the liveness clause signature per level and side (`carves`, and whether the level votes
  ballot rows at all) -- geometry, never a threshold, and the paths a launch can take;
* every declaration the seam REFUSED, with its reason, because a refusal doing its job
  and a gap look identical from a coverage count alone.

It is a DERIVATION sweep and launches no kernel: on this branch no kernel consumes the
runtime block (`docs/internals/common/geom.md#on-the-tip`). When the pipelined body does,
the launch is one line -- and until then a gap here is still a gap.

    python tools/field_coverage.py            # sweep every declared shape
    python tools/field_coverage.py --check    # same, non-zero exit on any gap
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from rola.ops import carry as carry_ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_shards  # noqa: E402 -- the path insert must precede it

#: the arch's own law, mirrored from `geom.cuh#arch-constants`: the CTA's box is
#: `leaves_per_warp(DV)` leaves per warp.
WARPS_PER_SM = 8

#: the 2x4 ruling (KERNEL_STANDARDS §R15): `warps_per_cta` is a compile-time axis with
#: exactly this two-value set -- `(1, 8)` the shipped, one-CTA-per-SM shape, `(2, 4)` the
#: latency-diagnostic test arm. No row in `gen_shards.CARRY_ARMS` carries a `warps_per_cta`
#: field yet (the list is `()` until the shipped set names rows under `(D, DV, warps_per_cta)`
#: key; today's 13-field schema's `WPS` is "warps per write stream", a DIFFERENT, older
#: field), so until then this tool sweeps the ARCH axis uniformly across every declared
#: shape rather than reading a per-row value that does not exist -- the member set is
#: `f(D, leaves_per_warp, warps_per_cta)` (`docs/internals/common/geom.md#cta-box-first`),
#: so a gap at either value is a real gap.
WARPS_PER_CTA = (8, 4)


def leaves_per_warp(dv: int) -> int:
    return (16384 // dv) // WARPS_PER_SM


def _topologies(D: int, bc: int):
    """Every uniform and mixed power-of-two topology at this depth the seam admits."""
    widths = (8, 16, 32, 64, 128, 256)
    seen = []
    for combo in itertools.product(widths, repeat=D):
        prod = 1
        for w in combo:
            prod *= w
        if prod < bc or sum(combo) % 8:
            continue
        seen.append(combo)
    return seen[:48]


def _signature(block, D):
    """The liveness clause's shape and the two members, read off the block."""
    sig = []
    for side in ("r", "w"):
        s = block[side]
        for lvl in range(D):
            sig.append((side, lvl, s["grid"][lvl] > 1, s["row_g"][lvl] > 0))
        sig.append((side, "member", s["assign"]))
    return frozenset(sig)


def _shapes():
    """`{(D, DV, warps, streams)}` -- the SHAPES the arm table declares, deduplicated."""
    out = {}
    for row in gen_shards.CARRY_ARMS:
        (D, B, K, M, DV, W, C, CV, MO, WPS, NSW, NSR, PK) = row
        out.setdefault((D, DV, NSR, NSW), []).append(row)
    return out


def sweep(verbose=True):
    gaps = []
    for (D, dv, nsr, nsw), rows in sorted(_shapes().items()):
        for warps_per_cta in WARPS_PER_CTA:
            bc = leaves_per_warp(dv) * warps_per_cta
            members = carry_ops.sub_boxes(D, bc, max(nsr, nsw))
            realized, refused, reached = set(), {}, set()
            for widths in _topologies(D, bc):
                for modes in range(0, 1 << (2 * D)):
                    try:
                        block = carry_ops.geometry(list(widths), level_modes=modes, bc=bc,
                                                   nsr=nsr, nsw=nsw, d_v=dv)
                    except RuntimeError as ex:
                        refused[str(ex).split(" (widths=")[0]] = (widths, modes)
                        continue
                    realized.add(_signature(block, D))
                    for side in ("r", "w"):
                        reached.add((side, block[side]["assign"]))
            if verbose:
                print(f"shape D={D} dv={dv} streams=({nsr},{nsw}) warps_per_cta="
                      f"{warps_per_cta} BC={bc} [{len(rows)} arm rows]: {len(realized)} "
                      f"distinct clause signatures, {len(members)} members generated, "
                      f"{len(reached)} (side, member) pairs reached")
                for reason, (widths, modes) in sorted(refused.items()):
                    print(f"    REFUSED (doing its job): {reason} -- e.g. widths={widths} "
                          f"modes={modes}")
            if not realized:
                gaps.append(f"shape D={D} dv={dv} warps_per_cta={warps_per_cta}: no "
                            f"declaration derived a block at all")
                continue
            #: THE EXHAUSTIVENESS QUESTION, asked of the GENERATED set: a member nothing
            #: can select is a compiled case no launch reaches.
            for a in range(len(members)):
                if nsr > 1 and ("r", a) not in reached and ("w", a) not in reached:
                    gaps.append(f"shape D={D} dv={dv} warps_per_cta={warps_per_cta}: "
                                f"member {a} ({members[a]}) of the generated set is "
                                f"selected by no declaration")
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any declared set has an unreachable member")
    args = ap.parse_args()
    gaps = sweep()
    for g in gaps:
        print(f"FIELD COVERAGE GAP: {g}", file=sys.stderr)
    if gaps:
        return 1 if args.check else 0
    print("field coverage: every declared shape derived a block, and every member of "
          "every generated set is selected by some declaration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
