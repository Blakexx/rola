"""Page commitment: the measurement that justifies the feature.

Two questions, and they are different questions.

**1. Does committed state scale with realized usage or with `N`?** Run a topology
where `N` is far larger than the owner set any step actually writes -- a wide
mixed radix with a genuinely sparse (exact-entmax) write router -- and report
resident MiB against the dense `[BH, N, cols]` counterfactual. If resident MiB
tracks `N`, paging has bought nothing; the claim is that it tracks the REALIZED
owner count instead.

**2. What does paging cost where it can buy nothing?** The flagship cells are
dense: every owner is live, so the page set is the whole page domain and the
resident footprint is exactly the dense one. Paging must cost ~nothing there, and
"must" is not a measurement.

That cost is reported **in isolation** (`plan_cold_ms` / `plan_warm_ms`: the
`prepare -> publish -> admit-mask` plane on its own), not only as a whole-layer
delta. Measured on the first pass, the whole-layer delta at the dense flagship
cell ranged from `-12.7%` to `+37.1%` across four cells -- a spread with a
NEGATIVE member, i.e. the quantity is launch-noise dominated at a 5-7 ms step and
the delta does not measure paging. The isolated plane is a real number; the delta
is reported beside it and should be read as bounded by it, not as an independent
result.

**Cold vs warm.** A FRESH state plans cold -- full reset, replan, republish --
and every continuation on it plans warm (`fresh=False`: no reset, so a repeat of
the same routing has zero new pages to commit). Both are what the shipped surface
runs, at different points in a sequence's life: the first call of a sequence pays
cold, every step after it pays warm. An
earlier revision of this benchmark measured only the warm number in a no-reset
loop and reported it as "the" paging cost; the two differ by 1.13x at the dense
flagship cell and by **4.74x** at the wide sparse cell (measured: 6.04 ms cold
vs 1.28 ms warm). Both are reported here, labelled, and neither is quoted alone.

The sparse level is a SYNTHETIC stand-in for training. `--logit-gain` sharpens an
untrained exact-entmax router into a real sparse support: the support is genuine
(exact zeros), its DEGREE is not a trained quantity and is labelled as
such wherever it is reported. Same convention, and the same caveat, as
`benchmark_rola_v3_layer.py`.

**REPOINTED.** Paging is not a layer configuration any more: the backing
belongs to the STATE (`rola.state()`), it is ON by default, and it engages when a
call carries a state at all. The two timed cells are therefore two STATEFUL calls --
one on a default (paged) state, one on a state pinned dense with
`PlanOverrides(paging=False)` -- which is the comparison the old paged/unpaged layer
pair was reaching for, with both arms now doing the same state I/O. The isolated
plane is `PageArena.plan_exact` over the kernel's OWN exact atom bitmap
(`rola.engine.facts.planes.atom_bits`), which is what the facade plans with.

Vocabulary: `N` = leaf count = `prod_l width_l`; `BC` = leaves per owner;
`owners_total` = `N / BC`; `BH` = `batch * heads`; `cols` = `d_v` plus one mass
column (the ratio readout's denominator).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from rola import LayerContinuation, RoLA, RouteProducer, dense_routing, union_routing
from rola import state as rola_state
from rola.engine.facts import planes
from rola.engine.facts.call import arm_of, widths_of
from rola.expert import PlanOverrides

_MiB = 1024 * 1024


def _median_ms(fn, *, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _layer(*, hidden, H, d_v, branches, router, decay, dense_pin=False, seed=0):
    from rola import LearnedDecay

    template = union_routing(1, 1.5) if router == "entmax" else dense_routing(1)
    torch.manual_seed(seed)
    levels = [template.at(w) for w in branches]
    routes = RouteProducer(levels, hidden_size=hidden, num_heads=H)
    decay_source = LearnedDecay(widths=routes.widths, num_heads=H) if decay else None
    layer = RoLA(
        routes, d_v=d_v, decay=decay_source, layer_idx=0,
        expert=PlanOverrides(paging=False) if dense_pin else None,
    ).cuda().eval()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def measure_residency(*, B, L, H, d_v, branches, router, logit_gain, decay,
                      reps, warmup) -> dict:
    hidden = H * d_v
    layer = _layer(hidden=hidden, H=H, d_v=d_v, branches=branches, router=router,
                   decay=decay)
    if logit_gain != 1.0:
        with torch.no_grad():
            layer.routes.route_W.mul_(logit_gain)
    x = torch.randn(B, L, hidden, device="cuda")

    def step(module, carried):
        # `torch.no_grad()` is load-bearing, not hygiene: a grad-enabled forward is
        # REFUSED. This measures the inference path, which is the path paging exists
        # on and the only one there is.
        def run():
            with torch.no_grad():
                module(x, continuation=LayerContinuation(state=carried))
        return run

    paged_state = rola_state()
    step(layer, paged_state)()
    assert layer.last_execution_backend == "cuda", (
        f"the paged cell took the {layer.last_execution_backend!r} backend, so this row "
        "measures the wrong kernel")
    assert paged_state.paged, (
        "the state is dense-backed; there is no paged cell to measure here")
    stats = paged_state._arena.last_result

    # The paging plane ALONE: `prepare -> publish -> admit mask` against the exact
    # schedule this cell's routing produces. This is the number that answers "what
    # does paging cost", because the whole-layer delta at these sizes is dominated
    # by launch noise (see the module docstring).
    arena = paged_state._arena
    #: The bitmap comes from the PRIMITIVE rather than from a plan: this bench has no
    #: value stream of its own -- the layer builds `v` internally -- so there is no
    #: chunk call to plan here, and `atom_bits` is a fact about the write side alone.
    #: Same launch the DAG's own node issues, on the same packed write plane.
    with torch.no_grad():
        routes = layer.routes(x)
        bitmap = planes.written_atoms(
            planes.atom_bits(planes.pack_side(routes.write), widths_of(routes)))

    # COLD is what the FIRST call of a sequence pays (full reset, replan, republish);
    # WARM is what every continuation pays (`fresh=False`, so a repeat of the same
    # routing commits nothing new). Both are shipped costs, at different points in a
    # sequence's life, and neither is quotable alone: they differed by 4.74x at the
    # wide sparse cell when this was measured.
    def plan_cold():
        with torch.no_grad():
            arena.plan_exact(bitmap)             # fresh=True (default): full reset + replan

    def plan_warm():
        with torch.no_grad():
            arena.plan_exact(bitmap, fresh=False)  # a continuation: no reset

    # `plan_warm` must start from a resident arena, or its own first call would be
    # doing the same new-page work `plan_cold` measures. One untimed cold call
    # establishes residency; every `fresh=False` call after commits nothing new.
    arena.plan_exact(bitmap)
    plan_warm_ms = _median_ms(plan_warm, reps=reps, warmup=warmup)
    plan_cold_ms = _median_ms(plan_cold, reps=reps, warmup=warmup)

    paged_ms = _median_ms(step(layer, paged_state), reps=reps, warmup=warmup)
    unpaged = _layer(hidden=hidden, H=H, d_v=d_v, branches=branches, router=router,
                     decay=decay, dense_pin=True)
    # `load_state_dict` carries the scaled `route_W` across, so both cells route
    # identically and the timing difference is the BACKING and nothing else.
    unpaged.load_state_dict(layer.state_dict())
    dense_state = rola_state()
    step(unpaged, dense_state)()
    assert unpaged.last_execution_backend == "cuda"
    assert not dense_state.paged, "the dense pin did not reach the state"
    unpaged_ms = _median_ms(step(unpaged, dense_state), reps=reps, warmup=warmup)

    N = layer.topology.N
    #: THE SAVINGS CLAIM IS THE DRIVER'S, AND IT IS ONLY MEANINGFUL WITH ITS POOL SIZE.
    #: `allocated_*` is what the arena handed out; `committed_*` is what the CUDA driver
    #: actually mapped, and the second is the number that states a saving. A pool smaller
    #: than one allocation granule (2 MiB on this device) commits its whole dense limit in
    #: a single mapping and saves NOTHING, so `dense_MiB` is printed beside every fraction
    #: rather than left implicit (docs/internals/paging/paging.md#granule).
    return {
        "router": router, "logit_gain": logit_gain, "norm": "global",
        "decay": "leaf_mass" if decay else "none",
        #: `BC` is the ARM's, and the arm is a manifest fact: the layer carries no
        #: owner-block pin any more (the `state_block` retirement).
        "branches": list(branches), "N": N, "BC": arm_of(routes).bc,
        "BH": B * H, "L": L,
        "owners_total": stats.owners_total,
        "realized_owners": stats.realized_owners,
        "allocated_pages": stats.allocated_pages,
        "resident_pages": arena.resident_pages,
        "peak_pages": stats.peak_pages,
        "allocated_MiB": stats.allocated_bytes / _MiB,
        "committed_MiB": stats.committed_bytes / _MiB,
        "dense_MiB": stats.dense_bytes / _MiB,
        "allocated_fraction": stats.allocated_fraction,
        "committed_fraction": stats.committed_fraction,
        "plan_cold_ms": plan_cold_ms,
        "plan_warm_ms": plan_warm_ms,
        "paged_ms": paged_ms,
        "unpaged_ms": unpaged_ms,
        "paging_overhead_pct": 100.0 * (paged_ms - unpaged_ms) / unpaged_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-v-dim", type=int, default=64)
    #: Default is the DEEPEST BUILT cell: `8^4 = 4096` leaves, where `N` is far larger
    #: than any step's realized owner set. The paged backing lives on the chunk arm, and
    #: the chunk arm is a BUILT MATRIX -- a topology outside it (`16,16,16,16`, which this
    #: default used to be) falls to the legacy tiled consumer, whose state is dense
    #: whatever the caller prefers, so there is no paged cell to measure at all. The
    #: assert in `measure_residency` says so by name rather than reporting a dense row.
    ap.add_argument("--branches", default="8,8,8,8")
    ap.add_argument("--router", choices=("softmax", "entmax"), default="entmax")
    ap.add_argument("--logit-gain", type=float, default=8.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")

    branches = [int(s) for s in args.branches.split(",")]
    #: TWO DEAD ARMS RETIRED, both older than this benchmark's repoint and both of
    #: which raised on their first cell: raw normalization (host axis deleted) and
    #: a decay-configured layer, which the chunk arm refuses (the decay retirement).
    rows = [measure_residency(
        B=args.batch, L=args.seq_len, H=args.heads, d_v=args.head_v_dim,
        branches=branches, router=args.router, logit_gain=args.logit_gain,
        decay=False, reps=args.reps, warmup=args.warmup)]

    head = rows[0]
    print(f"\n=== page commitment — branches={head['branches']} "
          f"N={head['N']} BC={head['BC']} BH={head['BH']} L={head['L']} "
          f"router={args.router} logit_gain={args.logit_gain} ===")
    for r in rows:
        print(f"decay={r['decay']:9s} norm={r['norm']:7s} | "
              f"atoms {r['resident_pages']:6d}/{r['owners_total'] * r['BH']:6d} | "
              f"allocated {r['allocated_MiB']:8.2f} / committed {r['committed_MiB']:8.2f} "
              f"vs dense {r['dense_MiB']:8.2f} MiB "
              f"({100 * r['committed_fraction']:5.1f}% committed) | "
              f"plan[cold {r['plan_cold_ms']:6.3f} / warm {r['plan_warm_ms']:6.3f}] ms | "
              f"paged {r['paged_ms']:7.3f} vs unpaged {r['unpaged_ms']:7.3f} ms "
              f"({r['paging_overhead_pct']:+5.1f}%)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
