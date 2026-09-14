"""The latency-vs-`L` curve: V3 whole layer against flash attention at matched shape.

The flash-attention baseline at large `L` is the decisive artifact: attention is quadratic in `L` and RoLA is linear in `L` against a
FIXED state, so the comparison is a CURVE and a single cell is not a result. At small
`L` attention wins and that is expected; the number that matters is the crossover and
the memory axis (attention's KV cache grows with `L`, RoLA's committed pages do not).

**What is measured.** One forward under `torch.no_grad()` on CUDA per arm per `L`:

* `v3`   -- `RoLA.forward`, plan-selected per step, paging optional, `P = 1`
           (the primary number: no sequence-parallel split is requested, so the
           consumer runs one block per owner). **THE READ SIDE IS DENSE AND
           THE WRITE SIDE IS ENTMAX-SPARSE by default** (ruling 2026-08-01):
           `candidate_frac ~ 0.2` sparse-both (item 0's cell) is the easy case and
           is not what the freeze criterion is stated at; a
           dense read is the arm the freeze is defined at, because winning on the
           sparse-both case is the trivial one. `--read-router`/`--write-router`
           override the two sides independently (`'softmax'` = dense, `'entmax'` =
           sparse); the default pair (`softmax`, `entmax`) IS the parity cell.
* `v3_sparse_both` -- the SAME arm with both sides forced `entmax`, i.e. exactly item
           0's original cell. Kept as a DIAGNOSTIC row, labelled `cell_role` in the
           output, never conflated with the `v3` parity row: it is the regime where
           the routing leaves ~20% of `(owner, tile)` pairs candidate, which is a
           different (easier) case than the freeze arm.
* `fa`   -- a matched attention layer: `q/k/v` projections, causal
           `scaled_dot_product_attention`, output projection. Same
           `(B, H, L, d_model)`, `d_head = d_model / H = d_v`.

**Local budget note on the dense-read arm.** A dense read walks every `(owner, tile)`
pair instead of the sparse-both cell's ~20%, so `v3` costs roughly 5x `v3_sparse_both`'s
consumer time at the same `L` -- and, at the planner's dense-degenerate path, can cost
far more than 5x in PLAN time (39.03 s of plan time was measured at
`candidate_frac = 1`, `N = L = 65,536`). Local runs of `v3` are therefore sized to
`L <= 16,384` under this task's 8 GiB budget, footprint printed before allocation; the
`L = 65,536` dense-read parity cell is a RENTAL cell (A100-class), as the whole-layer
call at that `L` has always been. If even `L = 16,384` turns out
pathological on the dense-degenerate planner path locally, that number is reported as
measured and stands as the dense-saturation lever's motivation --
it is not worked around.

The V3 forward is additionally decomposed into PLANES (producer / planner / consumer)
on identical inputs, exactly as `benchmark_rola_v3_layer.py` does, so an at-scale row
is attributed rather than just reported.

**Dtype is NOT matched and the mismatch favours attention.** The SDPA FLASH backend
rejects fp32; the V3 consumer accepts only fp32 (`v3_consumer.cu` `TORCH_CHECK`s
`at::kFloat`). So `fa` is timed in bf16 on the FLASH backend -- its fastest real
configuration -- and `v3` in fp32. A second `fa_fp32` row on the EFFICIENT backend
(the fastest fp32-capable SDPA backend) is emitted so the dtype axis is visible
instead of assumed. Which backend produced each row is recorded in the row itself,
never inferred.

**Memory is reported on the same rows as latency**, because the curve's second axis
is the point: `kv_cache_bytes` for attention (analytic, `2 * B * H * L * d_head *
itemsize`, the bytes a decoder would have to keep), against the `committed_bytes`
(measured, the state's own `PagingResult` -- the bytes the CUDA driver MAPPED, which is
the only quantity that states a saving; `allocated_bytes` beside it is what the arena
handed out) and `dense_state_bytes` (analytic, the provisioned `N`). A pool smaller than
one driver allocation granule commits its whole dense limit and saves nothing, which is
why the dense counterfactual is on every row rather than implied. `peak_alloc_gb` is torch's own peak for the whole
forward and is a THIRD, different quantity -- it includes activations and is reported
separately rather than conflated with either.

**OOM is a finding.** Every cell is wrapped: an OOM records a row with `oom` set and
the stage that failed, and the sweep continues at the next arm. Nothing is silently
shrunk.

Usage (invoked BARE -- `main()` takes `gpu_lock()` itself):

    PYTHONPATH=<worktree> python benchmarks/bench_scaling.py \
        --seq-lens 512,1024,2048,4096,8192 --json out.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from rola import LayerContinuation, RoLA, RouteProducer, union_routing
from rola import state as rola_state

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))
import clock_lock  # noqa: E402 -- path insert must precede this import
from gpu_lock import gpu_lock  # noqa: E402

#: Carried verbatim from `benchmark_rola_v3_layer.py`: a ~6x first-cell inflation was
#: MEASURED there (10.8 ms first vs 1.74 ms for the same cell measured second), from
#: something process-level that outlives a per-cell warmup. A 6x inflation on one cell
#: inverts a comparison, and on a CURVE it would invert exactly the smallest `L` --
#: the end the honesty rules say must not be cherry-picked -- so it is paid up front.
_FIRST_CELL_EXTRA_WARMUP = 20
_warmed = False


def _quartiles(samples: list[float]) -> tuple[float, float]:
    samples = sorted(samples)
    n = len(samples)
    return samples[n // 2], samples[(3 * n) // 4] - samples[n // 4]


def _measure(fn, reps: int, warmup: int) -> dict[str, float]:
    """Both instruments.

    `event_ms` is the `cuda.Event` median -- right for a GPU-bound region.
    `wall_ms` is the synchronized wall-clock median -- the honest one for a
    host-bound plane, where an event pair brackets a CPU stall rather than a kernel
    and reports it with large variance. Both are carried on every plane so neither is
    chosen silently.
    """
    global _warmed
    if not _warmed:
        for _ in range(_FIRST_CELL_EXTRA_WARMUP):
            fn()
        torch.cuda.synchronize()
        _warmed = True
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events, walls = [], []
    for _ in range(reps):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start = time.perf_counter()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1e3)
        events.append(e0.elapsed_time(e1))
    event_ms, event_iqr = _quartiles(events)
    wall_ms, wall_iqr = _quartiles(walls)
    return {"event_ms": event_ms, "event_iqr": event_iqr,
            "wall_ms": wall_ms, "wall_iqr": wall_iqr}


class MatchedAttention(nn.Module):
    """The attention arm, built to be the SAME LAYER SHAPE as the RoLA layer.

    `v_proj` + `o_proj` exist on both sides; attention additionally carries `q_proj`
    and `k_proj`, which is precisely the architecture's `content_addressing_width =
    d_qk` showing up as parameters. Causal, no positional embedding (RoLA has none
    either), no dropout, no mask tensor -- `is_causal=True` so the FLASH backend takes
    its fused path rather than materializing an `L x L` mask, which would make the
    memory column measure the harness instead of the architecture.
    """

    def __init__(self, hidden: int, heads: int, d_head: int):
        super().__init__()
        self.heads, self.d_head = heads, d_head
        self.q_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.k_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.v_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.o_proj = nn.Linear(heads * d_head, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        def split(t):
            return t.view(B, L, self.heads, self.d_head).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, self.heads * self.d_head))


_FA_BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}


def run_fa_cell(*, L, B, H, d_head, hidden, dtype, backend: str, reps, warmup) -> dict:
    torch.manual_seed(0)
    row = {"arm": f"fa_{backend}", "backend": backend, "L": L, "B": B, "H": H,
           "d_head": d_head, "hidden": hidden,
           "dtype": str(dtype).replace("torch.", ""),
           "reps": reps, "warmup": warmup,
           # ANALYTIC, and it is the decoder's bill, not this forward's allocation:
           # keys plus values for every position and head. This is the quantity that
           # grows with `L` and that RoLA's fixed state replaces.
           "kv_cache_bytes": 2 * B * H * L * d_head * torch.finfo(dtype).bits // 8}
    try:
        # The SDPA FLASH backend does not accept fp32 (atscale-review.md G2, reproduced
        # under `ncu`: FLASH-only context dispatches `pytorch_flash::flash_fwd_kernel`,
        # which is bf16/fp16 only, and fp32 under a FLASH-only context RAISES
        # "No available kernel" rather than silently falling back to another backend).
        # Assert it explicitly here rather than relying on that low-level message: a
        # caller who requests `backend == "flash"` with fp32 gets a benchmark-level
        # failure that names the constraint instead of a CUDA-side error string.
        if backend == "flash":
            assert dtype in (torch.bfloat16, torch.float16), (
                f"SDPA FLASH backend does not accept {dtype}; use fa_efficient for fp32 "
                "(atscale-review.md G2)")
        layer = MatchedAttention(hidden, H, d_head).to("cuda", dtype).eval()
        for param in layer.parameters():
            param.requires_grad_(False)
        x = torch.randn(B, L, hidden, device="cuda", dtype=dtype)

        def forward():
            with torch.no_grad(), sdpa_kernel(_FA_BACKENDS[backend]):
                return layer(x)

        forward()  # faults in the backend; raises here if this backend cannot run
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = _measure(forward, reps, warmup)
        row.update({
            "total_ms": round(stats["wall_ms"], 4),
            "total_iqr_ms": round(stats["wall_iqr"], 4),
            "event_ms": round(stats["event_ms"], 4),
            "event_iqr_ms": round(stats["event_iqr"], 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 4),
            "oom": False,
            # The addressing columns for the attention side: attention's addressing
            # is DATA-located and its width is `d_qk`; it has no routing hierarchy, so
            # the grid-located columns are structurally absent (None), not zero.
            "content_addressing_width": d_head,
            "routing_addressing_width": None,
            "address_space_bits": round(math.log2(L), 3),
            "address_space_basis": "attention addresses TOKENS: the space is L and it GROWS",
        })
        layer = x = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"oom": True, "oom_stage": "forward", "error": str(exc)[:200]})
    except RuntimeError as exc:
        row.update({"oom": False, "unsupported": True, "error": str(exc)[:200]})
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _addressing(branches, d_v, read_levels, write_levels) -> dict:
    """The addressing columns; same definitions as the layer bench's.

    `A_r` is a PRODUCT OF PER-LEVEL MEANS, exact at a dense operating point and an
    approximation at a sparse one (the levels' supports are correlated under sparse
    routing); `realized_read_address_bits` inherits that approximation. Labelled in
    the row itself rather than in this docstring alone.
    """
    N = math.prod(branches)

    def active(levels):
        return float(torch.stack([(t != 0).float().sum(-1).mean() for t in levels]).prod())

    a_r, a_w = active(read_levels), active(write_levels)
    return {
        "N": N,
        "state_floats_per_head": N * (d_v + 1),  # the mass column is unconditional
        "address_space_bits": round(math.log2(N), 3),
        "address_space_basis": "RoLA addresses LEAFS: the space is N and it is FIXED in L",
        "routing_addressing_width": sum(branches),
        "content_addressing_width": 0,
        "A_r": round(a_r, 4), "A_w": round(a_w, 4),
        "A_r_basis": "product-of-per-level-means (exact when dense, approximate when sparse)",
        "realized_read_address_bits": round(math.log2(a_r), 3),
        "realized_write_address_bits": round(math.log2(a_w), 3),
    }


def run_v3_cell(*, L, B, H, d_v, branches, hidden, read_router, write_router, logit_gain,
                paging, planes: bool, reps, warmup,
                arm="v3", cell_role="parity") -> dict:
    """`read_router`/`write_router` are independent (`'softmax'` = dense, `'entmax'`
    = exact entmax sparse), per the 2026-08-01 ruling: the freeze-defining parity
    cell is read-DENSE / write-SPARSE, not the sparse-both cell item 0 measured.
    `arm`/`cell_role` label the row so a dense-read parity row and a sparse-both
    diagnostic row (both produced by this same function) are never confused for
    each other downstream.
    """
    def routing_tag(read_router: str, write_router: str):
        """Both sides sparse -> `UnionRouting(alpha)` (`union_routing(1, 1.5)`);
        otherwise `IndependentRouting` with each side's own activation, which is
        what a MIXED read/write router (the read-dense/write-sparse parity cell)
        needs."""
        from rola import entmax, softmax
        from rola.routing.types import IndependentRouting

        def activation(router: str):
            return entmax(1.5) if router == "entmax" else softmax()

        if read_router == "entmax" and write_router == "entmax":
            return union_routing(1, 1.5)
        return IndependentRouting(width=1, read=activation(read_router),
                                  write=activation(write_router))

    row = {"arm": arm, "cell_role": cell_role, "L": L, "B": B, "H": H, "d_v": d_v,
           "hidden": hidden, "branches": list(branches),
           "read_router": read_router, "write_router": write_router,
           "logit_gain": logit_gain,
           #: TWO DEAD AXES RETIRED. Raw normalization used to be refused
           #: outright (the mass column is unconditional; the host axis itself is now
           #: deleted) -- and a decay-configured layer is refused by the chunk arm.
           #: Both raised on their first cell. The columns stay in the row, as the
           #: CONSTANTS they are, so a reader of an old JSON beside a new one is
           #: comparing the same arm.
           "norm": "global", "decay": "None",
           "paging": paging, "P": 1, "P_role": "PRIMARY",
           "dtype": "float32", "reps": reps, "warmup": warmup}
    try:
        torch.manual_seed(0)
        template = routing_tag(read_router, write_router)
        levels = [template.at(w) for w in branches]
        routes = RouteProducer(
            levels, hidden_size=hidden, num_heads=H)
        layer = RoLA(
            routes, d_v=d_v, decay=None, layer_idx=0,
        ).to("cuda", torch.float32).eval()
        if logit_gain != 1.0:
            with torch.no_grad():
                routes.route_W.mul_(logit_gain)
        for param in layer.parameters():
            param.requires_grad_(False)
        x = torch.randn(B, L, hidden, device="cuda", dtype=torch.float32)

        def whole_layer():
            with torch.no_grad():
                return layer(x)

        whole_layer()
        assert layer.last_execution_backend == "cuda", (
            f"cell L={L} took the {layer.last_execution_backend!r} backend, so this row "
            "measures the wrong kernel")
        # REPOINTED: paging is the STATE's backing, not a layer flag, and it
        # engages only on a call that carries a state. The TIMED call above stays
        # stateless -- that is the row this benchmark reports -- and residency is read
        # off one extra stateful call on the same routing, which commits exactly the
        # atoms that call writes.
        stats_paging = None
        if paging:
            with torch.no_grad():
                carried = rola_state()
                layer(x, continuation=LayerContinuation(state=carried))
            assert carried.paged, (
                f"cell L={L}: this topology has no built chunk arm, so its state is "
                "dense-backed and there is no residency to report")
            stats_paging = carried._arena.last_result
        if stats_paging is not None:
            # `allocated_*` is what the arena handed out; `committed_*` is what the CUDA
            # driver MAPPED, and only the second states a saving. A pool smaller than one
            # allocation granule commits its whole dense limit, so `dense_state_bytes` is
            # on the row beside every fraction (docs/internals/paging/paging.md#granule).
            row.update({
                "owners_total": stats_paging.owners_total,
                "realized_owners": stats_paging.realized_owners,
                "allocated_pages": stats_paging.allocated_pages,
                "resident_pages": carried._arena.resident_pages,
                "peak_pages": stats_paging.peak_pages,
                "allocated_bytes": stats_paging.allocated_bytes,
                "committed_bytes": stats_paging.committed_bytes,
                "dense_state_bytes": stats_paging.dense_bytes,
                "allocated_fraction": round(stats_paging.allocated_fraction, 6),
                "committed_fraction": round(stats_paging.committed_fraction, 6),
            })
        else:
            # Unpaged: the provisioned dense state IS the committed state. Analytic,
            # and labelled as such -- there is no page table to measure.
            row["dense_state_bytes"] = B * H * math.prod(branches) * (d_v + 1) * 4
            row["committed_bytes"] = row["dense_state_bytes"]
            row["committed_fraction"] = 1.0
            row["resident_basis"] = "analytic (paging off: every provisioned leaf is resident)"

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        total = _measure(whole_layer, reps, warmup)
        row.update({
            "total_ms": round(total["wall_ms"], 4),
            "total_iqr_ms": round(total["wall_iqr"], 4),
            "event_ms": round(total["event_ms"], 4),
            "event_iqr_ms": round(total["event_iqr"], 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 4),
            "oom": False,
        })

        if planes:
            # The ROUTING plane on the identical inputs the whole-layer call uses.
            # `rest` is a RESIDUAL (total minus producer), not an independent
            # measurement: the plane is timed as its own launch sequence, so its
            # launch/sync overhead is inside the plane and absent from the fused call,
            # and the residual can go negative.
            def producer():
                with torch.no_grad():
                    return layer.routes(x)

            with torch.no_grad():
                ro = producer()
            read_levels, write_levels = ro.read, ro.write
            plane_stats = {"producer": _measure(producer, reps, warmup)}
            row["planes"] = {name: {k: round(v, 4) for k, v in s.items()}
                             for name, s in plane_stats.items()}
            row["producer_ms"] = round(plane_stats["producer"]["wall_ms"], 4)
            row["rest_ms"] = round(row["total_ms"] - row["producer_ms"], 4)
            row["rest_basis"] = "RESIDUAL of total minus producer; may be negative"
            row.update(_addressing(branches, d_v, read_levels, write_levels))
            ro = read_levels = write_levels = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
        layer = x = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"oom": True, "oom_stage": "forward-or-planes", "error": str(exc)[:200]})
    except RuntimeError as exc:
        # A LAUNCHABILITY CEILING, not a crash and not an OOM. The consumer rejects
        # an unrealizable configuration loudly, so at some `L` the sweep hits a hard
        # kernel bound. That bound
        # is exactly the arm's max-`L`, so it is RECORDED as a row with its message
        # verbatim and the sweep continues at the next arm.
        row.update({"oom": False, "unsupported": True, "error": str(exc)[:400]})
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-lens", default="512,1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-v-dim", type=int, default=64)
    ap.add_argument("--branches", default="8,16")
    ap.add_argument("--read-router", choices=("softmax", "entmax"), default="softmax",
                    help="the `v3` arm's read side. Default 'softmax' (dense) -- the "
                         "2026-08-01 ruling: the freeze-defining parity cell is "
                         "read-dense/write-sparse, not the sparse-both cell item 0 "
                         "measured (winning sparse-both is the trivial case).")
    ap.add_argument("--write-router", choices=("softmax", "entmax"), default="entmax",
                    help="the `v3` arm's write side. Default 'entmax' (sparse); see "
                         "--read-router.")
    ap.add_argument("--logit-gain", type=float, default=8.0,
                    help="scales route_W to sharpen an UNTRAINED exact-entmax router into a "
                         "real sparse support; the support is genuine, its LEVEL is a "
                         "synthetic stand-in for training and is labelled as such")
    ap.add_argument("--paging", action="store_true")
    ap.add_argument("--planes", action="store_true",
                    help="time the routing plane separately from the rest of the forward")
    ap.add_argument("--arms", default="v3,v3_sparse_both,fa_flash,fa_efficient",
                    help="'v3' = the parity cell (--read-router/--write-router, default "
                         "dense-read/sparse-write). 'v3_sparse_both' = the SAME cell "
                         "with both sides forced entmax -- item 0's original cell, kept "
                         "as a labelled diagnostic row, not the freeze arm.")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    with gpu_lock():
        if not torch.cuda.is_available():
            raise SystemExit("this benchmark requires CUDA")
        # THE CLOCK LOCK (tools/clock_lock.py): the host holds the SM clock for the sweep, proven by the device's own
        # read, so the first cell is not timed on a ramping clock. A host with no clock declared runs unlocked, and
        # every row says which.
        from rola.ops import carry as carry_ops

        clock = clock_lock.engage(carry_ops.sm_clock_ghz)

        branches = [int(s) for s in args.branches.split(",")]
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        arms = args.arms.split(",")
        H, d_v, B = args.heads, args.head_v_dim, args.batch
        hidden = H * d_v
        rows: list[dict] = []

        for L in seq_lens:
            for arm in arms:
                if arm == "v3":
                    row = run_v3_cell(L=L, B=B, H=H, d_v=d_v, branches=branches, hidden=hidden,
                                      read_router=args.read_router, write_router=args.write_router,
                                      logit_gain=args.logit_gain,
                                      paging=args.paging, planes=args.planes,
                                      reps=args.reps, warmup=args.warmup,
                                      arm="v3", cell_role="parity (read-dense/write-sparse, "
                                      "the freeze-defining arm)")
                elif arm == "v3_sparse_both":
                    row = run_v3_cell(L=L, B=B, H=H, d_v=d_v, branches=branches, hidden=hidden,
                                      read_router="entmax", write_router="entmax",
                                      logit_gain=args.logit_gain,
                                      paging=args.paging, planes=args.planes,
                                      reps=args.reps, warmup=args.warmup,
                                      arm="v3_sparse_both",
                                      cell_role="DIAGNOSTIC: sparse-both (item 0's original "
                                      "cell; the easy regime, not the freeze arm)")
                elif arm.startswith("fa_"):
                    backend = arm[3:]
                    dtype = torch.bfloat16 if backend == "flash" else torch.float32
                    row = run_fa_cell(L=L, B=B, H=H, d_head=d_v, hidden=hidden, dtype=dtype,
                                      backend=backend, reps=args.reps, warmup=args.warmup)
                else:
                    raise SystemExit(f"unknown arm {arm!r}")
                ghz = carry_ops.sm_clock_ghz()
                row["clock_locked_ghz"] = clock["ghz"] if clock else None
                row["clock_ghz"] = ghz
                row["clock_held"] = clock is not None and clock_lock.within(ghz, clock)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if args.json:  # write incrementally: an OOM later must not lose earlier rows
                    with open(args.json, "w") as handle:
                        json.dump(rows, handle, indent=2)

    print("\n=== latency vs L (synchronized wall-clock median) ===")
    for r in rows:
        if r.get("oom"):
            print(f"L={r['L']:7d} {r['arm']:14s} OOM at {r.get('oom_stage')}")
        elif r.get("unsupported"):
            print(f"L={r['L']:7d} {r['arm']:14s} UNSUPPORTED: {r['error'][:80]}")
        else:
            print(f"L={r['L']:7d} {r['arm']:14s} total={r['total_ms']:9.3f} ms "
                  f"(IQR {r['total_iqr_ms']:7.3f}) event={r['event_ms']:9.3f} "
                  f"peak={r['peak_alloc_gb']:.3f} GB "
                  f"state/kv={r.get('committed_bytes', r.get('kv_cache_bytes', 0)) / 2 ** 20:8.2f} MiB")
    print("\nNOT A MATCHED-CAPACITY RATIO. Attention and RoLA are different architectures:\n"
          "attention's addressing is DATA-located over a token space that GROWS with L\n"
          "(content_addressing_width = d_head); RoLA's is GRID-located over a FIXED leaf\n"
          "space N (routing_addressing_width = sum_l b_l, content width 0). The columns\n"
          "are on every row so the cell is admissible as what it is: operational cost and\n"
          "committed state at matched (B, H, L, d_model).\n"
          "DTYPE IS NOT MATCHED AND THE MISMATCH FAVOURS ATTENTION: the SDPA FLASH backend\n"
          "rejects fp32 and the V3 consumer accepts only fp32, so fa_flash is bf16 and v3 is\n"
          "fp32. fa_efficient carries the fp32 attention row for that axis.")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
