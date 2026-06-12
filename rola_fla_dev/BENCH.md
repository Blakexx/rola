# Kernel-efficiency validation of the integrated fork path (2026-06-09)

`bench.py` — B=8 H=8 L=1024 d_qk=d_v=16 bf16, median of 10, idle GPU, warm autotune cache.
**Scope: post-projection KERNEL efficiency at matched total state** (nc·d_qk·d_v per head). The bet
under test: the shared gram (content gram computed once, routing gram rank-nc on top) scales state
more efficiently than monolithic d_qk scaling. Projection/param cost is model-level context, kept
out of this measurement (`--proj` prints it separately).

`act` = bytes actually retained for backward (saved_tensors_hooks, deduped by storage) — the honest
training-memory metric; an earlier draft under-counted the monolith (its giant saved q/k were
pre-allocated leaves and invisible to a post-fwd allocation delta). NOTE: a first run taken right
after an LM job measured ~2× slower across ALL impls (GPU clock/cache state); numbers below are the
clean idle-GPU rerun — rankings agreed in both.

| var | nc  | impl     | fwd ms | fwd+bwd ms | peak MiB | act MiB |
|-----|-----|----------|-------:|-----------:|---------:|--------:|
| RLA | 16  | rola     |   0.25 |       2.13 |       72 |      10 |
| RLA | 16  | vh       |   0.66 |       3.12 |      258 |     134 |
| RLA | 16  | monolith |   0.19 |       1.36 |      100 |      66 |
| RLA | 64  | rola     |   0.77 |       5.03 |      168 |      22 |
| RLA | 64  | vh       |   2.82 |      11.92 |     1032 |     530 |
| RLA | 64  | monolith |   0.83 |       3.37 |      388 |     258 |
| RLA | 256 | rola     |   2.48 |      16.31 |      552 |      70 |
| RLA | 256 | vh       |  13.16 |      51.61 |     4128 |    2114 |
| RLA | 256 | monolith |   2.84 |      11.06 |     1540 |    1026 |
| GLA | 16  | rola     |   0.28 |       6.85 |      166 |      12 |
| GLA | 16  | vh       |   0.75 |       3.49 |      266 |     138 |
| GLA | 16  | monolith |   0.21 |       1.67 |      101 |      66 |
| GLA | 64  | rola     |   0.65 |      16.79 |      545 |      30 |
| GLA | 64  | vh       |   3.19 |      12.95 |     1064 |     546 |
| GLA | 64  | monolith |   0.86 |       4.00 |      391 |     258 |
| GLA | 256 | rola     |   2.00 |      60.22 |     2107 |     102 |
| GLA | 256 | vh       |  14.99 |      55.97 |     4256 |    2178 |
| GLA | 256 | monolith |   3.89 |      13.69 |     1555 |    1026 |

## Verdict on the shared-gram bet (vs monolith, kernel-only)
1. **Forward: CONFIRMED, with crossover at nc≈64.** rola ties the monolith at nc=64 and WINS at
   nc=256 (RLA 2.48 vs 2.84; GLA 2.00 vs 3.89, 1.9×) — while doing ~15× fewer gram FLOPs per
   chunk-pair (C²(d_qk+nc)=272 MACs vs C²(nc·d_qk)=4096 at nc=256). The monolith burns its FLOP
   excess at near-peak tensor-core efficiency; shared-gram still beats it where the design targets.
   nc=16 the monolith is 1.3× faster (its dense gram is tiny there; routing overhead dominates).
2. **Activation memory: CONFIRMED, 6–15× less retained than the monolith** (70–102 vs 1026 MiB at
   nc=256) and 15–30× less than vh. The monolith must save its nc·d_qk-wide q/k; rola saves the thin
   q/k + gates. Peak: rola 2.8× below monolith for RLA (552 vs 1540); GLA@256 peak above (Sb buffer).
3. **Backward: the open gap — engineering, not architecture.** RLA fwd+bwd is 1.5× behind the
   monolith (16.3 vs 11.1 at nc=256); GLA 4.4× (60 vs 14: S_before HBM traffic + two sequential
   decayed scans + fp32-floor chunk=32). The same 15× gram-FLOP advantage applies to the backward's
   grams, so the deficit is implementation: grid (B·H, NB) serializes chunks inside each program →
   occupancy cliff at low NB, and Sb round-trips at high nc.
   Fix (scoped in TASKS): FLA-style restructure — sequential state-checkpoint scan + chunk-PARALLEL
   grad kernel, grid (B·H, NB, NCH). vs vh: rola already wins fwd+bwd at nc≥64 (RLA) and nc=256 ties.

## vs virtual-head (the practical alternative for routing)
rola beats vh on every column at nc≥64 and on memory everywhere (act 13–21× less). vh's only win is
the GLA backward at nc≤64 — same backward gap as above.

---

# UPDATE — chunk-parallel backward shipped (Phase F, same day)

Backward restructured: `_scan_S`/`_scan_dS` (tiny sequential state scans, store S_before/dS_after per
chunk) + `_par_grad_{rla,gla}_{qr,kwv}` (fully PARALLEL over (B·H, NB, NCH), one state tile each —
a combined single grad kernel was tried and REGRESSED on register spill; so did GLA-kwv until the
dLam Σ(dSa∘Sb) reduction moved to torch). Old fused bwd kernels deleted. Gate + parity green.

Final kernel table (hot-run, directly comparable to the fused-baseline run — vh/mono controls match):

| var | nc  | impl     | f+b ms (fused → split) | vs vh | vs mono |
|-----|-----|----------|------------------------|-------|---------|
| RLA | 16  | rola     | 11.46 → **2.26** (5.1×) | 3.26 ✅ | 1.45 ✗ |
| RLA | 64  | rola     | 29.16 → **10.41** (2.8×)| 17.90 ✅ | 5.42 ✗ |
| RLA | 256 | rola     | 28.31 → 35.63           | 71.74 ✅ | 17.76 ✗ |
| GLA | 16  | rola     | 11.11 → **6.67** (1.7×) | 5.90 ~ | 2.65 ✗ |
| GLA | 64  | rola     | 30.05 → **16.14** (1.9×)| 19.92 ✅ | 6.48 ✗ |
| GLA | 256 | rola     | 107.5 → **57.62** (1.9×)| 76.54 ✅ | 20.35 ✗ |

STAGE1+STAGE2 TOTALS (the full economics — monolith pays nc-scaling in projections too):
| nc  | RLA rola vs mono | GLA rola vs mono |
|-----|------------------|------------------|
| 16  | **3.94 vs 6.65** ✅ | 8.35 vs 7.85 (−6%) |
| 64  | **12.80 vs 23.83** ✅ | **18.54 vs 24.89** ✅ |
| 256 | **41.42 vs 86.88** ✅ | **63.41 vs 89.48** ✅ |
(+ params 7–15× fewer; act mem 6–15× less; vs vh: rola wins everything except GLA@16 ~par.)

Residual: GLA@16 total trails the monolith by ~6% (fixed costs: decayed scans + dld/dLam torch
assembly dominate at tiny nc). Lever if ever needed: fuse the dld reverse-cumsum into the kwv kernel.
RLA@256 kernel is 36 vs fused 28 (state-buffer HBM traffic) — totals still win 2.1×; chunk=64 would
halve the traffic but exceeds 100KB smem on this card.
