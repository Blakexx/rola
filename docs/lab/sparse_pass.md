# `sparse_pass`

Mirrors `measure/lab/experiments/sparse_pass.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The pass with each warp's live tokens drawn at a density (the CTA union every token, four passes a window): the
owner part's fill through a token permutation, the restage, the fold step at the arm's registers, the fp16 join
staggered, the window's lists two passes ahead. Forms: serial (today's owner part); a value tile with two pool slots;
values in place from swizzled pool rows with two or three pool slots; the fold side's rows listed (entries built ahead,
free here) or derived in the step from the warp's union words; and LOAD-AHEAD (fp32 arm only, 232-236 registers): a
warp takes up to three steps' operands into registers at a time and signals its slots drained after its last loads,
not its last MMA, its restage share after the loads of the chunk holding its mid-pass step. The later forms take the
fixed
work apart (fp32 arm): the factor products precomputed (an 8-byte plane read in place), those and the dense digits
precomputed as f16 (read in place by `ldmatrix` from 192-byte pool rows, chunk `c ^ ((r >> 1) & 3)`: no restage, no
tile, no staged signal), and the pass's rows built once at its start (word-parallel over its four words) in place of a
step at a time. The block's span a pass over the busiest scheduler's
paired steps at tensor time (busiest-scheduler steps a pass: 16.0 / 5.6 / 2.0):

| pass (fp32 arm / fp16 arm) | all live | 1 in 4 | 1 in 16 |
|---|---|---|---|
| serial, value tile, listed | 1.26 / 1.66 | 1.60 / 2.39 | 2.54 / 4.27 |
| pipelined, value tile, derived | 1.13 / 2.06 | 1.63 / 2.78 | 2.48 / 4.47 |
| pipelined, in place, 2 pool slots, derived | 1.33 / 2.05 | 1.85 / 3.05 | 2.88 / 5.09 |
| pipelined, in place, 3 pool slots, derived | 1.09 / 2.09 | 1.60 / 2.69 | 2.26 / 4.13 |
| pipelined, in place, 3 pool slots, listed | 1.21 / 1.97 | 1.55 / 2.35 | 2.05 / 3.53 |
| load-ahead, in place, 3 pool slots, derived | 1.13 / - | 1.30 / - | 2.06 / - |
| load-ahead, in place, 3 pool slots, listed | 1.14 / - | 1.26 / - | 1.88 / - |
| load-ahead, derived, factor products precomputed | 1.13 / - | 1.28 / - | 1.96 / - |
| load-ahead, derived, products and f16 dense precomputed (no restage) | 1.13 / - | 1.25 / - | 1.94 / - |
| no restage, rows built a pass at its start | 1.14 / - | 1.20 / - | 1.91 / - |

Three pool slots are what in-place reads need: with two, a slot refills only after its pass drains, so the fill and
the restage land on the pass boundary and the warps couple (1.33x dense fp32). Dense, three slots derived is the best
form (fp32 1.09x). Sparse, no pipeline beats serial at 1 in 4 (all 1.58-1.84x), and at 1 in 16 three slots listed is
best (2.06x against 2.53x): a warp's few steps leave the mid-pass point near its pass's end, so the half-pass slack
shrinks to nothing, and a warp alone on its scheduler cannot hide its own chain. Load-ahead is what sparse needed: a
slot frees when every warp
has loaded from it, so the fill and the restage stop waiting on the slowest warp's MMAs and the warps drift -- 1 in 4
1.27-1.30x against 1.55-1.60x, 1 in 16 1.87-2.07x against 2.05-2.26x, at ~3% dense (its chunked loads; 1.13x against
1.09x). Derived rows against listed: better dense, 3-10% worse sparse (a step scans up to four sparse words); the
listed rows' build is free here and is not in the owner part.

The per-warp phase ledger (`LEDGER` 1, a lap clock at each phase) says where a sparse pass goes:

| form, density (K cycles a warp a pass) | start waits | rows | ldmatrix | coefficient | mid-pass | MMA issue | end |
|---|---|---|---|---|---|---|---|
| load-ahead, derived, 1 in 4 | 0.52 | 1.88 | 0.13 | 0.35 | 0.89 | 4.07 | 0.04 |
| load-ahead, derived, 1 in 16 | 0.26 | 1.22 | 0.05 | 0.14 | 0.97 | 1.69 | 0.04 |
| load-ahead, listed (free entries), 1 in 4 | 0.60 | 0.70 | 0.13 | 0.75 | 1.03 | 4.13 | 0.07 |
| load-ahead, listed (free entries), 1 in 16 | 0.38 | 0.40 | 0.05 | 0.31 | 1.13 | 1.72 | 0.07 |
| no restage, derived, 1 in 4 | 0.25 | 1.87 | 0.14 | 0.40 | 0.78 | 4.07 | 0.05 |
| no restage, derived, 1 in 16 | 0.36 | 1.21 | 0.06 | 0.16 | 0.64 | 1.78 | 0.05 |
| no restage, rows a pass, 1 in 4 | 0.34 | 0.96 | 0.16 | 0.62 | 0.91 | 4.16 | 0.05 |
| no restage, rows a pass, 1 in 16 | 0.49 | 0.75 | 0.08 | 0.25 | 0.79 | 1.74 | 0.05 |

Rows derived a step at a time are the largest non-MMA piece (a word loop, warp syncs and scratch round trips each
step: ~650-940 cycles a step in sparse passes); built once a pass they cost about half, 1 in 4 1.25x -> 1.20x. The
precomputed forms take the restage out of the mid-pass work (-0.1 to -0.3K a warp-pass; 1 in 4 1.30x -> 1.25x, 1 in 16
2.06x -> 1.94x, dense unchanged). At 1 in 16 a warp has about one step a pass, and its fixed work a pass (rows ~0.75K,
mid-pass ~0.8K, waits ~0.5K) still outweighs that step: the 128-row pass is too small a unit there. The fp16 arm
reads 1.95-2.09x dense: every pipelined form sits at 251-254 registers, and the bookkeeping
(rows, cursor, lists, waits) leaves its steps no room -- the pipeline needs register headroom on that arm; its rows
also move between runs (1 in 4, three slots derived: 4.42x in an earlier 3-repetition run, 2.69x here). Record:
`measure/lab/results/sparse_pass/sm_86.json`.
