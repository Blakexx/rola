# `drain`

Mirrors `measure/lab/experiments/drain.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the readout's drain -- a pass's panel words added into the head's `num` by `red.global.add` at each slot's
token, the panel cleared for the next pass's adds -- cost a CTA-window, in today's form and in forms that pair
conflict-free slots, clear by exchange or by a 16-byte sweep, skip the requests past the union, and beside a stream's
copies? And what does a wider or floating `num` cost: each slot's 32 channels as fp32 or s32 (a full line a slot), or
f16 pairs (today's bytes)?

80 CTAs, box b of one head (the carry's order); the inputs (`drain.inputs.py`) are `read_stream`'s three cells, each
box-window's union in rank order (the facts pass's rank slots). Each pass every thread fills its slot's 16 panel words
(four 16-byte stores) and its token, a CTA barrier, the drain, a barrier. A warp drains its 32 slots two a
`red.global` (lanes 0-15 one slot's 16 words, 16-31 another's), each a token's half line of `num` (d_v 64: a token's
128-byte row, two channel tiles), so the 8 slice groups' boxes and every read group holding the token add into the
same words. Every draining form's `num` is checked word for word.

The forms: today's (slots (s, s + 1) at the panel's 20-word pitch: 4 banks shared, 2 wavefronts an access; a load and
a 4-byte clearing store); slots (s, s + 4) (16 banks apart: 1 wavefront); an `atom.shared.exch` reading and clearing
in one; no clear in the drain and a 16-byte sweep after its barrier; the sweep with a warp's groups of 4 requests past
the union not issued (a warp-uniform count); copies of 512 bytes a warp beside it every 4 adds or every add; bounds:
no clear, the fill alone.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, the drain's cycles a CTA-window (the form's less the fill's 589 /
952 / 946), every CTA draining at once (re-recorded 2026-10-19 with the output's formats below):

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| the first form: slots (s, s + 1), load and 4-byte clear | 1,645 | 3,372 | 3,561 |
| slots (s, s + 4), load and 4-byte clear | 1,646 | 3,358 | 3,562 |
| slots (s, s + 4), exchange | 1,459 | 3,069 | 3,394 |
| slots (s, s + 4), load; 16-byte clear sweep | 1,369 | 3,041 | 3,367 |
| bound: no clear | 1,429 | 3,047 | 3,369 |
| the sweep, groups past the union skipped (production's) | 1,045 | 2,542 | 2,441 |
| (s, s + 4) beside copies, 512 bytes every 4 adds | 2,212 | 4,470 | 4,530 |
| (s, s + 4) beside copies, 512 bytes every add | 4,494 | 8,600 | 8,614 |

Requests a CTA-window (ncu, `lts__t_requests_srcunit_tex_op_red`): 256 / 512 / 509 issuing every slot of each pass;
215 / 459 / 458 skipping the groups past the union (alt-k4's union 212 slots). L2's atomic unit runs ~36-41% of the
launch's elapsed cycles (about half the drain's); global-queue stalls ~0.

What it settles:

- THE DRAIN'S FLOOR IS THE SM'S GLOBAL REQUEST PATH: ~5 SM cycles a line request with every SM draining (the
  skipping form's row: 4.85 / 5.54 / 5.34 at the three cells (5.10 / 5.56 / 5.40 first recorded); alt-k4's is the global path's atomic price in
  `common/hardware.json`) -- the per-SM rate on record (2026-09-27), L2's atomic unit about half busy. The requests are
  the union's slots: one half line a slot at d_v 64. The drain is no shared-pipe or issue question on sm_86: today's
  slot pairing and the conflict-free one run alike.
- SKIP WHAT THE UNION DOES NOT HOLD: a pass's slots past the union issued as zero adds cost 16% more requests and 31%
  more time at alt-k4 (38% at cohort-k16, whose second pass is mostly empty). Never point dead lanes at one shared
  dummy row: the first form of this lab did, and every CTA's dead adds serialized on that one line -- the drain read
  11,300 cycles a window against 1,850 with a dummy row per CTA.
- A PREDICATED GLOBAL ADD IS A BRANCH: ptxas puts `@p red.global` in a convergence region (BSSY, BRA, BSYNC around it,
  ~4 instructions an add); the production drain's `red_nonzero` is that form. A dead lane's add is a select of a zero
  value and its own row, or the group is not issued.
- THE CLEAR: the exchange saves 11% of the first form's drain, the 16-byte sweep after the barrier 17% (below the no-clear bound:
  the sweep's stores leave the drain's loop shorter). On sm_80, where the shared pipe binds the readout, the
  conflict-free pairing also halves the drain's panel wavefronts.
- THE DRAIN AND THE STREAMS SHARE THE GLOBAL PATH: 512 bytes copied a warp every 4 adds cost the same form 34%
  more, every add 2.7x -- ~4-5 cycles a copied 128-byte line beside the adds. The streamed carry's copies and its drain are one
  per-SM budget (with L2's ~18 bytes a cycle an SM for copies alone, `smem_rates`): `tools/pipes.py`'s global pipe
  ([pipes.md](../internals/tools/pipes.md)).

THE OUTPUT'S FORMAT (2026-10-19): the skipping form with `num` in another format, the drain's cycles a CTA-window less
the fill. fp32 and s32 put a slot's 32 channels in one request (lane `l` channel `l`, from its pair word's half,
converted exactly through the magic offset and scaled by the slot's power of two); f16 pairs keep today's two slots a
request, each word's halves converted, scaled by the slot's (scale, bias) pair and packed by one `F2FP`, added by
`red.global.add.noftz.f16x2` (its panel filled with 5-bit halves so every f16 sum is an exact integer). Every slot
checked:

| `num` | alt-k4 | alt-k16 | cohort-k16 | L2 sectors (alt-k4) | L2 atomic unit busy |
|---|---|---|---|---|---|
| s16 pairs (the integer panel of the time) | 1,045 | 2,542 | 2,441 | 2.21M | 36% |
| f16 pairs | 1,515 (1.45x) | 3,284 (1.29x) | 3,083 (1.26x) | 2.21M | 30% |
| fp32 | 2,612 (2.50x) | 4,853 (1.91x) | 4,396 (1.80x) | 4.41M | 47% |
| fp32, one FFMA a channel | 2,586 | 4,804 | 4,318 | 4.41M | 47% |
| s32 | 2,524 | 4,762 | 4,300 | 4.41M | -- |

- A 32-BIT `num` DOUBLES THE DRAIN WHATEVER ITS TYPE: the same requests (1,103,104 at alt-k4), twice the sectors, fp32
  and s32 alike, and trimming the conversion to one FFMA a channel buys 2%: L2's atomic path charges the bytes.
- f16 PAIRS keep the bytes and the requests and cost 26-45% more: L2's atomic unit does ~20% more work a sector than
  for integer adds (30% of a 1.44x launch against 36%), the rest is the conversion in the request loop's latency
  (issue 740 against 484 a unit); the pipe model prices it at the s16 pairs' (the global path binds at 1,124 either way).

Records: `measure/lab/results/drain/sm_86.json`, `sm_86.counters.json`.
