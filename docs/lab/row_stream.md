# `row_stream`

Mirrors `measure/lab/experiments/row_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the fold's rows are streamed a list at a time into each warp's ring (merge width 1) instead of staged once a
window, what does the stream cost beyond the fold, at what ring depth, and in what form?

The fold step is the carry's: a warp folds its 4 lists `w + 8 j` into 128 fp32 registers, 16-token steps while more
than 8 entries remain, then an 8-token tail. The inputs (`row_stream.inputs.py`) are three cells' write lists:
nl64k-alt-k4 over 64 windows, and flagship-alt-k16-L16384 and flagship-cohort-k16-L16384 over 32. The 80 CTAs are
boxes 0 .. 79 of one head, channel tile fastest as the carry orders them. A box's row for a token is its level-1 slice
group's 32 bytes (the carry's u8 slice bytes, zero-filled to a 64-byte slot) and its channel tile's 64 bytes of values,
so L2 serves what one box fetched to the others, as in the carry. Every forming variant's state is checked exactly.

| a CTA-window | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| entries (union) | 255.6 (216.2) | 1,022.0 (459.1) | 1,022.3 (444.0) |
| stages of 8 rows: mean; the busiest warp's x 8 | 45.1; 46.5 | 141.8; 144.9 | 128.0; 153.0 |

- RESIDENT: each window's union rows copied once into a pool and its entries staged as slots (their lists and tokens
  read from global memory at their use), a CTA barrier, the fold, a barrier. This is not today's carry, whose fills of
  the next chunk run under the fold of this one.
- STREAMED: each warp's ring of `D` stages of 8 rows and their 8 coefficients (f16, in a 128-byte pad so every stage
  keeps the rows' 32-byte phase), each slot refilled as its step ends, across lists and windows. The producer walks its
  window's STAGE RECORD -- the warp's stages, each stage's 8 tokens and 8 coefficients (a pad row the stage's first:
  the same line, its coefficient zero), a plan a pre-pass emits -- copied into shared memory two windows ahead with the
  stage group it is issued in; a stage's coefficients ride in the slice copy's spare lane, its record, slot and
  per-window row bases are running pointers. The consumer's step reads its coefficients from the stage, its lane's row
  and swizzle offsets held in registers (opaque, so ptxas cannot re-form them each step); a list's entries arrive a
  window ahead in registers.
- STAGED (the stream's floor form): each warp's ring of 4 stages of 2,144 bytes -- a step's 16 rows (a tail's first
  8), its 16 coefficients lane-ready, the 16 tokens of the stage 4 on -- its stages one record stream across windows
  and digits (a pre-pass's plan, tokens global), so the producer reads a stage's tokens from the slot it refills,
  copies its rows (three 16-byte copies a lane) and its 96-byte record in one burst, and checks no window. A pad (an
  entry past a digit's, a tail's last 8 rows) is not copied: the slot keeps an earlier stage's finite row, which its
  zero coefficient cancels. A step reads its coefficients from its stage with one 8-byte load.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| resident (copies waited, lookups at their use) | 9,634 |  |  |
| streamed, ring 2 (a 16-token step's own two stages) | 3,562 |  |  |
| streamed, ring 3 | 3,210 |  |  |
| streamed, ring 4 | 3,062 | 8,018 | 8,706 |
| streamed, ring 6 | 3,072 |  |  |
| streamed, ring 4, records on mbarriers | 3,298 |  | 8,786 |
| streamed, ring 4, lookups from global memory at their use | 5,463 |  |  |
| bound: the fold with the producer, no row copies | 3,054 | 7,173 | 7,321 |
| bound: the same, records on mbarriers | 3,330 |  |  |
| bound: the fold alone, no producer | 1,894 | 5,121 | 5,138 |
| bound: the copies alone | 2,485 | 7,584 | 8,029 |
| bound: the copies alone, 40 boxes | 2,104 |  |  |
| bound: the copies alone, lookups at their use | 4,621 |  |  |
| bound: resident, the fold alone | 5,948 |  |  |
| bound: resident, the copies alone | 7,864 |  |  |
| resident, copies allocated in L1 (`.ca`) | 9,820 |  |  |
| streamed, ring 4, copies allocated in L1 (`.ca`) | 3,099 | 7,395 | 8,366 |
| STAGED, ring 4 (pads not copied) | 2,319 | 6,424 | 6,697 |
| staged, ring 4, pads copied as zeros | 2,394 | 6,457 | 6,701 |
| bound: staged, the fold with the producer, no row copies (pads as zeros) | 1,980 | 5,262 | 5,105 |
| bound: staged, the copies alone (pads as zeros) | 1,723 |  |  |

Instructions a MMA (ncu): the fold alone 6.3 at alt-k4, 6.2 at alt-k16, 7.0 at cohort-k16 (9.5 before its lane
offsets were held and its coefficients read from the stage: ptxas re-formed ~20 lane constants each step); streamed
14.5, 14.4, 16.6 -- the producer ~27 instructions a stage of 8 rows (a token load, two 64-bit sources, two
destinations, the two copies and ptxas's three fillers, the record pointers, the window check, the slot, the commit),
its structural floor ~22, against ~5.7 MMAs a stage at alt-k4. Staged: 8.7, 8.4, 8.9 -- its stream 2.4, 2.2, 1.8 a
MMA over the fold alone; a 16-token step 63 instructions (the producer 27 a 16-row stage: a token pair, three
sources -- ptxas forms each in a shift and a carry -- three pad tests, three copies and the record's, the record
pointer, ptxas's 3 fillers; the consumer's step 36). Shared-memory wavefronts a MMA (ncu bank reads and writes / 32):
the fold alone 1.6, 2.0, 2.2; streamed 3.4, 4.3, 5.0; staged 2.5, 3.4, 3.9 (4.1, 4.0, 3.9 with its pads copied as
zeros). L2 sectors a MMA: streamed 4.8, 6.1, 7.8; staged 3.4, 5.3, 6.4.

What it settles:

- The stream costs the fold 1.62x at alt-k4, 1.57x at alt-k16 and 1.69x at cohort-k16 (streamed against the fold with
  no producer); in instructions it is the producer's ~8 a MMA, its bookkeeping a stage of 8 rows.
- Staged, it costs 1.22x, 1.25x and 1.30x: a stage a step, one record stream (no window checks), the record riding
  in the stage's copy and a step refilling the slot the step before it read take the producer from ~8 instructions a
  MMA to ~2; what remains at alt-k16 and cohort-k16 is
  the copies' L2 traffic (84% and 77% of the L2's peak). Not copying pads takes a third of the shared-memory traffic
  at alt-k4 (2.5 wavefronts a MMA against 4.1).
- A copy's shared destination keeps its source's 32-byte phase: stages of 1,040 bytes (the coefficients appended
  unpadded, every other stage 16 bytes off) ran the stream 15-35% slower (unlocked runs, not recorded).
- Records land with the stage groups: on an mbarrier (`cp.async.mbarrier.arrive`, `ARRIVES.LDGSTSBAR`) the stream
  runs 8% slower at alt-k4, 1% at cohort-k16.
- A ring of 4 stages is enough: 6 gains nothing, 3 costs 5%, 2 (a 16-token step's own stages, nothing ahead) 17%.
- Lookups ahead are a requirement of the form: read from global memory at their use, the stream runs 1.79x.
- Allocating the copies in L1 (`.ca`) serves a box's re-fetches from L1 -- a pad row (its stage's first token's line)
  and a token two of its digits share: 36% fewer L2 sectors at alt-k4, 30% at alt-k16, 8% at cohort-k16 (ncu); where
  the copies bind on sm_86 it is faster -- alt-k16, its L2 at 80% of peak with `.cg`, 7.7%; cohort-k16 3.9% -- and at
  alt-k4, bound by the producer's instructions, +0.8% (resident +2%). But a `.ca` copy takes 5.4 shared-memory pipe
  cycles a 128 bytes against `.cg`'s 0.5 ([`smem_rates`](smem_rates.md)), and the pipe binds sm_80: `.cg`, the
  re-fetches removed at their source (a pad copied as zeros, reading nothing; shared tokens by the merge width).

Record: `measure/lab/results/row_stream/sm_86.json`.
