# `carry/carry_kernel.cuh` — the carry body

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`. The plan it runs on is
[`box.md`](box.md); the bytes are [`smem_ledger.md`](smem_ledger.md); the launch is
[`carry.md`](carry.md). The design's rulings are on card P81; this page is what the code
does, and why each piece is shaped the way it is.

<a id="frame"></a>
## The frame, in one picture

One CTA owns one BOX of `BC = 256` leaves for the whole call, eight warps, one CTA per SM.
The state for those leaves is fp32 in registers: a warp OWNS two of the box's sixteen
sixteen-leaf boxes (`w` and `w + 8`) and every channel of them. The call walks the sequence
in WINDOWS of 512 tokens; per window, in order:

1. the EDGE: the SNAPSHOT (the state's high halves written to shared memory once, the
   readout's operand), the tile counter reset, one CTA barrier; the fold's first chunks
   issued into their pool slots;
2. the READOUT over the read side's sorted tiles, dealt to warps by a shared counter;
3. the HEAD of the next window: the read side's sort, the write side's words;
4. the FOLD over the write side's union pool, each warp its own boxes, on landed chunks;
   then the barrier the next snapshot needs.

The two MMA phases are separate stages by ruling (P81, 2026-09-11): on sm_100 the TMEM that
holds the state during the fold cannot also hold the readout's staging, so the high halves
transit shared memory between the phases on every architecture. An interleaved form (a fold
step and a readout step alternating in every warp, dynamic tile dealing) was built and
measured a wash: the waits it hid were ~1K cycles of a sparse window once the sweep was
staged, and its step re-entry cost dense 5%.

The readout reads the snapshot and its own staged tiles; the fold reads the pool and writes
registers. Nothing in them touches until the window end: snapshot before either, fold
complete before the next snapshot, one CTA barrier at each. The head's sort has three sync
points of its own. Those five barriers are the window's. -- #window-loop

<a id="carve"></a>
## The carve, and why it is two different carves

The coefficient of token `t` on leaf `(b, pos)` is a Kronecker product,
`inner[t][pos] · outer[t][b] · gain[t]`, and it is never materialized.

- **The fold** is `S[leaf][ch] += Σ_t C[t][leaf] · V[t][ch]`: the contraction is tokens,
  the outputs are leaves and channels. Partitioning leaves across warps (the STATE CARVE)
  leaves every warp's result complete in its own registers, and lets one modulated A serve
  all eight n-tiles of V: the per-box work (two shuffles for the box's scalar pairs, four
  packed multiplies on A) is paid once per eight HMMAs.
- **The readout** is `y[tok][ch] = Σ_leaf C[tok][leaf] · S[leaf][ch]`: the contraction is
  leaves. Partitioning leaves would make every warp's y a partial needing a cross-warp sum;
  partitioning TOKENS (the POSITION CARVE) keeps each row complete in one warp. The price is
  that the readout needs every box's state, not just its own: the snapshot.

The rule underneath: partition an output axis and results are complete; partition the
contraction axis and they have to be summed. The value carve (channels, an output of both)
needs neither the snapshot nor a sum, but pays its per-box work once per HMMA; that is the
`kNT` knob's other end and the record is in FINDINGS §46.

<a id="layout"></a>
## A side's layout

`SideLayout` (`carry/layout.cuh`) classifies each level of a side's carve order by where its
digit falls against the sixteen-leaf position bits: INNER levels lie within them (their
runs make the box's tile, box independent), OUTER levels above (a scalar a box), a level
across bit four STRADDLES. It is derived on the host (`make_side_layout`) and handed to the
kernel in the parameter block, so every field is a constant-bank operand and none a register
— the earlier in-kernel struct cost 32 registers a side. Every array is indexed with a
compile-time level: a runtime index would put it in local memory.

The shipped arm's layouts are PLAIN: one inner level of sixteen digits, one outer level of
a digit a box. A composed or straddling arm is refused at compile time in the fill, the fold
and the readout (`static_assert`), where it would compose its runs per token.

<a id="factor-rows"></a>
## The factor rows

`abase[l]` is the box's first amplitude column at level `l`, which is also the liveness
words' digit row base. `run_bases` selects the inner and outer levels' byte bases from it
statically; `stage_box_words` stages both sides' words a window ahead by `cp.async`, ROUND-MAJOR
(`words(side, rho)`: a round's `kBoxRows` digit words contiguous), so the head's any-digit OR and
its sixteen box words are eight 16-byte loads a side rather than thirty-two scalar ones -- the
row-major form was MIO-bound (FINDINGS §46).

<a id="order"></a>
<a id="read-order"></a>
## The head

Read side: a counting sort of the window's tokens by FIRST LIVE BOX, the dead (and those
past the window) last; the ORDER (`order`, rank → token, u16), the TILE MASKS (`tilemask`, a
tile's live boxes) and the live count. A warp takes two (round) groups, lane = token: the
round's box words (below), the token's mask bit by bit, its bucket, the bucket counts by
`__match_any_sync`; warp 0 scans the count table in place into bases (a lane a bucket); the
scatter; the tile masks by reduction over the order. The `p.order` dial chooses the identity
order instead (one bucket for the live): the sort's A/B.

Write side: per round the box words, each warp's WORD (`warpwords`: tokens live in a box it
owns), the UNION word (`unionwords`: live in any box) and, by warp 1, the union's prefix by
round (`prefix`) and its total. No write-side sort, no lists: the fold compacts as it walks.

<a id="box-words"></a>
### The box words

`box_words` forms a round's sixteen box words from the staged liveness words: an AND over
the outer levels' digit words at the box's digits, over the inner levels' any-digit words,
and over the straddling level's class. The plain layout is two runtime row bases and static
offsets; a composed or straddling layout walks its level lists in uniform loops. It never
branches on a lane value, so the collectives that follow stay provably converged
(KERNEL_STANDARDS §20).

<a id="state"></a>
## The state

`State::c[kDealt][kNT][4]`: the owned boxes' state as m16n8k16 C-layout tiles (lane (r, q):
leaves `r`, `r + 8` of the box at channels `2q`, `2q + 1` of the n-tile). `State::m[kDealt][4]`:
each box's masses as an MMA accumulator whose column 0 is the mass (lane (r, 0): leaves `r`
in `[0]`, `r + 8` in `[2]`); the fold accumulates it with a ones column, so no reduction
ever forms it.

<a id="state-io"></a>
### The sweeps

The state crosses the pages as whole 128-byte rows, staged in shared memory: the high
halves in the snapshot, the low halves across the pool region (free at both edges of a call;
`kRegionBytes >= kSnapshotBytes`), the masses in the mass row. `state_copy<Store>` is the
CTA's coalesced copy between the pages and the stages, one 16-byte chunk an item dealt by
thread so eight lanes write one row, the masses a leaf a thread; `state_gather` joins a
warp's boxes off the stages into its accumulators, and `snapshot_publish<true>` is its
mirror at the exit (the window's publish is `<false>`, hi only). A page this call only reads
gives hi alone; a page without a slot, or inactive, is zero; the exit stores only the pages
this call writes. Once a call each, a rendezvous on each side of the stage.

The first form scattered 2-byte halves straight from the accumulator fragments: 128 stores a
lane, each address register's rewrite waiting on the previous store's scoreboard. At
alt-k4 that exit was 30% of the kernel's stall samples (12% dense), invisible to the phase
ledger until its launch carried a state plane.

<a id="snapshot"></a>
## The snapshot

`snapshot_publish`: each warp writes its READ-TOUCHED boxes' high halves into the snapshot as
`[leaf in read order][channel]` pairs (`chan_row_off`: the 16-byte chunk XORed with the row,
so the readout's `ldmatrix` over eight rows is bank free), and their masses into the mass
row in read order. Up to 32 KB once a window. A leaf's READ box is its read-order row's
sixteen; a row no read tile touches (`read_boxes`, the OR of the window's tile masks minted by
the head, in read-box numbering) is not published, and a write box with no touched row is
skipped whole behind `ops::once`; an unpublished row's mass is not written either (den reads
only live boxes' masses). In the alternating layouts a write box's leaves span all sixteen
read boxes, so nothing is skipped there; where the two sides' boxes coincide (cohort, the
structured cells) most of the 32 KB is. The exit's publish (`WithLo`) is every box.

<a id="pool"></a>
## The pool

The write side's union-live tokens, by rank, in chunks of `kPoolTok` (64 at DV = 64) across
`kPoolSlots` (2) slots, each slot a token's V row, inner run, outer run and GAIN PAIR (the
aligned word of the gain row holding it; the walker takes the half its token's parity
names), and a ZERO ROW (row `kPoolTok`) the tail of a list pads from, zeroed in the prologue (the
part harness's fill and fold drivers zero it too: a chunk with no idle lane never writes it). Slots are under
`mbarrier` pairs: `full` completes at `kThreads` landed-copy arrivals
(`cp.async.mbarrier.arrive.noinc`), `empty` at `kWarps` arrivals, a warp's after its LAST
GATHER from the slot, not after its MMAs — a row is free once loaded into registers.
`PoolCursor` carries a use count a slot for the parities (and `full_now` / `empty_now`, the
same asked once as a vote) -- a byte a slot, bumped WITHOUT A CARRY (`bump_byte`): a barrier
needs only a count's parity and, for a fill, whether the slot was ever filled, so a byte wraps in
place and a fill count wraps to 2, never to 0. A plain add carried the 256th fill of slot 0 into
slot 1's count and read slot 0 as never filled; that fill skipped its wait, and a dense call of 64
windows (`nl64k-dense`, 32768 tokens at this arm's pool of 8 chunks a window) hung on it until
2026-09-16. The ring cursor's bytes wrap the same way. A window's chunks take slots from 0 in turn; chunk 0 is
issued at the window's edge, under the head. The pool sits beside the readout's blocks, not
over them: both streams run at once (`smem_ledger.md#pool`).

`pool_fill`: the chunk's rounds by two ballots over the prefix, dealt to warps (offset
`(w + chunk) mod warps` from `r0`, and that plus `warps`; a warp without a round, or whose round has no live token -- its union word
is zero, which at N = L is most rounds a CTA sees -- skips it behind `ops::once`); a round is two passes
of sixteen tokens, lanes `j` and `j + 16` a token copying alternate 16-byte halves so each
instruction's pair covers whole sectors; a lane's token is live by the union word's bit,
its rank by popcount, in the chunk by range; an idle lane's zero-size copies go to the zero
row (a zero-size `cp.async` ZERO-FILLS its destination — the first form overwrote the next
warp's rows). One copy group; the arrive when it lands.

<a id="pool-fill-deal"></a>
The deal rotates with the chunk because a chunk's round span is a density property: at N = L
sparse a chunk's union tokens span up to sixteen rounds and any fixed offset deals them evenly,
but a dense 64-token chunk spans exactly two rounds, and the fixed offset (`r0 + w`) gave warps 0
and 1 every copy of every chunk while six warps skipped. That imbalance surfaced at the fold's end
barrier (fold rows 49.0K on warps 0,1 against 46.3K on 2,3,6,7 at flagship-dense, the fold plus its
end-edge wait constant across warps to 200 cycles). Rotating by the chunk keeps the deal a
bijection between the warps and the chunk's first sixteen rounds, so every round has one filler
and every warp still arrives at the slot's barrier.

<a id="fold"></a>
## The fold

`FoldStream`, a step function with a cursor. A step: if the warp still owes a fill (the
chunk after the one it last took), issue it when its slot is empty -- the barrier tested by
lane 0 and the answer parked in the warp's FOLD WORD, a shared load every lane reads back, so
the branch on it is a plain branch and not a vote; take the chunk's slot once full (else
return false: the caller does other work) and WALK the chunk once into the ring; then a
PAIR of fragments a step until the count is out; then the slot released, the next chunk.
Done when its chunks are folded AND its fills are issued (a warp that left with a fill owed
would hold the others). `fold` runs a stream alone to completion: the part harness's form.

**The walk** (`walk`): the chunk's rounds by the two ballots, TWO ROUNDS AN ITERATION; per
round each lane's token is KEPT when the warp's word has it and its rank is in the chunk, and
kept lanes write their entry -- the pool row in the low byte, the token's gain in the high
half, one word -- at the index their prefix popcount names in the ring (`ring`, `kPoolTokMax`
words a warp: the chunk's rows, fragment `n` at entry `16 n`). The two rounds' gain loads
issue together: one load's latency a round serialized the walk (at k = 4 a chunk spans a
dozen rounds, and the walk is the fold's largest item there). The entries past the count, up to a whole pair of fragments,
hold the zero row at gain zero; the fragment count reaches every lane through the fold word.
So the fragments after it are a COUNTED loop over static addresses, with no vote, ballot or
shuffle between two bursts: what an in-order warp can pipeline.

**The gather** (`frag_load`): a fragment's lanes load their rows off the ring at the
`ldmatrix` lane maps (`run_lane_row`, `v_lane_row`, a byte each) and their gain pairs off
two entries; A = the inner runs transposed (`ldmatrix.trans`: lane (r, q) holds positions
`r`, `r + 8` at tokens `2q..`, `2q + 8..`), the outer runs transposed the same way (lane
(r, q) holds boxes `r`, `r + 8`: pairs `[0]` (box r, low tokens), `[1]` (box r + 8, low),
`[2]` (box r, high), `[3]` (box r + 8, high)), V as B (four `ldmatrix.trans` for eight
n-tiles); the outer pairs scaled by the gains; then a box: its pairs by shuffle from the
lanes holding boxes `b & 7`, A scaled by them (four packed multiplies). The result is one
operand set (`FragOperands`: `ab` a box, the V tiles), 24 registers.

**The burst** (`frag_mma`): a box's `kNT` HMMAs into its state and one against a ones column
into its mass. A pair alternates two operand sets and INTERLEAVES the next gather with this
burst: box 0's fifth HMMA is ordered after the next set's V loads and box 1's first after
its whole gather (`after`, a `prmt` that selects the operand whole but depends on the other
set), so ptxas issues the loads inside the first half of the burst and the multiplies and
shuffles inside the second, in the pipe time this burst would have idled through. Why it is
needed: a warp issues one or two HMMAs ahead of the pipe and in order, so any chain longer
than a slot placed after a burst is exposed whole on a warp whose partner is not issuing
(calibration.md, the fragment rows); ptxas at 213 registers sinks loads to their uses and
regroups the bursts unless a dependency forbids it. Measured at nl64k-dense: a fragment 1,325
-> 1,128 cycles paired, the fold 51.4K -> 47.0K a warp a window (alt-k4 7.8K -> 7.5K),
registers 213 -> 239 (peak live ~157 in the fold; the allocation's peak sits in the fill
and the head), the ring 2 KB more shared memory.

<a id="readout"></a>
## The readout

`ReadoutStream`, a step function with a cursor that PERSISTS across windows. Tiles are DEALT:
lane 0 takes the next tile off the window's shared counter (`atom.shared.add`, one counter a
window parity) into the warp's take word and every lane reads it back -- a shared load is
warp-uniform to the compiler where a shuffle's or a reduction's result is not, and a value it
cannot prove uniform puts every collective after it on the convergence-checking path (51
calls, 19 divergence branches in the first form). `prime`, right after the next window's head:
ONE tile taken a warp (every warp has its first before any warp a second; a sparse window has
about as many tiles as warps, and the old two-a-warp deal left half the warps idle) and its
copies issued into the warp's free ring slot, a window ahead -- the rings are dead through the
fold, so the next readout starts on landed tiles instead of paying the ~2K-cycle gather of its
runs per tile exposed. `begin`, at the readout: the second tile taken and issued. `readout_issue`
stages a tile's inner and outer runs into the warp's private ring (`rring`; lanes `l`,
`l + 16` a row; a rank past the live count zero-filled); the slot's RING BARRIER (`ring_bar`,
32 landed-copy arrivals) says when it landed. A step: the landed tile (`readout_tile`), then
the next taken tile issued into the slot it freed; a tile not landed returns false and the
caller waits on the ring. A warp with light work takes more tiles. No CTA barrier in the phase.

`readout_tile`: A = the inner tile by `ldmatrix` once; per live box (the tile mask's set
bits): the rows' outer factors off the outer tile, A scaled per row (four packed
multiplies), B the box off the snapshot (`ldmatrix.trans`, two n-tiles a load), `kNT`
HMMAs into y and one against the box's mass column (`massrow`, lanes `r == 0`) into den.
Then the rows out: four rows a pass through the drain stage (`drain`), row-major, so each
reduction instruction covers one 128-byte line (a reduction costs by the sectors it touches:
the accumulator's own layout, eight rows' sectors a red, calibrates at ~200 cycles of issue a
red against ~10 -- `calibration.md`, and the register drain built on it lost 4% at dense). The
drain is instruction-bound, so every address in it is an immediate offset from a lane
constant (the lane's stage row, its stage column, its output column; `smem_ledger.md`), the
pass's loads issue before its reductions (both are asm with memory clobbers, so a load
written after a reduce would wait behind it), and a dead rank (past the live count, its row
zero-filled by the ring) reduces its zeros into the window's first token by a select, no
branch a row. den by reduction from the lanes holding it. `num` and `den` are summed across
owner CTAs, so every row leaves by reduction in every form.

<a id="edges"></a>
## Edges

`Edge::kCounts` and `Edge::kOrder` are the head's named barriers; `ops::rendezvous` the
window's two; `warp_edge` (barrier `kWarpEdge + warp`) orders a warp's private rows across
its lanes and publishes its landed copies. The pool's slots and the readout's ring slots are
`mbarrier`s, never a CTA barrier.

<a id="window-loop"></a>
## The window loop

`window_loop`: the edge -- the snapshot (when the read side has a live token), the previous
window's word copies landed (`cp.async.wait_group 0`), one barrier -- then the `FoldStream`
constructed (the first `kPoolSlots` chunks issued into their slots, free since the last
fold's releases), the readout run to completion on the tiles primed a window ago
(`readout`: `begin`, then the `ReadoutStream` stepped, waiting on its ring when a tile has not
landed), the next window's tile counter reset and its head, its readout PRIMED (one tile a
warp into the rings, dead through this fold), the fold run to completion (`fs.step()`,
`fs.wait()` when a slot is not ready), one barrier. The fold starts on landed chunks because
they were in flight under the readout and the head; the next readout starts on landed tiles
because they were in flight under this fold.

<a id="parts"></a>
## The parts (the composer's switches)

Each MMA phase is built from PARTS, compile-time switches read from the generated
`carry_parts.inc` (`ROLA_CARRY_PARTS`, `docs/build.md#carry-parts`). Production carries every
part. A STUB keeps its part's HMMAs, with the same count, atom and accumulators, and drops the
part's other work, so a composition's phase time minus another's is that part's cost in wall time:

| part | real | stub |
|---|---|---|
| `readout.stream` | tiles dealt by the shared counter, the ring's issue, landed test and wait | a warp reads every eighth tile by index; no take, no ring |
| `readout.loads` | A off the ring slot; per box the outer factors, A scaled, B off the snapshot, the masses | fixed bf16 operands in registers |
| `readout.drain` | the rows out through the stage by reduction, den | every accumulator summed once into a stored sink, so no HMMA is dead |
| `fold.pool` | the pool fills and the slots' full/empty barriers | no fills, no barriers: a chunk is taken at once |
| `fold.ring` | the walk's gain loads and ring stores, the fragment's entry off the ring | a fixed entry (pool row 0 at gain 1.0); the walk still counts the kept rows |
| `fold.loads` | the fragment's pool loads, gain shuffles and scaled A | fixed bf16 operands in registers |

What no stub removes is what sets the HMMA count: the head's masks and order, the tile's box mask
walk, and the fold's count of kept rows a chunk. The composer checks every composition's HMMA count
against the kernel's. The switches are `if constexpr` branches around the real statements, left
verbatim: with every part real the arm's instructions hash identically to a build without them;
only the debug line table moves.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` accumulates a warp's cycles per phase (`CarryPhase`) in its shared row by
lane 0 (`%clock64` laps), added into `CarryParams::ledger` at the kernel's end when bound
(`carry_ledger_bind`). A debug instrument; unbound in production.

The head laps at its own two edges: `head_words` is pass one (the box words, masks, buckets,
counts, warp and union words) up to the counts edge, `head_scans` is the count-table and union
scans up to the second counts edge, and `head` is what remains — the scatter, the tile masks, and
the next window's word staging. The three partition the old single `head` row, so totals stay
comparable.

Every CTA barrier in this kernel assembles as a deferred-blocking barrier: a warp arrives and keeps
issuing, the clock read of the next lap depends on nothing and issues at arrival, and the wait
materializes at the first instruction that depends on another warp's stores. A lap after a barrier
therefore charges that barrier's release to the NEXT phase, and per-warp rows beside a barrier are
not comparable. `head_scans` is the first counts edge's release plus the scans: at N = L alt-k4 it
reads ~1,740 cycles a warp a window, of which the scans are ~460 on warp 0 and ~250 on warp 1, and
the rest is the readout's per-warp imbalance arriving at that edge. A wait is read only from arrival
stamps taken BEFORE the barrier (the edge-wait meter, a measurement variant: the latest arrival minus
a warp's own, gathered after a later barrier; it costs the kernel 10-14%).

<a id="phase-trace"></a>
## The phase trace

The ledger says how long a warp spent in each phase; the trace says WHEN. Bound through
`carry_trace_bind` (a `[ctas][warps][cap]` int64 tensor and the warps per CTA), every lap of
the first `ctas` CTAs' warps and every step of their streams stamps `%clock64 << 8 | event`
into the warp's row, lane 0, one 64-bit store an event: a lap stamps the phase that ended
(`CarryPhase`), a stream stamps the activity that begins (`CarryTraceEvent`: the fold's walk,
fragment, wait and fill; the readout's tile, issue, wait and drain). A row that fills stops
recording. Unbound, the stamps are a predicated branch each and the kernel holds four more
registers (209 against 205 at flagship); one CTA's clock is one SM's, so a CTA's warps are on
one time base and CTAs are not.

`tools/phase_trace.py` reads it as three things the ledger cannot say: the finer ledger (a
warp's cycles a window by activity, with the readout's drain apart from its boxes), THE
SCHEDULER'S VIEW (the two warps a scheduler: the share of time zero, one or both are inside an
HMMA-issuing activity, and what they are doing when neither is), and the lockstep (the spread
of the warps' phase starts within a window). The scheduler pairing is `warp % 4`.

**Read at N = L dense (nl64k-dense, 2026-09-16).** A window is 109.6K cycles a warp; the
HMMA work in it, 1,152 an atom a warp at 32.5 cycles a scheduler for two warps, is 74.9K: the
pipe is 68% busy INSIDE a CTA. The warps are not out of step -- the fold's starts spread 139
cycles, the readout's 972 -- and both warps of a scheduler are inside HMMA-issuing activities
65% of the time. The pipe's idle time is not bursts of loads: the loads calibrate free under
a burst (`calibration.md`). It is (a) the 19% of the span where neither warp issues, of which
the readout's drain is half (both warps drain together, 2.8K cycles a tile), the head and
snapshot a quarter, the pool fill a tenth; and (b) the fragments' own rate, 1,325 cycles for
eighteen HMMAs against 1,170 for two warps sharing the pipe. The other loss is outside the
CTA: 256 boxes on 80 SMs at one CTA an SM is 3.2 waves, and the kernel runs four, so the
card sees 20% of the wall idle by wave quantization (the per-CTA 8.4 ms times four is the
33.7 ms measured under the lock).

<a id="budgets"></a>
## Budgets

The budget of each component is the design's own operation count, written in
`measure/carry_model.py` and generated into `tools/budgets/carry.json` per cell
(instructions per CTA-window; the formulas name every operation they count). The region
ledger (`tools/region_ledger.py`) attributes a launch's executed instructions and stalls to
the component each SASS instruction inlines from, with barrier spins split out, and is red
over budget. Flagship dense: head 4.0K, fill 5.6K, fold 21.0K (4,608 HMMAs), readout 20.4K
(4,608 HMMAs; the drain counted from the kernel: 12 nt + 64 a tile); alt-k4: 16.0K in all, the
head the largest block.

<a id="part-harness"></a>
## The part harness

Its driver (`measure/harness/carry_parts/carry_parts.cu`) is built ahead of time beside the arm
under `ROLA_BUILD_PARTS=1` (`docs/build.md#parts`), never by a JIT.

`measure/harness/bench_carry_parts.py` builds `measure/harness/carry_parts/carry_parts.cu`
(one driver kernel a component, each the kernel's own prologue plus that component, on the
call the kernel would run) and checks each against a pure reference: the head's order,
masks, words and prefix; the fill's slots byte for byte, zero row included; the fold's state
and masses against the Kronecker fold with the kernel's bf16 arithmetic emulated. `--ncu`
adds executed instructions per CTA-window beside the budget. KERNEL_STANDARDS §19.
