# `carry/ampere/schedule.cuh` — the window's plan and the rule

Mirrors `csrc/rola/src/carry/ampere/schedule.cuh`: each window's PLAN, made by the plan pass before the carry runs from the
facts pass's counts -- one word (its image mode, its read and fold chunks and their counts) and a 16-bit STEP word a
step -- and the RULE it is made by. A window's ENTRIES are its read chunks, then its fold chunks; each sits in one of
two SLOTS of the AREA (the shared memory past the tenants, `plan.md#plan`), the rule placing entry `t + 1` beside
entry `t` wherever the two fit side by side. The carry runs a window in one fixed order over the record's chunks
(`carry_kernel.md#kernel`, 2026-10-25): shared memory holds what is active now and what lands next, sized by the
window's actual counts, and a bigger card's area takes larger chunks with no other change (ruled 2026-10-14; the plan
as (phase, chunk size) words and the two-slot rule, 2026-10-16). The plan's Python model is `tests/oracle/carry_plan.py`, which a gate
holds the plan pass to word for word.

Symbols: `U` a side's union slots in the window, `C` a chunk's slots (a multiple of 64 below the union, or the whole
union), `E` entries, `B` the area's bytes, `n` a window's entries, `t` a step (`n` the step that only retires).

<a id="plan"></a>
## The plan

`PlanWord`: mode | read chunk / 64 << 1 | fold chunk / 64 << 9 | read chunks << 14 | fold chunks << 22 (`plan_word`):
5-bit fold fields, so a write union of a whole 1,024-token window is one word.
A window's RECORD is `plan_words(g)` words at `g` query heads a write head: the plan word, then the step words in
pairs (`max_steps(g)` = 8 `g` read chunks + 8 fold chunks + the retiring step; 12 words at one head). The CTA stages it with the window's inputs, by parity, and
reads it at each use: `stream_of` gives the window's mode, chunks, unions (its staged counts), entries and image;
`entry_at` entry `t`'s chunk (the read chunks first: its first slot `q C`, its slots `min(C, U - q C)`), two words a
consumer re-extracts its fields from; `step_of` step `t`'s word.

<a id="rule"></a>
## The rule

- THE SLOTS (`SlotIndex`) alternate down the CTA's stream across windows: the stream's `k`-th entry takes slot `k mod
  2`. The top slot's entry ends at the area's top (`B` less its bytes); the low slot's sits above its window's image
  where the image is live at its fill -- a read chunk; the first fold chunk filled while the last read computes -- else
  at the bottom (`slot_place`).
- THE IMAGE is whole (64 KB), stored at the window's store, or the halves (32 KB) each read chunk stores itself, at the
  bottom; it is live from the store to the last read chunk's readout.
- AT STEP `t` (the rule's model, which places the entries): entry `t - 1` is retired -- every warp done with it, a
  read chunk's panel drained; entry `t` is filled if it was not filled ahead; the NEXT entry -- `t + 1`, or at the
  window's last the next window's first -- is filled now where it fits beside entry `t` and the live image
  (`fits_beside`); entry `t` computes. A fill that does not fit waits for its own step: a pair that never fits runs
  serially and nothing else changes (R22, degrading smoothly).
- HOW THE CARRY RUNS IT (2026-10-25, `carry_kernel.md#kernel`): the chunks in the record's order at the rule's places,
  each read chunk's fill issued once the chunk before is drained (the first's at the window's top, under the maxima),
  each fold chunk's once every warp is done with the chunk before; a rendezvous before the retire, before a fold chunk's
  fill after the first, and before a fill that takes the drained panel's bytes (`reuse`). The places are the rule's,
  so a fill lands on nothing live but a panel still being drained, which `reuse` marks; the rule's ahead and next bits
  describe fills the carry runs in sequence instead.
- THE RENDEZVOUS: before the retire where a drain or a fill follows; before a fold chunk's fill after the first; and
  where the step word says a fill takes the bytes of the panel being drained (`reuse`, predicated:
  `../common/atoms/barrier.md#rendezvous-if`). Only the panel: past the readout's rendezvous a read chunk's rows and entries are dead,
  and its drain reads its panel, the union's tokens and the factors.
- A FILL FOLLOWS THE DRAIN (measured 2026-10-16, nl64k-alt-k4 on sm_86): a fill clear of the drained place issued
  before the drain made the drain contend with its copies (the drain 1,890 to 3,934 cycles a warp-window); after it,
  the launch is 5.61M cycles against 5.65M.
- THE STEPS: one an entry, then one that only retires the window's last (its read chunk's drain, the units where the
  window folds nothing); a window of no entries has the one step, and it fills the next window's first.

The rule needs two slots, never a ring: the carry runs one window ahead of its fills and no further, so its shared
memory holds two entries at a time whatever the call's length.

<a id="steps"></a>
## The step words

`StepWord`: the entry's place / 128 | filled ahead << 11 | the next entry filled now << 12 | a rendezvous after the
retired read's drain << 13. FILLED AHEAD: by the step before, or for a window's first by the window before (window 0's
before the loop). REUSE: the step's own fill meets the panel of the read entry the step retires (`drained_of`: its
offset past the rows, its bytes -- the num panel and the den words) -- the carry fills every entry at its own step, so a fill the rule
would issue ahead is not what waits on a drain. The carry reads each entry's PLACE from its step word once a window and
its REUSE where a fill follows a drain (`carry_kernel.md#kernel`); the ahead and next bits are the rule's own and not
read (the plan pass writes them, and the gate holds every bit to the model word for word).

<a id="buffers"></a>
## The buffers

A chunk's buffer is laid out at its OWN slots (`parts.cuh`'s `read_panel`, `read_dens`, `read_ents`, `fold_ents`): a
read buffer's rows (`C x row`), its num panel (`C x 64` at f16 pairs, `C x 128` at fp32: the call's reduce dtype,
`panel.md#words`), its den words (`C x 16`), its entries (`E x 8`; a dense read side's two exit words a slot in their
place, `readout_dense.md#row`), each part 128-byte aligned; a fold buffer's rows
(slices then values) and entries (a side with no list level: each slot's `(F, m)`, 8 bytes a slot, in their place).
`read_bytes` (at the call's `reduce`) and `fold_bytes` are their sizes, so a chunk's footprint is its actual count,
never the window's largest chunk; every size is a multiple of 128, so a place is too. The IMAGE (`ImageMode`) is whole
(32 planes, 64 KB) or a 16-plane half (32 KB). An entry's row offset is the union's, so a chunk from a multiple of 64
takes its pool back by its first slot, and its slot from the row (`../carry.md#lists`).

<a id="totals"></a>
## The segment sums

`side_counts`: a side's union slots and its segment sums (`Sums`): `S[j]`, its group's lists' entries before union slot
`64 (j + 1)`, 8 a query head of the union -- the facts pass's exact group totals, read where used
(`facts.md#lists-out`). Any chunk's entries are a difference of two of them (`entries_of`); `chunk_max` is the most any
chunk of `64 m` slots holds.

<a id="chunks"></a>
## The chunks

`largest_chunk`: the whole union where one buffer of it fits; else the largest `64 m` slots below the union whose
largest buffer fits -- the fewest chunks. A read chunk is sized beside the halves' 32 KB; a fold chunk in the whole
area. Every chunk pays its fill, its landed wait and its fixed work, and on a halves image each read chunk re-stores
both halves; a second buffer in flight (chunk `t + 1` landing while `t` computes, the rule until 2026-10-19) buys less
than the chunks it adds. Measured (2026-10-19, sm_86 locked, forced forms against that rule): flagship-alt-k16 3 read
chunks a window -> 2, 1,781 -> 1,520 us; flagship-cohort-k16 2.83 -> 1.88, 2,363 -> 1,983; nl64k-alt-k4-g4 7 -> 4,
12,217 -> 8,522; nl64k-dense 4 read and 3 fold chunks -> 2 and 2, 31,051 -> 29,028; a whole image beside smaller
chunks slower than halves beside fewer (alt-k16 whole at 128-slot chunks 1,778).

<a id="planner"></a>
## The planner

THE RULE'S PLAN (`rule_plan`): the image whole where it fits beside the largest read buffer (64 KB and one), else in
halves; the fold's largest chunk. A FORCED FORM
(`CarryParams::forced`, `rola.ops.carry.forced_word`: a mode, a read chunk and a fold chunk, each the rule's, the union
or a multiple of 64) is reserved for the gates and the measurements; a window takes it where every chunk of it fits
alone beside its image and a whole image has reads to serve (`forced_plan`), else the rule's. Every arm hands its
state off through the image (`handoff.md#same-split`). THE LEAST PLAN (`least_plan_fits`, at the call's reduce dtype, the
launch's refusal): the halves beside one 64-slot read buffer of lists full on every token, and one such fold buffer.

Why a rule, not a search (measured 2026-10-14, nl64k-alt-k4 on sm_86, every named form timed warm): the rule's form is
the fastest. Each extra fold chunk costs ~500 us there -- it splits every list's run into two partial steps, and the
fold's steps are latency-bound list by list. A search over the named forms by a per-item cost model ranked them
inverted on unmeasured prices and cost 661 us a call.

Worked, on the even box (64-byte read rows, 128-byte write rows; the area 88,832 bytes on sm_86): a nl64k-alt-k4 window of ~216-slot unions and ~260 entries a side takes the halves, one read chunk (33.3 KB at
f16 pairs) and the whole union's fold chunk (29.8 KB), which misses fitting beside the read chunk and the halves by
~6.9 KB: it is filled at its own step at the bottom, once the read is drained, and the next window's read chunk is
filled at the top beside it. On sm_80 the image is whole and the fold chunk lands beside the readout.

<a id="pass"></a>
## The plan pass

`window_plan_kernel`, launched before the carry on its stream: a block a (box class, batch-head), a thread a window of
each 128-window tile (`kPlanThreads`). Each thread its window's counts (`window_counts`; a BOX CLASS is the pair of the
read and write list levels' groups, which fixes a CTA's counts, `box_classes`) and plan word (`win_of`); the slots'
parity at each window, an exclusive scan of the windows' entry counts' parities (the tiles before carried); its first
entry filled ahead where the window before's last entry -- re-derived from that window's counts -- leaves it room;
then its steps, the next window's first decided from that window's counts the same way, so the two threads agree. The
record to `CarryParams.plans` (`[BH][windows][classes][plan_words(g)]`). 12.1 us at nl64k-alt-k4 (8,192 windows); gated
word for word against the model on every form cell under a spread of forced forms, and the model's rule held to never
filling over anything live on random windows (`tests/oracle/test_carry_plan.py`).

WHY THE PASS EVALUATES THE RULE, NOT THE CTA (measured 2026-10-16, nl64k-alt-k4 on sm_86): every warp re-deriving each
step's places from the counts cost ~3,500 cycles a warp-window, 13% of the window, on the critical path between the
phases; read off the step words, the step's bookkeeping is a few shifts.

Per architecture: the same rule everywhere, its area the card's (`hardware.json`); a larger area takes whole images and
larger chunks; on sm_90 the fills become TMA copies on the same slots' barriers. Scaling: the pass a thread a window,
O(1) each beside the scan; the record O(chunks) a window; nothing O(L) or O(N) in the carry.
