# `csrc/rola/src/facts/facts.cuh` / `facts.cu` — the facts pass

Mirror doc for the one pass that reads a call's bundle before its consumers do: both sides' class-1 liveness words
and their tile unions, the window's columns (the factor products), and each side's lists, an entry a slot and a byte,
in one launch. It is an INDEXER: it emits what is the data's alone, and nothing of a card's -- no exponent, no
coefficient in an operand's format, no layout offset (KERNEL_STANDARDS §29.5). What the words ARE — their layout,
the folds over them and why each is the grain it is — is [`liveness_contract.md`](liveness_contract.md), which this
kernel does not restate and must not contradict; its host seam is [`facts_api.md`](facts_api.md).
`rola.engine.facts.liveness.liveness_words` is the words' pure-torch MODEL, `rola.engine.facts.lists.final_lists` the
lists', and `tests/integration/test_facts_pass.py` holds the kernel to both.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B_l` = level `l`'s PADDED width; `b_l` =
its LOGICAL width, the rows' own; `L` = tokens in the call; `BH` = batch × heads; `rows = Σ_l B_l`; **word** = 32
consecutive tokens of one row; **window** = the card's W tokens (512 on sm_86: 16 words); **group** = 32 consecutive digits of a
level; **unit** = one (side, level, group) of a window, the work of one CTA; **list group** = the `G` consecutive
digits (16 or 32, the side's) one union and its prefix rows span ([below](#list-groups)).

## What it reads, and what it does not

The bundle as the producer wrote it (`docs/api.md` §2.1): each level's rows `[B, T, H, b_l]`, u8 digits, read where
they lie — a view of the producer's packed plane or a tensor of its own, by its strides, under the ROW ALIGNMENT LAW
([`liveness_contract.md`](liveness_contract.md#alignment)): every level's rows start on 32 bytes and step batch, token
and head by multiples of 32. It also reads the write levels' scales and the value scale for the
columns — a few words a token.

A DENSE level (the bundle's `sparse` flag unset; `rola.ops.facts.call_statics`) is neither read nor written: its words
are the width mask over the live tokens, which every reader takes from the side's statics
([`liveness_contract.md`](liveness_contract.md#statics)), so its rows and groups of `words` and `unions` are left as
allocated. Every sparse level's rows are read once, whole, whatever their support: the read is density-blind.

<a id="the-grid"></a>

## The grid: a CTA a unit

`kFactsWarps` = 8 warps a CTA, one CTA a UNIT; at W = 1,024 two CTAs an SM by `kFactsCtasMin` (a list lane holds
twice the words and boundary pairs, which spill under four CTAs' 64 registers); at W = 512 four CTAs an SM on sm_86 (its 18.7 KB of shared memory a CTA; its
registers a thread held to four CTAs by `kFactsCtasMin`, 46 used, which allow five) and five on sm_80 (registers): a
unit is a plan-fixed load
and its work, independent of every other unit, so the hardware's block scheduler balances the units' unequal costs and
several CTAs' loads and work overlap on an SM (R22's independent units). The grid is `(per_window, windows, BH)`, `BH`
the write heads' batch-heads (`B·H_w`; grouped queries, [`api.md`](../../api.md#grouped-queries)): a window's units by
x, innermost, so the CTAs at work together read one head's window of every level -- the bundle is stored head-major
(`[B, H, T, ...]`), a contiguous block of each side's plane -- and no CTA divides to find its unit (the host refuses
more than 65,535 windows or batch-heads).

<a id="the-plan"></a>

## The plan: a window's units

`plan_units` (host) lists a window's units once a call, the same for every window (`FactsParams::unit`, coded by
`unit_code`: kind, side, level, group, head): for each side and level in order, a `kUnitList` unit a group of the
side's LIST LEVEL (its one sparse level, `rola.ops.facts.list_level`) and a `kUnitVote` unit a group of every other
sparse level -- on the read side one a query head of the write head's `g`, its head in the code; and one
`kUnitColumns` unit for the window's columns. A read list unit takes
its write head's query heads IN TURN, one union over their rows (the ranks below): head `j`'s window loaded, voted and
its bits out at its slot, ranked and listed past the heads before it, the CTA's shared memory reused a head. A
side with two or more sparse levels gets its words and no lists: they would be COMPOSITE, keyed by a digit of each
level, which this pass does not build.

<a id="list-groups"></a>

## The list groups: a unit's whole group or each half

A side's LIST GROUP `G` (`FactsParams::list_group`, the call's: the digits of the side's list level that one box of
a carry arm reads, `rola.ops.carry.lists_form`) is the digits one union
and its prefix rows span: 32, a unit's whole group, or 16, each half (`kListGroupMin`; `kUnitGroups` = 2 at most a
unit); the host refuses any other. The unit stays a 32-digit group -- its load, vote, words and tile unions are the
liveness contract's at every `G` -- and at 16 it ranks two unions, list group `q` its digits `16q .. 16q + 15`, each
with its own slots, bounds, prefix rows (`G + 1`), totals, gap tail and size, at its index among the side's
`ceil(B / G)` groups of the window (`list_group_index`); a half past the level's width is none (`list_group_in`). The
vote ORs a block's words into both halves' union words (a half's words zero at 32); the ranks scan both halves' counts
as the 16-bit halves of one word, so one shuffle scan serves both, and a unit keeps both halves' slots before a head the
same way (`half_of`; a group's slots a window are `max_group` heads' at most) -- at W = 1,024 the pass sits at its
128-register cap, and a pair of counts a half held apart spilled; a list lane takes its own digit's half; warp `q`
writes half `q`'s prefix rows.

Why a half: a box that holds 16 digits of its list level reads a union of those 16; a union over 32 would hand it
another box's tokens as its own.

THE TWO PASSES: the kernel is compiled, at each build, for a unit of one list group (`facts_pass<BuildTag<B>, 1>`, every side at 32: the halves'
work compiled out, the code before them) and of two (`facts_pass<BuildTag<B>, 2>`), and the host launches the one the call's groups
need, each at its own tenants (`FactsSmemAt<W, G>`). A split call pays two warp reductions a block, a second gap
tail and a second union's bytes, the same at every density (unmeasured on the indexer); sm_80, sm_90 and sm_100 run the
same passes (the window's words double at W = 1,024, the halves' scan with them).

<a id="shared-memory"></a>

## The CTA's shared memory

`FactsSmemAt<W, G>` by offset, dynamic, at W = 512 18.6 KB a pass of one list group a unit and 18.7 KB of two:

| tenant | bytes | live |
|---|---|---|
| the window: 512 tokens × the group's 32 bytes | 16,384 | the load to the lists' last entry |
| the bit tile (`tile_at`) | 2,048 | the vote to the end |
| each list group's union words | 64 a group | the vote to the ranks |
| each list group's bounds (a boundary's first token) | 32 a group | the ranks to the lists |
| a head's lists' boundaries (`kTable`, u16) | 576 | the lists to the boundaries out |

<a id="load"></a>

## The load

A unit's window is 512 tokens × 32 bytes, the group's digits of each token's row: two `cp.async` 16-byte quanta a
token, a lane pair a token (one L2 sector request: [`ops.md`](../common/atoms/copy.md#stage-tile-n)), token `t` at region
`t / 32` (1 KB each) row `ring_row(t % 32)` = `4 (t % 8) + t / 8` of 32 bytes, so the vote's lanes meet 32 banks.
Bytes past the rows' width or a token past `L` are zero-filled (`ops::stage_tile_n`), never read. A list unit loads
nothing else: the scales are the columns unit's, which loads no window.

<a id="vote"></a>

## The vote: a lane a (token block, digit quad)

Lane `(tk, dq)` = `(lane / 8, lane % 8)` reads token `8 tk + i`'s digits `4 dq .. 4 dq + 3` as one word, `i` 0..7.
`((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x` sets bit 7 of each byte iff the byte is nonzero (`nonzero_bytes`); shifted right
by `7 - i` it lands at bit `i`, so byte `k` of the lane's register holds digit `4 dq + k`'s 8 token bits, and a 4 × 4
byte transpose across the token blocks (two `shfl_xor` exchanges) leaves lane `(tk, dq)` digit `4 dq + tk`'s 32 token
bits, masked by the digit's width-mask bit (`digit_in`). Warp `w` votes the window's words `2w` and `2w + 1`, and ORs
each over the warp into each list group's union word (two warp reductions: the digits below the side's list group, and
the rest).

<a id="window"></a>

## The window

The pass builds each window's lists at its build's W (`FactsPlan<B>`, its plan: [`../../architecture/families.md`](../../architecture/families.md)): `kWindowTokens` tokens, `kWindowWords` words
of 32 tokens, and the stages map onto that window by four counts derived from it -- a warp's 32-token blocks
(`kWarpBlocks`), a thread's tokens (`kThreadTokens`), a list lane's words (`kLaneWords`, eight lanes a list) and
boundaries (`kLaneBounds`) -- so a stage's loop runs over its count and at W = 512 every count is the 512-token pass's
own. The shared tenants are `FactsSmemAt<W>`; the device code is `FactsAt<FactsPlan<B>>` and the host launches the
card's build (`build_switch`) at that plan's tenants. A
write head's read side merges at most `max_group(W)` query heads: `kMaxGroup` = 16, the parameter block's bound and the
same in both compile passes, under what a union slot's head field holds at W (`window_heads`: 64 at 512, 32 at 1,024;
[`window.md`](../common/window.md#token-field)) -- 16 at both windows. MEASURED at W = 512 only (sm_86, 16 heads: 8,192 slots a union); a wider window's
16 heads first run on its card.

<a id="tiles"></a>

## The tiles and the bits out

A unit's words go into a shared TILE (`tile_at`: digit `d` word `j` at `w d + (j ^ x(d))` for the window's `w`
words, `x(d) = d / 2 % 16` at 16 words and `d % 32` at 32, so a warp's column writes, a digit a lane, and its row
reads each meet 32 banks); after the barrier each digit's window of words goes out as one run of `4 w` bytes and each
64-token tile's union, a digit a lane, as one ballot (warp `w` its `kWindowTiles / 8` tiles).

<a id="columns"></a>
<a id="factor-products"></a>

## The window's columns

    m(t) = Π_l scale_write_l(t)      (the levels in order: the write side's shared scale)      F(t) = m(t) × s(t)

`s` the value scale, a token past `L` zero. The window's `kUnitColumns` unit loads each token's scales and stores its
`(F, m)` to the `[BH, L, 2]` plane `fm` (the factor products: a consumer reads a token's two in one 8-byte load; at 8
bytes a token-head it adds no `O(L·N)` term). It touches no shared memory and votes nothing.

<a id="ranks"></a>

## The ranks: one slot map a (window, side, list group)

Every slot-addressed consumer -- the readout's pool and panel rows, the fold's pool rows --
reads one slot map a union (ruled 2026-10-12). A union token's SLOT is its RANK, its place among the union's tokens in
token order (ruled 2026-10-13). After the vote's barrier every warp scans the 16 union words' counts, both list
groups' as the 16-bit halves of one word (a 16-lane shuffle scan), and its two words' tokens go out to each group's
`union_tok` at their slots -- a word's tokens at consecutive slots, so the
stores coalesce -- every slot past the union `kGap` (token 0 with bit 15 set, so a row fetch masked
to the window stays inside it). The union's size is `union_count` (int32); a consumer
rounds it to its own step. The token at every 64th slot is a BOUND, posted in
shared memory (`kWindowTokens` past the union): a list's tokens before bound `k` are its entries before slot `64 k`,
which the lists count without walking.

A GROUPED read side ranks its write head's query heads' rows as one union a list group, head-major: head `j`'s union
tokens take the slots past the heads before it (`at`, a list group's own), each slot's ROW `W j + t` its `union_tok`
(the carry's fills and drains decode a row by a shift and a mask). A head's bounds are the tokens at the merged union's
64th slots that fall in its rows -- at most eight, from boundary `k0 = at / 64` -- the rest `kWindowTokens`; the gaps
and the size are written after the last head.

The slots are not colored. A coloring that put each step's 8 tokens on distinct residues mod 8 (the bank groups of
every slot-indexed row) made the carry's gathers conflict-free, but it padded each list's passes to its most-used
color and was most of this pass's time; on sm_86 the composed carry ran as fast on rank slots as on the best coloring,
its time following the lists' places rather than its shared-memory conflicts (2026-10-13). An architecture whose carry
is bound by its gathers' wavefronts would take the coloring back as one more slot-choosing step, measured in the
composed carry first.

<a id="lists-out"></a>

## The lists out

Per list (a digit of the list level), its tokens in token order -- slot order, the slots being ranks -- with no pads, at
most 512 a head. Its BOUNDARIES (`kListAt` = 8 a head) are its entries before slot `64 (j + 1)`, the last its length,
out in PREFIX FORM over the list group's lists (`pre`, `G + 1` rows: a row of 8 a list and a row of the group's after
them, row `l` the boundaries of lists `< l` summed) modulo 2^16, beside the group's exact TOTALS (`totals`, int32, the
planner's counts). Any chunk of the union whose bounds are multiples of 64 finds each list's run and its place among the
lists' runs by four loads, each a difference of two columns: a chunk's entries fit in shared memory, fewer than 2^16, so
the differences are exact although a union of grouped queries' rows passes 16 bits. A scan of the runs per chunk instead
cost the window 4% (2026-10-16); no chunk size of a consumer's reaches this pass (`kListBound`: a
consumer plans its chunks per window).

AN ENTRY IS A SLOT AND A BYTE, 4 bytes on either side: the token's union slot in the low 16 bits (a group's slots a
window fit them: the plan's assert) and its level byte in bits 16 to 23. Its token is the union's at the slot
(`union_tok[slot]`), and what a consumer's step needs beyond the byte -- a scale, a coefficient in its operand's
format, an address in its own layout -- the consumer forms where it expands the byte.

THE LANES, every warp its 4 lists `w + 8 j`, 8 lanes a list, lane `s` the list's words `2s` and `2s + 1`. Each lane
counts its tokens before the head's eight bounds and its total (a popcount of each word under a mask); a segmented
scan of those counts (nine u16 fields in five registers) gives, at the list's last lane, the list's boundaries --
lane `s` posts boundary `s` to the CTA's table past the list's entries in the heads before, lane 7 also its entries
through the head -- and each lane's tokens before its own, its first place. After the walks a barrier, and warp `q`
writes list group `q`'s columns `k0 ..` (a later head's own columns overwrite the ones past this head's rows; the last
head's entries through it fill every column past its eight), lane `c` column `k0 + c` in prefix form down the lists and
its sum, the group's total (`bounds_out`). A walk then writes the lane's entries at consecutive places from its
first, each's slot `pre + popc(union word & below)` from the lane's two union words and their ranks before, beside its
byte from the window.

<a id="access-table"></a>

## The access table (KERNEL_STANDARDS §30)

| buffer | writes | reads | wavefronts |
|---|---|---|---|
| window | `cp.async` 16-byte quanta, a lane pair a token | the vote: lane `(tk, dq)` row `4 i + tk` word `dq`; the walks: a byte at (token, digit) | the vote 1.0 a load (32 banks by `ring_row`); a walk's byte loads land on random banks, ~1.5; the copies' cost is their global pattern (`ldgsts-copy-ranking`) |
| tile | the vote: a digit a lane, one word | the bits out: a half-warp a digit row; the lists: a lane a list's two words | 1.0 except the lists' loads: their 4 lists share a bank half, 4.0 against 2.0 ideal, two loads a lane a unit |
| union words | the vote: lane 0, a word a list group | the ranks: lane `i` word `i % 16` of each | 1.0 (a broadcast pair) |
| bounds | the ranks: the lane holding a 64th slot | the lists: every lane, seven words | broadcast, 1.0 |
| table, prefix | the lists: a lane a list's boundary (u16); warp `q`'s sums (u16) | warp `q`: a lane a column of its list group's rows | 1.0 / the column walk's u16 at 16-byte stride, 2-way |

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[8][4]` int64 buffer (warp × load, vote, ranks, lists) that every CTA's warps add
their cycles into; `rola.ops.facts.ledger_bind` is the Python side. An unbound ledger issues no lap; a bound ledger
moves no output (`tests/integration/test_facts_pass.py`).

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView`, per write level and for the value scale a `ColumnView`, the outputs (a
`ListsOut` a side), the depth, the padded widths, row and group bases, each side's list level and list group, the
window's unit codes, per side the digit mask's words. Under a kilobyte and a half. No logical
width is in it: `b_l` reaches the mask at the seam and bounds nothing else — the host entry refuses a mask that names a
digit past a level's rows.

<a id="budget"></a>

## The budget line and the floor

THE LINE (ruled 2026-10-12, restated 2026-10-13): the whole pass at <= 1.15x its required bytes' DRAM floor. Its
required bytes are the sparse levels' rows and the scales read, and the words and tile unions, every list's entries (4
bytes each), the boundaries' prefix rows, the groups' totals, the unions by slot with their sizes and the factor
products written.

MEASURED (sm_86, RTX 3080 Ti, 2026-11-10, `tools/overlap_gate.py`'s rows, one `ncu` launch each on its locked clock,
4 heads at 65,536 tokens, whole list groups): `facts-alt-k4` 337.5 µs, 1.58x its DRAM floor of 213.8 µs -- OVER THE
LINE, RED, and what holds it there is not attributed (owed: the row's stall census); `facts-dense` 12.7 µs, 3.52x its
3.6 µs floor, RED -- a call with no sparse level runs the columns unit alone, and what holds that unit over its floor
is not measured either (owed: the row at a second length).

Per architecture: sm_80 runs the same form at five CTAs an SM (registers); on sm_90 the window fills by TMA (a
`[512 tokens × 32 bytes]` box of the head's plane in one instruction) under an mbarrier, the rest unchanged. Scaling:
O(L × rows) bytes read, O(L k) entries, O(L B / 32) unions; nothing O(L N).
