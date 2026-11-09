# `carry/ampere/fold.cuh` — the fold

Mirrors `csrc/rola/src/carry/ampere/fold.cuh`: a warp's write lists over a window's units into the fp32 state, in the fold's
declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`), gated per slot by
`tests/oracle/test_carry_fold.py`.

<a id="state"></a>
## The state

`State` (in `parts.cuh`) is a warp's accumulators: per fold list (`fold_list`: the box's list `w + 8 j`), its rows'
m-tiles (`kFoldMTiles` = rows / 16) by the channels' n-tiles (`kFoldNTiles` = channels / 8), fp32, each list in its own
UNIT (below): at 32 x 32 x 32 4 x 2 x 4 x 4 = 128 registers a lane, at every box `kStateValues / 256`. They are the
MMAs' C fragments for the whole call. Beside them the DEN'S, the v+1 channel's mass (`#den`), `kDenRegs` registers a
lane (the box's lists by rows once across the CTA: 4 at 32 rows, 2 at 16), absolute (ruled 2026-11-02: the den as one
more channel of the state, register-resident). A word's place (`den_place`) is the list's owner's on both side forms:
lane `4 g + t` takes its quad's (m-tile, list) pairs `kDenLists t ..` in m-tile-major order, a run of `kFoldLists /
kDenLists` lanes sharing an m-tile; word `i` is pair `i % kDenLists` at row `g + 8 (i / kDenLists)` -- at 32 x 32 list
`w + 8 (2 (t & 1) + (i & 1))`, digit `g + 8 (i >> 1) + 16 (t >> 1)` -- where `den_add` lands.

<a id="units"></a>
## The units

A record's c is the facts pass's `f16(fp32(b m s) 2^(4 - E))`, `E` its list's window exponent floored at `kUnitMin`
(`docs/internals/facts/facts.md#lists-out`). The state cannot follow `E` window by window: a quiet window would scale
it up past fp32's range. So each list keeps a running unit `u` in a shared-memory book (`UnitBook`: the unit and the
state's bound in units). `next_unit` is the kernel twin of `next_units`: a live list's `E`, held within the bound's
headroom (`kUnitHeadroom` = 100 binades), never below `kUnitMin`. A dead window keeps the unit. A side with no list
level has no records and no list exponents: every list takes the window's BOUND, `dense_exp` = `5 - shift` from the
facts pass's shift of the window's largest factor F (`facts.md#gains`: F under `2^(-3 - shift)`, a dense byte under
2^8), staged with the window's inputs in the exponents' place. A bound and not each list's own maximum: the dense
level's floor (a dense coefficient never under 2^-13 of its row's largest) holds every list's largest within 2^13 of it,
inside f16's 18 normal binades (measured 2026-10-19 against exact per-list exponents: the same state error on
flagship-dense, 2.8x it on corner-concentrated, both within the fold's bound). The state is rescaled by `2^(u_old - u_new)`, an exact power of two, zero below fp32's normals, beside the
list's first step.

c is then moved to the unit by an f16 power of two, `2^(E - u)` (`pow2_half2`: normal from 2^-14, subnormal to 2^-24,
zero below), two roundings where a contribution falls past f16's normals, which the reference mirrors. `fold_units`
runs once a window; the window's exponent pairs come in with its lists (`parts.md#lists-in`), so it never waits on a
global load. It posts each list's rescale and c factor to the list's scale (`ListScale`, the `kScale` tenant), which
the window's fold chunks read, the first applying the rescale: so no register holds them through the readout.

<a id="step"></a>
## The step

16 TOKENS WITH AN 8-TOKEN TAIL (ruled 2026-10-12): `steps` takes a list's entries in 16-token steps while more
than 8 remain, then an 8-token tail of m16n8k8 MMAs, half a step's tensor work. `entry_of` turns an entry into a
lane's row offset (its swizzle folded in, `parts.md#swizzle`) and its c paired with the next lane's, moved to the unit;
a pad past the list's count is row 0 with c = 0. `step16`: A is the slots' dense slices by `ldmatrix.trans` (the
box's rows' m-tiles: two at 32 rows, one at 16) times c by HMUL2, B the values (an n-tile a channel group: four at 32
channels, eight at 64), `kFoldMTiles x kFoldNTiles` = eight fp32-accumulate MMAs at either box. `step8`: the same over
8 tokens, the slice by one `ldmatrix.x4.trans` at two m-tiles and `.x2` at one, the values a `.x4` each four n-tiles. A list whose den the box carries takes its den
column's MMAs beside them ([`#den`](#den)).
Each loop's step is one block (KERNEL_STANDARDS §27), guarded by a vote and closed by one. A list's first step's lanes
(`off`, `cpair` for entry `lane & 15`, which serve a tail's lanes 0-7 as well) arrive computed, and each 16-token
step fetches the next step's entries in its own block, after its MMAs in program order, so their loads issue under
them.

<a id="chunk"></a>
## The chunk

A CHUNK is a range of the write union the window's plan sets (`schedule.md#chunks`): the whole union where its buffer
fits the area, else the largest `64 m` slots two of whose buffers fit, and the rest. `fold_fill` copies its rows and
its lists' runs into its buffer (`parts.md#entries`); `fold_chunk` reads every one of the warp's lists' scale, count,
base and first step's entries before any list's MMAs -- four independent chains where they had run one after the other
behind each list's burst (nl64k-alt-k4, sm_86: the fold phase 3.26K -> 2.86K cycles a warp-window, 2026-11-01) -- then
takes each list in turn, its run from the buffer, a step's lanes past it masked (the chunk's first row, c = 0), the
rows taken back by the chunk's first slot so an entry's union slot is its row. A side with no list level steps every
slot of the chunk once for all the warp's lists ([`#dense`](#dense)): its buffer holds each slot's token's `(F, m)` in
the entries' place, and c = `f16(fp32(b 255 F) 2^(4 - E))` with `b` the list's digit of the row's coefficient slice
(its second) and `E` the window's bound -- the facts records' arithmetic, rounded once, by the packed conversion
(`F2FP`, the ALU pipe: the scalar one runs on the conversion unit, which a step may not hold). The arm's fold side is
listed or dense at compile time (`kFoldFactors` 1 or 2), each arm compiling only its own; a window whose plan says
otherwise traps. Until 2026-10-19 this branch took c
as `b` alone, without F or the list's unit: the dense-dense carry's state was wrong from its second window, and no gate
read it (the whole-call gate refused read sides with no list level; the probe wrote such a side's lists at one leaf).
The list's rescale is applied before its first chunk's steps, and only where its unit moved: a list whose unit stood
(`f == 1`, the warp's one value) multiplies nothing -- the multiply is a one-trip loop on a warp vote, since 128
multiplies a lane every window (1.0K warp instructions a CTA-window at alt-k4) bought nothing in the steady state.

<a id="dense"></a>
## A dense side's steps

`steps_dense`, the ruled form for a side with no list level ("a declared-dense side merges maximally by structure",
2026-10-18; lab `list_fold`, 2026-10-05: merged 1.08x its tensor floor, per list reloading 1.37x the merged time): every
slot of the chunk once for all four of the warp's lists. A 16-slot step loads its slices (two `ldmatrix.trans`) and
values (two) once, and each list multiplies the slices by its c (four HMUL2 a list a digit tile) for its eight MMAs --
32 a step. Lanes `0..15` hold the step's slots: each its token's four lists' c from the row's coefficient slice and the
slot's F, two lists a packed conversion, paired with the next lane's by one shuffle a word and times each list's unit.
An 8-slot tail of m16n8k8 MMAs follows. Every accumulator takes the same MMAs in the same order as the per-list form
it replaces (which stepped each list over every slot, the four reloading the same rows), so the state is the same bit
for bit. BUDGET (predicted before the build, KERNEL_STANDARDS 32.2): a step 32 MMAs, 4 operand loads (16 wavefronts,
the per-list form's 64), 4 + 1 coefficient loads a lane, ~100 instructions; on sm_80 its memory ~22 cycles over 256
pipe cycles (the census read the per-list form at 2.66 cycles an MMA: ~0.7 now) and its step cover at the line; on
sm_86, tensor-bound either way, the dense fold 71.8K -> ~67-69K cycles a warp-window against its tensor 66.6K.

THE TWO-LIST FORM (a dense write side's 64-channel box, 16 lists x 32 rows x 64 channels: two lists a warp): the same
step over every slot once, each list's c a lane's one packed word (two lists, one conversion and one shuffle), eight
n-tiles of values a list -- 2 lists x 2 m-tiles x 8 n-tiles = 32 MMAs a step, the four-list form's count -- its write
row 224 bytes (64 of rows, 32 of the lists' coefficients, 128 of values: an odd number of 32-byte pairs,
`parts.md#swizzle`).

<a id="laps"></a>
## The laps

An instrument: `fold_chunk`'s `Laps` (default `NoLaps`, whose call is empty) is called at the chunk's marks -- 0 once
every list's scale, count and first step's lanes are read, `1 + 2 J` once list `J`'s rescale is guarded, `2 + 2 J` once
its steps are done; a dense side's (`#dense`) 0 before its rescales, 1 after them, 2 to `2 lists` once its steps are
done. The kernel passes none and its SASS is identical with the hook; the lab's
[`fold_alone`](../../../lab/fold_alone.md) passes a recorder.

<a id="den"></a>
## The den column

The den is the v+1 channel -- each token's value 1 at its gain `m` -- so its state is the MASS `sum_t x b m` a leaf
(ruled 2026-11-02, the den in the carry). A box's 32 channels fill the MMAs' tiles, so the den takes one of its own: a
fifth n-tile of one live column (column 0, lanes `0..3`), its A the step's RAW slices (the value tiles' A is `f16(x c)`,
the coefficient on the routing weight, the product order the reference declares), its B the entries' `c_m` pairs
(`entry_of`'s `dpair`: a write record's high half, `f16(b m 2^(4 - Km))` -- the facts pass's, `Km` the list's mass
exponent; one shuffle of the record's word pairs c and c_m both). Two more MMAs a
16-token step (one an 8-token tail), into a `DenColumn` held only through the list's steps: then `den_add` adds each
row's value times `2^(K - 4)` into the DEN'S REGISTERS (`State::den`, `#state`; zero at the call's start), so the
state is in `b m` units across windows with no unit book. B's every column is the c_m pair (no column-0 select), so
every lane holds its quad's rows: list `J` lies in the lanes whose pairs hold it (`t % run == J / kDenLists`; at 32 x 32
`t & 1 == J >> 1`), each adding its two rows (two selects, no branch) at words `J % kDenLists` and `kDenLists + J %
kDenLists`. The state left shared memory (4 KB of the area: nl64k-dense's read chunk 192 slots, three a
window, where 256 and two fit without it) and then the call's global workspace (a line of adds a list, a load a
window: 2026-11-02, 8c83835e) -- in registers it costs neither, and the window's tables read it there
(`handoff.md#den`: a window's read chunks run before its fold chunks, so they see the state before its writes).

A box carries the den of a warp's list `j` where `j boxes / 4` is its box (`den_lists`, `boxes = dv / 32`): each of
two boxes lists 0-15 or 16-31, of four a quarter -- every list once, the cost shared. The branch is the CTA's, an
`if` on the box's mask (uniform: formed from the block's place and the call's `dv`); the one-trip loop each way it
replaced cost ~12 instructions a list a chunk at nl64k-alt-k4 (the region ledger's 12.6M a launch); `lists` takes the
side's form at compile time, so a side with no list level compiles its lists' steps without the den. MEASURED
(`fold_alone`, sm_86, nl64k-alt-k4's draw, cycles a warp-window, the carry's shape, the first form): no den 2,471, half
the lists 2,747 (+276), every list 3,039 (+568). MEASURED composed (phase ledger, nl64k-alt-k4, sm_86): the fold 3.57K
with its den, 2.86K with the box's den lists forced off in the same build (+0.71K: the den MMAs' fp32 tensor work on a
tensor-bound step, the adds, and the instruction cache's misses on the den steps' second copy).

A SIDE WITH NO LIST LEVEL is one list, the den with it (Blake, 2026-11-03: "dense just uses one list covering
everything essentially, for readout and fold. And of course it would apply to den too"; amends the 2026-11-02 form's
one pass a chunk): the den state's tile `Den^T[digits][lists] += raw c_m` rides the merged steps ([`#dense`](#dense)).
Every warp of a dense side steps every slot and any list's coefficient is in each slot's row, so a pair (`w`, `w + 4`,
one scheduler and its tensor pipe) puts its eight lists in one n8 tile and its lower warp takes it: two more MMAs a
16-slot step in warp `w` on the raw slices it holds, none in `w + 4` (whose den registers sit idle) -- the separate
pass's MMA count, every column used. B is each slot's `c_m = f16(b f16(255 m 2^(4 - Em)))` (the weight rounded twice
as ruled: the fold gate's `DENSE_DEN_WEIGHT`), lanes `0..15` the warp's lists' and `16..31` its partner's (those lanes'
copies of the step's slots, their coefficient loads its partner's lists), four shuffles a B word. At the chunk's end
(`den_pairs`) the tile goes into the den's registers in the list owner's layout (`den_place<P>`, one layout on either
side: lane `(g, t)`'s word `i` is element `i` of digit tile `t >> 1` in lane `(g, t & 1)` for warp `w`'s lists, `(g, 2 +
(t & 1))` for its partner's), its own by a shuffle, its partner's through shared memory (2 KB over the den image, free
while folding, `handoff.md#den`; one barrier), at `2^(Em - 4)`, the box's lists alone. At one m-tile (four lists a
warp of 16 rows, `[16, 2, 16, 2, 64, 1]`; `den_pairs1`) a lane's den words are its list `w + 8 t` at rows `g`, `g + 8`:
column `c` of the tile is lane `(g, c >> 1)`'s elements `c & 1` and `2 + (c & 1)`, a doer's own by a quad shuffle,
its partner's through shared memory (1 KB; one barrier). At two lists a warp the n8 tile is a QUAD of warps' lists (`w`, `w + 2` in
lanes `0..15`, `w + 4`, `w + 6` in `16..31`, each lane's four coefficient loads two owners' two lists), warps `w < 2` its
doers, and every doer lane's tile goes through shared memory (`den_quads`: 2 doers x 2 m-tiles x 32 lanes x 16 bytes,
the den image's 2 KB; one barrier), each owner lane taking its list's column of its m-tile's two rows. A first form, each warp its own four lists in half
an n8 tile, ran twice the separate pass's den MMAs and read +0.8-1.5% on sm_86's tensor-bound dense fold.
A box whose lists' den it does
not carry takes the steps without it (`Den` false: a uniform choice a chunk). The separate pass it replaces
(2c8be890 / 82f9ce00, deleted here: a warp pair an n8 list tile over alternate 64-slot blocks, swapped through
shared memory) reloaded every slot's slices and read 28-30% step cover on sm_80, bound by the memory pipe (56 cycles
over a block's 64 pipe cycles), and ran in every channel box whether it carried the den or not.

<a id="probe"></a>
## The probe

An instrument (`carry_probe_bind`, `rola.ops.carry.probe_bind`). When bound, the final state is written absolute,
fp32 `[BH, N, d_v]` by global leaf in level order: each accumulator times `2^(u - 4) / 255` (c on the byte scale, the
dense operand at 1/255 of it); and the den's registers `[BH, N]` by the box carrying each list, times `1 / 255`. Unbound,
nothing is written. The fold's oracle gate reads both (the den against the reference's mass, `fold_state`).
