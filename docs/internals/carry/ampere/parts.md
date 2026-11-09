# `carry/ampere/parts.cuh` — the carry's components

Mirrors `csrc/rola/src/carry/ampere/parts.cuh`: the parts the fold, the hand-off and the readout share -- the CTA's place, a
window's lists in, the pool rows' swizzle, a chunk's fills and runs, the den's register places. Each is at its real
signature, tenants and copies, and keeps the work's counts; what a component leaves to its build is named in its
section and in `carry_kernel.md#kernel`. The fold is `fold.md`'s, the hand-off `handoff.md`'s, the readout
`readout.md`'s.

<a id="place"></a>
## The CTA's place

`place_of` is the CTA's batch and write head (`blockIdx.y`) and its channel group (the box index's innermost digit);
`origin_of` a level's first box digit. Before the windows, thread 0 works out every address that is the same each
window into the `Addr` tenant (`addr_in`): each side's STATICS (its union group, the arm's list group holding the box's
list digits, `plan.md#list-groups`; the box's first list digit; its lists; whether it lists) and its operand SLICES (each level's plane at
the CTA's batch and head, displaced to the box's first digit, its token stride and bytes), the values at the box's
channels, the channel group. The divisions run once: at nl64k-alt-k4 the per-fill address arithmetic they replaced was
a sixth of the carry's samples, and the carry ran 16% faster without it. The window loop reads the tenant at each use
(volatile), so none of it holds a register across the windows. `window_of` is a side's window: its tokens from the
window index, its statics from the tenant.

<a id="window"></a>
`ops::opaque` (`../../common/atoms/warp.md#opaque`) and `parity_of` pass the window index (and a thread's) through an opaque
move, so a window-invariant offset --
a window's first token, a list's row base, a tenant by parity, the drain's cell -- is formed at each use: computed once
before a window's steps, the compiler holds it in a register across the readout, the kernel's register peak, and at
255 a thread one such value spills. The fill's quad lane and first slot come from the thread index the same way, and
the kernel passes each part its warp and lane opaque (`carry_kernel.md#kernel`).

<a id="lists-in"></a>
## Lists in

A window ahead (`carry_kernel.md#staging`), into the window's parity of the tenants -- the read side's in the call's
read tables (`plan.md#read-tables`): a side's union tokens whole (512 slots a query head, 1 KB), its group's boundaries
in prefix form (the arm's list group plus one rows of `cols_of` columns, 8 a query head: `plan.md#list-groups`) and its
count (one 4-byte copy), its copies over the CTA's
threads; the planner alone reads the exact totals. `facts_in` copies the box's write lists' exponent pairs (4 bytes a list, never past the box's row) and
each query head's token units (512 bytes a head; head `j` of write head `bh` is the call's `bh g + j`). `union_of`,
`pre_of` reads a window's tenants back (`union_of` its tokens; `pre_of` forms its parity at each use, `#window`);
`chunk_cols` names a chunk's columns in the boundaries.

<a id="swizzle"></a>
## The pool rows' swizzle

A row of 128 bytes XORs its 16-byte chunk by the slot's residue (`slot & 7`); a row of an odd number of 64-byte quarters
(64 or 192 bytes) starts on alternating halves of the 128-byte bank line, so it XORs its chunk within each four by the
residue's high bits (`chunk_at`). Either way eight slots of distinct residues meet eight bank groups at one chunk: a
dense pass's step is 8 consecutive slots; a list's step is its next 8 rank slots, distinct residues only where they run
consecutively (the facts pass's [ranks](../../facts/facts.md#ranks)). `chunk_addr` is a logical
chunk's address from a row's chunk 0 with the swizzle folded in: an XOR inside the swizzle's group, the group's offset
added. An entry carries the row's chunk-0 offset, so a lane addresses its chunk with one XOR. A row of an odd
number of 32 bytes (the 64-channel boxes' write rows: 160 bytes, a 32-byte slice of 16 rows then 128 bytes of values;
224, a dense write side's 64 + 32 + 128) puts slots `s` and `s + 4` a 128-byte line apart at the same offset, so its
chunk's low bit flips by `(s >> 2) & 1`, and eight slots of distinct residues again meet eight bank groups.

<a id="fill"></a>
## The fill

A chunk's fill issues its copies as one commit group; the caller waits the group (`carry_kernel.md#staging`).
`fill` copies a range of union slots, each its token's row -- the side's operand slices (two bytes a digit of the box's
extent: 64 bytes at 32, 32 at 16, a quad's lanes past the slice idle), then on the write side its values at the box's
channels (a lane's 16-byte chunk at 32 channels, two 64 bytes apart at 64) -- swizzled, by `cp.async` over all 256 threads (a slice's 16-byte runs, 8
or 4 under 16, `stage_bytes`): a quad of lanes a slot, thread `4 q + r` slots `q + 64 k` of each 256. A trip of 256
slots is one block with no branch: its four tokens loaded on either side and selected (an unlisted side's slot is its
token), then every source (`row_at`: the slice's base plus the token's and the head's products, each one wide
multiply-add), then the copies (`copy16`, issued where the slot is under the trip's room in rows' bytes, its size zero
for a gap), so no copy's address register is rewritten while an earlier copy may still read it. A token load past the
chunk is not clamped: it reads the tenant after the union (the prefix rows), in bounds, and its copy is not issued. A
row's swizzle is its slot's low bits (`chunk_at`), the same at every `k` and every trip, so a thread's places are one
base and immediates. The first trip is straight-line; a chunk over 256 slots loops the rest. A read slot's row names
its query head (`../carry.md#sides`): its source is the slice at that head, a head stride on (`Slices::sh`), as a 32-bit
offset from the slice's base at the window's first token (the host holds a window's token and head strides under 2^31
together). A GAP slot (`kGap`, [`../carry.md`](../carry.md#lists)) -- its token negative as 16 bits -- is zero-filled and
reads nothing. `read_fill` adds the chunk's panel and den words (`panel.md`), zeroed by 16-byte stores four a thread
(each predicated on the quanta left, the first four straight-line, so the chunk's landing waits on its rows alone), and
the chunk's entries; `fold_fill` (`fold.md#chunk`) the write chunk's rows and entries, or on a side with no list level
each slot's token's `(F, m)` in the entries' place (8-byte copies from the facts pass's `fm`). Each part sits at the
chunk's own slots (`read_panel`, `read_dens`, `read_ents`, `fold_ents`; `schedule.md#buffers`).

THE FILL'S FLOOR is its bytes through the SM's global path, not its instructions: a CTA-window at nl64k-alt-k4 copies
~51 KB (the read rows 13.8 KB, the fold rows 27.6 KB, the entries 4 KB, the staging ~5 KB), and a `cp.async.cg` stream
of L2 hits runs at ~18 bytes a cycle an SM ([`smem_rates`](../../../lab/smem_rates.md), `common/hardware.json`'s line
price) -- ~2.85K cycles a window. The limit is the SM's, not the L2's: the grid's second wave (48 SMs busy) fills in the
first's cycles (80). The copies' issue backs up into the MIO queue (the copy's samples are half MIO throttle).

<a id="entries"></a>
## The entries

A buffer holds its chunk's entries as the group's lists' runs in list order: `chunk_runs` gives lane `l` list `l`'s
first entry of the chunk, its count, its count below the readout's token-group split (`group_split`,
[`readout.md#warps`](readout.md#warps): the prefix rows' count at the split's column) and its place (the group's lists'
counts before it) from its two prefix rows (list `l` and `l + 1`): the chunk's end in both, its start where the chunk
is not the first, each a difference within the chunk modulo 2^16, the same in every warp -- worked out once a chunk by
the kernel (`carry_kernel.md#kernel`) and passed to the chunk's fill and its readout or fold; a list's two rows are the
box's first list's place in its list group plus `l` (`digit0 mod G`). `entries_fill` copies a warp's own lists' runs
(`warp + 8 j`, of the side's box lists: 32 at four a warp, 16 at two), eight lanes a run at four (sixteen at two): lane
`l` takes run `l / lanes`'s first, count and place by one shuffle each from `run` and copies its entries `l mod
lanes`, `+ lanes`, ..., 8 bytes a lane (`ops::stage_run_lanes`: a lane past its run issues nothing), two a
trip, the trips the warp's longest run's (`__reduce_max_sync`), the first two straight-line. A consumer takes its
list's place and count from its own loads, so no warp waits on another for them.

<a id="den-place"></a>
## The den's register places

`den_place` is where a lane's `kDenRegs` den words sit in the fold's registers (`State::den`, `fold.md#state`): each of
the box's lists x rows once across the CTA, the list owner's on either side, a lane its quad's (m-tile, list) pairs in
order, two rows each (at 32 x 32 word `i` at fold list `w + 8 (2 (t & 1) + (i & 1))` and fold row `g + 8 (i >> 1) + 16
(t >> 1)`). The fold adds into them and the hand-off reads them (`handoff.md#den`).

