# `csrc/rola/src/facts/facts_api.cuh` — the facts pass's host seam

Mirror doc for the pass's host entries and for the refusals they make before a launch
exists. The kernel is [`facts.md`](facts.md); the words' shape and meaning are
[`liveness_contract.md`](liveness_contract.md). `rola/ops/facts.py` is the Python
caller: `call_layout` and `call_statics` read a bundle's layout and its two sides'
static operands off its `sparse` flags and topology, and `facts` packs them into this argument list and names what
comes back.

<a id="rola-facts-forward"></a>

## `rola_facts_forward`

    rola_facts_forward(read[], write[], scale_write[], v_scale, widths, dense_read, mask_read, dense_write,
                       mask_write, write_list_level, read_list_level, read_row, read_list_group,
                       write_list_group) -> (words, unions, list_exp, fm, gains16, gains_exp, write_lists[5],
                                             read_lists[5])

Each side's rows a level (read `[B, T, H, b_l]`, write `[B, T, H_w, b_l]` u8, `H_w` dividing `H` and `g = H / H_w`
at most 16 -- grouped queries, [`api.md`](../../api.md#grouped-queries); a dense level's are not read, so not checked),
the write levels' scales (their product the write side's shared scale) and the value scale (`[B, T, H_w]` fp32, any
strides). The entry finds its
card from the rows' device and ALLOCATES its outputs there at the card's window W (the pass's plan,
[`facts.md`](facts.md#window)), so no caller holds a window. They come back as one flat list, a write head's each
(`B·H_w`): `words` the contiguous int32 `[B·H_w, g + 1, rows, ceil(T/32)]`,
`unions` the contiguous int32 `[B·H_w, g + 1, groups, ceil(T/64)]` (slots `0 .. g - 1` its query heads' read side, `g`
its write side; [`liveness_contract.md`](liveness_contract.md#tile-unions)), `list_exp` the contiguous int16
`[B·H_w, windows, B_list, 2]` ([`facts.md`](facts.md#lists); `B_list` 0 when the write side lists nothing), `fm` the
contiguous fp32 `[B·H_w, T, 2]` factor products, `gains16` the f16 `[B·H_w, T, 2]` and `gains_exp` the int16
`[B·H_w, windows, 2]` ([`facts.md`](facts.md#gains)), and each side's five lists at its list level, the read side's
over its write head's `h = g` query heads' rows (the write side's `h = 1`), `groups = ceil(B / G)` at the side's list
group `G` -- entries `[B·H_w, windows, B, W h, 4]` int16, the boundaries' prefix rows
`[B·H_w, windows, groups, G + 1, W h / 64]` int16, the groups' totals
`[B·H_w, windows, groups, W h / 64]` int32, the union by slot `[B·H_w, windows, groups, W h]` int16 and the extents
`[B·H_w, windows, groups]` int32 ([`facts.md`](facts.md#lists-out)); width 0 for a side that lists nothing. The pass writes `fm` and the gains in
full, `list_exp`, the prefix rows and totals, the unions by slot and the extents in full, each list's
entries to its end, and of `words` and `unions`
every sparse level's rows and groups; a dense level's are left as allocated (its words are the width mask, which a
reader takes from the statics, [`liveness_contract.md`](liveness_contract.md#statics)). `widths` are the layout's PADDED
widths; `dense_*` is a bitmask over levels and `mask_*` one bit a row, packed by
`rola.engine.facts.liveness.side_statics` -- the only place `b_l` appears. `read_row` is the carry arm's read pool row
and `read_list_group`, `write_list_group` each side's list group (`rola.ops.carry.lists_form`), the read entries' row
offsets being in the row's 16-byte chunks and each union and its prefix rows spanning `G` digits
([`facts.md`](facts.md#list-groups)).

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* the depth is one to four levels, each side holds a row set a level and the write side a scale a level; a level is 1
  to 256 digits wide;
* a side's list level is a sparse level of the depth, or -1, and its list group 16 or 32 digits;
* a listed read side's pool row is 128 bytes or an odd number of 64 (the swizzles the carry folds in), a window's g
  heads' pool inside a 16-bit offset (a dense read side has no entries to place: its row is the carry's alone);
* the read side's heads are a whole number, at most 16, of the write side's;
* a sparse level's rows are u8 `[B, T, H, width]` with the first level's `B, T, H`, digits contiguous, under the row
  alignment law ([`liveness_contract.md`](liveness_contract.md#alignment));
* each side's dense bitmask names only levels inside the depth, and its digit mask is exactly `mask_words(rows)` words;
* **a mask bit names a digit inside its level's rows.** The kernel reads a digit iff its bit is set (on a sparse
  level), so a mask wider than the rows would read the next level's digits, or past the tensor; it is caught here, not
  by a wrong answer;
* the columns are CUDA fp32 `[B, T, H]`.

<a id="rola-facts-ledger-bind"></a>

## `rola_facts_ledger_bind`

Binds the `[8][7]` int64 phase ledger the next launches book into, or unbinds it with
`None` ([`facts.md`](facts.md#phase-ledger)); it issues no launch.
