# `csrc/rola/src/decode/decode_leaf.cuh` — one leaf's step, eight channels a lane

The arithmetic a decode step does to ONE leaf, stated once and run by both of the walk's
forms: a leaf a lane group (the read terms, and a write term whose innermost level is sparse)
and an atom a warp (a write term whose innermost level is dense by declaration,
[`decode.md#whole-atoms`](decode.md#whole-atoms)). A lane holds eight consecutive channels of
the leaf, so a leaf is `DV / 8` lanes and a 32-channel exponent group is four of them
([`decode.md#leaf-lanes`](decode.md#leaf-lanes)).

## <a id="leaf-addr"></a>The slice and its load

`leaf_addr` places a lane's slice in its page: its eight channels' hi words (16 bytes), its lo
words (8 bytes), its group's exponent byte and the leaf's mass word
([`../common/state_page.md#geometry`](../common/state_page.md#geometry)). `load_leaf` reads
them into a `LeafSlice`: the lo words only when the leaf is written or re-read in full, the
mass only on the leaf's first lane, and zeros for an absent atom's leaf -- an absent atom
reads as zeros and is never written.

## <a id="reads"></a>The reads

`read_hi` is a leaf this step only reads: its hi plane at `2^(e + 8)` (the s16 operand of
[`decode.md#int8-state`](decode.md#int8-state)) into the lane's eight partials. `read_full` is
a written leaf re-read without depositing -- a gated replay's third outcome -- at its full
stored value.

## <a id="deposit"></a>The deposit

`deposit_leaf` is UPDATE FIRST, THEN READ THE STORED VALUE, which is the step's semantics
([`decode.md#update-then-read`](decode.md#update-then-read)): a leaf's row is decayed by its clock
(the leaf's share of the token's write: [`decode_fold.md#row-sum`](decode_fold.md#row-sum)), takes its
deposit in fp32, and is rounded once into the stored state page; the readout reads the STORED value, so
the next step reads exactly what this one wrote and a step's `y` never sees precision its state lacks. The
group's exponent is shared by its four lanes, so a group rounds to one quantum; the mass rides the leaf's
first lane in fp32.

The store predicate is the caller's: `present` for the lane-group form, and for the atom form
`present` AND the leaf's write bit, since a leaf outside the write support belongs to a read
term. A leaf that does not store also takes no clock (its rate is passed as zero), which keeps
`0 * log1pf(-1)` out of the arithmetic.
