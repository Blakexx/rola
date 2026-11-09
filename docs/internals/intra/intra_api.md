# `csrc/rola/src/intra/intra_api.cuh` / `intra.cu` — the intra's host seam

Mirror doc for the intra's host entries and the refusals they make before a launch exists.
The kernel is [`ampere/intra_kernel.md`](ampere/intra_kernel.md), the unit that defines these entries
[`intra.md`](intra.md); the tile unions it gates on are
[`facts/liveness_contract.md`](../facts/liveness_contract.md#tile-unions). `rola/ops/intra.py`
is the Python caller.

<a id="rola-intra-forward"></a>

## `rola_intra_forward`

    rola_intra_forward(read[], write[], values, fm, unions, num!, den!, widths, dense_read,
                       dense_write, accumulate) -> ()

Each side's rows a level as the bundle stores them -- read `[B, T, H, w_l]`, write `[B, T, H_w, w_l]`, u8 for a
sparse level and f16 for a dense one, any strides on 16 bytes -- the values (f16 `[B, T, H_w, 64]`), the facts pass's
writer factors `fm` (contiguous fp32 `[B·H_w, T, 2]`, each token's `(m s, m)`: the write gain times the write levels'
scales, and that times the value scale), its tile unions (contiguous int32 `[B·H_w, H / H_w + 1, groups, T / 64]`:
a write head's query heads' read sides, then its write side), and the two output planes the pairs are REDUCED into:
`num` contiguous fp32 `[B·H, T, 64]` and `den` `[B·H, T]`. The caller owns their contents; the carry reduces into the
same planes. `widths` is the union table's layout, the facts pass's (`rola.ops.facts.call_layout`, the padded
widths), `dense_*` each side's dense-level bitmask, the same statics the facts pass took
(`rola.engine.facts.liveness.side_statics`), and `accumulate` the MMAs' accumulation, fp32 (0), fp16 (1) or -1 for
the device's rule, the one the carry's readout applies
([`../common/arch_runtime.md#readout-f16`](../common/arch_runtime.md#readout-f16)), so the inter and intra terms of a
call accumulate alike. The entry runs on its rows' device and takes its window from that card. A level's form --
which side is f16 -- is read from its rows' dtypes.

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* each side a row set a level, as many as the layout's levels, u8 or f16 each, and the values f16;
* `T` at least one token (a partial last window runs its tiles before the call's end:
  [`ampere/intra_kernel.md`](ampere/intra_kernel.md#partial-window));
* the write heads divide the query heads;
* the accumulation is fp32 (0), fp16 (1) or the device's rule (-1);
* one or two levels (the built arms serve depth two), each a whole number of 16-digit granules
  from 16 to 256, and no wider than the layout's level, which its union groups come from;
* a row set or the value plane starts and strides on 16 bytes, digits contiguous — the kernel's
  copies are 16-byte `cp.async` chunks (the producer's planes are);
* every token stride fits 31 bits of bytes — a copy adds its rows by one 32-bit stride;
* the factors, unions and outputs are the shapes above, and the dense bitmasks name levels inside the
  depth.

The footprint is the call's ([`intra_footprint`](#intra-footprint)); one past the 48 KB a kernel
gets unasked raises the arm's attribute to it, once a size (every footprint fits the card's: the walk is sized for the
widest rows); the grid is `(windows × pieces, B·H)`, a CTA a piece, the device's card's block and pieces
([`intra.md`](intra.md#block)).

<a id="rola-intra-ledger-bind"></a>

## `rola_intra_ledger_bind`

Binds the phase ledger (`[8][7]` contiguous int64, [`ampere/intra_kernel.md`](ampere/intra_kernel.md#phase-ledger))
the next launches add into, or unbinds it with `None`. A card whose block has fewer warps books its own rows.

<a id="intra-footprint"></a>

## `intra_footprint`

    intra_footprint(widths, forms, device) -> int

The shared-memory bytes a call at these level widths and forms launches with on `device`'s card, from the same
geometry the launch derives ([`intra.md`](intra.md#stages)), with the same refusals of widths it does not read.
The prefill's envelope asks `rola.ops.intra.admits` whether the intra reads a call's widths: the widths' law and the
value width, no device asked, since every width it reads fits every card's shared memory in every form (each build's
walk is sized for the widest rows, [`intra.md`](intra.md#block)).

<a id="intra-block-rows"></a>

## `intra_block_rows`

    intra_block_rows(device) -> int

The readers a CTA holds resident on `device`'s card ([`intra.md`](intra.md#block)).

<a id="intra-build-stamp"></a>

## `intra_build_stamp`

A fact only the loaded binary can produce: a one-thread kernel returns `(rows · 1000003 + sizeof(Smem)) · 1000003 +
sizeof(IntraParams)` -- the current device's build's walk: its block rows, its fixed tenants and the
operand block -- so the value moves
when any of them does. A path or hash check cannot catch a stale extension;
`test_the_loaded_binary_states_the_footprint_it_carries` reads it against `intra_footprint`'s law
(the stamp kernel's named namespace: [`intra.md`](intra.md#the-unit)).
