# Host-side padding: any level width, any value width, no kernel change

`rola/ops/padding.py` is the ONE translator between a caller's LOGICAL shape (any
level width `b_l >= 1`, any value width `d_v`) and an op's kernel-facing, PADDED shape
(a level width a power of two at or above 16, a shipped value width). The two
rulings ("PADDING AT THE PRODUCER", "d_v PADDING IS ALSO HOST-SIDE") name the
mechanism; this page states where it lives and why.

## Two contracts, kept separate (CONSTITUTION §R13)

**API CONTRACT.** The layer producer's bundle
(`RouteFactors`) and `rola.layer.RoLA` stay at the caller's own, unpadded widths --
`tests/unit/test_api_contract.py` pins this: one tensor shape per declared level, and
every public tensor a VIEW of the solve's packed layout with zero copy. Padding those
would break both pins for no reason: nothing downstream of the producer needs a padded
`RouteFactors` to exist as an object.

**KERNEL CONTRACT.** The two padding seams -- the prefill's carry leg (`rola.ops.prefill`)
and the decode step (`rola.ops.decode.derive_decode_geometry`/`_decode_step`) -- receive the
caller's logical tensors and widths and pad INTERNALLY, once, before any launch-facing
derivation. The kernel itself is never handed a width other than its own; `y` is sliced
back to the caller's own `d_v` before it leaves the op.

`rola/ops/padding.py`'s primitives, used identically by both seams:

* `pad_routes(levels, padded_widths)` -- per level, `[..., b_l] -> [..., B_l]`, the
  trailing columns exact `0.0`.
* `pad_v(v, DV)` -- `[..., d_v] -> [..., DV]`, exact `0.0` tail.
* `slice_y(y, d_v)` -- `[..., DV] -> [..., d_v]`, a view.
* `PaddedShape.of(widths, d_v, shipped_dv=...)` -- the descriptor: `widths`,
  `padded_widths`, `d_v`, `DV`, and `is_padded`. The one place "logical vs padded" is
  computed; every op builds one of these before it touches a tensor.

## Why a zero-pad AFTER the solve is the ruling's mechanism, not an approximation of it

The ruling states the mechanism as padding the LOGITS with `-inf` before entmax/softmax. This
module pads the SOLVED level with `0.0` after it -- the identical tensor, because both
softmax and entmax normalize over, and threshold against, only the finite entries of a
row: an `exp(-inf) == 0.0` addend changes neither a softmax's normalizer nor an
entmax's support-selecting sort, so the `b_l` real columns take the same value either
way and the extra columns are `0.0` either way.
`tests/oracle/test_padding_seam.py::test_prefill_padding_is_the_ruled_mechanism_not_an_approximation_of_it`
proves the two tensors are `torch.equal`. The zero-pad-after form is also SAFER:
autograd never runs through a literal `-inf` logit, so there is no `0 * -inf` NaN risk
in a backward pass, and no parameter reads a pad column, so its gradient is exactly
`0.0` by construction rather than by cancellation.

## Per-op scope: not every op pads every axis

* **the prefill's carry leg** (`rola.ops.prefill`'s `_padded_shape`/`_pack_operands`) pads
  BOTH axes: every level to `rola.routing.topology.padded_level_width` (next power of two
  >= 16, the register-shape floor), and `v` to the smallest shipped value width that
  admits it.
* **decode** (`rola.ops.decode.derive_decode_geometry`, `_decode_step`) pads ONLY
  `d_v` -> the smallest `DV` its own shipped arm table
  (`rola.ops.decode.arms`, keyed `(DV, D, decay)`) carries at that `(D, decay)`. Its
  per-level widths carry NO floor of their own -- `derive_lattice`/`box_shape`
  degenerate to `k = m = 1` for an unfactorable width rather than refusing, and the
  existing `D = 4, B = 8` conformance cell (`tests/oracle/test_decode_vs_oracle.py`)
  is proof this binary already runs a width the carry floor would refuse. Padding
  levels here would import the carry's constraint into a kernel that does not have it.
* **intra** (`rola.ops.intra.intra_forward`) pads nothing: it reads the bundle's rows
  where the producer stores them, at their own widths, each level a whole number of
  16-digit granules up to 256 -- a bundle outside that is refused by the prefill
  envelope (`rola.ops.intra.admits`), never padded at the seam. The values are the
  format's 64 channels.

## Where the shipped sets come from

Decode's shipped value widths are read off `rola.ops.decode.arms()` at call time -- the
binary's own declared arm set -- never a literal tuple, because a literal would drift the
moment the binary's arm table changed and a call would silently pad to a width the loaded
`.so` does not carry. The intra reads one value width, `rola.ops.intra.VALUE_WIDTH`.

The prefill's carry leg pads nothing either: the rebuilt carry takes the value width as
runtime addressing over the carve's channel extent
([`engine/rules.md#carve`](engine/rules.md#carve)), and the envelope refuses a carve no
built arm carries. The stateful band's `box` node still reads `carry_ops.SHIPPED_DV`, which
the rebuilt `rola.ops.carry` does not define; the envelope refuses a stateful call before
that node runs, and the node is re-pointed with the carry's state I/O
([`docs/open-work.md`](../open-work.md) C-e).

## What this does not do

Nothing here changes `RouteFactors`, `rola.layer.RoLA`, or `rola._state.StateFormat`'s
fields (`docs/internals/state.md#binding`): the public surface keeps the caller's own
widths, and padding exists only below the op seams.
