# Host-side padding: any level width, any value width, no kernel change

`rola/ops/padding.py` is the ONE translator between a caller's LOGICAL shape (any
level width `b_l >= 1`, any value width `d_v`) and an op's kernel-facing, PADDED shape
(a level width a power of two at or above 16, a shipped value width). The two
rulings ("PADDING AT THE PRODUCER", "d_v PADDING IS ALSO HOST-SIDE") name the
mechanism; this page states where it actually lives on the `k31/carry-clean` line and
why it moved there.

## Two contracts, kept separate (KERNEL_STANDARDS §R13)

**API CONTRACT.** `rola.routing.producer.RouteProducer`'s bundle
(`RouteFactors`) and `rola.layer.RoLA` stay at the caller's own, unpadded widths --
`tests/unit/test_api_contract.py` pins this: one tensor shape per declared level, and
every public tensor a VIEW of the solve's packed layout with zero copy. Padding those
would break both pins for no reason: nothing downstream of the producer needs a padded
`RouteFactors` to exist as an object.

**KERNEL CONTRACT.** Every op's call surface -- `rola.ops.carry.carry_forward`,
`rola.ops.decode.derive_decode_geometry`/`_decode_step`, `rola.ops.intra`'s
`intra_forward_padded` -- receives the caller's logical tensors and widths directly and
pads INTERNALLY, once, before touching `box_shape`, the extension, or any launch-facing
derivation. The kernel itself is never handed a width other than its own; `y` (or
`num`/`o`) is sliced back to the caller's own `d_v` before it leaves the op.

`rola/ops/padding.py`'s three primitives, used identically by all three ops:

* `pad_routes(levels, padded_widths)` -- per level, `[..., b_l] -> [..., B_l]`, the
  trailing columns exact `0.0`.
* `pad_v(v, DV)` -- `[..., d_v] -> [..., DV]`, exact `0.0` tail.
* `slice_y(y, d_v)` -- `[..., DV] -> [..., d_v]`, a view.
* `PaddedShape.of(widths, d_v, shipped_dv=...)` -- the descriptor: `widths`,
  `padded_widths`, `d_v`, `DV`, and `is_padded`. The one place "logical vs padded" is
  computed; every op builds one of these before it touches a tensor.

## Why a zero-pad AFTER the solve is the ruling's mechanism, not an approximation of it

K50 states the mechanism as padding the LOGITS with `-inf` before entmax/softmax. This
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

* **carry** (`rola.ops.prefill`'s `_descriptor`/`_pack_operands`, the operand-packing
  seam onto `rola.ops.carry`'s C1c launch surface) pads BOTH axes: every level to
  `rola.routing.topology.padded_level_width` (next power of two >= 16 -- the
  box/warp-sub-box register-shape floor), and `v` to the smallest of
  `rola.ops.carry.SHIPPED_DV` that admits it. The clean-slate rebuild has since landed a
  real kernel body (`extension().carry_forward`, `docs/internals/carry/carry_kernel.md`), so
  a lawful padded call now reaches a real launch rather than the deleted extension
  symbol's absence -- but only at the one arm the shipped set carries (below): a call
  padded to a `DV` outside it meets the arm-table refusal instead.
* **decode** (`rola.ops.decode.derive_decode_geometry`, `_decode_step`) pads ONLY
  `d_v` -> the smallest `DV` its own shipped arm table
  (`rola.ops.decode.arms`, keyed `(DV, D, decay)`) carries at that `(D, decay)`. Its
  per-level widths carry NO floor of their own -- `derive_lattice`/`box_shape`
  degenerate to `k = m = 1` for an unfactorable width rather than refusing, and the
  existing `D = 4, B = 8` conformance cell (`tests/oracle/test_decode_vs_oracle.py`)
  is proof this binary already runs a width the carry floor would refuse. Padding
  levels here would be importing carry's constraint into a kernel that does not have
  it; an earlier draft of this stage did exactly that and broke that cell, which is
  how this scope line was found.
* **intra** (`rola.ops.intra.intra_forward_padded`) pads MORE aggressively than carry:
  its kernel takes one uniform `level_width` for every level (`intra_forward`'s own
  `width // levels`), so every level pads to the SAME value -- the shipped width at
  `(D, window)` read off `rola.ops.intra.arms()` -- not to its own independent next
  power of two. `v` pads to the single shipped `VALUE_WIDTH`.

## Where the shipped sets come from

`rola.ops.carry.SHIPPED_DV` is the carry's declaration, `(32, 64, 128)`: `DV` is a FREE
AXIS because it is the register shape, and a logical `d_v` outside the set is served by
padding at the value projection, above this boundary, never by a kernel that masks.
Decode's and intra's shipped
sets are read off `arms()` at call time -- the device's own declared matrix -- never a
literal tuple, because a literal would drift the moment either binary's arm table
changed and a call would silently pad to a width the loaded `.so` does not carry.
The carry's own `(D, DV, warps_per_cta)` table now exists and is a ONE-ROW table:
`tools/gen_shards.py`'s `CARRY_ARMS` is `((2, 64, 8),)`, so of the three declared widths
`DV = 64` is the only one a call can actually reach today, and the other two are a
declaration the arm list has not caught up with.

## What this does not do

Nothing here changes `RouteFactors`, `rola.layer.RoLA`, or `rola._state.StateFormat`'s
seven fields (`docs/internals/state.md#binding`). A caller of the public layer/producer
surface does not reach any of this yet: `rola.rola_op`'s CUDA arm is deleted pending
K31 R2, and nothing in `rola.engine`/`rola.layer` reaches `rola.ops.carry` (its own
module docstring says so). This page describes the OP-LEVEL seam the eventual C3
kernel work and the eventual layer-to-op wiring both build on, not a feature reachable
from `rola.RoLA` today.
