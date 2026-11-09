# `carry/carry_api.cuh` — the carry's host entries

Mirrors `csrc/rola/src/carry/carry_api.cuh`, the declarations the operator registration unit declares on the stable
ABI (`csrc/rola/rola_api.cpp`); the Python caller is `rola/ops/carry.py`.

<a id="carry-forward"></a>
## `carry_forward`

One launch over a call's windows. In: both sides' level rows and the values (the bundle, read where it lies), both
sides' lists in the facts pass's final form (`(entries, pre, totals, union, count)` each, empty tensors for a side with no
list level, `carry.md#lists`), the write list exponents, the facts pass's factors and their window shifts (`fm` and
`gains_exp`: read only where the write side has no list level, `ampere/fold.md#units`), the cross-CTA sum's scale bytes a window wide (`scale`, [`#carry-reduce`](#carry-reduce)), and the carve flattened: widths, extents,
channels, each side's list level (-1 for none) and operand levels, the arm key, the readout's accumulation (0
fp32, 1 fp16, -1 the device's rule, [`#carry-readout-accumulate`](#carry-readout-accumulate)), the cross-CTA sum's
dtype (`reduce`: 0 f16 pairs, 1 fp32), the named form every window takes where it builds (-1 none, else one of the 60
in the plan pass's order, `ampere/schedule.md#planner`). The entry runs on its values' device: its card gives the
window, and the launch is guarded onto that device. Out: `num`, `den` and `check` (the caller's), and the plan records,
which the entry allocates at the card's window and returns -- int32 `[BH_w, windows, box classes, plan words]`, read
only by a gate. The refusals are `carry.md#refusals`.

<a id="carry-readout-accumulate"></a>
## `carry_readout_accumulate`

The readout's accumulation on a device, the shared rule's answer
([`../common/arch_runtime.md#readout-f16`](../common/arch_runtime.md#readout-f16)) that each entry also applies itself
when its caller passes -1: fp16 (1) where its fp32-accumulate MMA is slower than its fp16-accumulate one by
its rates -- its product's row by its name, else its architecture's
([`../common/arch_runtime.md#mma-quanta`](../common/arch_runtime.md#mma-quanta)) -- else fp32 (0): a GeForce card's fp32
accumulate runs at half its fp16 rate, a professional or datacenter part's at its full rate (ruled 2026-10-07). The two
forms' numerics differ, GeForce's the coarser; both are gated per slot (`tests/oracle/test_carry_readout.py`).


<a id="carry-reduce"></a>
## The cross-CTA sum: `reduce_dtype`

The CTAs holding a token's read leaves add their partials into one `num` row. Inside a CTA the partials are exact
integers at the CTA's own unit (`ampere/parts.md#units`); across CTAs the sum's dtype is declared (ruled 2026-10-19): F16
PAIRS (`torch.float16`; `num` int32 `[BH, T, d_v / 2]`, two channels a word in the panel's channel order) at each
token's `scale` byte -- a power of two bounding |num(t)|, every CTA's the same, which f16's exponent absorbs with
slack to ~2^27, so a loose bound costs nothing (the scale pass's write-aligned bound, `scale/scale.md`: its slack
2^14 at the layer's gain 16, measured 2026-11-02) -- or FP32 (`torch.float32`; `num` fp32 `[BH, T, d_v]`, no scale read), twice the drain's L2
sectors. The error of the f16 sum grows as ~0.012% x sqrt(n) a token for `n` CTAs adding into a slot (0.095% at 64,
0.41% at 1,024, a cliff past ~2,048: partials under half an ulp), fp32's ~1e-4% at any `n`. AUTO (`None`,
`reduce_dtype_for`): f16 pairs while `reduce_ctas(carve)` -- the read side's boxes along every level -- stays within
`REDUCE_F16_MAX_CTAS` = 177, where the f16 error sits under the bf16 output store's own 0.16% a token (y crosses into
W_O in bf16, `rola/layer.py`); fp32 beyond. Both pass the per-slot gate on every modeled cell
(`tests/oracle/test_carry_vs_oracle.py` runs both). The epilogue reads either (`readout_reference.py`'s `unpack_num`).

<a id="carry"></a>
## `rola.ops.carry.carry`

The Python call returns `(num, den, check, plans)` -- `den` each query head's token's den, fp32 `[BH, T]`, the v+1 channel
(`ampere/handoff.md#den`). It launches over the bundle's rows and values, the facts pass's lists and exponents and the scale bytes
(`scale`; read only for f16 pairs), its cross-CTA sum at `reduce_dtype` or the auto rule's, its readout in the device's accumulation (the entry's rule; `readout_accumulate` reports it) or the one a gate names, each window's plan its
own unless a gate or a measurement forces a form (`forced`, as `forced_word` encodes it), the facts pass run with
this arm's read pool row (`form`, read off the binary's census for the device's card). `plans` are the plan records a
gate reads (`plan_words` a window, `tests/oracle/test_carry_plan.py`). `arm_key` is the compiled body
a carve selects: each side's dense rows a warp and its operand's level count, the channel extent and the publish
field.

`carry_ledger_bind`, `carry_build_stamp`, `carry_census` and `carry_arms` are the instruments: the phase ledger
(`ampere/carry_kernel.md#phases`), the device stamp, and the binary's own arm list and census.

<a id="carry-capture-bind"></a>
## `carry_capture_bind`

Binds a host u8 tensor of `carry_capture_bytes()` bytes; the next launch copies a `CarryCapture` into it -- its byte
count, the grid's boxes and batch-heads, then the operand block as the launch built it -- before it launches. A lab
experiment runs a window form on a real call's inputs from it: the caller saves each tensor the call read or wrote
beside the capture, and the experiment moves every pointer word of the block that falls in a saved tensor to its own
copy. Unbound (`None`) in production; `rola.ops.carry.capture_bind`
and `capture_bytes` are the Python calls.
